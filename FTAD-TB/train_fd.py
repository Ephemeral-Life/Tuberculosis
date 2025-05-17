import os
import json
import random
import time
import torch
import math
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict
from mmcv import Config
from mmcv.utils import Registry, build_from_cfg
from mmdet.datasets import CocoDataset, build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed, single_gpu_test
from pycocotools.coco import COCO
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, Parameters, Scalar
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from mmcv.parallel import MMDataParallel
from coco_classification import CocoClassificationDataset
import gc

import warnings

warnings.filterwarnings("ignore")
import logging

logging.basicConfig(level=logging.WARNING)

# 加载配置
cfg = Config.fromfile('FTAD-TB/configs/symformer/symformer_retinanet_p2t_cls_fpn_1x_TBX11K.py')
cfg.gpu_ids = [0]
num_clients = cfg.num_clients

# 注册数据集
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
DATASETS.register_module(CocoDataset)
DATASETS.register_module(CocoClassificationDataset)

# 构建训练数据集
train_dataset = build_from_cfg(cfg.data.train, DATASETS, default_args=None)
image_ids = train_dataset.coco.getImgIds()
random.shuffle(image_ids)

# 为客户端划分图像ID
num_clients = cfg.num_clients
random.shuffle(image_ids)
partition_size = len(image_ids) // num_clients
partitions = [image_ids[i * partition_size:(i + 1) * partition_size] for i in range(num_clients)]

# 预生成的客户端注解文件路径
client_ann_files = [os.path.join('client_ann', f'client_{partition_id}_ann.json')
                    for partition_id in range(num_clients)]


# Flower客户端类
class FlowerClient(NumPyClient):
    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.net = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        self.ann_file = client_ann_files[partition_id]
        self.trainloader = None
        print(f"客户端 {self.partition_id} 初始化，注解文件: {self.ann_file}")

    def freeze_parameters(self):
        for name, param in self.net.named_parameters():
            if "backbone" in name or "neck" in name or "bbox_head" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def load_data(self):
        client_cfg = dict(
            type='CocoClassificationDataset',
            ann_file=self.ann_file,
            img_prefix=cfg.data.train.img_prefix,
            pipeline=cfg.data.train.pipeline,
            classes=cfg.data.train.classes
        )
        try:
            client_dataset = build_from_cfg(client_cfg, DATASETS)
            print(f"客户端 {self.partition_id} 加载数据集，图像数量: {len(client_dataset)}")
            return client_dataset
        except Exception as e:
            print(f"客户端 {self.partition_id} 加载数据集失败: {e}")
            raise

    def get_parameters(self, config) -> List[np.ndarray]:
        with torch.no_grad():
            params = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        print(f"客户端 {self.partition_id} 返回 {len(params)} 个参数")
        return params

    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = self.net.state_dict()
        params_dict = zip(state_dict.keys(), parameters)
        new_state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        try:
            self.net.load_state_dict(new_state_dict, strict=True)
            print(f"客户端 {self.partition_id} 参数设置完成")
        except Exception as e:
            print(f"客户端 {self.partition_id} 参数设置失败: {e}")
            raise

    def fit(self, parameters: List[np.ndarray], config) -> Tuple[List[np.ndarray], int, dict]:
        self.set_parameters(parameters)
        seed = cfg.get('seed', None)
        if seed is not None:
            set_random_seed(seed)

        self.trainloader = self.load_data()
        server_round = config.get("server_round", 1)
        print(f"客户端 {self.partition_id} 训练（轮次 {server_round}）...")

        self.freeze_parameters()
        original_load_from = cfg.get('load_from', None)
        cfg.load_from = None

        # 获取全局参数（训练前的参数）
        global_params = [torch.from_numpy(np.copy(p)).to('cuda' if torch.cuda.is_available() else 'cpu')
                         for p in parameters]
        global_state_dict = OrderedDict(zip(self.net.state_dict().keys(), global_params))

        try:
            train_detector(
                self.net,
                [self.trainloader],
                cfg,
                distributed=False,
                validate=False,
                timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
                meta={'seed': seed}
            )
            print(f"客户端 {self.partition_id} 训练完成")
        except Exception as e:
            print(f"客户端 {self.partition_id} 训练失败: {e}")
            raise
        finally:
            cfg.load_from = original_load_from

        # 计算更新差异
        local_params = self.get_parameters(config)
        local_state_dict = self.net.state_dict()
        update_diff = {key: global_state_dict[key] - local_state_dict[key]
                       for key in global_state_dict}

        # 生成一致性掩码（简单实现，假设服务器维护历史）
        # 这里仅返回更新差异和参数，掩码逻辑移到服务器端
        masked_update = update_diff  # 掩码在服务器端应用

        # 计算欧几里得距离
        euclidean_distance = 0
        for key in masked_update:
            euclidean_distance += torch.norm(masked_update[key]).item()

        num_examples = len(self.trainloader)
        metrics = {"euclidean_distance": euclidean_distance}

        # 将 masked_update 转换为 numpy 数组并返回
        masked_update_list = [val.cpu().numpy() for val in masked_update.values()]

        del self.trainloader
        self.trainloader = None
        gc.collect()

        return local_params, num_examples, metrics


# 客户端工厂函数
def client_fn(context: Context) -> Client:
    partition_id = context.node_config["partition-id"]
    return FlowerClient(partition_id).to_client()


# 创建ClientApp
client = ClientApp(client_fn=client_fn)


# 权重平均用于指标
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples) if sum(examples) > 0 else 0.0}


# 自定义FedAvg策略（改为FedAvGHEAL）
class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        self.model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        self.start_round = 1

        # FedAvGHEAL 特定属性
        self.client_update = {}
        self.increase_history = {}
        self.mask_dict = {}
        self.euclidean_distance = {}
        self.previous_weights = {}
        self.previous_delta_weights = {}
        self.threshold = 0.5  # 可调参数
        self.beta = 0.1  # 可调参数

        if cfg.get('load_from', None):
            try:
                checkpoint = torch.load(cfg.load_from)
                state_dict = checkpoint.get('state_dict', checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"服务器加载初始模型: {cfg.load_from}")
            except Exception as e:
                print(f"服务器加载初始模型失败: {e}")
                raise
        else:
            print("未指定cfg.load_from，使用随机初始化的模型")
            self.model.init_weights()

        try:
            model_files = [f for f in os.listdir(cfg.work_dir) if
                           f.startswith("aggregated_model_round_") and f.endswith(".pth")]
        except FileNotFoundError:
            model_files = None
        if model_files:
            rounds = [int(f.split('_')[-1].split('.')[0]) for f in model_files]
            latest_round = max(rounds)
            latest_model_path = os.path.join(cfg.work_dir, f"aggregated_model_round_{latest_round}.pth")
            print(f"加载最新的聚合模型: {latest_model_path}")
            checkpoint = torch.load(latest_model_path)
            state_dict = checkpoint.get('state_dict', checkpoint)
            self.model.load_state_dict(state_dict, strict=True)
            self.start_round = latest_round + 1

        initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for val in self.model.state_dict().values()])
        super().__init__(*args, initial_parameters=initial_parameters, **kwargs)

        val_cfg = cfg.data.val
        val_cfg['type'] = 'CocoClassificationDataset'
        try:
            self.val_dataset = build_dataset(val_cfg)
            self.val_dataloader = build_dataloader(
                self.val_dataset,
                samples_per_gpu=cfg.data.samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=False,
                shuffle=False
            )
            print("服务器端加载评估数据集完成")
        except Exception as e:
            print(f"服务器端加载评估数据集失败: {e}")
            raise

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        for instruction in client_instructions:
            instruction[1].config["server_round"] = server_round
        return client_instructions

    def consistency_mask(self, client_id, update_diff, server_round):
        updates = update_diff
        device = next(iter(updates.values())).device
        if client_id not in self.increase_history or server_round == 1:
            self.increase_history[client_id] = {key: torch.zeros_like(val, device=device)
                                                for key, val in updates.items()}
            for key in updates:
                self.increase_history[client_id][key] = (updates[key] >= 0).float()
            return {key: torch.ones_like(val, device=device) for key, val in updates.items()}

        mask = {}
        for key in updates:
            positive_consistency = self.increase_history[client_id][key]
            negative_consistency = 1 - positive_consistency
            consistency = torch.where(updates[key] >= 0, positive_consistency, negative_consistency)
            mask[key] = (consistency > self.threshold).float()

        for key in updates:
            increase = (updates[key] >= 0).float()
            self.increase_history[client_id][key] = (self.increase_history[client_id][key] * (
                        server_round - 1) + increase) / server_round

        return mask

    def compute_distance(self, client_id, update_diff):
        euclidean_distance = 0
        for key in update_diff:
            euclidean_distance += torch.norm(update_diff[key]).item()
        self.euclidean_distance[client_id] = euclidean_distance

    def get_params_diff_weights(self, online_clients):
        weight_dict = {}
        total_distance = sum(self.euclidean_distance.values()) or 1e-10  # 防止除零
        online_num = len(online_clients)

        for client in online_clients:
            client_distance = self.euclidean_distance.get(client, 0)
            delta_weight = (1 - self.beta) * self.previous_delta_weights.get(client, 0) + \
                           self.beta * (client_distance / total_distance)
            new_weight = self.previous_weights.get(client, 1 / online_num) + delta_weight
            weight_dict[client] = max(new_weight, 0)  # 确保权重非负

            self.previous_weights[client] = weight_dict[client]
            self.previous_delta_weights[client] = delta_weight

        total_weight = sum(weight_dict.values()) or 1e-10  # 防止除零
        for client in online_clients:
            weight_dict[client] /= total_weight

        return weight_dict

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"服务器轮次 {server_round}: 聚合 {len(results)} 个结果, {len(failures)} 个失败")
        if not results or failures:
            print(f"轮次 {server_round} 无有效结果或存在失败")
            return None, {}

        # 获取全局模型参数
        global_state_dict = self.model.state_dict()
        device = next(iter(global_state_dict.values())).device

        # 收集客户端更新
        online_clients = []
        for client_proxy, fit_res in results:
            client_id = str(client_proxy)
            online_clients.append(client_id)

            # 解析客户端参数
            local_params = parameters_to_ndarrays(fit_res.parameters)
            local_state_dict = OrderedDict({
                k: torch.from_numpy(np.copy(v)).to(device)
                for k, v in zip(global_state_dict.keys(), local_params)
            })

            # 计算更新差异
            update_diff = {key: global_state_dict[key] - local_state_dict[key]
                           for key in global_state_dict}
            self.client_update[client_id] = update_diff

            # 生成一致性掩码
            mask = self.consistency_mask(client_id, update_diff, server_round)
            self.mask_dict[client_id] = mask

            # 应用掩码
            masked_update = {key: update_diff[key] * mask[key]
                             for key in update_diff}
            self.client_update[client_id] = masked_update

            # 计算距离
            self.compute_distance(client_id, masked_update)

            # 从 metrics 中获取客户端计算的距离（可选验证）
            client_distance = fit_res.metrics.get("euclidean_distance", 0)
            print(f"客户端 {client_id} 距离: {client_distance}")

        # 计算聚合权重
        freq = self.get_params_diff_weights(online_clients)

        # 聚合参数
        global_params_new = {key: torch.zeros_like(val, device=device)
                             for key, val in global_state_dict.items()}
        for client_id in online_clients:
            weight = freq[client_id]
            for key in global_params_new:
                global_params_new[key] += self.client_update[client_id][key] * weight

        # 更新全局模型
        for key in global_state_dict:
            global_state_dict[key] -= global_params_new[key]
        self.model.load_state_dict(global_state_dict, strict=True)

        # 保存聚合模型
        actual_round = self.start_round + server_round - 1
        model_path = os.path.join(cfg.work_dir, f"aggregated_model_round_{actual_round}.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"保存聚合模型到 {model_path}")

        # 转换为 Flower 参数格式
        aggregated_parameters = ndarrays_to_parameters([val.cpu().numpy()
                                                        for val in global_state_dict.values()])

        gc.collect()
        return aggregated_parameters, {}

    def evaluate_aggregated_model(self):
        print("开始评估聚合模型...")
        self.model.eval()
        try:
            model = MMDataParallel(self.model, device_ids=[0])
            outputs = single_gpu_test(model, self.val_dataloader)

            preds = []
            gts = []
            for i, output in enumerate(outputs):
                pred = torch.argmax(torch.tensor(output)).item()
                gt = self.val_dataset[i]['ann_info']['labels'][0]
                preds.append(pred)
                gts.append(gt)

            accuracy = sum(1 for p, g in zip(preds, gts) if p == g) / len(gts)
            print(f"聚合模型评估完成，准确率: {accuracy:.4f}")
        except Exception as e:
            print(f"评估聚合模型失败: {e}")
            raise


# 服务器工厂函数
def server_fn(context: Context) -> ServerAppComponents:
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=math.ceil(num_clients / 2),
        min_evaluate_clients=math.ceil(num_clients / 2),
        min_available_clients=math.ceil(num_clients / 2),
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    start_round = strategy.start_round if hasattr(strategy, 'start_round') else 1
    num_rounds = cfg.num_rounds
    if start_round > 1:
        num_rounds -= (start_round - 1)
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# 客户端资源
backend_config = {"client_resources": {"num_cpus": 1.0, "num_gpus": 0.0}}
if torch.cuda.is_available():
    backend_config["client_resources"] = {"num_cpus": 1.0, "num_gpus": 1.0}

# 运行模拟
if __name__ == "__main__":
    print("开始模拟...")
    run_simulation(
        server_app=ServerApp(server_fn=server_fn),
        client_app=client,
        num_supernodes=num_clients,
        backend_config=backend_config,
    )
