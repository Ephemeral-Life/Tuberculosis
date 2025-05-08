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
total_rounds = cfg.num_rounds

lambda_0 = 0.05
momentum = 0.9
initial_lr = 0.001
decay_rate = 0.1
momentum_coefficient = 0.9

# 注册数据集
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
DATASETS.register_module(CocoDataset)
DATASETS.register_module(CocoClassificationDataset)

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
        self.tb_ratio = None  # TB图像占比
        self.image_size = None  # 图像尺寸
        self.calculate_local_stats()  # 初始化时计算本地统计信息
        print(f"客户端 {self.partition_id} 初始化，注解文件: {self.ann_file}")

    def calculate_local_stats(self):
        """计算本地数据的TB图像占比和图像尺寸"""
        # 加载注解文件
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)

        # 统计TB图像数量（根据file_name中是否包含'tb'判断）
        tb_images = [img for img in coco_data['images'] if 'tb' in img['file_name'].lower()]
        total_images = len(coco_data['images'])
        self.tb_ratio = len(tb_images) / total_images if total_images > 0 else 0.0

        # 获取图像尺寸（假设所有图像尺寸相同，取第一张图像的尺寸）
        if total_images > 0:
            self.image_size = (coco_data['images'][0]['width'], coco_data['images'][0]['height'])
        else:
            self.image_size = (0, 0)

        print(f"客户端 {self.partition_id}: TB图像占比 {self.tb_ratio:.4f}, 图像尺寸 {self.image_size}")

    def freeze_parameters(self):
        """冻结部分参数，只训练特定层"""
        for name, param in self.net.named_parameters():
            if "backbone" in name or "neck" in name or "bbox_head" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def load_data(self):
        """加载客户端本地数据集"""
        client_cfg = dict(
            type='CocoClassificationDataset',
            ann_file=self.ann_file,
            img_prefix=cfg.data.train.img_prefix,
            pipeline=cfg.data.train.pipeline,
            classes=cfg.data.train.classes
        )
        try:
            # 添加统一预处理
            target_size = (512, 512)
            client_cfg['pipeline'].append(dict(type='Resize', size=target_size))
            client_dataset = build_from_cfg(client_cfg, DATASETS)
            print(f"客户端 {self.partition_id} 加载数据集，图像数量: {len(client_dataset)}")
            return client_dataset
        except Exception as e:
            print(f"客户端 {self.partition_id} 加载数据集失败: {e}")
            raise

    def get_parameters(self, config) -> List[np.ndarray]:
        """获取模型参数"""
        with torch.no_grad():
            params = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        print(f"客户端 {self.partition_id} 返回 {len(params)} 个参数")
        return params

    def set_parameters(self, parameters: List[np.ndarray]):
        """设置模型参数"""
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
        """客户端本地训练并返回参数、样本数和统计信息"""
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

        params = self.get_parameters(config)
        num_examples = len(self.trainloader)
        del self.trainloader
        self.trainloader = None
        gc.collect()

        # 返回本地统计信息
        metrics = {
            "tb_ratio": self.tb_ratio,
            "image_width": self.image_size[0],
            "image_height": self.image_size[1]
        }
        return params, num_examples, metrics


# 客户端工厂函数
def client_fn(context: Context) -> Client:
    partition_id = context.node_config["partition-id"]
    return FlowerClient(partition_id).to_client()


# 创建ClientApp
client = ClientApp(client_fn=client_fn)


# 权重平均用于指标聚合
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples) if sum(examples) > 0 else 0.0}


# 自定义FedAvg策略
class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        self.model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        self.start_round = 1
        self.client_stats = {}  # 存储客户端统计信息
        self.momentum = None  # 服务器动量向量，初始为None，稍后初始化为全零

        # 加载初始模型
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

        # 检查是否存在已保存的聚合模型
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

        # 加载验证数据集
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
        """配置客户端训练"""
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        for instruction in client_instructions:
            instruction[1].config["server_round"] = server_round
        return client_instructions

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"服务器轮次 {server_round}: 聚合 {len(results)} 个结果, {len(failures)} 个失败")
        if not results:
            print("没有客户端结果可供聚合")
            return None, {}

        # 获取当前全局模型参数
        global_state_dict = self.model.state_dict()
        global_params = list(global_state_dict.values())

        # 存储客户端参数差异和统计信息
        client_diffs = []
        client_stats = []
        total_weighted_samples = 0.0
        total_weighted_tb = 0.0
        total_samples = 0

        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            client_params = parameters_to_ndarrays(fit_res.parameters)
            # 确保客户端参数与全局参数对齐
            if len(client_params) != len(global_params):
                raise ValueError(
                    f"客户端 {client_id} 参数数量不匹配: {len(client_params)} vs {len(global_params)}")
            diff = []
            for client_p, global_p in zip(client_params, global_params):
                client_tensor = torch.from_numpy(client_p).to('cpu')
                if client_tensor.shape != global_p.shape:
                    raise ValueError(
                        f"客户端 {client_id} 参数形状不匹配: {client_tensor.shape} vs {global_p.shape}")
                diff.append(client_tensor - global_p)
            client_diffs.append(diff)

            # 收集客户端统计信息
            num_examples = fit_res.num_examples
            tb_ratio = fit_res.metrics.get("tb_ratio", 0.0)
            image_width = fit_res.metrics.get("image_width", 0)
            image_height = fit_res.metrics.get("image_height", 0)

            # 根据图像尺寸映射质量权重
            if image_width >= 1024 and image_height >= 1024:
                quality_weight = 1.2  # 大尺寸
            elif image_width >= 512 and image_height >= 512:
                quality_weight = 1.1  # 中尺寸
            else:
                quality_weight = 1  # 小尺寸

            # 计算加权样本数
            weighted_samples = num_examples * quality_weight
            total_weighted_samples += weighted_samples
            total_samples += num_examples
            total_weighted_tb += tb_ratio * num_examples

            client_stats.append({
                "num_examples": num_examples,
                "quality_weight": quality_weight,
                "weighted_samples": weighted_samples,
                "tb_ratio": tb_ratio
            })

        # 计算全局TB占比
        global_tb_ratio = total_weighted_tb / total_samples if total_samples > 0 else 0.0
        print(f"全局病灶占比: {global_tb_ratio:.4f}")

        # 计算惩罚系数
        # lambda_0 = 1.0  # 基础惩罚强度，可调initial_lambda_0
        initial_lambda_0 = 0.05
        # lambda_0 = initial_lambda_0 * (1 + server_round / total_rounds)
        # lambda_penalty = lambda_0 * (1 - global_tb_ratio)
        lambda_0 = initial_lambda_0  # 保持常数
        lambda_penalty = lambda_0 * (1 - global_tb_ratio)
        print(f"惩罚系数 λ: {lambda_penalty:.4f}")

        # 计算初始客户端权重
        initial_weights = [stats["weighted_samples"] / total_weighted_samples for stats in client_stats]

        # 计算客户端偏差（参数差异的欧几里得范数）
        client_deviations = []
        for diff in client_diffs:
            # deviation = sum(torch.norm(d).item() for d in diff)
            deviation = sum(
                torch.norm(d).item() / (torch.norm(global_p).item() + 1e-8) for d, global_p in zip(diff, global_params))
            client_deviations.append(deviation)

        # 应用指数惩罚并归一化权重
        adjusted_weights = []
        for initial_weight, deviation in zip(initial_weights, client_deviations):
            # penalty = math.exp(-lambda_penalty * deviation)
            penalty = 1 / (1 + lambda_penalty * deviation)
            adjusted_weight = initial_weight * penalty
            adjusted_weights.append(adjusted_weight)

        # 归一化最终权重
        total_adjusted_weight = sum(adjusted_weights)
        final_weights = [w / total_adjusted_weight for w in adjusted_weights]

        # 计算本轮聚合更新向量
        aggregated_diff = [torch.zeros_like(p) for p in global_params]
        for diff, weight in zip(client_diffs, final_weights):
            for i, d in enumerate(diff):
                aggregated_diff[i] += weight * d

        # 自适应调整动量系数
        base_momentum = 0.8  # 基础动量系数，范围 0.8-0.99
        momentum_factor = 0.1  # 动量调整因子，范围 0.1-0.5
        # momentum_coefficient = base_momentum + momentum_factor * global_tb_ratio
        # momentum_coefficient = min(max(momentum_coefficient, 0.8), 0.99)  # 限制在 0.8-0.99 之间
        momentum_coefficient = 0.9 * (1 - 0.5 ** (server_round / 10))
        print(f"本轮动量系数: {momentum_coefficient:.4f}")

        # 初始化或更新服务器动量向量
        if self.momentum is None:
            self.momentum = [torch.zeros_like(p) for p in global_params]
        self.momentum = [
            momentum_coefficient * m + (1 - momentum_coefficient) * agg_d
            for m, agg_d in zip(self.momentum, aggregated_diff)
        ]

        # 服务器端学习率
        # server_lr = 0.005  # 范围 0.0005-0.005，默认 0.001
        server_lr = initial_lr * (1 - decay_rate) ** server_round
        # 使用动量更新全局模型参数
        new_global_params = [global_p + server_lr * m for global_p, m in zip(global_params, self.momentum)]

        # 更新全局模型的状态字典
        for key, param in zip(global_state_dict.keys(), new_global_params):
            global_state_dict[key] = param
        self.model.load_state_dict(global_state_dict)

        # 保存聚合后的模型参数
        actual_round = self.start_round + server_round - 1
        model_path = os.path.join(cfg.work_dir, f"aggregated_model_round_{actual_round}.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"保存聚合模型到 {model_path}")

        # 更新下一轮的初始参数
        self.initial_parameters = ndarrays_to_parameters([p.cpu().numpy() for p in new_global_params])

        gc.collect()
        return self.initial_parameters, {}


# 服务器工厂函数
def server_fn(context: Context) -> ServerAppComponents:
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=math.ceil(num_clients),
        min_evaluate_clients=math.ceil(num_clients),
        min_available_clients=math.ceil(num_clients),
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    start_round = strategy.start_round if hasattr(strategy, 'start_round') else 1
    num_rounds = cfg.num_rounds
    if start_round > 1:
        num_rounds -= (start_round - 1)
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# 客户端资源配置
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
