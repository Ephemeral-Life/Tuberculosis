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
        """聚合客户端训练结果并处理统计信息"""
        print(f"服务器轮次 {server_round}: 聚合 {len(results)} 个结果, {len(failures)} 个失败")
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            try:
                # 更新服务器模型参数
                aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
                params_dict = zip(self.model.state_dict().keys(),
                                  [torch.from_numpy(np.copy(p)) for p in aggregated_ndarrays])
                state_dict = OrderedDict({k: v for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=True)

                # 保存聚合模型
                actual_round = self.start_round + server_round - 1
                model_path = os.path.join(cfg.work_dir, f"aggregated_model_round_{actual_round}.pth")
                torch.save(self.model.state_dict(), model_path)
                print(f"保存聚合模型到 {model_path}")

                # 处理并打印客户端统计信息
                for client_proxy, fit_res in results:
                    client_id = client_proxy.cid
                    tb_ratio = fit_res.metrics.get("tb_ratio", 0.0)
                    image_width = fit_res.metrics.get("image_width", 0)
                    image_height = fit_res.metrics.get("image_height", 0)
                    print(f"\n服务器信息：客户端 {client_id}: TB图像占比 {tb_ratio:.4f}, 图像尺寸 ({image_width}, {image_height})\n")
                    # 存储统计信息以供后续使用
                    self.client_stats[client_id] = {
                        "tb_ratio": tb_ratio,
                        "image_size": (image_width, image_height)
                    }

                # 可选：评估聚合模型
                # self.evaluate_aggregated_model()
            except Exception as e:
                print(f"聚合或评估模型失败: {e}")
        else:
            print(f"轮次 {server_round} 没有聚合参数, 结果: {len(results)}, 失败: {len(failures)}")

        gc.collect()
        return aggregated_parameters, metrics

    def evaluate_aggregated_model(self):
        """评估聚合模型（当前未启用）"""
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
        min_fit_clients=math.ceil(3),
        min_evaluate_clients=math.ceil(3),
        min_available_clients=math.ceil(3),
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
