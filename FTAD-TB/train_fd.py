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
from mmdet.datasets import CocoDataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed
from pycocotools.coco import COCO
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, Parameters, Scalar
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import gc

import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.WARNING)

# 加载配置
cfg = Config.fromfile('FTAD-TB/configs/symformer/symformer_retinanet_p2t_cls_fpn_1x_TBX11K.py')
cfg.runner = dict(type='EpochBasedRunner', max_epochs=cfg.max_epochs)
cfg.gpu_ids = [0]
cfg.data.samples_per_gpu = 4
cfg.optimizer_config = dict(grad_clip=None)

# 注册数据集和管道
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
DATASETS.register_module(CocoDataset)

# 构建训练数据集
train_dataset = build_from_cfg(cfg.data.train, DATASETS, default_args=None)
image_ids = train_dataset.coco.getImgIds()
random.shuffle(image_ids)

# 为客户端划分图像ID
num_clients = cfg.num_clients
random.shuffle(image_ids)
partition_size = len(image_ids) // num_clients
partitions = [image_ids[i * partition_size:(i + 1) * partition_size] for i in range(num_clients)]

# 临时目录用于客户端注解
temp_dir = 'temp_client_ann'
os.makedirs(temp_dir, exist_ok=True)

# 预计算客户端注解文件（在循环外加载COCO annotation）
client_ann_files = []
coco = COCO(cfg.data.train.ann_file)  # 只加载一次COCO annotation文件
for partition_id in range(num_clients):
    client_image_ids = partitions[partition_id]
    client_imgs = [img for img in coco.imgs.values() if img['id'] in client_image_ids]
    client_ann_ids = coco.getAnnIds(imgIds=client_image_ids)
    client_anns = [coco.anns[ann_id] for ann_id in client_ann_ids]
    client_coco = {
        'images': client_imgs,
        'annotations': client_anns,
        'categories': coco.dataset['categories'],
        'info': coco.dataset.get('info', {}),
        'licenses': coco.dataset.get('licenses', [])
    }
    temp_ann_file = os.path.join(temp_dir, f'client_{partition_id}_ann.json')
    with open(temp_ann_file, 'w') as f:
        json.dump(client_coco, f)
    client_ann_files.append(temp_ann_file)
del coco  # 释放COCO对象
gc.collect()

# Flower客户端类
class FlowerClient(NumPyClient):
    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.net = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        # 移除init_weights()调用，确保从服务器参数开始
        self.ann_file = client_ann_files[partition_id]  # 使用预计算的注解文件
        self.trainloader = None  # 延迟加载数据集
        print(f"客户端 {self.partition_id} 初始化，注解文件: {self.ann_file}")

    def freeze_parameters(self):
        for name, param in self.net.named_parameters():
            if "backbone" in name or "neck" in name or "bbox_head" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def load_data(self):
        client_cfg = dict(
            type='CocoDataset',
            ann_file=self.ann_file,
            img_prefix=cfg.data.train.img_prefix,
            pipeline=cfg.data.train.pipeline,
            filter_empty_gt=cfg.data.train.filter_empty_gt,
            classes=cfg.data.train.classes
        )
        try:
            client_dataset = build_from_cfg(client_cfg, DATASETS, default_args=None)
            print(f"客户端 {self.partition_id} 加载数据集，图像数量: {len(client_dataset)}")
            return client_dataset
        except Exception as e:
            print(f"客户端 {self.partition_id} 加载数据集失败: {e}")
            raise

    def get_parameters(self, config) -> List[np.ndarray]:
        with torch.no_grad():  # 减少内存使用
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
        self.set_parameters(parameters)  # 在训练前设置参数
        seed = cfg.get('seed', None)
        if seed is not None:
            set_random_seed(seed)

        # 加载数据集
        self.trainloader = self.load_data()
        server_round = config.get("server_round", 1)
        print(f"客户端 {self.partition_id} 训练（轮次 {server_round}）...")

        # Freeze backbone and detection head parameters
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
        num_examples = len(self.trainloader)  # 表示图像数量，而非批次数量
        # 释放数据集
        del self.trainloader
        self.trainloader = None
        gc.collect()

        return params, num_examples, {}

    def evaluate(self, parameters: List[np.ndarray], config) -> Tuple[float, int, dict]:
        self.set_parameters(parameters)
        self.trainloader = self.load_data()
        num_examples = len(self.trainloader)  # 表示图像数量，而非批次数量
        del self.trainloader
        self.trainloader = None
        gc.collect()
        # TODO: 实现实际的评估逻辑
        return 0.0, num_examples, {"accuracy": 0.0}

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

# 自定义FedAvg策略
class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        self.model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        self.start_round = 1

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

        # 将模型参数传递给initial_parameters
        initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for val in self.model.state_dict().values()])
        super().__init__(*args, initial_parameters=initial_parameters, **kwargs)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
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
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            try:
                aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
                params_dict = zip(self.model.state_dict().keys(),
                                  [torch.from_numpy(np.copy(p)) for p in aggregated_ndarrays])
                state_dict = OrderedDict({k: v for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=True)
                actual_round = self.start_round + server_round - 1
                model_path = os.path.join(cfg.work_dir, f"aggregated_model_round_{actual_round}.pth")
                torch.save(self.model.state_dict(), model_path)
                print(f"保存聚合模型到 {model_path}")
            except Exception as e:
                print(f"保存聚合模型失败: {e}")
        else:
            print(f"轮次 {server_round} 没有聚合参数, 结果: {len(results)}, 失败: {len(failures)}")

        gc.collect()  # 聚合后清理
        return aggregated_parameters, metrics

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
