import os
import json
import random
import time
import torch
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

# 加载配置文件
cfg = Config.fromfile('FTAD-TB/configs/symformer/symformer_retinanet_p2t_cls_fpn_1x_TBX11K.py')

# 设置训练轮数为 1 以快速验证
cfg.runner = dict(type='EpochBasedRunner', max_epochs=1)

# 设置 GPU ID（与原项目一致）
cfg.gpu_ids = [0]

# 减小批次大小以节省内存
cfg.data.samples_per_gpu = 2  # 从默认值减小到 2

# 设置 optimizer_config，仅保留 grad_clip
cfg.optimizer_config = dict(grad_clip=None)

# 注册 DATASETS 和 PIPELINES
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
DATASETS.register_module(CocoDataset)

# 构建完整训练数据集
train_dataset = build_from_cfg(cfg.data.train, DATASETS, default_args=None)

# 获取所有图像的 ID
image_ids = train_dataset.coco.getImgIds()

# 打乱图像 ID，确保随机性
random.shuffle(image_ids)

# 将图像 ID 分割为子集
num_clients = 2
partition_size = len(image_ids) // num_clients
partitions = [image_ids[i * partition_size:(i + 1) * partition_size] for i in range(num_clients)]

# 创建临时目录存储客户端标注文件
temp_dir = 'temp_client_ann'
os.makedirs(temp_dir, exist_ok=True)

# 定义一个函数，为每个客户端生成临时的标注文件
def create_client_ann_file(partition_id: int, client_image_ids: List[int]) -> str:
    coco = COCO(cfg.data.train.ann_file)
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
    return temp_ann_file

# 定义 Flower 客户端类
class FlowerClient(NumPyClient):
    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        # 构建模型（与原项目一致，不手动包装）
        self.net = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        self.net.init_weights()
        # 加载客户端数据集
        self.trainloader = self.load_data()
        print(f"Client {self.partition_id} initialized with dataset size: {len(self.trainloader)}")

    def load_data(self):
        client_image_ids = partitions[self.partition_id]
        client_ann_file = create_client_ann_file(self.partition_id, client_image_ids)
        client_cfg = dict(
            type='CocoDataset',
            ann_file=client_ann_file,
            img_prefix=cfg.data.train.img_prefix,
            pipeline=cfg.data.train.pipeline,
            filter_empty_gt=cfg.data.train.filter_empty_gt,
            classes=cfg.data.train.classes
        )
        try:
            client_dataset = build_from_cfg(client_cfg, DATASETS, default_args=None)
            print(f"Client {self.partition_id} loaded dataset with {len(client_dataset)} images")
            return client_dataset
        except Exception as e:
            print(f"Client {self.partition_id} failed to load dataset: {e}")
            raise

    def get_parameters(self, config) -> List[np.ndarray]:
        params = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        print(f"Client {self.partition_id} returning {len(params)} parameters")
        return params

    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = self.net.state_dict()
        params_dict = zip(state_dict.keys(), parameters)
        new_state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        try:
            self.net.load_state_dict(new_state_dict, strict=True)
            print(f"Client {self.partition_id} parameters set successfully")
        except Exception as e:
            print(f"Client {self.partition_id} failed to set parameters: {e}")
            raise

    def fit(self, parameters: List[np.ndarray], config) -> Tuple[List[np.ndarray], int, dict]:
        self.set_parameters(parameters)
        seed = cfg.get('seed', None)
        if seed is not None:
            set_random_seed(seed)
        try:
            print(f"Client {self.partition_id} starting training...")
            # 模仿原项目调用 train_detector
            train_detector(
                self.net,
                [self.trainloader],
                cfg,
                distributed=False,
                validate=False,
                timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
                meta={'seed': seed}
            )
            print(f"Client {self.partition_id} training completed")
        except Exception as e:
            print(f"Client {self.partition_id} training failed: {e}")
            raise
        params = self.get_parameters(config)
        return params, len(self.trainloader), {}

    def evaluate(self, parameters: List[np.ndarray], config) -> Tuple[float, int, dict]:
        self.set_parameters(parameters)
        return 0.0, len(self.trainloader), {"accuracy": 0.0}

# 定义 client_fn
def client_fn(context: Context) -> Client:
    partition_id = context.node_config["partition-id"]
    return FlowerClient(partition_id).to_client()

# 创建 ClientApp
client = ClientApp(client_fn=client_fn)

# 定义 weighted_average 函数
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples) if sum(examples) > 0 else 0.0}

# 自定义 FedAvg 策略以保存聚合模型
class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"Server round {server_round}: aggregating {len(results)} results, {len(failures)} failures")
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            try:
                params_dict = zip(self.model.state_dict().keys(), [torch.tensor(p) for p in aggregated_parameters.tensors])
                state_dict = OrderedDict({k: v for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=True)
                model_path = f"aggregated_model_round_{server_round}.pth"
                torch.save(self.model.state_dict(), model_path)
                print(f"Saved aggregated model to {model_path}")
            except Exception as e:
                print(f"Error saving aggregated model: {e}")
        else:
            print(f"No aggregated parameters for round {server_round}, results: {len(results)}, failures: {len(failures)}")
        return aggregated_parameters, metrics

# 定义 server_fn
def server_fn(context: Context) -> ServerAppComponents:
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=1)
    return ServerAppComponents(strategy=strategy, config=config)

# 指定客户端资源
backend_config = {"client_resources": {"num_cpus": 1.0, "num_gpus": 0.0}}
if torch.cuda.is_available():
    backend_config["client_resources"] = {"num_cpus": 1.0, "num_gpus": 1.0}

# 运行模拟
if __name__ == "__main__":
    print("Starting simulation...")
    run_simulation(
        server_app=ServerApp(server_fn=server_fn),
        client_app=client,
        num_supernodes=num_clients,
        backend_config=backend_config,
    )
