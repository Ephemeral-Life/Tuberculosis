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
cfg = Config.fromfile('FTAD-TB\\configs\\symformer\\symformer_retinanet_p2t_cls_fpn_1x_TBX11K.py')

# 注册 DATASETS 和 PIPELINES
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
DATASETS.register_module(CocoDataset)  # 注册 CocoDataset

# 构建完整训练数据集
train_dataset = build_from_cfg(cfg.data.train, DATASETS, default_args=None)

# 获取所有图像的 ID
image_ids = train_dataset.coco.getImgIds()

# 打乱图像 ID，确保随机性
random.shuffle(image_ids)

# 将图像 ID 分割为子集
num_clients = 5
partition_size = len(image_ids) // num_clients
partitions = [image_ids[i * partition_size:(i + 1) * partition_size] for i in range(num_clients)]

# 创建临时目录存储客户端标注文件
temp_dir = 'temp_client_ann'
os.makedirs(temp_dir, exist_ok=True)

# 定义一个函数，为每个客户端生成临时的标注文件
def create_client_ann_file(partition_id: int, client_image_ids: List[int]) -> str:
    """为客户端生成临时的标注文件（JSON）。

    Args:
        partition_id (int): 客户端的分区编号 (0-9)。
        client_image_ids (list): 客户端的图像 ID 列表。

    Returns:
        str: 临时标注文件的路径。
    """
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
        """初始化 Flower 客户端。

        Args:
            partition_id (int): 客户端的分区编号 (0-9)。
        """
        self.partition_id = partition_id
        # 构建模型
        self.net = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        self.net.init_weights()
        # 加载客户端数据集
        self.trainloader = self.load_data()

    def load_data(self):
        """加载客户端的本地数据集。

        Returns:
            dataset: 客户端的本地数据集。
        """
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
        client_dataset = build_from_cfg(client_cfg, DATASETS, default_args=None)
        return client_dataset

    def get_parameters(self, config) -> List[np.ndarray]:
        """获取模型的当前参数。

        Returns:
            List[np.ndarray]: 模型参数的 NumPy 数组列表。
        """
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """设置模型参数。

        Args:
            parameters (List[np.ndarray]): 来自服务器的模型参数。
        """
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config) -> Tuple[List[np.ndarray], int, dict]:
        """训练模型。

        Args:
            parameters (List[np.ndarray]): 来自服务器的初始模型参数。
            config: 训练配置。

        Returns:
            Tuple: 更新后的参数、样本数和附加信息。
        """
        self.set_parameters(parameters)
        # 设置随机种子
        seed = cfg.get('seed', 42)
        set_random_seed(seed)
        # 指定设备为单 GPU 或 CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = self.net.to(device)
        # 训练模型（移除 device 参数）
        train_detector(
            self.net,
            [self.trainloader],
            cfg,
            distributed=False,
            validate=False,  # 在客户端不进行验证
            timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            meta={'seed': seed}
        )
        return self.get_parameters(config), len(self.trainloader), {}

    def evaluate(self, parameters: List[np.ndarray], config) -> Tuple[float, int, dict]:
        """评估模型（占位符）。

        Args:
            parameters (List[np.ndarray]): 来自服务器的模型参数。
            config: 评估配置。

        Returns:
            Tuple: 损失、样本数和评估指标。
        """
        self.set_parameters(parameters)
        # 注意：此处为占位符，您可根据需要添加实际评估逻辑
        return 0.0, len(self.trainloader), {"accuracy": 0.0}

# 定义 client_fn
def client_fn(context: Context) -> Client:
    """创建 Flower 客户端实例。

    Args:
        context (Context): Flower 上下文，包含节点配置。

    Returns:
        Client: Flower 客户端实例。
    """
    partition_id = context.node_config["partition-id"]
    return FlowerClient(partition_id).to_client()

# 创建 ClientApp
client = ClientApp(client_fn=client_fn)

# 定义 weighted_average 函数
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """聚合客户端的评估指标。

    Args:
        metrics (List[Tuple[int, Metrics]]): 客户端返回的样本数和指标。

    Returns:
        Metrics: 聚合后的指标。
    """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples) if sum(examples) > 0 else 0.0}

# 自定义 FedAvg 策略以保存聚合模型
class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 创建一个模型实例用于保存聚合参数
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
        """聚合客户端模型并保存聚合后的模型。

        Args:
            server_round (int): 当前轮次。
            results: 客户端训练结果。
            failures: 失败的客户端。

        Returns:
            Tuple: 聚合后的参数和指标。
        """
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            # 将聚合参数转换为 PyTorch 状态字典
            params_dict = zip(self.model.state_dict().keys(), [torch.tensor(p) for p in aggregated_parameters.tensors])
            state_dict = OrderedDict({k: v for k, v in params_dict})
            # 更新模型的 state_dict
            self.model.load_state_dict(state_dict, strict=True)
            # 保存模型
            model_path = f"aggregated_model_round_{server_round}.pth"
            torch.save(self.model.state_dict(), model_path)
            print(f"Saved aggregated model to {model_path}")
        return aggregated_parameters, metrics

# 定义 server_fn
def server_fn(context: Context) -> ServerAppComponents:
    """定义服务器配置。

    Args:
        context (Context): Flower 上下文。

    Returns:
        ServerAppComponents: 服务器配置组件。
    """
    strategy = CustomFedAvg(
        fraction_fit=1.0,  # 训练时采样 100% 的客户端
        fraction_evaluate=0.5,  # 评估时采样 50% 的客户端
        min_fit_clients=5,  # 最少训练客户端数
        min_evaluate_clients=5,  # 最少评估客户端数
        min_available_clients=5,  # 等待所有 5 个客户端可用
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=5)  # 运行 5 轮联邦学习
    return ServerAppComponents(strategy=strategy, config=config)

# 创建 ServerApp
server = ServerApp(server_fn=server_fn)

# 指定客户端资源
backend_config = {"client_resources": {"num_cpus": 0.5, "num_gpus": 0.0}}
if torch.cuda.is_available():
    backend_config["client_resources"] = {"num_cpus": 0.5, "num_gpus": 1.0}

# 运行模拟
if __name__ == "__main__":
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=num_clients,
        backend_config=backend_config,
    )
