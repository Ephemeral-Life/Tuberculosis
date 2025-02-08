import argparse
import copy
import os
import os.path as osp
import time
import warnings

import torch
import flwr as fl
from mmcv import Config
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmdet.apis import train_detector

warnings.filterwarnings("ignore")  # 不打印警告

# 在客户端启动时限制GPU内存占用，避免同时运行多个客户端导致显卡资源耗尽
if torch.cuda.is_available():
    try:
        device = torch.cuda.current_device()
        # 将每个客户端进程的GPU内存占用限制为总量的 30%
        torch.cuda.set_per_process_memory_fraction(0.3, device)
        print(f"成功将设备 {device} 的GPU内存占用限制设置为 30%")
    except Exception as e:
        print("设置GPU内存占用限制失败：", e)


class MMTBClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, cfg: Config):
        self.client_id = client_id
        self.cfg = cfg

        # 构建模型并初始化权重
        self.model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        self.model.init_weights()

        # 构建完整的训练数据集，并按照客户端编号划分数据
        full_dataset = build_dataset(cfg.data.train)
        total_samples = len(full_dataset)
        samples_per_client = total_samples // 30
        start_idx = client_id * samples_per_client
        if client_id == 29:
            end_idx = total_samples
        else:
            end_idx = (client_id + 1) * samples_per_client
        indices = list(range(start_idx, end_idx))
        # 使用 torch.utils.data.Subset 构建局部数据集
        from torch.utils.data import Subset
        self.client_dataset = Subset(full_dataset, indices)

        # 如果配置中包含验证数据，则构建验证集（可选）
        if 'val' in cfg.data:
            self.val_dataset = build_dataset(cfg.data.val)
        else:
            self.val_dataset = None

    def get_parameters(self):
        # 将模型参数转换为 numpy 数组列表
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        # 根据参数列表更新模型参数
        state_dict = dict(zip(self.model.state_dict().keys(),
                              [torch.tensor(param) for param in parameters]))
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        # 更新全局参数到本地模型
        self.set_parameters(parameters)
        # 为了模拟联邦学习中的局部训练，将配置复制后设置仅训练 1 个 epoch
        local_cfg = copy.deepcopy(self.cfg)
        local_cfg.total_epochs = 1  # 每个联邦轮次只训练 1 个 epoch
        # 修改工作目录，避免不同客户端之间日志、模型文件冲突
        local_cfg.work_dir = osp.join(local_cfg.work_dir, f'client_{self.client_id}')
        os.makedirs(local_cfg.work_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        meta = {}
        # 调用 train_detector 进行局部训练，此处不采用分布式训练，也不做验证
        train_detector(
            self.model,
            [self.client_dataset],
            local_cfg,
            distributed=False,
            validate=False,
            timestamp=timestamp,
            meta=meta
        )
        new_parameters = self.get_parameters()
        # 返回更新后的参数、参与训练的样本数及空的指标字典
        return new_parameters, len(self.client_dataset), {}

    def evaluate(self, parameters, config):
        # 客户端评估逻辑（示例中返回默认的 0 损失，可按需扩展）
        self.set_parameters(parameters)
        loss = 0.0
        return float(loss), len(self.client_dataset), {"loss": loss}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flower Client for Revisiting Computer-Aided Tuberculosis Diagnosis"
    )
    parser.add_argument("--cid", type=int, default=0, help="客户端ID（0~29）")
    parser.add_argument("config", help="配置文件路径")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    client = MMTBClient(args.cid, cfg)
    # 启动 Flower 客户端，连接到服务器（请根据实际情况修改 server_address）
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
