import argparse
import flwr as fl
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from mmcv import Config
from mmdet.models import build_detector
from mmdet.datasets import build_dataset

class EvaluateAndSaveModelStrategy(FedAvg):
    def __init__(
        self,
        model,
        num_rounds: int,
        save_path: str = "global_model.pt",
        val_dataset=None,
        patience: int = 5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model                  # 服务器端模型实例
        self.num_rounds = num_rounds        # 总轮数
        self.save_path = save_path          # 模型保存路径
        self.val_dataset = val_dataset      # 验证数据集
        self.patience = patience            # 早停耐心值
        self.best_metric = float('-inf')    # 最佳验证指标（越大越好，例如 mAP）
        self.rounds_without_improvement = 0 # 无提升轮数计数器

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.common.Parameters, int]],
        failures: List[BaseException],
    ) -> Tuple[fl.common.Parameters, Dict]:
        """聚合客户端的训练结果并更新全局模型"""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            # 将聚合参数加载到模型
            parameters_tensors = [torch.from_numpy(np_array) for np_array in aggregated_parameters]
            state_dict = self.model.state_dict()
            param_keys = list(state_dict.keys())
            for key, param in zip(param_keys, parameters_tensors):
                if state_dict[key].shape == param.shape:
                    state_dict[key] = param
                else:
                    print(f"Warning: Shape mismatch for {key}, skipping...")
            self.model.load_state_dict(state_dict)
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[float, int]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """聚合客户端的评估结果，执行早停逻辑"""
        if not results:
            return None, {}

        # 聚合评估指标（这里假设客户端返回的是损失，实际应改为 mAP 等指标）
        total_loss = sum([loss for loss, _ in results])
        total_samples = sum([num_samples for _, num_samples in results])
        avg_loss = total_loss / total_samples
        metrics = {"avg_loss": avg_loss}

        # 假设验证指标是 mAP（越大越好），这里用 -avg_loss 模拟（越小越好需调整逻辑）
        current_metric = -avg_loss  # 实际应用中应替换为 mAP

        # 早停逻辑
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.rounds_without_improvement = 0
            # 保存最优模型
            torch.save(self.model.state_dict(), self.save_path)
            print(f"Round {server_round}: New best model saved with metric: {current_metric}")
        else:
            self.rounds_without_improvement += 1
            print(f"Round {server_round}: No improvement, rounds without improvement: {self.rounds_without_improvement}")
            if self.rounds_without_improvement >= self.patience:
                print(f"Early stopping triggered at round {server_round}")
                # 停止服务器（Flower暂无直接API，可通过退出进程模拟）
                raise SystemExit("Early stopping triggered")

        return avg_loss, metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Flower Server for Federated Learning with MMDetection')
    parser.add_argument('--config', type=str, required=True, help='Path to the MMDetection configuration file')
    parser.add_argument('--num-rounds', type=int, default=5, help='Number of federated learning rounds')
    parser.add_argument('--save-path', type=str, default='global_model.pt', help='Path to save the global model')
    parser.add_argument('--val-dir', type=str, required=True, help='Path to the validation directory (g_val)')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    return parser.parse_args()

def main():
    args = parse_args()

    # 加载配置文件
    cfg = Config.fromfile(args.config)

    # 更新验证集配置
    cfg.data.val.ann_file = f"{args.val_dir}/annotations.json"  # JSON标注文件路径
    cfg.data.val.img_prefix = f"{args.val_dir}/imgs"            # 图像文件夹路径

    # 构建模型
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    model.init_weights()

    # 构建验证数据集
    val_dataset = build_dataset(cfg.data.val)

    # 定义联邦学习策略
    strategy = EvaluateAndSaveModelStrategy(
        model=model,
        num_rounds=args.num_rounds,
        save_path=args.save_path,
        val_dataset=val_dataset,
        patience=args.patience,
        fraction_fit=1.0,           # 每轮训练时选择所有客户端
        fraction_evaluate=0.5,      # 每轮评估50%的客户端
        min_fit_clients=2,          # 每轮训练所需的最小客户端数
        min_evaluate_clients=1,     # 每轮评估所需的最小客户端数
        min_available_clients=2,    # 服务器启动前所需的最小可用客户端数
    )

    # 启动Flower服务器
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
