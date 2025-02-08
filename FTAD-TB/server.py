import argparse
import os
import os.path as osp
import torch
import flwr as fl
from mmcv import Config
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmdet.apis import single_gpu_test
from torch.utils.data import DataLoader
from mmcv.parallel import MMDataParallel


class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, global_model, val_dataset, save_dir, patience=3, **kwargs):
        super().__init__(**kwargs)
        self.global_model = global_model
        self.val_dataset = val_dataset
        self.best_metric = None
        self.rounds_without_improvement = 0
        self.patience = patience
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.should_stop = False

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, _ = super().aggregate_fit(rnd, results, failures)
        self.save_global_model(aggregated_parameters, rnd)
        eval_metric = self.evaluate_global_model(aggregated_parameters)
        print(f"Round {rnd}: Evaluation metric (mAP): {eval_metric}")
        if self.best_metric is None or eval_metric > self.best_metric:
            self.best_metric = eval_metric
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1
        if self.rounds_without_improvement >= self.patience:
            print(f"Early stopping triggered at round {rnd}.")
            self.should_stop = True
        return aggregated_parameters

    def configure_fit(self, rnd, parameters, client_manager):
        if self.should_stop:
            print("早停条件满足，不再选择客户端进行训练。")
            return []  # 返回空列表，表示不再启动下一轮训练
        return super().configure_fit(rnd, parameters, client_manager)

    def evaluate_global_model(self, parameters):
        # 将聚合参数更新到全局模型
        state_dict = dict(zip(self.global_model.state_dict().keys(),
                              [torch.tensor(param) for param in parameters]))
        self.global_model.load_state_dict(state_dict)
        self.global_model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.global_model.to(device)
        ddp_model = MMDataParallel(self.global_model, device_ids=[0])
        # 构建验证集 DataLoader，批大小设为1
        val_dataloader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=2)
        outputs = single_gpu_test(ddp_model, val_dataloader, show_score_thr=0.05)
        eval_results = self.val_dataset.evaluate(outputs, metric='bbox')
        # 假设评估结果中存在 'mAP' 项，mAP 越高越好
        return eval_results.get('mAP', 0)

    def save_global_model(self, parameters, rnd):
        state_dict = dict(zip(self.global_model.state_dict().keys(),
                              [torch.tensor(param) for param in parameters]))
        self.global_model.load_state_dict(state_dict)
        save_path = os.path.join(self.save_dir, f"global_round_{rnd}.pth")
        torch.save(self.global_model.state_dict(), save_path)
        print(f"Saved global model to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Flower Server with Early Stopping and Model Saving")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--save-dir", type=str, default="./global_model", help="全局模型保存目录")
    parser.add_argument("--num-rounds", type=int, default=10, help="联邦学习总轮次")
    parser.add_argument("--patience", type=int, default=3, help="早停容忍轮次")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    global_model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    # 检查是否存在之前保存的模型，若存在则加载以继续训练
    latest_checkpoint = None
    if os.path.exists(args.save_dir):
        checkpoints = [f for f in os.listdir(args.save_dir) if f.endswith(".pth")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            checkpoint_path = os.path.join(args.save_dir, latest_checkpoint)
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            global_model.load_state_dict(state_dict)
            print(f"Loaded global model from {checkpoint_path}")

    # 如果配置文件中没有定义 "val"，则尝试使用 "test" 数据集作为验证数据
    if "val" in cfg.data:
        val_dataset = build_dataset(cfg.data.val)
    elif "test" in cfg.data:
        val_dataset = build_dataset(cfg.data.test)
    else:
        raise ValueError("配置文件中未定义验证数据（cfg.data.val 或 cfg.data.test），无法进行早停评估。")

    strategy = CustomFedAvg(
        global_model=global_model,
        val_dataset=val_dataset,
        save_dir=args.save_dir,
        patience=args.patience,
        fraction_fit=1.0,
        fraction_eval=0.0,
        min_fit_clients=30,
        min_available_clients=30,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy
    )


if __name__ == "__main__":
    main()
