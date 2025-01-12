import flwr as fl
import torch
from mmdet.models import build_detector
from mmcv import Config


class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def aggregate_fit(self, rnd, results, failures):
        # 获取客户端返回的权重
        weights = [torch.tensor(res.parameters) for res in results]

        # 聚合权重
        aggregated_weights = torch.mean(torch.stack(weights), dim=0)

        # 将聚合的权重加载到模型中
        state_dict = {k: v for k, v in zip(self.model.state_dict().keys(), aggregated_weights)}
        self.model.load_state_dict(state_dict)

        return super().aggregate_fit(rnd, results, failures)


def main():
    # 加载配置文件
    config = Config.fromfile("config/s_symformer_retinanet_p2t_fpn_2x_TBX11K.py")
    model = build_detector(config.model, train_cfg=config.get('train_cfg'), test_cfg=config.get('test_cfg'))
    model.init_weights()

    # 自定义策略
    strategy = CustomStrategy(model=model)

    # 启动服务器
    fl.server.start_server(server_address="localhost:8080", strategy=strategy)


if __name__ == "__main__":
    main()
