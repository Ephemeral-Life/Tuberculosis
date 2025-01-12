import os
import torch
import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel

from mmdet.apis import train_detector
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
import flwr as fl
import argparse
import os.path as osp


class FederatedClient(fl.client.NumPyClient):  # 继承 NumPyClient 而非 Client
    def __init__(self, cid, config, dataset, model):
        self.cid = cid
        self.config = config
        self.model = model
        self.dataset = dataset
        self.model.CLASSES = dataset.CLASSES  # Set classes for the model

    def get_parameters(self, config=None):  # 接收额外参数 config
        # Return the model's parameters
        return [tensor.detach().cpu().numpy() for tensor in self.model.parameters()]

    def set_parameters(self, parameters):
        # Set the model's parameters
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.from_numpy(new_param).to(param.device)

    def fit(self, parameters, config):
        # Set the model parameters from the global model
        self.set_parameters(parameters)
        # Train the model
        datasets = [self.dataset]  # Your dataset
        train_detector(self.model, datasets, self.config, distributed=False, validate=False)
        return self.get_parameters(), len(self.dataset), {}



def parse_args():
    parser = argparse.ArgumentParser(description='Federated Training')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('--client-id', type=int, required=True, help='ID of the client')
    parser.add_argument('--server-address', type=str, default='localhost:8080', help='Address of the FL server')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU id to use for training')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Load config
    cfg = Config.fromfile(args.config)

    # Update config for single GPU
    cfg.gpu_ids = [0]  # Ensure single GPU is used

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # Build model
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # Wrap model with MMDataParallel for single GPU
    model = MMDataParallel(model, device_ids=[0])

    # Build dataset
    dataset = build_dataset(cfg.data.train)

    # Create and start federated client
    client = FederatedClient(args.client_id, cfg, dataset, model)

    # Start Flower client
    fl.client.start_client(server_address=args.server_address, client=client)



if __name__ == "__main__":
    main()
