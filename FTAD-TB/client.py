# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

warnings.filterwarnings("ignore")  # 不打印警告

import mmcv
import torch
import numpy as np
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from torch.utils.data import Subset

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

import flwr as fl
from flwr.common import Context

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector with Flower')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--num-clients', type=int, default=10, help='number of clients')
    parser.add_argument('--num-rounds', type=int, default=5, help='number of federated learning rounds')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()

    # 加载配置文件
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 设置cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 设置工作目录
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # 初始化分布式环境
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # 创建工作目录并保存配置
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # 初始化日志
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # 日志记录环境信息
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # 设置随机种子
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    # 加载数据集并分区
    train_dataset = build_dataset(cfg.data.train)
    N = args.num_clients
    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)
    client_indices = np.array_split(indices, N)
    client_datasets = [Subset(train_dataset, indices) for indices in client_indices]

    # 定义Flower客户端类
    class FlowerClient(fl.client.NumPyClient):
        def __init__(self, model, dataset, cfg, distributed, validate, timestamp, meta):
            self.model = model
            self.dataset = dataset
            self.cfg = copy.deepcopy(cfg)  # 深拷贝防止修改全局配置
            self.distributed = distributed
            self.validate = validate
            self.timestamp = timestamp
            self.meta = meta

        def get_parameters(self, config=None):
            """返回模型参数给服务器"""
            # 添加 config 参数并忽略，以兼容 Flower 1.11.1
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        def fit(self, parameters, config):
            """本地训练"""
            # 将服务器参数加载到模型
            self.model.load_state_dict(
                {k: torch.from_numpy(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
            )
            # 调整配置，每轮训练1个epoch
            self.cfg.total_epochs = 1
            train_detector(
                self.model,
                [self.dataset],
                self.cfg,
                distributed=self.distributed,
                validate=self.validate,
                timestamp=self.timestamp,
                meta=self.meta
            )
            return self.get_parameters(), len(self.dataset), {}

        def evaluate(self, parameters, config):
            """评估逻辑（可选）"""
            # 这里暂时为空，您可根据需要添加评估逻辑
            return 0.0, len(self.dataset), {}

    # 定义客户端生成函数（适应 Flower 1.11.1）
    def client_fn(context: Context):
        cid = context.node_config["cid"]  # 从 context 获取客户端 ID
        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        # 加载预训练模型，忽略不匹配的键
        if cfg.get('load_from'):
            checkpoint = torch.load(cfg.load_from, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
        model.init_weights()
        dataset = client_datasets[int(cid)]
        model.CLASSES = dataset.dataset.CLASSES  # 设置类别
        return FlowerClient(model, dataset, cfg, distributed, not args.no_validate, timestamp, meta)

    # 定义联邦学习策略
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,           # 每轮训练所有客户端
        fraction_evaluate=0.0,      # 暂时不评估
        min_fit_clients=N,          # 最小训练客户端数
        min_available_clients=N,    # 最小可用客户端数
    )

    # 运行联邦学习模拟
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=N,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

if __name__ == '__main__':
    main()
