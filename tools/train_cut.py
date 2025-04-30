import copy
import os.path as osp
import time
import warnings
warnings.filterwarnings("ignore")  # 不打印警告

import mmcv
import torch
from mmcv import Config
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

def main():
    # 直接设置启动指令中指定的参数
    config = 'configs/symformer/symformer_retinanet_p2t_cls_fpn_1x_TBX11K.py'
    work_dir = 'work_dirs/symformer_retinanet_p2t_cls/'
    no_validate = True
    resume_from = None
    gpu_ids = [0]  # 默认使用 GPU 0
    seed = None
    deterministic = False

    # 加载配置文件
    cfg = Config.fromfile(config)

    # 设置 cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 设置工作目录
    cfg.work_dir = work_dir

    # 设置恢复训练的检查点
    cfg.resume_from = resume_from

    # 设置 GPU ID
    cfg.gpu_ids = gpu_ids

    # 设置分布式训练（由于 launcher='none'，这里为非分布式）
    distributed = False

    # 创建工作目录
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # 转储配置文件到工作目录
    cfg.dump(osp.join(cfg.work_dir, osp.basename(config)))

    # 初始化日志记录器
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # 初始化 meta 字典，用于记录环境信息等
    meta = dict()
    # 记录环境信息
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # 记录基本信息
    # logger.info(f'Distributed training: {distributed}')
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # 设置随机种子（如果 seed 不为 None）
    if seed is not None:
        logger.info(f'Set random seed to {seed}, deterministic: {deterministic}')
        set_random_seed(seed, deterministic=deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(config)

    # 构建检测器模型
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # 构建训练数据集
    datasets = [build_dataset(cfg.data.train)]

    # 如果 workflow 长度为 2，则构建验证数据集
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # 设置检查点元信息
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    model.CLASSES = datasets[0].CLASSES

    # 训练模型
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=not no_validate,  # 根据 no_validate 设置验证
        timestamp=timestamp,
        meta=meta)

if __name__ == '__main__':
    main()
