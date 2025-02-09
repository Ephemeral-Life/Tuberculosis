# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.apis.test import single_gpu_test_cls
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--txt', help='output class result')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             ' useful when you want to format the result to a specific format and '
             ' submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, e.g., "accuracy", "precision", "recall", "f1_score" for classification')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function (deprecate), '
             'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--cls-filter',
        type=bool,
        default=False,
        help='filter the bbox with cls (default: False). 当只关注图像是否患结核病时，建议开启该参数')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        'Please specify at least one operation (save/eval/format/show the results / save the results) '
        'with the argument "--out", "--eval", "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max([ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # 当只关注图像是否患结核病时，只输出二分类结果，不需要检测框信息
    if not args.cls_filter:
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir, args.show_score_thr)
        else:
            model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    else:
        # 目前仅支持单 GPU 模式下的分类结果输出
        if distributed:
            raise NotImplementedError("当前仅支持单 GPU 模式下的分类输出，请关闭--cls-filter参数或使用单 GPU模式。")
        else:
            model = MMDataParallel(model, device_ids=[0])
            # 调用单 GPU 模式下的分类测试函数，获得检测结果及分类预测
            _, class_preds = single_gpu_test_cls(model, data_loader, args.show, args.show_dir, args.show_score_thr)
            import torch.nn as nn
            softmax = nn.Softmax(dim=1)
            class_preds_tensor = softmax(torch.cat(class_preds).view(-1, 3))
            _, max_idx = class_preds_tensor.max(dim=1)
            # 根据预测类别生成二分类结果：若预测类别为2（非结核病），则标记为 0；否则标记为 1
            binary_preds = [0 if int(max_idx[i]) != 2 else 1 for i in range(len(max_idx))]
            outputs = binary_preds
            if args.txt:
                os.makedirs(os.path.dirname(args.txt), exist_ok=True)
                with open(args.txt, 'w') as f:
                    for pred in outputs:
                        f.write(str(pred) + '\n')

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            # 修改评估部分：如果使用分类结果，则计算二分类指标
            # 假设每个样本的 ground truth 标签存储在 dataset.data_infos 中的 'gt_label' 字段中
            if args.cls_filter:
                if len(outputs) != len(dataset.data_infos):
                    raise AssertionError(
                        f'Dataset and results have different sizes: {len(dataset.data_infos)} vs {len(outputs)}')
                gt_labels = [info['gt_label'] for info in dataset.data_infos]
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                accuracy = accuracy_score(gt_labels, outputs)
                precision = precision_score(gt_labels, outputs, pos_label=1, zero_division=0)
                recall = recall_score(gt_labels, outputs, pos_label=1, zero_division=0)
                f1 = f1_score(gt_labels, outputs, pos_label=1, zero_division=0)
                cm = confusion_matrix(gt_labels, outputs)
                eval_results = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'confusion_matrix': cm.tolist()
                }
                print("Classification evaluation results: ", eval_results)
                metric = eval_results
            else:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))
                metric = dataset.evaluate(outputs, **eval_kwargs)
                print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main()
