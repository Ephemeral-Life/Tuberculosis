import os
import json
import random
import time
import torch
import math
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict
from mmcv import Config
from mmcv.utils import Registry, build_from_cfg
from mmdet.datasets import CocoDataset, build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed, single_gpu_test
from pycocotools.coco import COCO
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, Parameters, Scalar
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from mmcv.parallel import MMDataParallel
from coco_classification import CocoClassificationDataset
import gc
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
cfg = Config.fromfile('configs/symformer/symformer_retinanet_p2t_cls_fpn_1x_TBX11K.py')
cfg.gpu_ids = [0]
num_clients = cfg.num_clients

# Register datasets
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
DATASETS.register_module(CocoDataset)
DATASETS.register_module(CocoClassificationDataset)

# Pre-generated client annotation files
client_ann_files = [os.path.join('../client_ann', f'client_{partition_id}_ann.json')
                    for partition_id in range(num_clients)]

# Flower client class
class FlowerClient(NumPyClient):
    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.net = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        self.ann_file = client_ann_files[partition_id]
        self.trainloader = None
        logger.info(f"Client {self.partition_id} initialized, annotation file: {self.ann_file}")

    def freeze_parameters(self):
        for name, param in self.net.named_parameters():
            if "backbone" in name or "neck" in name or "bbox_head" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def load_data(self):
        client_cfg = dict(
            type='CocoClassificationDataset',
            ann_file=self.ann_file,
            img_prefix=cfg.data.train.img_prefix,
            pipeline=cfg.data.train.pipeline,
            classes=cfg.data.train.classes
        )
        try:
            client_dataset = build_from_cfg(client_cfg, DATASETS)
            logger.info(f"Client {self.partition_id} loaded dataset, image count: {len(client_dataset)}")
            return client_dataset
        except Exception as e:
            logger.error(f"Client {self.partition_id} failed to load dataset: {e}")
            raise

    def get_parameters(self, config) -> List[np.ndarray]:
        with torch.no_grad():
            params = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        logger.info(f"Client {self.partition_id} returning {len(params)} parameters")
        return params

    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = self.net.state_dict()
        params_dict = zip(state_dict.keys(), parameters)
        new_state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        try:
            self.net.load_state_dict(new_state_dict, strict=True)
            logger.info(f"Client {self.partition_id} parameters set successfully")
        except Exception as e:
            logger.error(f"Client {self.partition_id} failed to set parameters: {e}")
            raise

    def fit(self, parameters: List[np.ndarray], config) -> Tuple[List[np.ndarray], int, dict]:
        self.set_parameters(parameters)
        seed = cfg.get('seed', None)
        if seed is not None:
            set_random_seed(seed)

        self.trainloader = self.load_data()
        server_round = config.get("server_round", 1)
        logger.info(f"Client {self.partition_id} training (round {server_round})...")

        self.freeze_parameters()
        original_load_from = cfg.get('load_from', None)
        cfg.load_from = None

        try:
            train_detector(
                self.net,
                [self.trainloader],
                cfg,
                distributed=False,
                validate=False,
                timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
                meta={'seed': seed}
            )
            logger.info(f"Client {self.partition_id} training completed")
        except Exception as e:
            logger.error(f"Client {self.partition_id} training failed: {e}")
            raise
        finally:
            cfg.load_from = original_load_from

        params = self.get_parameters(config)
        num_examples = len(self.trainloader)
        del self.trainloader
        self.trainloader = None
        gc.collect()

        return params, num_examples, {}

# Client factory function
def client_fn(context: Context) -> Client:
    partition_id = context.node_config["partition-id"]
    return FlowerClient(partition_id).to_client()

# Create ClientApp
client = ClientApp(client_fn=client_fn)

# Weighted average for metrics
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples) if sum(examples) > 0 else 0.0}

# Custom FedAvg strategy
class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        self.model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg')
        )
        self.start_round = 1

        if cfg.get('load_from', None):
            try:
                checkpoint = torch.load(cfg.load_from)
                state_dict = checkpoint.get('state_dict', checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"Server loaded initial model: {cfg.load_from}")
            except Exception as e:
                logger.error(f"Server failed to load initial model: {e}")
                raise
        else:
            logger.info("No cfg.load_from specified, using randomly initialized model")
            self.model.init_weights()

        try:
            model_files = [f for f in os.listdir(cfg.work_dir) if
                           f.startswith("aggregated_model_round_") and f.endswith(".pth")]
        except FileNotFoundError:
            model_files = None
        if model_files:
            rounds = [int(f.split('_')[-1].split('.')[0]) for f in model_files]
            latest_round = max(rounds)
            latest_model_path = os.path.join(cfg.work_dir, f"aggregated_model_round_{latest_round}.pth")
            logger.info(f"Loading latest aggregated model: {latest_model_path}")
            checkpoint = torch.load(latest_model_path)
            state_dict = checkpoint.get('state_dict', checkpoint)
            self.model.load_state_dict(state_dict, strict=True)
            self.start_round = latest_round + 1

        initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for val in self.model.state_dict().values()])
        super().__init__(*args, initial_parameters=initial_parameters, **kwargs)

        val_cfg = cfg.data.val
        val_cfg['type'] = 'CocoClassificationDataset'
        try:
            self.val_dataset = build_dataset(val_cfg)
            self.val_dataloader = build_dataloader(
                self.val_dataset,
                samples_per_gpu=cfg.data.samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=False,
                shuffle=False
            )
            logger.info("Server loaded evaluation dataset successfully")
        except Exception as e:
            logger.error(f"Server failed to load evaluation dataset: {e}")
            raise

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        for instruction in client_instructions:
            instruction[1].config["server_round"] = server_round
        return client_instructions

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        logger.info(f"Server round {server_round}: Aggregating {len(results)} results, {len(failures)} failures")
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            try:
                aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
                params_dict = zip(self.model.state_dict().keys(),
                                  [torch.from_numpy(np.copy(p)) for p in aggregated_ndarrays])
                state_dict = OrderedDict({k: v for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=True)

                actual_round = self.start_round + server_round - 1
                model_path = os.path.join(cfg.work_dir, f"aggregated_model_round_{actual_round}.pth")
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"Saved aggregated model to {model_path}")

                self.evaluate_aggregated_model()
            except Exception as e:
                logger.error(f"Aggregation or evaluation failed: {e}")
                raise
        else:
            logger.warning(f"Round {server_round} produced no aggregated parameters, results: {len(results)}, failures: {len(failures)}")

        gc.collect()
        return aggregated_parameters, metrics

    def evaluate_aggregated_model(self):
        logger.info("Starting aggregated model evaluation...")
        self.model.eval()
        try:
            model = MMDataParallel(self.model, device_ids=[0])
            outputs = single_gpu_test(model, self.val_dataloader)

            preds = []
            gts = []
            for i, output in enumerate(outputs):
                pred = torch.argmax(torch.tensor(output)).item()
                gt = self.val_dataset[i]['ann_info']['labels'][0]
                preds.append(pred)
                gts.append(gt)

            accuracy = sum(1 for p, g in zip(preds, gts) if p == g) / len(gts)
            logger.info(f"Aggregated model evaluation completed, accuracy: {accuracy:.4f}")
        except Exception as e:
            logger.error(f"Failed to evaluate aggregated model: {e}")
            raise

# Server factory function
def server_fn(context: Context) -> ServerAppComponents:
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=math.ceil(num_clients / 2),
        min_evaluate_clients=math.ceil(num_clients / 2),
        min_available_clients=math.ceil(num_clients / 2),
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    start_round = strategy.start_round if hasattr(strategy, 'start_round') else 1
    num_rounds = cfg.num_rounds
    if start_round > 1:
        num_rounds -= (start_round - 1)
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

# Client resources
backend_config = {"client_resources": {"num_cpus": 1.0, "num_gpus": 0.0}}
if torch.cuda.is_available():
    backend_config["client_resources"] = {"num_cpus": 1.0, "num_gpus": 1.0}

# Configuration updates
cfg.seed = 42
cfg.work_dir = '../work_dirs/symformer_retinanet_p2t_cls_flower'
cfg.num_clients = 3
cfg.num_rounds = 1
cfg.max_epochs = 1
cfg.log_level = 'INFO'

cfg.model = dict(
    type='RetinaNetClsAtt',
    backbone=dict(
        type='p2t_small',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        style='pytorch',
        pretrained='../pretrained/p2t_small.pth',
        init_cfg=dict(type='Pretrained', checkpoint='../pretrained/p2t_small.pth')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaGuideAttHead',
        num_classes=2,
        num_query=500,
        dims_radio=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        transformer=dict(
            type='DeformableDetrTransformerConv',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=1,
                transformerlayers=dict(
                    type='SymDetrTransformerEncoderLayer',
                    attn_cfgs=dict(
                        type='SymMultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=1),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='GuidePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5,
            left=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    classifier=dict(input_dim=512),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
        stage='resnet_classify'),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

cfg.dataset_type = 'COCODataset'
cfg.data_root = '../data/TBX11K/'
cfg.classes = ('ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis')
cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_classes', 'gt_bboxes', 'gt_labels'])
]

# Updated test pipeline to ensure consistent image dimensions
cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='Pad', size=(512, 512)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=['img_shape', 'scale_factor', 'pad_shape', 'filename', 'ori_shape'])
        ])
]

cfg.data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file='../data/TBX11K/annotations/json/all_trainval_without_extra.json',
        img_prefix='../data/fed/clients/',
        pipeline=cfg.train_pipeline,
        filter_empty_gt=False,
        classes=cfg.classes),
    val=dict(
        type='CocoClassificationDataset',
        ann_file='../data/g/g_val/val_dataset.json',
        img_prefix='../data/g/g_val/img/',
        pipeline=cfg.test_pipeline,
        classes=('healthy', 'sick_non_tb', 'tb')))

cfg.evaluation = dict(interval=1, metric='bbox')
cfg.optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001, stage='resnet_finetune')
cfg.optimizer_config = dict(grad_clip=None)
cfg.lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[3, 4])
cfg.runner = dict(type='EpochBasedRunner', max_epochs=cfg.max_epochs)
cfg.checkpoint_config = dict(interval=1)
cfg.log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
cfg.custom_hooks = [dict(type='NumClassCheckHook')]
cfg.dist_params = dict(backend='nccl')
cfg.log_level = 'INFO'
cfg.load_from = '../work_dirs/symformer_retinanet_p2t/latest.pth'
cfg.resume_from = None
cfg.workflow = [('train', 1)]
cfg.gpu_ids = [0]
cfg.find_unused_parameters = True

# Run simulation
if __name__ == "__main__":
    logger.info("Starting simulation...")
    run_simulation(
        server_app=ServerApp(server_fn=server_fn),
        client_app=client,
        num_supernodes=num_clients,
        backend_config=backend_config,
    )
