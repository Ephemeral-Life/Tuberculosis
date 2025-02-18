# symformer_retinanet_p2t_cls_fpn_1x_TBX11K_test.py
# Copyright (c) OpenMMLab. All rights reserved.

# model settings
model = dict(
    type='RetinaNetClsAtt',
    backbone=dict(
        type='p2t_small',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        style='pytorch',
        pretrained='pretrained/p2t_small.pth',
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/p2t_small.pth')
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5
    ),
    bbox_head=dict(
        type='RetinaGuideAttHead',
        num_classes=2,  # 检测头类别数仍保留2（一般用于检测任务），但后续测试时只关注分类结果
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
            strides=[8, 16, 32, 64, 128]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
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
                        num_levels=1
                    ),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')
                )
            )
        ),
        positional_encoding=dict(
            type='GuidePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5,
            left=True
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    classifier=dict(
        input_dim=512,
        num_classes=2  # 确保分类头输出二分类结果，0 表示阴性，1 表示阳性
    ),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
        stage='resnet_classify'
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100
    )
)

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/ChinaSet_AllFiles/'
# 新测试集只区分阴性与阳性，因此 classes 定义为二分类
classes = ('Negative', 'Positive')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# pipelines
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_classes', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]
    )
]

# data 配置
data = dict(
    samples_per_gpu=1,  # 测试时通常使用单卡，每个 GPU 一张图片
    workers_per_gpu=4,
    # 训练集部分保留，仅供训练时参考
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/json/all_trainval.json',
        img_prefix=data_root + 'CXR_png/',
        pipeline=train_pipeline,
        filter_empty_gt=False,
        classes=('ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis')
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'dataset.json',  # 新测试集标注文件路径
        img_prefix=data_root + 'CXR_png',         # 新测试集图像目录
        pipeline=test_pipeline,
        classes=classes  # 使用二分类的类别定义
    )
)

# evaluation 配置
# 这里 evaluation 部分主要在训练时使用，测试时若仅关注分类评估，会在 test.py 中调用自定义评估逻辑
evaluation = dict(interval=1, metric=['bbox'])

# 以下部分为训练时的优化器、学习率调度器等配置，如仅用于测试可忽略
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001, stage='resnet_finetune')
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11]
)
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=6)
log_config = dict(interval=150, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dirs/symformer_retinanet_p2t/latest.pth'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 2)
find_unused_parameters = True
