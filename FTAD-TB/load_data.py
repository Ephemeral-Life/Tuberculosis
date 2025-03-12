import random
import os
import json
from mmcv import Config
from mmdet.datasets import CocoDataset
from mmdet.datasets.builder import DATASETS, PIPELINES
from mmcv.utils import Registry, build_from_cfg
from pycocotools.coco import COCO

# 加载配置文件
cfg = Config.fromfile('configs/symformer/symformer_retinanet_p2t_cls_fpn_1x_TBX11K.py')

# 注册 DATASETS 和 PIPELINES
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
DATASETS.register_module(CocoDataset)  # 注册 CocoDataset

# 构建完整训练数据集
train_dataset = build_from_cfg(cfg.data.train, DATASETS, default_args=None)

# 获取所有图像的 ID
image_ids = train_dataset.coco.getImgIds()

# 打乱图像 ID，确保随机性
random.shuffle(image_ids)

# 将图像 ID 分割为 10 个子集
num_clients = 10
partition_size = len(image_ids) // num_clients
partitions = [image_ids[i * partition_size:(i + 1) * partition_size] for i in range(num_clients)]

# 定义一个函数，为每个客户端生成临时的标注文件
def create_client_ann_file(partition_id, client_image_ids):
    """
    为客户端生成临时的标注文件（JSON）。

    Args:
        partition_id (int): 客户端的分区编号 (0-9)。
        client_image_ids (list): 客户端的图像 ID 列表。

    Returns:
        str: 临时标注文件的路径。
    """
    # 加载原始 COCO 标注
    coco = COCO(cfg.data.train.ann_file)

    # 提取客户端图像的标注
    client_imgs = [img for img in coco.imgs.values() if img['id'] in client_image_ids]
    client_ann_ids = coco.getAnnIds(imgIds=client_image_ids)
    client_anns = [coco.anns[ann_id] for ann_id in client_ann_ids]

    # 创建新的 COCO 标注字典
    client_coco = {
        'images': client_imgs,
        'annotations': client_anns,
        'categories': coco.dataset['categories'],  # 保留原始类别
        'info': coco.dataset.get('info', {}),
        'licenses': coco.dataset.get('licenses', [])
    }

    # 保存为临时 JSON 文件
    temp_ann_file = f'temp_client_{partition_id}_ann.json'
    with open(temp_ann_file, 'w') as f:
        json.dump(client_coco, f)

    return temp_ann_file

# 定义一个函数，为每个客户端加载数据集
def load_client_dataset(partition_id):
    """
    根据分区 ID 加载对应客户端的数据集。

    Args:
        partition_id (int): 客户端的分区编号 (0-9)。

    Returns:
        dataset: 客户端的 CocoDataset 实例。
    """
    # 获取当前客户端的图像 ID 子集
    client_image_ids = partitions[partition_id]

    # 生成客户端的临时标注文件
    client_ann_file = create_client_ann_file(partition_id, client_image_ids)

    # 构建客户端数据集的配置
    client_cfg = dict(
        type='CocoDataset',
        ann_file=client_ann_file,  # 使用临时标注文件
        img_prefix=cfg.data.train.img_prefix,
        pipeline=cfg.data.train.pipeline,
        filter_empty_gt=cfg.data.train.filter_empty_gt,
        classes=cfg.data.train.classes
    )

    # 使用 build_from_cfg 构建客户端数据集
    client_dataset = build_from_cfg(client_cfg, DATASETS, default_args=None)

    return client_dataset

# 示例：加载每个客户端的数据集
client_datasets = []
for i in range(num_clients):
    client_dataset = load_client_dataset(i)
    client_datasets.append(client_dataset)
    print(f"客户端 {i} 数据集大小: {len(client_dataset)}")

    # 可选：删除临时标注文件
    os.remove(f'temp_client_{i}_ann.json')
