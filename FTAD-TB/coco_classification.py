from mmdet.datasets import DATASETS, CocoDataset
from pycocotools.coco import COCO
import numpy as np

@DATASETS.register_module()
class CocoClassificationDataset(CocoDataset):
    """自定义COCO分类数据集类，用于加载图像级别的分类标注"""

    def __init__(self, ann_file, pipeline, img_prefix='', classes=None, **kwargs):
        """
        初始化CocoClassificationDataset
        Args:
            ann_file (str): COCO格式的JSON标注文件路径
            pipeline (list): 数据处理流程
            img_prefix (str): 图像文件路径前缀
            classes (tuple): 类别名称元组，例如 ('healthy', 'sick_non_tb', 'tb')
            **kwargs: 其他参数
        """
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        super(CocoClassificationDataset, self).__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            img_prefix=img_prefix,
            classes=classes,
            filter_empty_gt=False,  # 禁用默认的图像过滤
            **kwargs
        )

    def load_annotations(self, ann_file):
        """
        加载COCO格式的标注文件
        Args:
            ann_file (str): 标注文件路径
        Returns:
            list: 图像信息字典列表
        """
        data_infos = []
        for img_id in self.img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            # 将 'file_name' 重命名为 'filename' 以适配MMDetection管道
            img_info['filename'] = img_info.pop('file_name')
            data_infos.append(img_info)
        return data_infos

    def get_ann_info(self, idx):
        """
        获取指定索引的图像标注信息
        Args:
            idx (int): 图像索引
        Returns:
            dict: 包含标签的标注信息，例如 {'labels': np.array([0])}
        """
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        if len(anns) > 0:
            category_id = anns[0]['category_id']
            label = self.cat2label[category_id]
        else:
            label = -1  # 无标注情况

        # 将 label 包装成单元素的 NumPy 数组
        labels = np.array([label], dtype=np.int64)
        # 创建一个空的 float32 NumPy 数组，形状为 (0, 4)
        empty_bboxes = np.zeros((0, 4), dtype=np.float32)

        # 返回包含 'labels' 和空的 'bboxes' 的字典
        return dict(labels=labels, bboxes=empty_bboxes)

    def prepare_train_img(self, idx):
        """
        为训练准备图像和标注数据
        Args:
            idx (int): 图像索引
        Returns:
            dict: 处理后的图像和标注数据
        """
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(
            img_info=img_info,
            ann_info=ann_info,
            img_prefix=self.img_prefix,
            seg_prefix=self.seg_prefix,  # 通常需要，即使为空
            proposal_file=self.proposal_file,  # 通常需要，即使为 None
            # --- 初始化 fields 列表 ---
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[]
        )
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """
        为测试准备图像数据
        Args:
            idx (int): 图像索引
        Returns:
            dict: 处理后的图像数据
        """
        img_info = self.data_infos[idx]
        data = dict(
            img_info=img_info,
            img_prefix=self.img_prefix
        )
        return self.pipeline(data)

    def __len__(self):
        """返回数据集大小"""
        return len(self.data_infos)

    def _filter_imgs(self, min_size=32):
        """
        重写过滤图像方法，直接返回所有图像索引
        Args:
            min_size (int): 最小图像尺寸（此处未使用）
        Returns:
            list: 所有有效图像的索引列表
        """
        return list(range(len(self.data_infos)))
