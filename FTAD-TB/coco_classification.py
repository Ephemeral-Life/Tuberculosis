from mmdet.datasets import DATASETS, CocoDataset
from pycocotools.coco import COCO

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
        # 先创建 COCO 对象以获取必要信息
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        # 调用父类初始化，禁用过滤空标注
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
            img_info = self.coco.loadImgs(img_id)[0]  # 获取图像信息字典
            data_infos.append(img_info)
        return data_infos

    def get_ann_info(self, idx):
        """
        获取指定索引的图像标注信息
        Args:
            idx (int): 图像索引
        Returns:
            dict: 包含标签的标注信息，例如 {'labels': 0}
        """
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        # 假设每张图像只有一个分类标签
        if len(anns) > 0:
            category_id = anns[0]['category_id']
            label = self.cat2label[category_id]
        else:
            label = -1  # 无标注情况
        return dict(labels=label)

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
        data = dict(
            img_info=img_info,
            ann_info=ann_info,
            img_prefix=self.img_prefix
        )
        return self.pipeline(data)

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
