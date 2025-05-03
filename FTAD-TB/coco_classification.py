from mmdet.datasets import DATASETS, CocoDataset
from pycocotools.coco import COCO
import numpy as np
import torch

@DATASETS.register_module()
class CocoClassificationDataset(CocoDataset):
    def __init__(self, ann_file, pipeline, img_prefix='', classes=None, **kwargs):
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        super(CocoClassificationDataset, self).__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            img_prefix=img_prefix,
            classes=classes,
            filter_empty_gt=False,
            **kwargs
        )

    def load_annotations(self, ann_file):
        data_infos = []
        for img_id in self.img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_info['filename'] = img_info.pop('file_name')
            data_infos.append(img_info)
        return data_infos

    def get_ann_info(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        if len(anns) > 0:
            category_id = anns[0]['category_id']
            label = self.cat2label[category_id]
        else:
            label = -1

        labels = np.array([label], dtype=np.int64)
        empty_bboxes = np.zeros((0, 4), dtype=np.float32)
        return dict(labels=labels, bboxes=empty_bboxes)

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(
            img_info=img_info,
            ann_info=ann_info,
            img_prefix=self.img_prefix,
            seg_prefix=self.seg_prefix,
            proposal_file=self.proposal_file,
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[]
        )
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        data = dict(
            img_info=img_info,
            img_prefix=self.img_prefix
        )
        results = self.pipeline(data)

        print(f"Image {idx}: img_shape={results['img'].shape}, pad_shape={results['img_metas']['pad_shape']}")

        # 确保 'img' 是独立的张量
        img = results['img']
        if not img.is_contiguous():
            img = img.contiguous()
        results['img'] = img
        return results

    def __len__(self):
        return len(self.data_infos)

    def _filter_imgs(self, min_size=32):
        return list(range(len(self.data_infos)))
