import os
import json
from PIL import Image
from datetime import datetime


def generate_coco_json(img_dir, output_json):
    """
    生成COCO格式的JSON文件，用于评估数据集。

    Args:
        img_dir (str): 评估图像目录，例如 'data/g/g_val/imgs/'
        output_json (str): 输出JSON文件路径，例如 'data/g/g_val/val_dataset.json'
    """
    images = []
    annotations = []
    # 定义类别：只包含健康 (0) 和结核病 (2)
    categories = [
        {"id": 0, "name": "healthy", "supercategory": "classification"},
        {"id": 2, "name": "tb", "supercategory": "classification"}
    ]

    img_id = 1
    ann_id = 1

    # 遍历图像目录
    for img_file in os.listdir(img_dir):
        if img_file.endswith('.png'):
            # 从文件名提取标签
            parts = img_file.split('_')
            file_label = int(parts[-1].split('.')[0])  # 最后一个下划线后的数字
            # 映射文件名中的标签到模型类别ID
            if file_label == 0:
                category_id = 0  # 健康
            elif file_label == 1:
                category_id = 2  # 结核病 (映射 1 -> 2)
            else:
                print(f"警告: 图像 {img_file} 的标签 {file_label} 不符合预期，跳过")
                continue

            # 获取图像尺寸
            img_path = os.path.join(img_dir, img_file)
            with Image.open(img_path) as img:
                width, height = img.size

            # 添加图像信息
            images.append({
                "id": img_id,
                "file_name": img_file,
                "width": width,
                "height": height,
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "license": 1,
                "coco_url": "",
                "flickr_url": ""
            })

            # 添加标注信息
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id
            })

            img_id += 1
            ann_id += 1

    # 构造COCO格式的JSON
    coco_json = {
        "info": {
            "contributor": "Your Name",
            "date_created": datetime.now().strftime("%Y/%m/%d"),
            "description": "Evaluation Dataset for TB Classification",
            "url": "",
            "version": "1.0",
            "year": 2023
        },
        "licenses": [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ],
        "categories": categories,
        "images": images,
        "annotations": annotations
    }

    # 保存JSON文件
    with open(output_json, 'w') as f:
        json.dump(coco_json, f, indent=4)
    print(f"已生成COCO JSON文件: {output_json}，包含 {len(images)} 张图像")


# 使用示例
if __name__ == "__main__":
    img_dir = 'data/g/g_val/imgs/'
    output_json = 'data/g/g_val/val_dataset.json'
    generate_coco_json(img_dir, output_json)
