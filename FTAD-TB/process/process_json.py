import os
import json
from PIL import Image
import time

# 定义文件夹路径和输出文件
# root = '../../data/Pakistan/'
# root = '../../data/ChinaSet_AllFiles/'
# root = '../../data/mc+shenzhen/'
root = '../../data/test/'
# root = '../../data/lowres_mc+shenzhen/'
# root = '../../data/g/g_test/'

img_folder = root + 'img'  # 图片文件夹路径
output_json = root + 'test_dataset.json'  # 输出的JSON文件

# 获取文件夹中的所有图片
image_files = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 获取当前时间戳
timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

# 定义JSON结构
data = {
    "info": [
        {
            "contributor": "Yun Liu, Yu-Huan Wu, Yunfeng Ban, Huifang Wang, Ming-Ming Cheng",
            "date_created": "2020/06/22",
            "description": "TBX11K Dataset",
            "url": "http://mmcheng.net/tb",
            "version": "1.0",
            "year": 2020
        }
    ],
    "licenses": [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "ActiveTuberculosis",
            "supercategory": "Tuberculosis"
        },
        {
            "id": 2,
            "name": "ObsoletePulmonaryTuberculosis",
            "supercategory": "Tuberculosis"
        },
        {
            "id": 3,
            "name": "PulmonaryTuberculosis",
            "supercategory": "Tuberculosis"
        }
    ],
    "images": []
}

# 遍历图片文件，获取图片的宽度和高度
for idx, image_file in enumerate(image_files):
    image_path = os.path.join(img_folder, image_file)
    with Image.open(image_path) as img:
        width, height = img.size

    image_info = {
        "id": idx + 1,
        "file_name": f"{image_file}",
        "width": width,
        "height": height,
        "date_captured": timestamp,
        "license": 1,
        "coco_url": "",
        "flickr_url": ""
    }
    data["images"].append(image_info)

# 将结果写入JSON文件
with open(output_json, 'w') as f:
    json.dump(data, f, indent=4)

print(f"JSON文件已保存为 {output_json}")
