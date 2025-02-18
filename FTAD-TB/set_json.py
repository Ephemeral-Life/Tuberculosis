import os
import json
from datetime import datetime

# 目录路径
image_dir = "../data/ChinaSet_AllFiles/CXR_png"

# 获取图片文件名列表
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 获取当前时间
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 构建JSON数据
data = {
    "info": [
        {
            "contributor": "Your Name or Contributor Name",
            "date_created": "2025/01/19",
            "description": "Tuberculosis Image Dataset",
            "url": "Your dataset URL",
            "version": "1.0",
            "year": 2025
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
            "name": "Positive",
            "supercategory": "Tuberculosis"
        },
        {
            "id": 2,
            "name": "Negative",
            "supercategory": "Tuberculosis"
        }
    ],
    "images": []
}

# 遍历所有图片文件并添加到JSON数据
for idx, image_file in enumerate(image_files, start=1):
    image_path = os.path.join(image_dir, image_file)
    image_info = {
        "id": idx,
        "file_name": f"img\\{image_file}",
        "date_captured": current_time,
        "license": 1,
        "coco_url": "",
        "flickr_url": ""
    }
    data["images"].append(image_info)

# 将生成的数据保存为JSON文件
output_file = "dataset_info.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"JSON文件已保存至 {output_file}")
