import json
import os
from PIL import Image

# 图片所在目录
img_dir = r"D:\python workspace\Tuberculosis\data\g\g_val\img"
# 输出 JSON 文件路径
output_json = "val_dataset.json"

# 初始化 COCO 格式结构
data = {
    "images": [],
    "categories": [
        {"id": 0, "name": "healthy"},
        {"id": 1, "name": "sick_non_tb"},
        {"id": 2, "name": "tb"}
    ]
}

# label 到 category_id 的映射：0->0, 1->2, 2->1
label2cat = {
    0: 0,
    1: 2,
    2: 1
}

for img_file in os.listdir(img_dir):
    if not img_file.lower().endswith(".png"):
        continue

    # 从文件名中解析标签，比如 "xxx_1.png" -> label=1
    try:
        label = int(img_file.split('_')[-1].split('.')[0])
    except ValueError:
        print(f"跳过无法解析标签的文件: {img_file}")
        continue

    # 根据映射表获取 category_id
    if label not in label2cat:
        print(f"跳过标签不在映射表中的文件: {img_file}")
        continue
    category_id = label2cat[label]

    # 读取图片实际尺寸
    img_path = os.path.join(img_dir, img_file)
    with Image.open(img_path) as img:
        width, height = img.size

    # 追加到 images 列表
    data["images"].append({
        "id": len(data["images"]) + 1,
        "file_name": img_file,
        "width": width,
        "height": height,
        "category_id": category_id
    })

# 写入 JSON，支持中文并格式化
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"已生成 {output_json}，共 {len(data['images'])} 张图片信息。")
