import os
import json
from tqdm import tqdm

def generate_client_annotations(clients_root_dir, all_ann_file, client_sizes):
    """
    为每个客户端生成标注JSON文件，基于总的标注文件。
    修改点：匹配逻辑改为仅根据文件名（而非完整路径）匹配标注信息

    Args:
        clients_root_dir (str): 包含客户端文件夹（0, 1, ..., n）的根目录。
        all_ann_file (str): 总的JSON标注文件的路径。
        client_sizes (dict): 将客户端ID映射到（宽度，高度）元组的字典。
    """
    # 加载总的标注文件
    if not os.path.exists(all_ann_file):
        raise FileNotFoundError(f"总的标注文件未找到: {all_ann_file}")
    with open(all_ann_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
        info = all_data.get('info', [])
        licenses = all_data.get('licenses', [])
        categories = all_data.get('categories', [])
        # 修改点1：使用文件名（而非完整路径）作为字典键
        images_dict = {os.path.basename(img['file_name']): img for img in all_data['images']}
        annotations_dict = {}
        for ann in all_data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in annotations_dict:
                annotations_dict[image_id] = []
            annotations_dict[image_id].append(ann)

    # 动态检测客户端文件夹
    client_dirs = [d for d in os.listdir(clients_root_dir) if
                   os.path.isdir(os.path.join(clients_root_dir, d)) and d.isdigit()]
    if not client_dirs:
        raise ValueError(f"在 {clients_root_dir} 中未找到客户端文件夹")

    # 处理每个客户端
    for client_id in sorted(client_dirs, key=int):
        client_ann = '../../client_ann/'
        client_dir = os.path.join(clients_root_dir, client_id)
        client_ann_file = os.path.join(client_ann, f'client_{client_id}_ann.json')
        client_images = []
        client_annotations = []
        size = client_sizes.get(int(client_id), (512, 512))  # 如果未指定，默认为512x512

        # 处理每个类别
        for category in ['health', 'sick', 'tb']:
            category_dir = os.path.join(client_dir, category)
            if not os.path.exists(category_dir):
                print(f"警告: 目录 {category_dir} 不存在，跳过")
                continue

            # 列出图像文件
            images = [f for f in os.listdir(category_dir) if
                      f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            if not images:
                print(f"警告: 在 {category_dir} 中未找到图像")
                continue

            # 匹配图像与标注
            for image in tqdm(images, desc=f"客户端 {client_id} - {category}"):
                # 修改点2：仅使用文件名部分匹配
                base_name = os.path.basename(image)
                if base_name in images_dict:
                    img_info = images_dict[base_name].copy()
                    # 修改点3：构造客户端专属路径（使用跨平台路径分隔符）
                    img_info['file_name'] = os.path.join(client_id, category, image).replace('\\', '/')
                    img_info['width'], img_info['height'] = size
                    client_images.append(img_info)
                    # 检查并添加相关的annotations信息
                    image_id = img_info['id']
                    if image_id in annotations_dict:
                        client_annotations.extend(annotations_dict[image_id])
                else:
                    print(f"警告: 图像 {base_name} 在 {all_ann_file} 中未找到")

        # 保存客户端的标注文件
        if client_images:
            client_data = {
                'info': info,
                'licenses': licenses,
                'categories': categories,
                'images': client_images,
                'annotations': client_annotations
            }
            os.makedirs(os.path.dirname(client_ann_file), exist_ok=True)
            with open(client_ann_file, 'w', encoding='utf-8') as f:
                json.dump(client_data, f, ensure_ascii=False, indent=4)
            print(f"保存了 {len(client_images)} 张图像到 {client_ann_file}")
        else:
            print(f"客户端 {client_id} 没有处理任何图像，跳过文件创建")

def main():
    # 配置（根据需要调整）
    clients_root_dir = '../../data/fed/clients'
    all_ann_file = '../../data/TBX11K/annotations/json/all_trainval_without_shenzhen_val.json'
    client_sizes = {
        0: (512, 512),
        1: (256, 256),
        2: (128, 128),
    }

    # 运行生成
    try:
        generate_client_annotations(clients_root_dir, all_ann_file, client_sizes)
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()
