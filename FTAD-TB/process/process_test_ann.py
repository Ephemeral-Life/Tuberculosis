import os
import argparse


def generate_annotation_files(image_dir, output_dir):
    """
    根据图片文件名生成对应的标注txt文件
    Args:
        image_dir (str): 包含图片的目录路径
        output_dir (str): 输出标注文件的目录路径
    """
    # 支持的图片扩展名列表
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm']

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历图片目录
    for filename in os.listdir(image_dir):
        # 分离文件名和扩展名
        base_name, ext = os.path.splitext(filename)

        # 只处理支持的图片格式
        if ext.lower() not in valid_extensions:
            continue

        # 解析文件名中的标签
        parts = base_name.split('_')
        if len(parts) < 2:
            print(f"警告：文件名 {filename} 不符合命名规范，跳过处理")
            continue

        # 获取最后一个下划线后的部分作为标签
        label_str = parts[-1]

        # 验证标签有效性
        if label_str not in ['0', '1']:
            print(f"警告：文件名 {filename} 包含无效标签 '{label_str}'，应为0或1，跳过处理")
            continue

        # 生成标注文件路径
        annotation_filename = f"{base_name}.txt"
        annotation_path = os.path.join(output_dir, annotation_filename)

        # 写入标注内容
        try:
            with open(annotation_path, 'w', encoding='utf-8') as f:
                f.write(label_str)
            print(f"已生成：{annotation_filename}")
        except Exception as e:
            print(f"错误：无法写入文件 {annotation_path} - {str(e)}")


if __name__ == "__main__":
    # 输入参数示例：
    # --image-dir /path/to/images
    # --output-dir /path/to/annotations
    generate_annotation_files("../../data/test/img", "../../data/test/annotations")

