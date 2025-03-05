import os
from PIL import Image


def resize_images(input_folder, output_folder, size=(256, 256)):
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # 仅处理文件，忽略子文件夹
        if os.path.isfile(input_path):
            # 判断是否为图片文件（根据扩展名判断）
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                try:
                    with Image.open(input_path) as img:
                        # 调整图片大小为256x256，使用LANCZOS滤波器提高缩放质量
                        img_resized = img.resize(size, Image.LANCZOS)
                        output_path = os.path.join(output_folder, filename)
                        # 保存图片，格式自动根据扩展名推断
                        img_resized.save(output_path)
                        print(f"{filename} 处理完成")
                except Exception as e:
                    print(f"处理 {filename} 时出错: {e}")
            else:
                print(f"{filename} 不是图片文件，已跳过")
        else:
            print(f"{filename} 不是文件，已跳过")


if __name__ == "__main__":
    # 定义输入和输出文件夹路径，请根据实际情况修改
    input_folder = "../data/mc+shenzhen/img"  # 存放原始图片的文件夹
    output_folder = "../data/lowres_mc+shenzhen/img"  # 存放处理后图片的文件夹

    resize_images(input_folder, output_folder)
