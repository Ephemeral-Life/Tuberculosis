import json
import os


# 定义函数来读取json文件
def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 读取标注文件夹中的txt文件内容的最后一行
def get_last_line_of_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return lines[-1].strip() if lines else ""


# 主函数
def process_annotations(json_path, annotations_folder, output_folder):
    # 加载json文件
    data = load_json(json_path)

    # 遍历json中的images部分
    for image in data['images']:
        file_name = image['file_name']
        txt_file_name = file_name.replace('.png', '.txt')  # 将.png改为.txt
        txt_path = os.path.join(annotations_folder, txt_file_name)

        if os.path.exists(txt_path):
            last_line = get_last_line_of_txt(txt_path)
            # 如果最后一行是"normal"，写入0，否则写入1
            label = 0 if last_line == 'normal' else 1
            output_file_path = os.path.join(output_folder, 'output.txt')

            # 将结果写入输出文件
            with open(output_file_path, 'a', encoding='utf-8') as output_file:
                output_file.write(f"{label}\n")
        else:
            print(f"警告: {txt_path} 不存在，跳过")


# 调用函数
root = '../../data/mc+shenzhen/'
json_path = root + 'test_dataset.json'  # 修改为你的json文件路径
annotations_folder = root + 'annotations'  # 修改为你的annotations文件夹路径
output_folder = root + ''  # 修改为你想保存输出的文件夹路径

process_annotations(json_path, annotations_folder, output_folder)
