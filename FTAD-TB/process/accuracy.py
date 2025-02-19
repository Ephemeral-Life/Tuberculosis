def compare_txt_files(file1_path, file2_path):
    # 读取文件内容
    with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
        file1_lines = f1.readlines()
        file2_lines = f2.readlines()

    # 确保两个文件行数相等
    if len(file1_lines) != len(file2_lines):
        print("文件行数不相等，请检查文件")
        return

    # 初始化统计数据
    total = len(file1_lines)
    correct_normal = 0
    correct_other = 0
    false_normal_to_other = 0
    false_other_to_normal = 0

    # 比对两文件的每一行
    for line1, line2 in zip(file1_lines, file2_lines):
        # 解析每一行的标签（0为阴性，1为阳性）
        label1 = int(line1.strip())
        label2 = int(line2.strip())

        # 根据判断结果分类
        if label1 == label2:
            if label1 == 0:
                correct_normal += 1
            else:
                correct_other += 1
        else:
            if label1 == 0 and label2 == 1:
                false_normal_to_other += 1
            elif label1 == 1 and label2 == 0:
                false_other_to_normal += 1

    # 计算准确率
    correct = correct_normal + correct_other
    accuracy = correct / total * 100

    # 计算各类百分比
    correct_normal_percent = (correct_normal / total) * 100
    correct_other_percent = (correct_other / total) * 100
    false_normal_to_other_percent = (false_normal_to_other / total) * 100
    false_other_to_normal_percent = (false_other_to_normal / total) * 100

    # 输出结果
    print(f"总个数: {total}")
    print(f"正确判断为阴性的个数: {correct_normal} ({correct_normal_percent:.2f}%)")
    print(f"正确判断为阳性个数: {correct_other} ({correct_other_percent:.2f}%)")
    print(f"把阳性判断成阴性的个数: {false_normal_to_other} ({false_normal_to_other_percent:.2f}%)")
    print(f"把阴性判断成阳性的个数: {false_other_to_normal} ({false_other_to_normal_percent:.2f}%)")
    print(f"准确率: {accuracy:.2f}%")


# 使用示例
file1_path = '../../data/mc+shenzhen/result.txt'  # 替换为第一个txt文件的路径
file2_path = '../../work_dirs/symformer_retinanet_p2t_cls/result/cls_result.txt'  # 替换为第二个txt文件的路径

compare_txt_files(file1_path, file2_path)
