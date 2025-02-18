import os


# 读取实际标签
def read_actual_labels(folder_path):
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                content = f.read().strip().splitlines()
                # 获取实际标签（最后一行可能是标签，去除空行）
                actual_label = content[-1].strip().lower()  # 取最后一行作为标签
                if actual_label:
                    labels.append(actual_label)
    return labels


# 读取模型预测结果
def read_model_predictions(prediction_file):
    with open(prediction_file, 'r', encoding='utf-8') as f:
        predictions = [int(line.strip()) for line in f.readlines()]
    return predictions


# 计算准确性
def calculate_accuracy(actual_labels, predictions):
    correct_normal = 0
    correct_other = 0
    false_normal_to_other = 0
    false_other_to_normal = 0

    total = len(actual_labels)

    for actual, predicted in zip(actual_labels, predictions):
        # 实际标签为 normal，预测为 normal
        if actual == "normal" and predicted == 0:
            correct_normal += 1
        # 实际标签为 normal，预测为其他
        elif actual == "normal" and predicted == 1:
            false_normal_to_other += 1
        # 实际标签为其他，预测为 1（其他）
        elif actual != "normal" and predicted == 1:
            correct_other += 1
        # 实际标签为其他，预测为 normal
        elif actual != "normal" and predicted == 0:
            false_other_to_normal += 1

    # 计算准确率
    correct_predictions = correct_normal + correct_other
    accuracy = correct_predictions / total * 100 if total > 0 else 0

    return correct_normal, correct_other, false_normal_to_other, false_other_to_normal, accuracy

# 输出结果
def print_results(correct_normal, correct_other, false_normal_to_other, false_other_to_normal, accuracy, total):
    correct_normal_percent = (correct_normal / total) * 100
    correct_other_percent = (correct_other / total) * 100
    false_normal_to_other_percent = (false_normal_to_other / total) * 100
    false_other_to_normal_percent = (false_other_to_normal / total) * 100

    print(f"总个数: {total}")
    print(f"正确判断为阴性的个数: {correct_normal} ({correct_normal_percent:.2f}%)")
    print(f"正确判断为阳性个数: {correct_other} ({correct_other_percent:.2f}%)")
    print(f"把阳性判断成阴性的个数: {false_normal_to_other} ({false_normal_to_other_percent:.2f}%)")
    print(f"把阴性判断成阳性的个数: {false_other_to_normal} ({false_other_to_normal_percent:.2f}%)")
    print(f"准确率: {accuracy:.2f}%")


# 修改后的主函数
def main():
    # 路径设置
    data_folder = "../data/ChinaSet_AllFiles/ClinicalReadings"  # 实际标签的文件夹路径
    prediction_file = "../work_dirs/symformer_retinanet_p2t_cls/result/cls_result.txt"  # 模型预测结果文件路径

    # 读取数据
    actual_labels = read_actual_labels(data_folder)
    predictions = read_model_predictions(prediction_file)

    # 确保实际标签和预测结果数量一致
    if len(actual_labels) != len(predictions):
        print("实际标签和预测结果数量不匹配！")
        return

    # 计算结果
    correct_normal, correct_other, false_normal_to_other, false_other_to_normal, accuracy = calculate_accuracy(
        actual_labels, predictions)

    # 输出结果并加入百分比
    total = len(actual_labels)  # 样本总数
    print_results(correct_normal, correct_other, false_normal_to_other, false_other_to_normal, accuracy, total)


if __name__ == "__main__":
    main()
