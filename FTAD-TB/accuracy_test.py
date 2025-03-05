#!/usr/bin/env python
# -*- coding: utf-8 -*-

def main():
    file_path = "../work_dirs/symformer_retinanet_p2t_cls_256/result/cls_result.txt"
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"读取文件 {file_path} 失败：{e}")
        return

    total = len(lines)
    correct = 0
    wrong = 0
    false_positive = 0  # 误将阴性判断为阳性
    false_negative = 0  # 误将阳性判断为阴性
    total_negative = 514  # 阴性样本数
    total_positive = total - total_negative  # 阳性样本数

    for idx, line in enumerate(lines):
        line = line.strip()
        try:
            pred = int(line)
        except ValueError:
            print(f"第 {idx+1} 行内容无法转换为整数：{line}")
            wrong += 1
            continue

        expected = 0 if idx < total_negative else 1
        if pred == expected:
            correct += 1
        else:
            wrong += 1
            if expected == 0:
                false_positive += 1
            else:
                false_negative += 1

    accuracy = correct / total if total > 0 else 0
    false_positive_rate = false_positive / total_negative if total_negative > 0 else 0
    false_negative_rate = false_negative / total_positive if total_positive > 0 else 0

    print(f"总行数: {total}")
    print(f"正确预测行数: {correct}")
    print(f"错误预测行数: {wrong}")
    print(f"准确率: {accuracy:.2%}")
    print(f"阴性误判为阳性的数量: {false_positive}")
    print(f"阳性误判为阴性的数量: {false_negative}")
    print(f"阴性误判率: {false_positive_rate:.2%}")
    print(f"阳性误判率: {false_negative_rate:.2%}")
    # print(f"{1 - false_negative_rate:.2%}")


if __name__ == '__main__':
    main()

