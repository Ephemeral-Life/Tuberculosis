#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
import subprocess

# —— 如果你的虚拟环境目录名不是 venv，请修改下面这行路径 —— #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # 获取上一级目录
VENV_PYTHON = os.path.join(PROJECT_ROOT, 'venv', 'Scripts', 'python.exe')

# —— 准确度计算函数 —— #
def read_actual_labels(folder_path):
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f.read().splitlines() if l.strip()]
                if lines:
                    labels.append(lines[-1].lower())
    return labels

def read_model_predictions(prediction_file):
    with open(prediction_file, 'r', encoding='utf-8') as f:
        return [int(line.strip()) for line in f if line.strip()]

def calculate_accuracy(actual_labels, predictions):
    correct_normal = correct_other = 0
    false_normal_to_other = false_other_to_normal = 0
    total = len(actual_labels)
    for a, p in zip(actual_labels, predictions):
        if a == "0" and p == 0:
            correct_normal += 1
        elif a == "0" and p == 1:
            false_normal_to_other += 1
        elif a != "0" and p == 1:
            correct_other += 1
        elif a != "0" and p == 0:
            false_other_to_normal += 1
    accuracy = (correct_normal + correct_other) / total * 100 if total > 0 else 0.0
    return correct_normal, correct_other, false_normal_to_other, false_other_to_normal, accuracy

# —— 主流程 —— #
def main():
    # 路径配置
    config_path       = os.path.join(PROJECT_ROOT, 'configs', 'symformer',
                                     'symformer_retinanet_p2t_cls_fpn_1x_TBX11K_test_shenzhen_val.py')
    pth_dir           = os.path.join(PROJECT_ROOT, 'work_dirs', 'symformer_retinanet_p2t_cls_flower')
    result_dir        = os.path.join(pth_dir, 'result')
    annotation_folder = os.path.join(PROJECT_ROOT, 'data', 'test', 'annotations')

    os.makedirs(result_dir, exist_ok=True)
    summary_file = os.path.join(result_dir,'evaluation_summary.txt')

    # 写入表头
    with open(summary_file, 'w', encoding='utf-8') as sf:
        sf.write('round\ttotal\tcorrect_normal\tcorrect_other\tfalse_normal_to_other\tfalse_other_to_normal\taccuracy(%)\n')

    # 收集并排序所有聚合模型
    # pattern = os.path.join(pth_dir, 'epoch_*.pth')
    pattern = os.path.join(pth_dir, 'aggregated_model_round_*.pth')
    pth_files = glob.glob(pattern)
    def get_round(fp):
        # m = re.search(r'epoch_(\d+)\.pth$', fp)
        m = re.search(r'aggregated_model_round_(\d+)\.pth$', fp)
        return int(m.group(1)) if m else -1
    pth_files = sorted(pth_files, key=get_round)

    # 逐轮测试并记录
    for pth_path in pth_files:
        rnd = get_round(pth_path)
        out_pkl  = os.path.join(result_dir, f'result_round_{rnd}.pkl')
        txt_pred = os.path.join(result_dir, f'cls_result_round_{rnd}.txt')

        cmd = [
            VENV_PYTHON,
            '-W', 'ignore',
            os.path.join(PROJECT_ROOT, 'tools', 'test1.py'),
            config_path,
            pth_path,
            '--cls-filter', 'True',
            '--out', out_pkl,
            '--txt', txt_pred,
        ]
        print(f'\n\n[INFO] 正在测试第 {rnd} 轮模型：{os.path.basename(pth_path)}')
        # 确保在项目根目录下执行
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

        # 读取并计算
        actual = read_actual_labels(annotation_folder)
        pred   = read_model_predictions(txt_pred)
        if len(actual) != len(pred):
            print(f'\n\n[WARNING] 第 {rnd} 轮：标签数 ({len(actual)}) 与预测数 ({len(pred)}) 不一致，跳过统计。')
            continue

        cn, co, fn2o, fo2n, acc = calculate_accuracy(actual, pred)
        total = len(actual)

        # 追加到汇总文件
        with open(summary_file, 'a', encoding='utf-8') as sf:
            sf.write(f'{rnd}\t{total}\t{cn}\t{co}\t{fn2o}\t{fo2n}\t{acc:.2f}\n')

    print(f'\n\n[DONE] 全部轮次评估完成，结果保存在：{summary_file}')

if __name__ == '__main__':
    main()
