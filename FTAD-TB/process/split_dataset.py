import os
import shutil
import random


def split_dataset(img_dir='../../data/mc+shenzhen/img', val_dir='g_val', test_dir='g_test'):
    # 设置随机种子以保证结果可重复
    random.seed(42)

    # 创建存放验证集和测试集图片的文件夹（如果不存在）
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    negatives = []
    positives = []

    # 遍历 img 文件夹内所有文件
    for filename in os.listdir(img_dir):
        file_path = os.path.join(img_dir, filename)
        if os.path.isfile(file_path):
            # 提取不含扩展名的文件名，判断最后一个字符
            base, _ = os.path.splitext(filename)
            if base and base[-1] == '0':
                negatives.append(filename)
            else:
                positives.append(filename)

    # 分别对阴性和阳性数据进行随机排序
    random.shuffle(negatives)
    random.shuffle(positives)

    # 根据各自数量按照 2:1 的比例划分（验证集:测试集）
    num_neg = len(negatives)
    num_pos = len(positives)
    # 这里使用四舍五入保证大多数情况下比例较为合理
    split_neg = int(round(2 * num_neg / 3))
    split_pos = int(round(2 * num_pos / 3))

    neg_val = negatives[:split_neg]
    neg_test = negatives[split_neg:]
    pos_val = positives[:split_pos]
    pos_test = positives[split_pos:]

    # 将验证集图片复制到 val 文件夹
    for filename in neg_val + pos_val:
        src_path = os.path.join(img_dir, filename)
        dst_path = os.path.join(val_dir, filename)
        shutil.copy(src_path, dst_path)

    # 将测试集图片复制到 test 文件夹
    for filename in neg_test + pos_test:
        src_path = os.path.join(img_dir, filename)
        dst_path = os.path.join(test_dir, filename)
        shutil.copy(src_path, dst_path)


if __name__ == '__main__':
    split_dataset()
