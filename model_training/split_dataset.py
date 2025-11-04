# -*- coding: utf-8 -*-
# @Time: 2025/10/22 11:31
# @Author: Joharify
# @File: split_dataset.py
# @Software: PyCharm

import os
import shutil
import random
from sklearn.model_selection import train_test_split


def split_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    将未划分的数据集自动划分为训练集、验证集和测试集

    参数:
    - data_dir: 原始数据集目录（包含类别子文件夹）
    - output_dir: 输出目录
    - train_ratio: 训练集比例
    - val_ratio: 验证集比例
    - test_ratio: 测试集比例
    - seed: 随机种子
    """
    # 检查比例总和是否为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"

    random.seed(seed)

    # 创建输出目录
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # 获取所有类别（排除隐藏文件）
    classes = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]

    if not classes:
        raise ValueError(f"在目录 {data_dir} 中没有找到任何类别文件夹")

    print(f"📂 发现 {len(classes)} 个类别: {classes}")

    total_files = 0

    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)

        # 获取该类别的所有图像文件
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]

        if not image_files:
            print(f"⚠️  类别 '{class_name}' 中没有图像文件，跳过")
            continue

        print(f"📁 处理类别 '{class_name}': {len(image_files)} 个图像")
        total_files += len(image_files)

        # 第一次划分：分离测试集
        train_val_files, test_files = train_test_split(
            image_files, test_size=test_ratio, random_state=seed, shuffle=True
        )

        # 第二次划分：分离训练集和验证集
        val_ratio_adj = val_ratio / (train_ratio + val_ratio)  # 调整比例
        train_files, val_files = train_test_split(
            train_val_files, test_size=val_ratio_adj, random_state=seed, shuffle=True
        )

        # 复制文件到相应目录
        for split, files in zip(splits, [train_files, val_files, test_files]):
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for file in files:
                src_path = os.path.join(class_dir, file)
                dst_path = os.path.join(split_class_dir, file)

                # 复制文件
                shutil.copy2(src_path, dst_path)

            print(f"   ✅ {split}: {len(files)} 个图像")

    print(f"\n🎉 数据集划分完成！")
    print(f"📊 总计处理 {total_files} 个图像文件")
    print(f"📁 原始数据: {data_dir}")
    print(f"📁 划分后数据: {output_dir}")
    print(f"📊 划分比例: 训练集 {train_ratio * 100}% | 验证集 {val_ratio * 100}% | 测试集 {test_ratio * 100}%")

    return True


def check_and_split_dataset(data_dir, output_dir=None, auto_split=True):
    """
    检查数据集结构并自动划分

    参数:
    - data_dir: 数据目录
    - output_dir: 输出目录（如果为None，则使用data_dir）
    - auto_split: 是否自动划分
    """
    if output_dir is None:
        output_dir = data_dir

    # 检查是否已经划分
    has_train = os.path.exists(os.path.join(data_dir, 'train'))
    has_val = os.path.exists(os.path.join(data_dir, 'val'))
    has_test = os.path.exists(os.path.join(data_dir, 'test'))

    if has_train and has_val:
        print("✅ 数据集已经划分好")
        if has_test:
            print("📁 数据集结构: train/val/test")
        else:
            print("📁 数据集结构: train/val")
        return False  # 不需要重新划分

    # 检查是否是未划分的结构（直接包含类别文件夹）
    classes = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]

    if not classes:
        raise ValueError(f"无法识别数据集结构。请在 {data_dir} 中放置类别文件夹")

    print("📁 检测到未划分的数据集结构:")
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        image_count = len([f for f in os.listdir(cls_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        print(f"   {cls}: {image_count} 个图像")

    if auto_split:
        print("\n🔄 开始自动划分数据集...")
        # 如果输出目录与数据目录相同，需要特殊处理
        if output_dir == data_dir:
            # 创建临时目录进行划分
            temp_dir = data_dir + "_temp"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

            split_dataset(data_dir, temp_dir)

            # 移动文件回原目录
            for split in ['train', 'val', 'test']:
                split_src = os.path.join(temp_dir, split)
                split_dst = os.path.join(data_dir, split)
                if os.path.exists(split_src):
                    if os.path.exists(split_dst):
                        shutil.rmtree(split_dst)
                    shutil.move(split_src, data_dir)

            # 清理临时目录
            shutil.rmtree(temp_dir)
        else:
            split_dataset(data_dir, output_dir)

        return True
    else:
        print("ℹ️  检测到未划分的数据集，但未启用自动划分")
        return False


if __name__ == "__main__":
    # 使用示例
    original_data_dir = "./dataset_root"  # 您的数据目录

    if os.path.exists(original_data_dir):
        check_and_split_dataset(original_data_dir, auto_split=True)
    else:
        print(f"❌ 数据目录不存在: {original_data_dir}")
        print("💡 请将您的图像数据按类别组织在相应的子文件夹中")