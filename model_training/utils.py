# -*- coding: utf-8 -*-
# @Time: 2025/10/22 11:31
# @Author: Joharify
# @File: utils.py
# @Software: PyCharm

import argparse
from config import get_available_models


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='图像分类模型训练脚本')

    parser.add_argument('--model', type=str, default='mobilenetv3_small',
                        choices=get_available_models(),
                        help='选择要训练的模型')
    parser.add_argument('--epochs', type=int, default=2,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批量大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--data-dir', type=str, default='./dataset_root',
                        help='数据集目录路径')
    parser.add_argument('--save-path', type=str, default='best_model.pth',
                        help='模型保存路径')
    parser.add_argument('--log-dir', type=str, default='runs/experiment',
                        help='TensorBoard日志目录')
    parser.add_argument('--k-folds', type=int, default=5,
                        help='K折交叉验证的折数')
    parser.add_argument('--cross-validation', action='store_true',
                        help='是否使用K折交叉验证')
    parser.add_argument('--cv-seed', type=int, default=42,
                        help='交叉验证随机种子')
    parser.add_argument('--no-auto-split', action='store_true',
                        help='禁用自动数据集划分')
    parser.add_argument('--disease-info', type=str, default='',
                        help='病害信息JSON文件路径（可选）')

    return parser.parse_args()


def update_config_from_args(config, args):
    """根据命令行参数更新配置"""
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.data_dir = args.data_dir
    config.model_save_path = args.save_path
    config.tensorboard_log_dir = args.log_dir

    return config