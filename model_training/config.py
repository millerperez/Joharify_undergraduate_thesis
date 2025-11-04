# -*- coding: utf-8 -*-
# @Time: 2025/10/22 11:31
# @Author: Joharify
# @File: config.py
# @Software: PyCharm

import torch
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)
import json
import os


class Config:
    """训练配置参数"""

    # 数据配置
    data_dir = './dataset_root'
    batch_size = 32
    num_workers = 0

    # 数据集划分比例 (train:val:test)
    split_ratio = (0.7, 0.15, 0.15)

    # 交叉验证配置
    k_folds = 5
    use_cross_validation = False
    cv_random_seed = 42

    # 图像预处理参数
    image_size = 224
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]

    # 训练参数
    num_epochs = 10
    learning_rate = 0.001
    step_size = 7
    gamma = 0.1

    # 模型保存
    model_save_path = 'best_model.pth'
    metadata_save_path = 'model_metadata.json'  # 新增：元数据保存路径

    # TensorBoard配置
    tensorboard_log_dir = 'runs/experiment'


# 支持的模型配置
MODEL_CONFIGS = {
    'mobilenetv3_small': {
        'model_fn': mobilenet_v3_small,
        'weights': MobileNet_V3_Small_Weights.DEFAULT,
        'feature_dim': 1024
    },
    'mobilenetv3_large': {
        'model_fn': mobilenet_v3_large,
        'weights': MobileNet_V3_Large_Weights.DEFAULT,
        'feature_dim': 960
    },
    'resnet18': {
        'model_fn': resnet18,
        'weights': ResNet18_Weights.DEFAULT,
        'feature_dim': 512
    },
    'resnet50': {
        'model_fn': resnet50,
        'weights': ResNet50_Weights.DEFAULT,
        'feature_dim': 2048
    },
    'efficientnet_b0': {
        'model_fn': efficientnet_b0,
        'weights': EfficientNet_B0_Weights.DEFAULT,
        'feature_dim': 1280
    }
}


def get_available_models():
    return list(MODEL_CONFIGS.keys())


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_model_metadata(model_path, metadata):
    """保存模型元数据到JSON文件"""
    metadata_path = model_path.replace('.pth', '_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata_path


def load_model_metadata(model_path):
    """从JSON文件加载模型元数据"""
    metadata_path = model_path.replace('.pth', '_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None