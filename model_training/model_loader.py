# -*- coding: utf-8 -*-
# @Time: 2025/10/31 09:40
# @Author: Joharify
# @File: model_loader
# @Software: PyCharm 
# @Comment: None

import torch.nn as nn
from config import MODEL_CONFIGS, get_device


class ModelLoader:
    """模型加载器"""

    def __init__(self, config):
        self.config = config
        self.device = get_device()

    def load_model(self, model_name, num_classes):
        """加载指定模型"""
        if model_name not in MODEL_CONFIGS:
            available_models = list(MODEL_CONFIGS.keys())
            raise ValueError(f"不支持的模型: {model_name}。可用模型: {available_models}")

        model_config = MODEL_CONFIGS[model_name]

        # 加载预训练模型
        print(f"🔄 正在加载 {model_name}...")
        model = model_config['model_fn'](weights=model_config['weights'])

        # 根据模型类型调整分类器
        if model_name.startswith('mobilenetv3'):
            # MobileNetV3 调整
            model.classifier[3] = nn.Linear(model_config['feature_dim'], num_classes)
        elif model_name.startswith('resnet'):
            # ResNet 调整
            model.fc = nn.Linear(model_config['feature_dim'], num_classes)
        elif model_name.startswith('efficientnet'):
            # EfficientNet 调整
            model.classifier[1] = nn.Linear(model_config['feature_dim'], num_classes)
        else:
            raise ValueError(f"未知的模型类型: {model_name}")

        # 移动到设备
        model = model.to(self.device)

        print(f"✅ 已加载 {model_name}，设备: {self.device}")
        print(f"🎯 分类任务类别数: {num_classes}")

        return model

    def get_model_summary(self, model):
        """获取模型摘要"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"📊 模型参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")

        return {
            'total_params': total_params,
            'trainable_params': trainable_params
        }