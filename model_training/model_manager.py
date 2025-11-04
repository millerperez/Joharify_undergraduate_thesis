# -*- coding: utf-8 -*-
# @Time: 2025/10/22 11:31
# @Author: Joharify
# @File: model_manager.py
# @Software: PyCharm

import torch
import json
import os
from datetime import datetime


class ModelManager:
    """模型管理器，负责模型的保存和加载"""

    @staticmethod
    def save_model(model, save_path, class_names, disease_info=None, training_info=None):
        """
        保存模型和元数据

        参数:
        - model: 要保存的模型
        - save_path: 保存路径
        - class_names: 类别名称列表
        - disease_info: 病害详细信息字典
        - training_info: 训练信息字典
        """
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        # 准备模型检查点
        checkpoint = {
            # 模型参数
            'model_state_dict': model.state_dict(),
            'model_architecture': str(type(model).__name__),

            # 类别信息
            'class_names': class_names,
            'num_classes': len(class_names),

            # 病害详细信息
            'disease_info': disease_info or {},

            # 训练信息
            'training_info': training_info or {},

            # 元数据
            'metadata': {
                'save_time': datetime.now().isoformat(),
                'pytorch_version': torch.__version__,
                'model_type': type(model).__name__
            }
        }

        # 保存模型检查点
        torch.save(checkpoint, save_path)

        # 保存独立的元数据文件（便于查看）
        metadata_path = save_path.replace('.pth', '_metadata.json')
        metadata = {
            'class_names': class_names,
            'num_classes': len(class_names),
            'disease_info': disease_info or {},
            'training_info': training_info or {},
            'model_architecture': str(type(model).__name__),
            'save_time': checkpoint['metadata']['save_time']
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"💾 模型已保存: {save_path}")
        print(f"📄 元数据已保存: {metadata_path}")

        return save_path, metadata_path

    @staticmethod
    def load_model(model_path, model_class=None, device=None):
        """
        加载模型和元数据

        参数:
        - model_path: 模型文件路径
        - model_class: 模型类（可选，用于重建模型结构）
        - device: 设备

        返回:
        - model: 加载的模型
        - metadata: 模型元数据
        """
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载检查点
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # 提取元数据
        metadata = {
            'class_names': checkpoint.get('class_names', []),
            'num_classes': checkpoint.get('num_classes', 0),
            'disease_info': checkpoint.get('disease_info', {}),
            'training_info': checkpoint.get('training_info', {}),
            'model_architecture': checkpoint.get('model_architecture', 'Unknown')
        }

        # 如果提供了模型类，重建模型结构
        if model_class is not None:
            # 创建新模型实例
            model = model_class(num_classes=metadata['num_classes'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()

            print(f"✅ 模型加载成功: {model_path}")
            print(f"📊 类别信息: {metadata['class_names']}")
            print(f"🔢 类别数量: {metadata['num_classes']}")

            return model, metadata
        else:
            # 只返回检查点，让调用者处理模型重建
            return checkpoint, metadata

    @staticmethod
    def create_disease_info(class_names, descriptions=None, symptoms=None, treatments=None):
        """
        创建病害详细信息

        参数:
        - class_names: 类别名称列表
        - descriptions: 病害描述字典
        - symptoms: 病害症状字典
        - treatments: 治疗方法字典

        返回:
        - disease_info: 病害信息字典
        """
        disease_info = {}

        for i, class_name in enumerate(class_names):
            disease_info[class_name] = {
                'class_id': i,
                'description': descriptions.get(class_name,
                                                f"{class_name}的详细描述") if descriptions else f"{class_name}的详细描述",
                'symptoms': symptoms.get(class_name, []) if symptoms else [],
                'treatments': treatments.get(class_name, []) if treatments else [],
                'prevention': treatments.get(class_name, []) if treatments else [],
                'severity': 'medium'  # 默认严重程度
            }

        return disease_info

    @staticmethod
    def create_training_info(dataset_info, training_config, performance_metrics):
        """
        创建训练信息

        参数:
        - dataset_info: 数据集信息
        - training_config: 训练配置
        - performance_metrics: 性能指标

        返回:
        - training_info: 训练信息字典
        """
        return {
            'dataset': {
                'total_samples': sum(dataset_info.get('dataset_sizes', {}).values()),
                'class_distribution': dataset_info.get('dataset_sizes', {}),
                'image_size': training_config.get('image_size', 224)
            },
            'training': {
                'epochs': training_config.get('num_epochs', 10),
                'batch_size': training_config.get('batch_size', 32),
                'learning_rate': training_config.get('learning_rate', 0.001),
                'optimizer': 'Adam'
            },
            'performance': performance_metrics,
            'timestamp': datetime.now().isoformat()
        }