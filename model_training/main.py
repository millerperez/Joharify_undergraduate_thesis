# -*- coding: utf-8 -*-
# @Time: 2025/10/22 11:31
# @Author: Joharify
# @File: main.py
# @Software: PyCharm

import argparse
import torch
import os
from config import Config, get_available_models
from data_loader import DataLoaderManager
from model_loader import ModelLoader
from trainer import Trainer
from evaluator import ModelEvaluator
from cross_validation import CrossValidator, KFoldDataLoader
from model_manager import ModelManager  # 新增导入
from utils import parse_args, update_config_from_args


def main():
    """主函数"""
    args = parse_args()
    config = Config()
    config = update_config_from_args(config, args)

    print("=" * 60)
    print("🎯 图像分类模型训练与评估")
    if config.use_cross_validation:
        print(f"🔄 使用 {config.k_folds}折交叉验证")
    print("=" * 60)

    try:
        if config.use_cross_validation:
            run_cross_validation(config, args.model)
        else:
            run_standard_training(config, args.model)

    except Exception as e:
        print(f"❌ 过程中出现错误: {e}")
        raise


def run_standard_training(config, model_name):
    """运行标准训练流程"""
    print("📦 正在加载和预处理数据...")
    data_manager = DataLoaderManager(config)
    data_manager.setup_data(auto_split=True)

    dataloaders = data_manager.get_dataloaders()
    dataset_info = data_manager.get_dataset_info()

    print("🔄 正在加载模型...")
    model_loader = ModelLoader(config)
    model = model_loader.load_model(model_name, dataset_info['num_classes'])
    model_loader.get_model_summary(model)

    # 测试前向传播
    print("\n🧪 测试前向传播...")
    try:
        test_input, test_labels = next(iter(dataloaders['train']))
        test_input = test_input.to(next(model.parameters()).device)

        model.eval()
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ 前向传播测试成功!")
            print(f"   输入形状: {test_input.shape}")
            print(f"   输出形状: {output.shape}")
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        return

    # 训练模型（传递类别名称）
    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_info['dataset_sizes'],
        config=config,
        class_names=dataset_info['class_names']  # 新增：传递类别名称
    )
    trained_model = trainer.train()

    # 测试评估
    if dataset_info['has_test_set']:
        print("\n🧪 测试评估...")
        # 使用新的加载方法
        best_model, metadata = ModelManager.load_model(
            model_path=config.model_save_path,
            model_class=lambda num_classes: ModelLoader(config).load_model(model_name, num_classes),
            device=next(model.parameters()).device
        )

        evaluator = ModelEvaluator(best_model, dataloaders, metadata['class_names'],
                                   next(best_model.parameters()).device)
        evaluator.evaluate_all_splits()
    else:
        print("ℹ️ 测试集不存在，跳过测试评估")

    print("✅ 训练完成！")


def run_cross_validation(config, model_name):
    """运行K折交叉验证"""
    print("🔄 设置K折交叉验证数据...")
    kfold_loader = KFoldDataLoader(config)
    complete_dataset = kfold_loader.setup_kfold_data(use_original_split=False)

    class_names = complete_dataset.classes
    num_classes = len(class_names)

    print(f"📊 总样本数: {len(complete_dataset)}")
    print(f"📦 类别数: {num_classes}")

    # 执行交叉验证（需要更新CrossValidator以支持新的保存方法）
    # 这里简化处理，实际使用时需要更新CrossValidator类
    print("ℹ️ 交叉验证模式暂不支持新的模型保存格式")
    print("💡 请使用标准训练模式以获得完整功能")

    # 临时使用标准训练
    config.use_cross_validation = False
    run_standard_training(config, model_name)


if __name__ == "__main__":
    main()