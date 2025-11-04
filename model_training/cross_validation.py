# -*- coding: utf-8 -*-
# @Time: 2025/10/22 11:31
# @Author: Joharify
# @File: cross_validation.py
# @Software: PyCharm

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from data_loader import DataLoaderManager
from model_loader import ModelLoader
from trainer import Trainer
from evaluator import ModelEvaluator


class CrossValidator:
    """K折交叉验证器"""

    def __init__(self, config, model_name, num_classes, class_names):
        self.config = config
        self.model_name = model_name
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.k_folds = config.k_folds
        self.results = {
            'fold_accuracies': [],
            'fold_losses': [],
            'best_epochs': [],
            'train_histories': [],
            'val_histories': []
        }

    def prepare_kfold_data(self, dataset):
        """准备K折交叉验证数据"""
        # 使用分层K折交叉验证，保持每个折的类别分布一致
        skf = StratifiedKFold(
            n_splits=self.k_folds,
            shuffle=True,
            random_state=self.config.cv_random_seed
        )

        # 获取所有样本的标签
        labels = [label for _, label in dataset]

        return skf, labels

    def run_cross_validation(self, dataset):
        """执行K折交叉验证"""
        print(f"🔄 开始 {self.k_folds}折交叉验证...")
        print(f"📊 总样本数: {len(dataset)}")

        # 准备K折数据
        skf, labels = self.prepare_kfold_data(dataset)

        fold_results = []

        # 修复语法错误：在enumerate中添加括号
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
            print(f"\n{'=' * 50}")
            print(f"🔄 训练第 {fold + 1}/{self.k_folds} 折")
            print(f"{'=' * 50}")

            # 创建当前折的数据集
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            # 创建数据加载器
            train_loader = DataLoader(
                train_subset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers
            )

            dataloaders = {'train': train_loader, 'val': val_loader}
            dataset_sizes = {'train': len(train_subset), 'val': len(val_subset)}

            print(f"📊 训练集样本数: {len(train_subset)}")
            print(f"📊 验证集样本数: {len(val_subset)}")

            # 训练当前折的模型
            fold_result = self._train_fold(fold, dataloaders, dataset_sizes)
            fold_results.append(fold_result)

            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 汇总结果
        self._aggregate_results(fold_results)
        self._visualize_results()

        return self.results

    def _train_fold(self, fold, dataloaders, dataset_sizes):
        """训练单个折的模型"""
        # 创建新模型实例（每个折使用独立的模型）
        model_loader = ModelLoader(self.config)
        model = model_loader.load_model(self.model_name, self.num_classes)

        # 调整TensorBoard日志目录
        original_log_dir = self.config.tensorboard_log_dir
        self.config.tensorboard_log_dir = f"{original_log_dir}_fold{fold + 1}"

        # 调整模型保存路径
        original_model_path = self.config.model_save_path
        self.config.model_save_path = original_model_path.replace('.pth', f'_fold{fold + 1}.pth')

        # 训练模型
        trainer = Trainer(model, dataloaders, dataset_sizes, self.config)
        trained_model = trainer.train(self.config.num_epochs)

        # 评估当前折的最佳模型
        best_model_path = self.config.model_save_path
        best_model = ModelLoader(self.config).load_model(self.model_name, self.num_classes)

        if os.path.exists(best_model_path):
            best_model.load_state_dict(torch.load(best_model_path))

            # 评估验证集性能
            evaluator = ModelEvaluator(best_model, dataloaders, self.class_names, self.device)
            val_result = evaluator.evaluate('val')

            # 保存当前折的结果
            fold_result = {
                'fold': fold + 1,
                'model': best_model,
                'accuracy': val_result['accuracy'] if val_result else 0.0,
                'best_model_path': best_model_path
            }
        else:
            print(f"⚠️ 最佳模型文件不存在: {best_model_path}")
            fold_result = {
                'fold': fold + 1,
                'model': None,
                'accuracy': 0.0,
                'best_model_path': best_model_path
            }

        # 恢复原始配置
        self.config.tensorboard_log_dir = original_log_dir
        self.config.model_save_path = original_model_path

        return fold_result

    def _aggregate_results(self, fold_results):
        """汇总各折结果"""
        accuracies = [result['accuracy'] for result in fold_results if result['accuracy'] > 0]

        if not accuracies:
            print("❌ 所有折的训练都失败了")
            return

        self.results['fold_accuracies'] = accuracies
        self.results['mean_accuracy'] = np.mean(accuracies)
        self.results['std_accuracy'] = np.std(accuracies)
        self.results['best_fold'] = np.argmax(accuracies) + 1
        self.results['best_accuracy'] = np.max(accuracies)
        self.results['all_fold_results'] = fold_results

        print(f"\n{'=' * 60}")
        print(f"🎯 {self.k_folds}折交叉验证结果汇总")
        print(f"{'=' * 60}")

        for i, acc in enumerate(accuracies, 1):
            print(f"第 {i} 折验证集准确率: {acc:.4f}")

        print(f"\n📊 平均准确率: {self.results['mean_accuracy']:.4f} ± {self.results['std_accuracy']:.4f}")
        print(f"🏆 最佳折: 第 {self.results['best_fold']} 折, 准确率: {self.results['best_accuracy']:.4f}")
        print(f"{'=' * 60}")

    def _visualize_results(self):
        """可视化交叉验证结果"""
        os.makedirs('results', exist_ok=True)

        if not self.results['fold_accuracies']:
            print("⚠️ 没有可用的准确率数据用于可视化")
            return

        # 绘制准确率分布图
        plt.figure(figsize=(10, 6))
        folds = range(1, len(self.results['fold_accuracies']) + 1)

        plt.bar(folds, self.results['fold_accuracies'], alpha=0.7, color='skyblue')
        plt.axhline(y=self.results['mean_accuracy'], color='red', linestyle='--',
                    label=f'平均准确率: {self.results["mean_accuracy"]:.4f}')

        plt.xlabel('折数')
        plt.ylabel('验证集准确率')
        plt.title(f'{self.k_folds}折交叉验证结果')
        plt.xticks(folds)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig('results/cross_validation_results.png', bbox_inches='tight', dpi=300)
        plt.close()

        # 保存详细结果到JSON文件
        results_dict = {
            'model_name': self.model_name,
            'k_folds': self.k_folds,
            'mean_accuracy': float(self.results['mean_accuracy']),
            'std_accuracy': float(self.results['std_accuracy']),
            'fold_accuracies': [float(acc) for acc in self.results['fold_accuracies']],
            'best_fold': int(self.results['best_fold']),
            'best_accuracy': float(self.results['best_accuracy'])
        }

        with open('results/cross_validation_details.json', 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        print(f"📈 结果可视化已保存: results/cross_validation_results.png")
        print(f"📄 详细结果已保存: results/cross_validation_details.json")


class KFoldDataLoader:
    """K折交叉验证数据加载器"""

    def __init__(self, config):
        self.config = config

    def setup_kfold_data(self, use_original_split=False):
        """
        设置K折交叉验证数据

        参数:
        - use_original_split: 是否使用原始的训练/验证集划分
        """
        if use_original_split:
            # 使用现有的训练集进行K折交叉验证
            data_manager = DataLoaderManager(self.config)
            data_manager.setup_data(auto_split=False)
            train_dataset = data_manager.image_datasets['train']
            return train_dataset
        else:
            # 使用完整数据集进行K折交叉验证
            return self._create_complete_dataset()

    def _create_complete_dataset(self):
        """创建完整数据集（合并训练集和验证集）"""
        from torchvision import transforms, datasets

        # 加载所有数据，不区分训练/验证
        complete_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])

        # 检查数据目录结构
        data_dir = self.config.data_dir

        # 如果存在train目录，使用train目录
        if os.path.exists(os.path.join(data_dir, 'train')):
            complete_dataset = datasets.ImageFolder(
                root=os.path.join(data_dir, 'train'),
                transform=complete_transform
            )
        else:
            # 直接使用数据根目录
            complete_dataset = datasets.ImageFolder(
                root=data_dir,
                transform=complete_transform
            )

        print(f"📊 完整数据集样本数: {len(complete_dataset)}")
        print(f"📦 数据集类别: {complete_dataset.classes}")

        return complete_dataset