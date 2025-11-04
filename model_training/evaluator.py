# -*- coding: utf-8 -*-
# @Time: 2025/10/22 11:31
# @Author: Joharify
# @File: evaluator.py
# @Software: PyCharm

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model, dataloaders, class_names, device):
        self.model = model
        self.dataloaders = dataloaders
        self.class_names = class_names
        self.device = device

    def evaluate(self, phase='test'):
        """评估模型在指定数据集上的性能"""
        if phase not in self.dataloaders:
            print(f"⚠️  {phase}集不存在，跳过评估")
            return None

        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloaders[phase], desc=f'评估{phase}集'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 计算评估指标
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

        print(f"\n🎯 {phase}集评估结果:")
        print(f"准确率: {accuracy:.4f}")

        # 详细分类报告
        print("\n📊 详细分类报告:")
        print(classification_report(all_labels, all_preds, target_names=self.class_names))

        # 混淆矩阵
        self._plot_confusion_matrix(all_labels, all_preds, phase)

        return {
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }

    def _plot_confusion_matrix(self, y_true, y_pred, phase):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {phase} Set')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        # 保存图像
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/confusion_matrix_{phase}.png', bbox_inches='tight', dpi=300)
        plt.close()

        print(f"📈 混淆矩阵已保存: results/confusion_matrix_{phase}.png")

    def evaluate_all_splits(self):
        """评估所有数据分割（训练集、验证集、测试集）"""
        results = {}

        for phase in ['train', 'val', 'test']:
            if phase in self.dataloaders:
                results[phase] = self.evaluate(phase)
            else:
                print(f"⚠️  {phase}集不存在，跳过评估")

        return results