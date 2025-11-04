# -*- coding: utf-8 -*-
# @Time: 2025/10/22 11:31
# @Author: Joharify
# @File: trainer.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from model_manager import ModelManager  # 新增导入


class Trainer:
    """模型训练器"""

    def __init__(self, model, dataloaders, dataset_sizes, config, class_names=None):
        self.model = model
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.config = config
        self.device = next(model.parameters()).device
        self.class_names = class_names or []  # 新增：类别名称

        # 训练组件
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config.step_size, gamma=config.gamma
        )

        # 创建日志目录
        os.makedirs(os.path.dirname(config.tensorboard_log_dir), exist_ok=True)
        self.writer = SummaryWriter(log_dir=config.tensorboard_log_dir)

        self.best_acc = 0.0

    def train_epoch(self, epoch, phase):
        """训练一个epoch"""
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_corrects = 0

        dataloader = self.dataloaders[phase]
        loop = tqdm(dataloader, desc=f'{phase} Epoch {epoch + 1}')

        for inputs, labels in loop:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # 更新进度条
            loop.set_postfix(
                loss=loss.item(),
                acc=torch.sum(preds == labels.data).double() / inputs.size(0)
            )

        epoch_loss = running_loss / self.dataset_sizes[phase]
        epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

        return epoch_loss, epoch_acc

    def train(self, num_epochs=None):
        """训练模型"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        print("🚀 开始训练模型...")

        # 训练历史记录
        train_history = []
        val_history = []

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            # 训练阶段
            train_loss, train_acc = self.train_epoch(epoch, 'train')
            print(f'train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')

            # 验证阶段
            val_loss, val_acc = self.train_epoch(epoch, 'val')
            print(f'val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

            # 记录历史
            train_history.append({'epoch': epoch, 'loss': train_loss, 'accuracy': train_acc})
            val_history.append({'epoch': epoch, 'loss': val_loss, 'accuracy': val_acc})

            # 更新学习率
            if epoch < num_epochs - 1:
                self.scheduler.step()

            # 记录到TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)

            # 保存验证集上性能最好的模型（使用新的保存方法）
            if val_acc > self.best_acc:
                self.best_acc = val_acc

                # 创建病害信息（示例）
                disease_info = ModelManager.create_disease_info(self.class_names)

                # 创建训练信息
                training_info = ModelManager.create_training_info(
                    dataset_info={'dataset_sizes': self.dataset_sizes},
                    training_config={
                        'num_epochs': num_epochs,
                        'batch_size': self.config.batch_size,
                        'learning_rate': self.config.learning_rate,
                        'image_size': self.config.image_size
                    },
                    performance_metrics={
                        'best_val_accuracy': float(self.best_acc),
                        'best_epoch': epoch
                    }
                )

                # 使用新的保存方法
                ModelManager.save_model(
                    model=self.model,
                    save_path=self.config.model_save_path,
                    class_names=self.class_names,
                    disease_info=disease_info,
                    training_info=training_info
                )

                print(f"💾 已保存最佳模型，验证集Acc: {self.best_acc:.4f}")

            print()

        self.writer.close()

        # 训练完成后的汇总信息
        final_training_info = ModelManager.create_training_info(
            dataset_info={'dataset_sizes': self.dataset_sizes},
            training_config={
                'num_epochs': num_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'image_size': self.config.image_size
            },
            performance_metrics={
                'best_val_accuracy': float(self.best_acc),
                'final_train_accuracy': float(train_history[-1]['accuracy']),
                'final_val_accuracy': float(val_history[-1]['accuracy'])
            }
        )

        # 保存最终模型
        final_model_path = self.config.model_save_path.replace('.pth', '_final.pth')
        ModelManager.save_model(
            model=self.model,
            save_path=final_model_path,
            class_names=self.class_names,
            disease_info=ModelManager.create_disease_info(self.class_names),
            training_info=final_training_info
        )

        print(f"🎉 训练完成！最佳验证集准确率: {self.best_acc:.4f}")
        print(f"📊 TensorBoard日志: {self.config.tensorboard_log_dir}")
        print(f"💾 最佳模型已保存: {self.config.model_save_path}")
        print(f"💾 最终模型已保存: {final_model_path}")

        return self.model