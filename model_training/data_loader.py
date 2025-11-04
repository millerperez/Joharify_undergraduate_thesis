# -*- coding: utf-8 -*-
# @Time: 2025/10/22 11:31
# @Author: Joharify
# @File: data_loader.py
# @Software: PyCharm

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import Config
from split_dataset import check_and_split_dataset


class DataLoaderManager:
    """数据加载管理器"""

    def __init__(self, config):
        self.config = config
        self.data_transforms = self._get_transforms()
        self.image_datasets = {}
        self.dataloaders = {}
        self.class_names = []
        self.num_classes = 0
        self.dataset_sizes = {}

    def _get_transforms(self):
        """获取数据预处理流程"""
        return {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.config.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.normalize_mean,
                    std=self.config.normalize_std
                )
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.normalize_mean,
                    std=self.config.normalize_std
                )
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.normalize_mean,
                    std=self.config.normalize_std
                )
            ])
        }

    def _detect_dataset_structure(self):
        """检测数据集结构"""
        data_dir = self.config.data_dir

        # 检查是否已经划分
        has_train = os.path.exists(os.path.join(data_dir, 'train'))
        has_val = os.path.exists(os.path.join(data_dir, 'val'))
        has_test = os.path.exists(os.path.join(data_dir, 'test'))

        if has_train and has_val:
            print("✅ 检测到已划分的数据集结构")
            return 'split'

        # 检查是否是未划分的结构（直接包含类别文件夹）
        items = os.listdir(data_dir)
        class_folders = [d for d in items
                         if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]

        # 检查这些文件夹中是否包含图像文件
        is_classification_structure = False
        for folder in class_folders:
            folder_path = os.path.join(data_dir, folder)
            image_files = [f for f in os.listdir(folder_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            if image_files:
                is_classification_structure = True
                break

        if is_classification_structure:
            print("📁 检测到未划分的分类数据集结构")
            return 'unsplit'
        else:
            raise ValueError(f"无法识别数据集结构: {data_dir}")

    def setup_data(self, auto_split=True):
        """设置数据加载器，自动检测并处理数据集结构"""
        data_dir = self.config.data_dir

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        # 检测数据集结构
        structure_type = self._detect_dataset_structure()

        if structure_type == 'unsplit' and auto_split:
            print("🔄 自动划分数据集...")
            # 自动划分数据集
            was_split = check_and_split_dataset(data_dir, auto_split=True)
            if was_split:
                print("✅ 数据集划分完成")
            else:
                print("ℹ️  数据集已存在划分，跳过自动划分")

        # 现在加载划分后的数据
        splits = ['train', 'val', 'test']
        available_splits = []

        for split in splits:
            split_dir = os.path.join(data_dir, split)
            if os.path.exists(split_dir):
                available_splits.append(split)
            else:
                print(f"⚠️  {split}目录不存在: {split_dir}")

        if 'train' not in available_splits:
            raise ValueError("训练集不存在，无法进行训练")

        print(f"📂 可用的数据分割: {available_splits}")

        # 加载数据集
        for split in available_splits:
            self.image_datasets[split] = datasets.ImageFolder(
                root=os.path.join(data_dir, split),
                transform=self.data_transforms[split]
            )

        # 创建数据加载器
        for split in available_splits:
            self.dataloaders[split] = DataLoader(
                self.image_datasets[split],
                batch_size=self.config.batch_size,
                shuffle=(split == 'train'),
                num_workers=self.config.num_workers
            )

        # 获取数据集信息
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in available_splits}
        self.class_names = self.image_datasets['train'].classes
        self.num_classes = len(self.class_names)

        self._print_dataset_info()

    def _print_dataset_info(self):
        """打印数据集信息"""
        print(f"📦 数据集类别: {self.class_names}")
        print(f"🔢 类别数量: {self.num_classes}")

        total_size = sum(self.dataset_sizes.values())
        for split, size in self.dataset_sizes.items():
            ratio = (size / total_size * 100) if total_size > 0 else 0
            print(f"📂 {split}集样本数: {size} ({ratio:.1f}%)")

    def get_dataloaders(self):
        """获取数据加载器"""
        return self.dataloaders

    def get_dataset_info(self):
        """获取数据集信息"""
        return {
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'dataset_sizes': self.dataset_sizes,
            'has_test_set': 'test' in self.image_datasets
        }

    def get_complete_dataset(self, phase='train'):
        """获取完整数据集（用于K折交叉验证）"""
        if phase not in ['train', 'val']:
            raise ValueError("phase必须是'train'或'val'")

        complete_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])

        dataset = datasets.ImageFolder(
            root=os.path.join(self.config.data_dir, phase),
            transform=complete_transform
        )

        return dataset