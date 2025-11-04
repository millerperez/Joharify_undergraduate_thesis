# -*- coding: utf-8 -*-
# @Time: 2025/10/31 22:07
# @Author: Joharify
# @File: use_models
# @Software: PyCharm 
# @Comment: None
# -*- coding: utf-8 -*-
# @Time: 2025/10/22 11:31
# @Author: Joharify
# @File: use_model.py
# @Software: PyCharm
import os

import torch
from torchvision import transforms
from PIL import Image
from model_manager import ModelManager
from model_loader import ModelLoader
from config import Config


def load_trained_model(model_path, config):
    """加载训练好的模型"""
    print(f"🔍 加载模型: {model_path}")

    # 加载模型和元数据
    checkpoint, metadata = ModelManager.load_model(model_path)

    # 打印模型信息
    print("📋 模型信息:")
    print(f"   类别名称: {metadata['class_names']}")
    print(f"   类别数量: {metadata['num_classes']}")
    print(f"   模型架构: {metadata['model_architecture']}")
    print(f"   保存时间: {metadata.get('save_time', '未知')}")

    # 打印病害信息
    if metadata.get('disease_info'):
        print("\n🌱 病害信息:")
        for disease, info in metadata['disease_info'].items():
            print(f"   {disease}: {info.get('description', '无描述')}")

    # 重建模型
    model_loader = ModelLoader(config)
    model = model_loader.load_model('mobilenetv3_small', metadata['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, metadata


def predict_image(model, image_path, metadata, config):
    """使用模型预测单张图像"""
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std
        )
    ])

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # 添加batch维度

    # 预测
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, 1).item()
        confidence = probabilities[0][predicted_class].item()

    # 获取预测结果
    class_name = metadata['class_names'][predicted_class]
    disease_info = metadata['disease_info'].get(class_name, {})

    print(f"\n🔍 预测结果:")
    print(f"   图像: {image_path}")
    print(f"   预测类别: {class_name} (ID: {predicted_class})")
    print(f"   置信度: {confidence:.4f}")
    print(f"   描述: {disease_info.get('description', '无描述')}")

    if disease_info.get('symptoms'):
        print(f"   症状: {', '.join(disease_info['symptoms'])}")

    if disease_info.get('treatments'):
        print(f"   治疗方法: {', '.join(disease_info['treatments'])}")

    return predicted_class, confidence, class_name


if __name__ == "__main__":
    # 配置
    config = Config()
    model_path = "best_model.pth"

    try:
        # 加载模型
        model, metadata = load_trained_model(model_path, config)

        # 预测示例图像
        test_image = "test_image.jpg"  # 替换为您的测试图像路径
        if os.path.exists(test_image):
            predict_image(model, test_image, metadata, config)
        else:
            print(f"⚠️ 测试图像不存在: {test_image}")

    except Exception as e:
        print(f"❌ 错误: {e}")