# -*- coding: utf-8 -*-
# @Time: 2025/10/31 20:16
# @Author: Joharify
# @File: plant_disease_classifier
# @Software: PyCharm 
# @Comment: None

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


class PlantDiseaseClassifier:
    def __init__(self, model_path, device=None):
        """
        初始化植物病害分类器
        """
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 加载保存的模型信息
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # 提取类别信息
        self.class_names = checkpoint['class_names']
        self.num_classes = checkpoint['num_classes']
        self.disease_info = checkpoint.get('disease_info', {})  # 病害详细信息

        # 加载模型结构
        self.model = self._load_model()

        # 加载训练好的权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"✅ 植物病害模型加载完成，使用设备: {self.device}")
        print(f"🎯 病害类别数量: {self.num_classes}")
        print(f"📊 病害类别: {self.class_names}")

    def _load_model(self):
        """加载模型结构"""
        model = models.mobilenet_v3_small(pretrained=False)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, self.num_classes)
        return model

    def predict(self, image_path):
        """
        对单张植物叶片图像进行病害预测
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 预处理
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)  # 添加batch维度

        # 移动到设备
        input_batch = input_batch.to(self.device)

        # 预测
        with torch.no_grad():
            output = self.model(input_batch)

        # 应用softmax获取概率
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # 获取预测结果
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class_name = self.class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx].item()

        # 获取所有类别的概率
        all_probabilities = {
            self.class_names[i]: round(probabilities[i].item(), 4)
            for i in range(self.num_classes)
        }

        # 判断是否为健康叶片
        is_healthy = "健康" in predicted_class_name or "healthy" in predicted_class_name.lower()

        return {
            'predicted_class': predicted_class_name,
            'confidence': round(confidence, 4),
            'all_probabilities': all_probabilities,
            'class_index': predicted_class_idx,
            'is_healthy': is_healthy,
            'disease_info': self.disease_info.get(predicted_class_name, {}),
            'recommendation': self._get_recommendation(predicted_class_name, confidence)
        }

    def predict_from_image(self, image):
        """
        从PIL Image对象进行植物病害预测
        """
        # 预处理
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)

        # 移动到设备
        input_batch = input_batch.to(self.device)

        # 预测
        with torch.no_grad():
            output = self.model(input_batch)

        # 应用softmax获取概率
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # 获取预测结果
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class_name = self.class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx].item()

        # 获取所有类别的概率
        all_probabilities = {
            self.class_names[i]: round(probabilities[i].item(), 4)
            for i in range(self.num_classes)
        }

        # 判断是否为健康叶片
        is_healthy = "健康" in predicted_class_name or "healthy" in predicted_class_name.lower()

        return {
            'predicted_class': predicted_class_name,
            'confidence': round(confidence, 4),
            'all_probabilities': all_probabilities,
            'class_index': predicted_class_idx,
            'is_healthy': is_healthy,
            'disease_info': self.disease_info.get(predicted_class_name, {}),
            'recommendation': self._get_recommendation(predicted_class_name, confidence)
        }

    def _get_recommendation(self, disease_name, confidence):
        """根据病害类型生成建议"""
        if confidence < 0.6:
            return "置信度较低，建议重新拍摄清晰叶片图片或咨询专业农技人员"

        if "健康" in disease_name or "healthy" in disease_name.lower():
            return "叶片健康状况良好，建议继续保持当前管理措施"

        # 根据不同病害类型给出建议
        recommendations = {
            "白粉病": "建议使用三唑酮或硫磺粉进行防治，注意通风透光",
            "锈病": "建议使用粉锈宁或代森锰锌进行防治，及时清除病叶",
            "霜霉病": "建议使用甲霜灵或代森锰锌，注意控制田间湿度",
            "叶斑病": "建议使用多菌灵或百菌清，及时清除病残体",
            "炭疽病": "建议使用咪鲜胺或苯醚甲环唑，加强肥水管理",
            "病毒病": "建议防治传毒媒介，使用宁南霉素等抗病毒剂",
            "缺素症": "建议进行土壤检测，补充相应营养元素"
        }

        for key, value in recommendations.items():
            if key in disease_name:
                return value

        return "建议及时采取防治措施，可咨询当地农技部门获取具体防治方案"