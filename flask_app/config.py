# -*- coding: utf-8 -*-
# @Time: 2025/10/31 20:13
# @Author: Joharify
# @File: config
# @Software: PyCharm 
# @Comment: None

import os
from datetime import datetime


class Config:
    """应用配置类"""
    SECRET_KEY = 'your-secret-key-here'
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
    MODEL_PATH = 'plant_disease_model.pth'  # 修改模型文件名

    # 应用信息
    APP_NAME = "植物叶片病害识别系统"
    APP_DESCRIPTION = "基于深度学习的植物叶片病害识别系统"
    APP_VERSION = "1.0.0"

    @staticmethod
    def get_timestamp():
        """获取当前时间戳"""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# 创建必要的目录
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)