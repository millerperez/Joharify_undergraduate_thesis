# -*- coding: utf-8 -*-
# @Time: 2025/10/31 20:10
# @Author: Joharify
# @File: app
# @Software: PyCharm 
# @Comment: None

import os
from flask import Flask
from config import Config
from plant_disease_classifier import PlantDiseaseClassifier  # 更新导入
from routes.api_routes import ApiRoutes
from routes.view_routes import ViewRoutes

# 全局分类器实例
classifier = None


def create_app():
    """创建Flask应用"""
    app = Flask(__name__)
    app.config.from_object(Config)

    # 初始化分类器
    global classifier
    try:
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(f"植物病害模型文件不存在: {Config.MODEL_PATH}")

        classifier = PlantDiseaseClassifier(Config.MODEL_PATH)
        print("✅ 植物病害分类器初始化完成")
        print(f"🌿 应用名称: {Config.APP_NAME}")
        print(f"📋 支持病害类别数: {classifier.num_classes}")
    except Exception as e:
        print(f"❌ 植物病害分类器初始化失败: {e}")
        return None

    # 初始化路由
    api_routes = ApiRoutes(classifier)
    view_routes = ViewRoutes()

    api_routes.init_routes(app)
    view_routes.init_routes(app)

    return app


def init_classifier():
    """初始化分类器（兼容旧版本）"""
    global classifier
    try:
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(f"植物病害模型文件不存在: {Config.MODEL_PATH}")

        classifier = PlantDiseaseClassifier(Config.MODEL_PATH)
        print("✅ 植物病害分类器初始化完成")
        return True
    except Exception as e:
        print(f"❌ 植物病害分类器初始化失败: {e}")
        return False


if __name__ == '__main__':
    app = create_app()

    if app is not None:
        print("🚀 启动植物叶片病害识别系统...")
        print("📝 前端界面: http://127.0.0.1:5000")
        print("🔍 健康检查: http://127.0.0.1:5000/api/health")
        print("📊 病害信息: http://127.0.0.1:5000/api/classes")
        print("ℹ️ 应用信息: http://127.0.0.1:5000/api/info")

        # 启动Flask应用
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ 植物病害识别系统启动失败")