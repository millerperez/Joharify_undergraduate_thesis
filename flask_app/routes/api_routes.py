# -*- coding: utf-8 -*-
# @Time: 2025/10/31 20:26
# @Author: Joharify
# @File: api_routes
# @Software: PyCharm 
# @Comment: None

import os
import base64
import io
from flask import jsonify, request
from PIL import Image
from werkzeug.utils import secure_filename
from flask_app.config import Config
from flask_app.plant_disease_classifier import PlantDiseaseClassifier  # 更新导入


class ApiRoutes:
    def __init__(self, classifier):
        self.classifier = classifier

    def init_routes(self, app):
        """初始化API路由"""

        @app.route('/api/predict', methods=['POST'])
        def predict():
            """植物病害预测接口 - 通过文件上传"""
            return self._predict_file()

        @app.route('/api/predict_base64', methods=['POST'])
        def predict_base64():
            """植物病害预测接口 - 通过Base64编码"""
            return self._predict_base64()

        @app.route('/api/health', methods=['GET'])
        def health_check():
            """健康检查接口"""
            return jsonify({
                'status': 'healthy',
                'model_loaded': self.classifier is not None,
                'app_name': Config.APP_NAME,
                'version': Config.APP_VERSION,
                'timestamp': Config.get_timestamp()
            })

        @app.route('/api/classes', methods=['GET'])
        def get_classes():
            """获取所有病害类别信息"""
            if self.classifier is None:
                return jsonify({
                    'success': False,
                    'error': '分类器未初始化',
                    'timestamp': Config.get_timestamp()
                }), 500

            return jsonify({
                'success': True,
                'classes': self.classifier.class_names,
                'num_classes': self.classifier.num_classes,
                'disease_info': self.classifier.disease_info,
                'timestamp': Config.get_timestamp()
            })

        @app.route('/api/info', methods=['GET'])
        def get_app_info():
            """获取应用信息"""
            return jsonify({
                'app_name': Config.APP_NAME,
                'description': Config.APP_DESCRIPTION,
                'version': Config.APP_VERSION,
                'supported_diseases': len(self.classifier.class_names) if self.classifier else 0,
                'timestamp': Config.get_timestamp()
            })

    def _allowed_file(self, filename):
        """检查文件扩展名是否允许"""
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

    def _predict_file(self):
        """处理文件上传预测"""
        try:
            # 检查是否有文件被上传
            if 'file' not in request.files:
                return jsonify({
                    'success': False,
                    'error': '没有上传叶片图片文件',
                    'timestamp': Config.get_timestamp()
                }), 400

            file = request.files['file']

            # 检查文件名
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': '没有选择文件',
                    'timestamp': Config.get_timestamp()
                }), 400

            # 检查文件类型
            if file and self._allowed_file(file.filename):
                # 安全保存文件名
                filename = secure_filename(file.filename)
                filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
                file.save(filepath)

                try:
                    # 进行病害预测
                    result = self.classifier.predict(filepath)

                    # 清理临时文件
                    os.remove(filepath)

                    return jsonify({
                        'success': True,
                        'prediction': result,
                        'filename': filename,
                        'timestamp': Config.get_timestamp()
                    })

                except Exception as e:
                    # 确保在出错时也清理文件
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    raise e
            else:
                return jsonify({
                    'success': False,
                    'error': '不支持的文件格式，请上传植物叶片图片文件（png, jpg, jpeg, bmp, gif）',
                    'timestamp': Config.get_timestamp()
                }), 400

        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'病害预测过程中发生错误: {str(e)}',
                'timestamp': Config.get_timestamp()
            }), 500

    def _predict_base64(self):
        """处理Base64图片预测"""
        try:
            data = request.get_json()

            if not data or 'image' not in data:
                return jsonify({
                    'success': False,
                    'error': '没有提供叶片图片数据',
                    'timestamp': Config.get_timestamp()
                }), 400

            # 解码Base64图片
            try:
                if 'image' in data:
                    # 去除可能的data:image前缀
                    image_data = data['image']
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]

                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                else:
                    return jsonify({
                        'success': False,
                        'error': '无效的图片数据',
                        'timestamp': Config.get_timestamp()
                    }), 400

            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'叶片图片解码失败: {str(e)}',
                    'timestamp': Config.get_timestamp()
                }), 400

            # 进行病害预测
            result = self.classifier.predict_from_image(image)

            return jsonify({
                'success': True,
                'prediction': result,
                'timestamp': Config.get_timestamp()
            })

        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'病害预测过程中发生错误: {str(e)}',
                'timestamp': Config.get_timestamp()
            }), 500