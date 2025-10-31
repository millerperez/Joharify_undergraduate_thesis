# -*- coding: utf-8 -*-
# @Time: 2025/10/31 20:45
# @Author: Joharify
# @File: view_routes
# @Software: PyCharm 
# @Comment: None

from flask import render_template, jsonify
from flask_app.config import Config


class ViewRoutes:
    def init_routes(self, app):
        """初始化页面路由"""

        @app.route('/')
        def index():
            """首页"""
            return render_template('index.html')

        # 错误处理
        @app.errorhandler(413)
        def too_large(e):
            """文件过大错误处理"""
            return jsonify({
                'success': False,
                'error': '文件大小超过限制（16MB）',
                'timestamp': Config.get_timestamp()
            }), 413

        @app.errorhandler(404)
        def not_found(e):
            """404错误处理"""
            return jsonify({
                'success': False,
                'error': '接口不存在',
                'timestamp': Config.get_timestamp()
            }), 404

        @app.errorhandler(500)
        def internal_error(e):
            """500错误处理"""
            return jsonify({
                'success': False,
                'error': '服务器内部错误',
                'timestamp': Config.get_timestamp()
            }), 500