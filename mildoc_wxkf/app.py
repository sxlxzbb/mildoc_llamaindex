# -*- coding: utf-8 -*-
"""mildoc_wxkf 服务入口

RAG 检索问答服务，提供：登录、检索、重排序、流式响应、记忆。
当前已实现登录（含登出），其余模块已预留接口，后续逐步填充。
"""
from flask import Flask, redirect, url_for, session

from config.config import Config
from extensions import db
from logger.logger import logger
from auth.routes import auth_bp
from chat.routes import chat_bp


def create_app(config_class: type = Config) -> Flask:
    """应用工厂"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.secret_key = app.config['SECRET_KEY']

    # 让 session 默认 7 天过期
    from datetime import timedelta
    app.permanent_session_lifetime = timedelta(days=7)

    # 初始化数据库
    db.init_app(app)
    # 确保表存在（若不存在则按模型自动创建；用户已手动建表则跳过）
    with app.app_context():
        db.create_all()

    # 注册各功能模块蓝图
    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)

    @app.route('/')
    def index():
        if 'user_id' in session:
            return redirect(url_for('chat.chat_page'))
        return redirect(url_for('auth.login'))

    @app.route('/health')
    def health():
        return {'status': 'healthy', 'service': 'mildoc_wxkf'}

    logger.info("mildoc_wxkf 应用初始化完成")
    return app


if __name__ == '__main__':
    app = create_app()
    # 注意：务必关闭 reloader。本服务使用 SSE 长连接流式返回，
    # reloader 会监视 site-packages 等文件，任何变动都会重启进程、
    # 掐断正在进行的流式连接，导致前端报 "[网络错误] Failed to fetch"。
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG'],
        use_reloader=False,
        threaded=True,
    )
