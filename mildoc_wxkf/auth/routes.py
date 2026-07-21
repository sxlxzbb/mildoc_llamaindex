"""登录 / 登出 路由

鉴权方式：基于 Flask session（与 mildoc_admin 保持一致）。
后续如需改为 JWT，只需替换 login_required 与 login/logout 内部逻辑，
对外接口路径保持不变。
"""
from functools import wraps

from flask import (
    Blueprint, render_template, request, redirect,
    url_for, session, flash, jsonify, g
)

from auth.service import authenticate, get_user_by_id
from logger import logger

# 认证蓝图
auth_bp = Blueprint('auth', __name__)


def login_required(f):
    """登录验证装饰器：未登录跳转登录页（页面）；API 返回 401"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            # 判断是否为 API 请求
            if request.path.startswith('/api/') or request.is_json \
                    or request.headers.get('Accept') == 'application/json':
                return jsonify({'error': '未登录或登录已过期'}), 401
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面与登录处理"""
    # 已登录直接进聊天页
    if 'user_id' in session:
        return redirect(url_for('chat.chat_page'))

    if request.method == 'POST':
        # 支持表单提交（页面）与 JSON（API）
        if request.is_json:
            data = request.get_json(silent=True) or {}
            username = data.get('username')
            password = data.get('password')
        else:
            username = request.form.get('username')
            password = request.form.get('password')

        user = authenticate(username, password)
        if user:
            session['user_id'] = user.id
            session['username'] = user.username
            session['nickname'] = user.nickname
            session.permanent = True

            if request.is_json:
                return jsonify({'message': '登录成功', 'username': user.username})
            return redirect(url_for('chat.chat_page'))

        if request.is_json:
            return jsonify({'error': '用户名或密码错误'}), 401
        flash('用户名或密码错误', 'error')

    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    """退出登录"""
    username = session.get('username')
    session.clear()
    logger.info(f"用户退出登录: {username}")

    if request.is_json:
        return jsonify({'message': '已退出登录'})
    return redirect(url_for('auth.login'))


@auth_bp.route('/api/me')
@login_required
def me():
    """获取当前登录用户信息（API）"""
    return jsonify({
        'id': session.get('user_id'),
        'username': session.get('username'),
        'nickname': session.get('nickname'),
    })
