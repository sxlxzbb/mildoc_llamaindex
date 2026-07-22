"""聊天相关路由（流式响应，基于 LlamaIndex ChatEngine）

RAG 链路（由 CondenseQuestionChatEngine 内部完成）：
1. 接收用户问题
2. 基于记忆压缩为独立问题 → 混合检索 → RerankPostprocessor 精排 → 响应合成器流式生成
3. 记忆：由 ChatEngine 通过 MemoryServiceMemory 自动写入 MemoryService（Redis + MySQL），
   无需在此手动保存。
"""
import json

from flask import (
    Blueprint, render_template, request, jsonify, Response, session, current_app
)

from auth.routes import login_required
from core.base_retriever import sources_from_nodes, get_chat_engine
from logger.logger import logger

chat_bp = Blueprint('chat', __name__)


@chat_bp.route('/chat')
@login_required
def chat_page():
    """聊天页面（需登录）"""
    return render_template('chat.html')


@chat_bp.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """聊天接口（RAG + 流式响应）

    请求体（JSON）：
        {
            "message": "用户问题",
            "stream": true  # 是否流式
        }
    """
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    if not message:
        return jsonify({'error': '消息内容不能为空'}), 400

    use_stream = bool(data.get('stream', True))
    username = session.get('username')

    logger.info(f"用户问题：{message}")

    # 构建对话引擎
    try:
        engine = get_chat_engine(username, app=current_app._get_current_object())
    except Exception as e:
        logger.error(f"构建对话引擎失败: {e}")
        return jsonify({'error': f'初始化失败: {str(e)}'}), 500

    # 非流式：同步返回完整答案与来源
    if not use_stream:
        try:
            resp = engine.chat(message)
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return jsonify({'error': f'生成失败: {str(e)}'}), 500

        sources = sources_from_nodes(resp.source_nodes)
        return jsonify({'answer': resp.response, 'sources': sources})


    # 流式（SSE）
    def generate():
        try:
            resp = engine.stream_chat(message)
            # source_nodes 在构造时已由 ToolOutput.raw_output 提取，可立即取用
            sources = sources_from_nodes(resp.source_nodes)
            for delta in resp.response_gen:
                if delta:
                    payload = json.dumps(
                        {'content': delta, 'finished': False}, ensure_ascii=False
                    )
                    yield f'data: {payload}\n\n'
            # 结束信号（附带引用来源，已按 file_path 去重）
            yield f'data: {json.dumps({"content": "", "finished": True, "sources": sources}, ensure_ascii=False)}\n\n'
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            err = json.dumps({'error': str(e), 'finished': True}, ensure_ascii=False)
            yield f'data: {err}\n\n'

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    )
