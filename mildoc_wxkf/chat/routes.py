"""聊天相关路由（流式响应）

完整 RAG 链路：
1. 接收用户问题
2. 检索（retrieval）：向量召回候选分片
3. 重排序（rerank）：精排候选分片
4. 拼接上下文，调用 LLM 流式返回（SSE）
5. 记忆（memory）：保存多轮对话（流式结束后后台线程写入，避免阻塞响应关闭）
"""
import json
import threading

from flask import (
    Blueprint, render_template, request, jsonify,
    Response, session, current_app
)

from auth.routes import login_required
from core.retrieval import retrieve, dedupe_chunks
from core.rerank import RerankService
from chat.service import build_messages, stream_completion, complete_completion
from memory.service import MemoryService
from logger.logger import logger
from config.config import Config

chat_bp = Blueprint('chat', __name__)
memory = MemoryService()


def _save_memory_async(username: str, message: str, answer: str) -> None:
    """在后台线程保存对话记忆，避免阻塞 SSE 响应关闭。

    使用独立的 app_context，不依赖原请求上下文（防止原请求结束后
    被复制的 request context 中的 session 已被关闭）。
    """
    try:
        app = current_app._get_current_object()
    except RuntimeError:
        logger.warning("无法获取 current_app，跳过后台记忆保存")
        return

    def _run():
        with app.app_context():
            try:
                memory.append(username, 'user', message)
                memory.append(username, 'assistant', answer)
                logger.debug(f"已后台保存用户 {username} 的对话记忆")
            except Exception as e:
                logger.error(f"后台保存记忆失败: {e}")

    threading.Thread(target=_run, daemon=True).start()


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
            "history": [{"role": "user", "content": "..."}],  # 可选，缺省用服务端记忆
            "stream": true  # 是否流式
        }
    """
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    if not message:
        return jsonify({'error': '消息内容不能为空'}), 400

    history = data.get('history') or []
    use_stream = bool(data.get('stream', True))
    username = session.get('username')

    # 1. 检索
    try:
        chunks = retrieve(message)
    except Exception as e:
        logger.error(f"检索失败: {e}")
        return jsonify({'error': f'检索失败: {str(e)}'}), 500

    # 2. 重排序
    chunks = RerankService.rerank(message, chunks, top_n=Config.TOP_N)
    # 参考来源去重（同一文档只保留一条，按得分最高）
    sources = dedupe_chunks(chunks)

    # 3. 记忆：前端未传历史时，使用服务端保存的历史
    if not history:
        history = memory.get_history(username)

    # 4. 组装 prompt
    messages = build_messages(message, chunks, history=history)

    # 5. 响应
    if not use_stream:
        try:
            answer = complete_completion(messages)
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return jsonify({'error': f'生成失败: {str(e)}'}), 500
        # 非流式：同步保存记忆并返回完整答案
        memory.append(username, 'user', message)
        memory.append(username, 'assistant', answer)
        return jsonify({'answer': answer, 'sources': [c.to_source_dict() for c in sources]})

    # 流式（SSE）
    def generate():
        accumulated = ''
        success = False
        try:
            for piece in stream_completion(messages):
                accumulated += piece
                payload = json.dumps(
                    {'content': piece, 'finished': False}, ensure_ascii=False
                )
                yield f'data: {payload}\n\n'
            # 结束信号（附带引用来源，已去重，且只传轻量元数据）
            yield f'data: {json.dumps({"content": "", "finished": True, "sources": [c.to_source_dict() for c in sources]}, ensure_ascii=False)}\n\n'
            success = True
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            err = json.dumps({'error': str(e), 'finished': True}, ensure_ascii=False)
            yield f'data: {err}\n\n'
        # 流式正常结束后，后台线程保存记忆；生成器可立即返回，响应关闭更干净
        if success:
            _save_memory_async(username, message, accumulated)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    )
