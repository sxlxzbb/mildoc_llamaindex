"""记忆模块

职责：
- 短期记忆：Redis。按用户保存最近多轮对话（JSON 列表），带 TTL 自动过期，
  作为每次请求拼装 prompt 的实时上下文。
- 长期记忆：MySQL（chat_history 表）。每一轮对话都永久落库，用于审计/回溯，
  后续可扩展为摘要、跨会话检索等。

设计要点：
- Redis 不可用时自动降级为内存字典，保证服务不中断（仅丢失跨进程共享与持久化能力）。
- 每次 append 都会写入 MySQL，与 Redis 写入解耦（MySQL 失败不影响主流程）。
- session_id 用于把多轮对话归为一次完整会话，清空记忆时切换新会话。
"""
import json
import uuid

import redis
from extensions import db

from config.config import Config
from logger.logger import logger


class MemoryService:
    """对话记忆服务：短期 Redis + 长期 MySQL"""

    def __init__(self):
        self._redis = None
        # Redis 不可用时的内存兜底
        self._fallback: dict = {}
        self._fallback_sessions: dict = {}

    # ===================== Redis（懒加载） =====================
    def _get_redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                password=Config.REDIS_PASSWORD or None,
                db=Config.REDIS_DB,
                decode_responses=True,
            )
            logger.info("Redis 客户端初始化完成（短期记忆）")
        return self._redis

    def _key(self, username: str) -> str:
        return f"memory:{username}"

    def _session_key(self, username: str) -> str:
        return f"memory:{username}:session"

    def _current_session(self, username: str) -> str:
        """获取当前会话 ID，不存在则生成新的并写入 Redis（带 TTL）"""
        try:
            r = self._get_redis()
            sid = r.get(self._session_key(username))
            if not sid:
                sid = uuid.uuid4().hex
                r.set(self._session_key(username), sid, ex=Config.REDIS_TTL_SECONDS)
            return sid
        except Exception as e:
            logger.warning(f"Redis 读取会话失败，使用内存兜底: {e}")
            sid = self._fallback_sessions.get(username)
            if not sid:
                sid = uuid.uuid4().hex
                self._fallback_sessions[username] = sid
            return sid

    # ===================== 公开接口（兼容原调用方） =====================
    def get_history(self, username: str, max_turns: int = None) -> list:
        """获取短期记忆（最近 max_turns 条），用于拼装上下文"""
        max_turns = max_turns or Config.MEMORY_MAX_TURNS
        try:
            r = self._get_redis()
            raw = r.lrange(self._key(username), -max_turns, -1)
            return [json.loads(x) for x in raw]
        except Exception as e:
            logger.warning(f"Redis 读取历史失败，使用内存兜底: {e}")
            return self._fallback.get(username, [])[-max_turns:]

    def append(self, username: str, role: str, content: str) -> None:
        """追加一条对话记录：同时写入短期记忆(Redis) 与长期记忆(MySQL)"""
        item = json.dumps({'role': role, 'content': content}, ensure_ascii=False)
        # 1. 短期记忆：Redis 列表，每次写入刷新 TTL
        try:
            r = self._get_redis()
            r.rpush(self._key(username), item)
            r.expire(self._key(username), Config.REDIS_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"Redis 写入失败，使用内存兜底: {e}")
            self._fallback.setdefault(username, []).append({'role': role, 'content': content})
        # 2. 长期记忆：MySQL 永久归档（失败不影响主流程）
        self._persist_mysql(username, role, content)

    def clear(self, username: str) -> None:
        """清空短期记忆并切换到新会话（长期记忆保留归档）"""
        try:
            r = self._get_redis()
            r.delete(self._key(username))
            r.delete(self._session_key(username))
        except Exception as e:
            logger.warning(f"Redis 清空失败，使用内存兜底: {e}")
            self._fallback.pop(username, None)
            self._fallback_sessions.pop(username, None)
        logger.info(f"已清空用户短期对话记忆: {username}")

    # ===================== 长期记忆（MySQL） =====================
    def _persist_mysql(self, username: str, role: str, content: str) -> None:
        """把每一轮对话归档到 MySQL chat_history 表"""
        try:
            from models.chat_history import ChatHistory
            rec = ChatHistory(
                username=username,
                session_id=self._current_session(username),
                role=role,
                content=content,
            )
            db.session.add(rec)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.error(f"长期记忆写入 MySQL 失败: {e}")
