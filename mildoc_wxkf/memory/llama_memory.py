"""LlamaIndex 记忆适配层

把 LlamaIndex ChatEngine 的 BaseMemory 接口，委托给现有的 MemoryService
（Redis 短期 + MySQL 长期）。ChatEngine 的读写经由此处落到 MemoryService，
无需在路由层手动保存记忆。
"""
from typing import Any, List, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import BaseMemory
from pydantic import ConfigDict

from logger.logger import logger
from memory.service import MemoryService


def _to_chat_message(m: dict) -> ChatMessage:
    role_str = m.get("role", "user")
    try:
        role = MessageRole(role_str)
    except Exception:
        role = MessageRole.USER
    return ChatMessage(role=role, content=m.get("content", ""))


class MemoryServiceMemory(BaseMemory):
    """把 LlamaIndex ChatEngine 的记忆读写委托给现有 MemoryService（Redis + MySQL）。

    实现了 BaseMemory 的抽象方法；ChatEngine 的写入（用户/助手消息）会自动
    经由本类落到 MemoryService，无需在路由层手动保存记忆。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    username: str
    memory_service: MemoryService
    app: Any = None  # Flask app，用于后台线程写入 MySQL 时提供 app context

    def _append(self, role: str, content: str) -> None:
        # 优先使用传入的 Flask app 提供上下文（后台线程写 MySQL 必须）；
        # 否则尝试当前请求上下文；都没有则降级为仅调用（Redis 通常可用）。
        ctx_app = self.app
        if ctx_app is None:
            try:
                from flask import current_app
                ctx_app = current_app._get_current_object()
            except Exception:
                ctx_app = None

        if ctx_app is not None:
            with ctx_app.app_context():
                self.memory_service.append(self.username, role, content)
            return

        # 无可用 app 上下文：直接调用，MySQL 持久化可能失败，但不阻塞主流程
        try:
            self.memory_service.append(self.username, role, content)
        except Exception as e:
            logger.error(f"记忆写入失败（无 app 上下文，已忽略）: {e}")

    def get(self, input: Optional[str] = None, **kwargs) -> List[ChatMessage]:
        history = self.memory_service.get_history(self.username)
        return [_to_chat_message(m) for m in history]

    def get_all(self, **kwargs) -> List[ChatMessage]:
        return self.get()

    def put(self, message: ChatMessage) -> None:
        role = message.role.value if hasattr(message.role, "value") else str(message.role)
        content = message.content or ""
        self._append(role, content)

    def put_messages(self, messages: List[ChatMessage]) -> None:
        for m in messages:
            self.put(m)

    def set(self, messages: List[ChatMessage]) -> None:
        self.reset()
        self.put_messages(messages)

    def reset(self) -> None:
        self.memory_service.clear(self.username)

    @classmethod
    def from_defaults(cls, username, memory_service, app=None, **kwargs):
        return cls(username=username, memory_service=memory_service, app=app)
