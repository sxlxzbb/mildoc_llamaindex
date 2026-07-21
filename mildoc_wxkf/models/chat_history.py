"""长期对话记忆模型，对应数据库 chat_history 表（mildoc 库）

与短期记忆（Redis）的区别：
- 短期记忆：Redis 中按用户保存最近多轮对话，带 TTL 自动过期，用于实时上下文。
- 长期记忆：MySQL 中永久归档每一轮对话，可审计、可回溯、可后续做摘要/检索。
"""
from extensions import db


class ChatHistory(db.Model):
    __tablename__ = 'chat_history'

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True, comment='主键ID')
    username = db.Column(db.String(50), nullable=False, index=True, comment='用户名')
    session_id = db.Column(db.String(64), nullable=False, default='', index=True, comment='会话ID（一次完整对话）')
    role = db.Column(db.String(20), nullable=False, comment='角色：user / assistant / system')
    content = db.Column(db.Text, nullable=False, comment='消息内容')
    created_at = db.Column(db.DateTime, server_default=db.func.now(), comment='创建时间')

    def __repr__(self):
        return f"<ChatHistory {self.username} {self.role} {self.created_at}>"

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'username': self.username,
            'session_id': self.session_id,
            'role': self.role,
            'content': self.content,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
