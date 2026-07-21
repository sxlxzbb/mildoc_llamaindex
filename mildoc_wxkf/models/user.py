"""用户模型，对应数据库 users 表"""
from extensions import db


class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True, comment='主键ID')
    username = db.Column(db.String(50), nullable=False, unique=True, comment='用户名')
    password = db.Column(db.String(255), nullable=False, comment='密码（MD5 加密）')
    nickname = db.Column(db.String(50), nullable=False, default='', comment='昵称')
    status = db.Column(db.SmallInteger, nullable=False, default=1, comment='状态：1-正常，0-删除')
    created = db.Column(db.DateTime, comment='创建时间')
    updated = db.Column(db.DateTime, comment='最后更新时间')

    def __repr__(self):
        return f"<User {self.username}>"

    def to_dict(self) -> dict:
        """返回脱敏后的用户信息"""
        return {
            'id': self.id,
            'username': self.username,
            'nickname': self.nickname,
            'status': self.status,
        }
