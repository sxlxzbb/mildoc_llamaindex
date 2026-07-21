"""登录鉴权业务逻辑

说明：
- 数据库中的密码字段 password 使用 MD5 加密存储，因此校验时
  对用户输入的明文密码做 MD5 后再与库中值比对。
- MD5 安全性较弱，仅与现有数据库结构保持一致；后续如需增强，
  可在此层替换为更安全的哈希算法而不影响调用方。
"""
import hashlib

from extensions import db
from models.user import User
from logger import logger


def md5_hash(password: str) -> str:
    """对明文密码进行 MD5 加密，返回 32 位小写十六进制字符串"""
    return hashlib.md5(password.encode('utf-8')).hexdigest()


def authenticate(username: str, password: str) -> User | None:
    """校验用户名与密码。

    Args:
        username: 用户输入的用户名
        password: 用户输入的明文密码

    Returns:
        校验成功返回 User 对象，失败返回 None
    """
    if not username or not password:
        return None

    try:
        user = User.query.filter_by(username=username, status=1).first()
    except Exception as e:
        logger.error(f"查询用户失败: {e}")
        return None

    if not user:
        logger.warning(f"登录失败，用户不存在或已禁用: {username}")
        return None

    if user.password != md5_hash(password):
        logger.warning(f"登录失败，密码错误: {username}")
        return None

    logger.info(f"用户登录成功: {username}")
    return user


def get_user_by_id(user_id: int) -> User | None:
    """根据主键 id 获取用户（用于会话恢复/鉴权依赖）"""
    try:
        return User.query.filter_by(id=user_id, status=1).first()
    except Exception as e:
        logger.error(f"查询用户失败: {e}")
        return None
