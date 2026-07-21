"""全局扩展对象，避免循环依赖"""
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
