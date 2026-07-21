import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """应用配置"""

    # ===================== Flask =====================
    SECRET_KEY: str = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-me')
    DEBUG: bool = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    HOST: str = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT: int = int(os.getenv('FLASK_PORT', '8872'))

    # ===================== MySQL =====================
    MYSQL_HOST: str = os.getenv("MYSQL_HOST")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT"))
    MYSQL_USER: str = os.getenv("MYSQL_USER")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE")

    # SQLAlchemy 连接串（mysql+pymysql）
    SQLALCHEMY_DATABASE_URI: str = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    SQLALCHEMY_ENGINE_OPTIONS: dict = {
        "pool_pre_ping": True,
        "pool_recycle": 3600,
    }

    # ===================== MinIO =====================
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET")
    MINIO_REGION: str = os.getenv("MINIO_REGION")
    MINIO_USE_VIRTUAL_HOST: bool = os.getenv('MINIO_USE_VIRTUAL_HOST', 'false').lower() == 'true'
    MINIO_USE_SSL: bool = os.getenv("MINIO_USE_SSL", 'false').lower() == 'true'

    # ===================== Milvus =====================
    MILVUS_HOST: str = os.getenv("MILVUS_HOST")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT"))
    MILVUS_USER: str = os.getenv("MILVUS_USER")
    MILVUS_PASSWORD: str = os.getenv("MILVUS_PASSWORD")
    MILVUS_DATABASE: str = os.getenv("MILVUS_DATABASE")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION")
    MILVUS_INDEX_NAME: str = os.getenv("MILVUS_INDEX_NAME")
    MILVUS_VECTOR_DIM: int = int(os.getenv("MILVUS_VECTOR_DIM"))

    # ===================== LLM（生成回答） =====================
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", '2000'))

    # ===================== Embedding =====================
    OPENAI_API_KEY: str = os.getenv("LLM_EMBEDDING_API_KEY")
    OPENAI_BASE_URL: str = os.getenv("LLM_EMBEDDING_BASE_URL")
    ENBEDDING_MODEL: str = os.getenv("LLM_EMBEDDING_MODEL_NAME")

    # ===================== 重排序 =====================
    RERANK_PROVIDER: str = os.getenv("RERANK_PROVIDER")
    RERANK_API_KEY: str = os.getenv("RERANK_API_KEY")
    RERANK_MODEL_NAME: str = os.getenv("RERANK_MODEL_NAME", 'gte-rerank-hybrid')
    RERANK_ENDPOINT: str = os.getenv("RERANK_ENDPOINT")

    # ===================== Redis（短期记忆） =====================
    REDIS_HOST: str = os.getenv("REDIS_HOST")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_TTL_SECONDS: int = int(os.getenv("REDIS_TTL_SECONDS", "1800"))  # 短期记忆过期时间，默认 30 分钟

    # ===================== 记忆 =====================
    MEMORY_MAX_TURNS: int = int(os.getenv("MEMORY_MAX_TURNS", "20"))  # 短期记忆保留的最大消息条数

    # 向量检索返片段数
    TOP_K: int = int(os.getenv("TOP_K", "20"))
    # 重排序后返回的片段数
    TOP_N: int = int(os.getenv("TOP_N", "10"))
