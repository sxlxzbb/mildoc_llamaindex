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
    # 给层次解析使用
    MINIO_BUCKET_HIER: str = os.getenv("MINIO_BUCKET_HIER")
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
    # 以下两个配置给层次解析使用
    MILVUS_COLLECTION_HIER: str = os.getenv("MILVUS_COLLECTION_HIER")
    MILVUS_INDEX_NAME_HIER: str = os.getenv("MILVUS_INDEX_NAME_HIER")
    MILVUS_VECTOR_DIM: int = int(os.getenv("MILVUS_VECTOR_DIM"))

    # ===================== Milvus 索引 / 检索调优 =====================
    # 前两个参数仅在摄取侧使用，检索侧可以不用配
    # 均有默认值，不配 .env 也能跑（等价于当前行为）。
    # index_type / nlist / metric_type 属「建索引时」参数，仅集合首次创建时生效；
    #   若集合已存在（overwrite=False），改了不会重建索引，必须删集合重摄取。
    # nprobe 属「查询时」参数，立即生效、无需重建。
    # 推荐：embedding=text-embedding-v4 时把 metric_type 改成 COSINE；
    #       数据量大时把 index_type 改成 IVF_FLAT 并配合 nlist（经验 ≈ sqrt(N)）。
    MILVUS_INDEX_TYPE: str = os.getenv("MILVUS_INDEX_TYPE", "FLAT")
    MILVUS_NLIST: int = int(os.getenv("MILVUS_NLIST", "1024"))
    MILVUS_METRIC_TYPE: str = os.getenv("MILVUS_METRIC_TYPE", "IP")
    MILVUS_NPROBE: int = int(os.getenv("MILVUS_NPROBE", "10"))

    # ===================== LLM（生成回答） =====================
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", '2000'))
    # LLM 上下文窗口（token）。OpenAILike 默认仅 3900，会导致 COMPACT 合成器
    # 预算为负而报错，必须按真实模型窗口显式设置（qwen 系列通常为 128k）。
    LLM_CONTEXT_WINDOW: int = int(os.getenv("LLM_CONTEXT_WINDOW", "128000"))

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

    # ===================== 检索方式（A/B 对比用） =====================
    # default      = 原混合检索（VectorStoreIndex.as_retriever + hybrid）
    # hierarchical = AutoMergingRetriever（配合 mildoc_index 侧 NODE_PARSER_MODE=hierarchical 的层级节点）
    RETRIEVER_MODE: str = os.getenv("RETRIEVER_MODE", "default")
    # 层次解析写入的 Redis docstore / index store 命名空间
    # 必须与 mildoc_index 完全一致，
    # AutoMergingRetriever 才能通过节点关系找到父节点并合并。
    REDIS_DOC_NAME_SPACE_HIER: str = os.getenv("REDIS_DOC_NAME_SPACE_HIER")
    REDIS_INDEX_NAME_SPACE_HIER: str = os.getenv("REDIS_INDEX_NAME_SPACE_HIER")
