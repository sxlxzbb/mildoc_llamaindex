import os
from typing import List

from dotenv import load_dotenv

load_dotenv()

class Config:
    """应用配置"""
    # 文件存储临时路径
    TMP_FILE_DIR: str = os.getenv("TMP_FILE_DIR")

    # minio
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET")
    MINIO_REGION: str = os.getenv("MINIO_REGION")
    MINIO_USE_VIRTUAL_HOST: bool = os.getenv('MINIO_USE_VIRTUAL_HOST', 'false').lower() == 'true'
    MINIO_USE_SSL: bool = os.getenv("MINIO_USE_SSL", 'false').lower() == 'true'
    MINIO_BUCKET_HIER: str = os.getenv("MINIO_BUCKET_HIER")

    # minvus
    MILVUS_HOST: str = os.getenv("MILVUS_HOST")
    MILVUS_PORT: int = os.getenv("MILVUS_PORT")
    MILVUS_USER: str = os.getenv("MILVUS_USER")
    MILVUS_PASSWORD: str = os.getenv("MILVUS_PASSWORD")
    MILVUS_DATABASE: str = os.getenv("MILVUS_DATABASE")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION")
    MILVUS_INDEX_NAME: str = os.getenv("MILVUS_INDEX_NAME")
    MILVUS_VECTOR_DIM: int = int(os.getenv("MILVUS_VECTOR_DIM"))
    MILVUS_COLLECTION_HIER: str = os.getenv("MILVUS_COLLECTION_HIER")
    MILVUS_INDEX_NAME_HIER: str = os.getenv("MILVUS_INDEX_NAME_HIER")


    # llm
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL")
    TEMPERATURE: float = 0.1

    # embedding
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL")
    ENBEDDING_MODEL: str = os.getenv("ENBEDDING_MODEL")

    # oss
    OSS_ACCESS_KEY_ID: str = os.getenv("OSS_ACCESS_KEY_ID")
    OSS_ACCESS_KEY_SECRET: str = os.getenv("OSS_ACCESS_KEY_SECRET")
    OSS_ENDPOINT: str = os.getenv("OSS_ENDPOINT")
    OSS_BUCKET_NAME: str = os.getenv("OSS_BUCKET_NAME")
    OSS_IMAGE_PATH: str = os.getenv("OSS_IMAGE_PATH", '')
    DOC_IMAGE_DIR: str = os.getenv("DOC_IMAGE_DIR", 'images')

    # soffice doc文件转pdf文档工具路径
    SOFFICE_PATH: str = os.getenv("SOFFICE_PATH")
    USE_MINREU: bool = os.getenv("USE_MINREU", 'false') == 'true'
    USE_PYMUPDF4LLM: bool = os.getenv("USE_PYMUPDF4LLM", 'false') == 'true'
    USE_LIBRE_OFFICE: bool = os.getenv("USE_LIBRE_OFFICE", 'false') == 'true'

    # 文档分块
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    OVERLAP_SIZE: int = int(os.getenv("OVERLAP_SIZE", "128"))
    TITLE_EXTRACTOR_NODES: int = int(os.getenv("TITLE_EXTRACTOR_NODES"))

    # ===================== 节点解析方式（A/B 对比用） =====================
    # default      = 原逻辑（SentenceSplitter / MarkdownNodeParser）
    # hierarchical = 层次节点解析器（HierarchicalNodeParser，生成「父-子」层级节点）
    NODE_PARSER_MODE: str = os.getenv("NODE_PARSER_MODE", "default")
    # 层次解析的分块层级（由大到小，单位 token），逗号分隔；
    # 最小层会作为 HierarchicalNodeParser 的 base parser 的叶子块大小参考。
    HIERARCHICAL_CHUNK_SIZES: List[int] = [
        int(x) for x in os.getenv("HIERARCHICAL_CHUNK_SIZES", "2048,512").split(",") if x.strip()
    ]
    # 以下可留空：留空则复用上面的原集合 / Redis 命名空间。
    # 做 A/B 对比时建议设成独立值，避免两套节点互相覆盖。
    # HIERARCHICAL_COLLECTION: str = os.getenv("HIERARCHICAL_COLLECTION", "")
    # HIERARCHICAL_REDIS_DOC_NS: str = os.getenv("HIERARCHICAL_REDIS_DOC_NS", "")
    # HIERARCHICAL_REDIS_INDEX_NS: str = os.getenv("HIERARCHICAL_REDIS_INDEX_NS", "")
    # HIERARCHICAL_REDIS_CACHE: str = os.getenv("HIERARCHICAL_REDIS_CACHE", "")

    # redis
    REDIS_HOST: str = os.getenv("REDIS_HOST")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT"))
    REDIS_INDEX_NAME_SPACE: str = os.getenv("REDIS_INDEX_NAME_SPACE")
    REDIS_DOC_NAME_SPACE: str = os.getenv("REDIS_DOC_NAME_SPACE")
    REDIS_CACHE: str = os.getenv("REDIS_CACHE")
    REDIS_INDEX_NAME_SPACE_HIER: str = os.getenv("REDIS_INDEX_NAME_SPACE_HIER")
    REDIS_DOC_NAME_SPACE_HIER: str = os.getenv("REDIS_DOC_NAME_SPACE_HIER")
    REDIS_CACHE_HIER: str = os.getenv("REDIS_CACHE_HIER")

    # mysql
    MYSQL_HOST: str = os.getenv("MYSQL_HOST")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT"))
    MYSQL_USER: str = os.getenv("MYSQL_USER")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE")

    # 解析偏 CPU、embedding/Milvus 偏 I/O，默认取 CPU 核数；可用 MINIO_PROCESS_MAX_WORKERS 覆盖。
    MINIO_PROCESS_MAX_WORKERS: int = int(os.getenv("MINIO_PROCESS_MAX_WORKERS", str(os.cpu_count() or 1)))
    # I/O 密集型任务，默认线程数为 CPU 核数 * 2；可通过 OSS_UPLOAD_MAX_WORKERS 显式覆盖
    OSS_UPLOAD_MAX_WORKERS: int = int(os.getenv("OSS_UPLOAD_MAX_WORKERS", str((os.cpu_count() or 1) * 2)))

    # 是否将文档中的图片上传OSS
    UPLOAD_IMAGE_TO_OSS = os.getenv('UPLOAD_IMAGE_TO_OSS', 'false') == 'true'


