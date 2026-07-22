import os

from dotenv import load_dotenv

load_dotenv()

class Config:
    # 管理员账号
    ADMIN_USERNAME = os.getenv('ADMIN_USERNAME')
    ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')

    # Minio 配置
    MINIO_BUCKET = os.getenv('MINIO_BUCKET')
    MINIO_BUCKET_HIER = os.getenv('MINIO_BUCKET_HIER')
    MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT')
    MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
    MINIO_REGION = os.getenv('MINIO_REGION')
    MINIO_USE_VIRTUAL_HOST = os.getenv('MINIO_USE_VIRTUAL_HOST', 'false').lower() == 'true'
    MINIO_USE_SSL = os.getenv('MINIO_USE_SSL', 'false').lower() == 'true'

    NODE_PARSER_MODE = os.getenv('NODE_PARSER_MODE', 'default')

    # Milvus 配置
    MILVUS_HOST = os.getenv("MILVUS_HOST")
    MILVUS_PORT = os.getenv("MILVUS_PORT")
    MILVUS_USER = os.getenv("MILVUS_USER")
    MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
    MILVUS_DATABASE = os.getenv("MILVUS_DATABASE")
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")
    MILVUS_INDEX_NAME = os.getenv("MILVUS_INDEX_NAME")
    MILVUS_COLLECTION_HIER = os.getenv("MILVUS_COLLECTION_HIER")
    MILVUS_INDEX_NAME_HIER = os.getenv("MILVUS_INDEX_NAME_HIER")

    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "false") == 'true'
    FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "'default-secret-key'")
    FLASK_HOST = os.getenv("FLASK_HOST")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT"))


