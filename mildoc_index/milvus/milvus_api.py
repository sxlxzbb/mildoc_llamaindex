from pymilvus import MilvusClient

from config.config import Config
from logger.logging import setup_logging
from milvus import milvus_config

logger = setup_logging()

class MilvusApi:
    def __init__(self):
        self.collection_name = Config.MILVUS_COLLECTION

        self.client = MilvusClient(
            uri=f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}",
            user=Config.MILVUS_USER,
            password=Config.MILVUS_PASSWORD,
            db_name=Config.MILVUS_DATABASE
        )


    def _load_collection(self) -> bool:
        """加载集合到内存"""
        try:
            self.client.load_collection(collection_name=self.collection_name)
            logger.info(f"集合 '{self.collection_name}' 加载成功")
            return True
        except Exception as e:
            logger.exception(f"加载集合失败:{self.collection_name}")
            return False


    def check_document_exists(self, doc_path_name: str) -> bool:
        """
        检查文档是否已经存在
        :param doc_path_name: 文档路径
        :return: 文档是否已经存在
        """
        try:
            # 先确保集合已加载
            self._load_collection()

            # 根据路径查询
            filter_expr = f'file_path == "{doc_path_name}"'

            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=['id', 'doc_id'],
                limit=1
            )

            return len(results) > 0
        except Exception as e:
            logger.exception(f"检查文档是否存在失败:{doc_path_name}")
