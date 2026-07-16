from typing import List, Optional, Tuple
from pathlib import Path
import torch
import os

from lightrag.utils import setup_logger
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    SimpleDirectoryReader,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser, MarkdownElementNodeParser
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache, DocstoreStrategy
from llama_index.core.schema import Document
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.vector_stores.milvus import MilvusVectorStore

import milvus_config
from config.config import Config


logger = setup_logger(__name__)

class DocumentIngestionPipeline:
    """文档摄取管道"""
    def __init__(self):
        self._setup_models()
        self.index: Optional[VectorStoreIndex] = None
        self.milvus_vector_store: Optional[MilvusVectorStore] = None
        self.text_pipeline: Optional[IngestionPipeline] = None
        self.markdown_pipeline: Optional[IngestionPipeline] = None

        # 1. 初始化存储组件 (Redis & Chroma)
        self._initialize_storage_components()

        # 2. 创建不同类型的 Pipeline
        self._create_pipelines()

        # 3. 初始化 PDF 处理器
        # self.pdf_processor = MultimodalPDFProcessor()

        # 创建 StorageContext
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.milvus_vector_store,
            docstore=self.redis_document_store,
            index_store=self.redis_index_store
        )


    def _setup_models(self):
        """
        设置LLM嵌入模型
        :return:
        """
        # 大语言模型
        Settings.llm = DashScope(
            api_key=Config.LLM_API_KEY,
            api_base=Config.LLM_BASE_URL,
            model_name=Config.LLM_MODEL_NAME,
            temperature=Config.TEMPERATURE
        )

        # 嵌入模型
        Settings.embed_model = DashScopeEmbedding(
            model_name=Config.ENBEDDING_MODEL,
            api_key=Config.OPENAI_API_KEY,
            dimensions=Config.MILVUS_VECTOR_DIM
        )


    def _initialize_storage_components(self):
        """初始化存储组件"""
        # 初始化索引存储
        self.redis_index_store = RedisIndexStore.from_host_and_port(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            namespace=Config.REDIS_INDEX_NAME_SPACE
        )
        # 初始化文档存储
        self.redis_document_store = RedisDocumentStore.from_host_and_port(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            namespace=Config.REDIS_DOC_NAME_SPACE
        )

        # 初始化向量存储
        self.milvus_vector_store = MilvusVectorStore(
            uri=f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}",
            token=None,  # 用的是 user/password 认证，所以 token 留空
            user=Config.MILVUS_USER,
            password=Config.MILVUS_PASSWORD,
            db_name=Config.MILVUS_DATABASE,  # 指定数据库名
            collection_name=Config.MILVUS_COLLECTION,  # 集合名称，建议明确指定
            dim=Config.MILVUS_VECTOR_DIM,    # 必须与嵌入模型维度一致（比如百炼 text-embedding-v4）
            # 传入自定义 schema
            schema=milvus_config.initialize_milvus_schema(),
            # 关键：告诉 LlamaIndex 使用哪个字段作为文本内容
            text_field=milvus_config.MilvusDocumentField.CONTENT,
            # 关键：告诉 LlamaIndex 使用哪个字段作为向量字段
            # 默认是 "embedding"，这里改为我自己定义的 "content_vector"
            vector_field=milvus_config.MilvusDocumentField.CONTENT_VECTOR,
            # 如果集合已存在，不覆盖
            overwrite=False,   # 设为 False 防止误删已有数据
        )

    def _create_pipelines(self):
        """创建不同类型的摄取管道"""
        # 通用配置
        common_config = {
            "vector_store": self.milvus_vector_store,
            "docstore": self.redis_document_store,
            "cache": IngestionCache(
                cache=RedisCache.from_host_and_port(Config.REDIS_HOST,Config.REDIS_PORT),
                collection=Config.REDIS_CACHE,
            ),
            "docstore_strategy": DocstoreStrategy.UPSERTS_AND_DELETE
        }

        # 1. 普通文本文件的管道 (使用 SentenceSplitter)
        self.text_pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=Config.CHUNK_SIZE,
                    chunk_overlap=Config.OVERLAP_SIZE
                ),
                TitleExtractor(nodes=Config.TITLE_EXTRACTOR_NODES),
                Settings.embed_model,
            ],
            **common_config
        )

        # 2. Markdown 文件的管道 (使用 MarkdownNodeParser)
        # 对于没有目录层级的文档，优化MarkdownNodeParser配置
        # MarkdownNodeParser 也会分隔文档，但只是粗略的分隔（粒度可能比较大），需要借助SentenceSplitter更精细的切割
        self.markdown_pipeline = IngestionPipeline(
            transformations=[
                MarkdownNodeParser(
                    include_metadata=True,       # 保留文档元数据
                    include_prev_next_rel=True,  # 保留节点之间的关系，有助于上下文理解
                ),
                # 使用更适合Markdown的分块策略，避免在图片标签中间切分
                SentenceSplitter(
                    chunk_size=Config.CHUNK_SIZE,
                    chunk_overlap=Config.OVERLAP_SIZE,
                    # 使用更安全的分隔符，避免在图片标签中间切分
                    separator="\n\n",
                    # 禁用默认的句子切分，使用段落级别切分
                    paragraph_separator="\n\n"
                ),
                TitleExtractor(nodes=Config.TITLE_EXTRACTOR_NODES),
                Settings.embed_model,
            ],
            **common_config
        )

    def update_model_config(self, model_name: str, temperature: float, max_tokens: int):
        """动态更新 LLM 模型和温度"""
        Settings.llm = DashScope(
            api_key=Config.LLM_API_KEY,
            api_base=Config.LLM_BASE_URL,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        logger.info(f"模型已更新: model={model_name}, temperature={temperature}")


    def ingest_document(self, doc: Document) -> Tuple:
        """
        摄取文档并创建索引
        :param doc:
        :return:
        """
        try:
            if not doc:
                logger.info(f"文档摄取，入参文档为空")
                return "error", "入参文档为空"

            file_name = doc.metadata.get("file_name")
            content_type = doc.metadata.get("content_type")
            if not content_type and file_name:
                content_type = os.path.splitext(os.path.basename(file_name))[0]

            if not content_type:
                logger.info(f"无法确定文件类型,file_name:{file_name}")
                return "error", "未知的文件类型"

            logger.info(f"开始处理文档:{file_name}")

            pipeline_nodes = []
            if content_type == 'markdown':
                pipeline_nodes = self.markdown_pipeline.run(documents=[doc], show_progress=True)
                logger.info(f"文件[{file_name}]处理完成,生成了 {len(pipeline_nodes)} 个节点")

            if content_type == 'text' or content_type == 'txt':
                pipeline_nodes = self.text_pipeline.run(documents=[doc], show_progress=True)
                logger.info(f"文件[{file_name}]处理完成,生成了 {len(pipeline_nodes)} 个节点")

            # 检查是否有新节点生成（如果文档重复，pipeline_nodes将为空）
            if not pipeline_nodes:
                result = "文档重复上传，无需更新索引"
                logger.info(f"{result},file_name:{file_name}")
                return "success", result, pipeline_nodes

            result = f"文档摄取成功，生成了 {len(pipeline_nodes)} 个节点"

            return "success", result, pipeline_nodes
        except Exception as e:
            logger.exception('文档摄取异常')
            error_msg = f"文档摄取失败: {str(e)}"
            return "error", error_msg, ""


if __name__ == '__main__':
    file_path = '../test_data/MongoDB-test.md'
    docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
    ingestion = DocumentIngestionPipeline()
    result = ingestion.ingest_document(docs[0])
    print(f"文档摄取结果:{result}")
