from typing import List, Optional, Tuple
import os

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    SimpleDirectoryReader
)
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache, DocstoreStrategy
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.llms.dashscope import DashScope
from llama_index.vector_stores.milvus import MilvusVectorStore

from milvus import milvus_config
from config.config import Config
from logger.logging import setup_logging

logger = setup_logging()

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
        # 用 OpenAIEmbedding 走百炼的 OpenAI 兼容接口（OPENAI_BASE_URL），
        # 该接口支持 dimensions 参数，可强制输出维度与 Milvus 的 MILVUS_VECTOR_DIM 对齐。
        # 必须用 model_name= 而非 model=，否则会触发 OpenAIEmbeddingModelType 枚举校验报错。
        Settings.embed_model = OpenAIEmbedding(
            model_name=Config.ENBEDDING_MODEL,      # "text-embedding-v4"，绕过枚举校验
            dimensions=Config.MILVUS_VECTOR_DIM,    # 768，透传给 API，确保与 Milvus 维度一致
            api_key=Config.OPENAI_API_KEY,
            api_base=Config.OPENAI_BASE_URL,
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
        # 注意：llama-index-vector-stores-milvus >= 1.0 改为参数式建 schema，
        # 不再接收 schema=/text_field=/vector_field= 参数，由 store 自动构建。
        self.milvus_vector_store = MilvusVectorStore(
            uri=f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}",
            token=None,  # 用的是 user/password 认证，所以 token 留空
            user=Config.MILVUS_USER,        # 经 **kwargs 透传给 MilvusClient
            password=Config.MILVUS_PASSWORD,  # 经 **kwargs 透传给 MilvusClient
            db_name=Config.MILVUS_DATABASE,  # 指定数据库名，经 **kwargs 透传
            collection_name=Config.MILVUS_COLLECTION,  # 集合名称，建议明确指定
            dim=Config.MILVUS_VECTOR_DIM,    # 必须与嵌入模型维度一致（比如百炼 text-embedding-v4）
            # 关键：文本字段名必须是 "text"（即节点 node.dict() 的键），
            # MilvusVectorStore 内部用 node.dict()[text_key] 取文本；同时 BM25 的 input_field_names
            # 也指向它，才能对该字段开启分词器。这里通过枚举统一为 "text"。
            text_key=milvus_config.MilvusDocumentField.CONTENT.value,
            # 关键：告诉 LlamaIndex 使用哪个字段作为向量字段
            # 默认是 "embedding"，这里改为我自己定义的 "content_vector"
            embedding_field=milvus_config.MilvusDocumentField.CONTENT_VECTOR.value,
            # 稠密 + 稀疏（BM25）混合检索
            enable_dense=True,
            enable_sparse=True,
            sparse_embedding_field=milvus_config.MilvusDocumentField.CONTENT_SPARSE.value,
            sparse_embedding_function=milvus_config.build_bm25_function(),
            # 额外的业务标量字段
            scalar_field_names=milvus_config.SCALAR_FIELD_NAMES,
            scalar_field_types=milvus_config.SCALAR_FIELD_TYPES,
            # 如果集合已存在，不覆盖
            overwrite=False,   # 设为 False 防止误删已有数据
        )

    def _create_pipelines(self):
        """创建不同类型的摄取管道"""
        # 通用配置
        ingestion_cache = IngestionCache(
            cache=RedisCache.from_host_and_port(Config.REDIS_HOST, Config.REDIS_PORT),
            collection=Config.REDIS_CACHE,
        )
        common_config = {
            "vector_store": self.milvus_vector_store,
            "docstore": self.redis_document_store,
            "cache": ingestion_cache,
            "docstore_strategy": DocstoreStrategy.UPSERTS_AND_DELETE
        }
        # 单独持有 cache 引用，供删除文档时清理 ingestion cache 使用
        self.ingestion_cache = ingestion_cache

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
            file_type = doc.metadata.get("file_type")
            if not file_type and file_name:
                file_type = os.path.splitext(os.path.basename(file_name))[1]

            if not file_type:
                logger.info(f"无法确定文件类型,file_name:{file_name}")
                return "error", "未知的文件类型"

            logger.info(f"开始处理文档:{file_name}")

            # 内容 hash 缓存失效：若文档内容较上次有变化，先清除该 doc 的 ingestion cache，
            # 强制重新切分/向量化，避免 IngestionCache 返回旧节点（同路径覆盖上传场景）
            self._bust_cache_if_content_changed(doc)

            if file_type in ('.markdown', '.md'):
                pipeline_nodes = self.markdown_pipeline.run(documents=[doc], show_progress=True)
                logger.info(f"文件[{file_name}]处理完成,生成了 {len(pipeline_nodes)} 个节点")

            else:
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


    def ingest_documents(self, docs: List[Document]):
        """
        批量摄取文档
        :param docs:
        :return:
        """
        if not docs:
            logger.info(f"批量文档摄取，入参文档列表为空")
            return None, None

        fail_docs = []
        success_docs = []
        for doc in docs:
            is_success, error_msg, nodes = self.ingest_document(doc)
            if 'success' != is_success:
                fail_docs.append(doc.metadata.get('file_name', 'unknown doc name'))
            else:
                success_docs.append(doc.metadata.get('file_name', 'unknown doc name'))

        logger.info(f"文档摄取完成，成功文档：{success_docs}, 失败文档：{fail_docs}")

        return success_docs, fail_docs


    def _bust_cache_if_content_changed(self, doc: Document) -> None:
        """
        内容 hash 缓存失效：对比本次文档与 docstore 中已存文档的 doc_md5，
        若内容变化（或旧数据无 doc_md5）则清除该 doc_id 的 ingestion cache，
        使后续 pipeline.run 重新切分/向量化，避免缓存返回旧节点。
        """
        doc_id = doc.doc_id
        curr_md5 = doc.metadata.get("doc_md5")
        if not curr_md5:
            return

        # 读取 docstore 中已存在的同 doc_id 文档
        prev_doc = None
        try:
            prev_doc = self.redis_document_store.get_document(doc_id)
        except KeyError:
            # 文档不存在（首次摄取），cache 本就是空的，无需清除
            prev_doc = None
        except Exception:
            logger.exception(f"读取 docstore 旧文档失败，跳过缓存失效判断：{doc_id}")
            prev_doc = None

        if prev_doc is None:
            return

        prev_md5 = prev_doc.metadata.get("doc_md5") if getattr(prev_doc, "metadata", None) else None
        # 内容变化 或 旧数据无 doc_md5：清除缓存强制重算
        if prev_md5 != curr_md5:
            try:
                self.ingestion_cache.delete(doc_id)
                logger.info(f"文档内容已变化，已清除 ingestion cache 强制重新向量化：{doc_id}")
            except Exception:
                logger.exception(f"清除 ingestion cache 失败：{doc_id}")


    def delete_document(self, doc_path_name: str) -> None:
        """
        删除指定文档在向量库、docstore、index_store、ingestion cache 中的全部数据。
        doc_id 使用确定性的 doc_path_name（与摄取时保持一致），因此可直接据此删除。
        :param doc_path_name: 文档在 minio 的路径（bucket/object），即 doc_id
        """
        doc_id = doc_path_name

        # 0. 删除前确保 Milvus 集合已加载到内存：Milvus 的 delete 要求集合 loaded，
        #    否则会报错；尤其服务刚启动、尚未处理过任何新增事件时集合可能未 load。
        #    重复 load 是幂等的，已 load 时不会报错。
        try:
            self.milvus_vector_store.client.load_collection(
                self.milvus_vector_store.collection_name
            )
        except Exception:
            logger.exception(f"加载 Milvus 集合失败（删除前）：{doc_path_name}")

        # 1. 删除 Milvus 向量数据（按 doc_id 过滤删除，不依赖 docstore 状态，最可靠）
        try:
            self.milvus_vector_store.delete(doc_id)
            logger.info(f"已从 Milvus 删除向量数据：{doc_path_name}")
        except Exception:
            logger.exception(f"从 Milvus 删除向量数据失败：{doc_path_name}")

        # 2. 删除 docstore + index_store 中的文档引用信息
        try:
            self.storage_context.delete_ref_doc(doc_id, delete_from_vector_store=False)
            logger.info(f"已从 docstore/index_store 删除文档引用：{doc_path_name}")
        except Exception:
            logger.exception(f"从 docstore/index_store 删除失败：{doc_path_name}")

        # 3. 单独清理 ingestion cache（delete_ref_doc 不会清理 cache，
        #    不清理会导致同名文件重传时命中旧缓存而不重新解析）
        try:
            self.ingestion_cache.delete(doc_id)
            logger.info(f"已清理 ingestion cache：{doc_path_name}")
        except Exception:
            logger.exception(f"清理 ingestion cache 失败：{doc_path_name}")



if __name__ == '__main__':
    file_path = '../test_data/MongoDB-test.md'
    docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
    ingestion = DocumentIngestionPipeline()
    result = ingestion.ingest_document(docs[0])
    print(f"文档摄取结果:{result}")

