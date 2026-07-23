"""文档摄取管道公共基类。

把 DocumentIngestionPipeline（原逻辑）与 HierarchicalDocumentIngestionPipeline
（层次节点解析）之间的重复代码上提到此处：
- 模型初始化（LLM / 嵌入）
- 存储组件初始化（Milvus 稠密+BM25 混合、Redis docstore / index store / ingestion cache）
- 文档摄取、批量摄取、删除、重传保护
- 动态更新模型

子类只需聚焦「文档如何解析→切成哪些节点」这一步，实现：
- _create_pipelines()：构建 self.text_pipeline / self.markdown_pipeline
- 可选覆写 _resolve_storage_namespaces()：返回独立的集合 / Redis 命名空间（A/B 对比用）

原两者行为在重构后逐字节等价（日志统一为中性文字，不再带“层次解析”字样）。
"""
from typing import List, Optional, Tuple
import os

from llama_index.core import VectorStoreIndex, StorageContext, Settings
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


class BaseDocumentIngestionPipeline:
    """文档摄取管道公共基类。子类实现 _create_pipelines() 即可。"""

    def __init__(self):
        self._setup_models()
        self.index: Optional[VectorStoreIndex] = None
        self.milvus_vector_store: Optional[MilvusVectorStore] = None
        self.text_pipeline: Optional[IngestionPipeline] = None
        self.markdown_pipeline: Optional[IngestionPipeline] = None

        # 1. 初始化存储组件
        self._initialize_storage_components()

        # 2. 创建解析管道（由子类决定用哪种 node parser）
        self._create_pipelines()

        # 3. 构建 StorageContext
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.milvus_vector_store,
            docstore=self.redis_document_store,
            index_store=self.redis_index_store,
        )

    # --------------------------- 模型 / 存储 ---------------------------
    def _setup_models(self):
        """设置 LLM / 嵌入模型（与原逻辑保持一致）。"""
        Settings.llm = DashScope(
            api_key=Config.LLM_API_KEY,
            api_base=Config.LLM_BASE_URL,
            model_name=Config.LLM_MODEL_NAME,
            temperature=Config.TEMPERATURE,
        )

        # 用 OpenAIEmbedding 走百炼的 OpenAI 兼容接口（OPENAI_BASE_URL），
        # 支持 dimensions 参数，强制输出维度与 Milvus 的 MILVUS_VECTOR_DIM 对齐。
        # 必须用 model_name= 而非 model=，否则会触发 OpenAIEmbeddingModelType 枚举校验报错。
        Settings.embed_model = OpenAIEmbedding(
            model_name=Config.ENBEDDING_MODEL,      # "text-embedding-v4"，绕过枚举校验
            dimensions=Config.MILVUS_VECTOR_DIM,    # 与 Milvus 维度一致
            api_key=Config.OPENAI_API_KEY,
            api_base=Config.OPENAI_BASE_URL,
            embed_batch_size=10,  # 限制每批嵌入数量，避免超过接口上限
        )

    def _resolve_storage_namespaces(self) -> Tuple[str, str, str, str]:
        """返回 (collection, doc_ns, index_ns, cache_col)。

        默认使用原始配置；子类可覆写，使其指向独立的集合 / Redis 命名空间，
        从而在不覆盖原数据的前提下做 A/B 对比。
        """
        return (
            Config.MILVUS_COLLECTION,
            Config.REDIS_DOC_NAME_SPACE,
            Config.REDIS_INDEX_NAME_SPACE,
            Config.REDIS_CACHE,
        )


    def _initialize_storage_components(self):
        """初始化存储组件（集合 / Redis 命名空间支持子类覆写）。"""
        collection, doc_ns, index_ns, cache_col = self._resolve_storage_namespaces()

        self.redis_index_store = RedisIndexStore.from_host_and_port(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            namespace=index_ns,
        )

        self.redis_document_store = RedisDocumentStore.from_host_and_port(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            namespace=doc_ns,
        )

        # llama-index-vector-stores-milvus >= 1.0 改为参数式建 schema，
        # 不再接收 schema=/text_field=/vector_field= 参数，由 store 自动构建。
        self.milvus_vector_store = MilvusVectorStore(
            uri=f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}",
            token=None,  # 用的是 user/password 认证，所以 token 留空
            user=Config.MILVUS_USER,        # 经 **kwargs 透传给 MilvusClient
            password=Config.MILVUS_PASSWORD,  # 经 **kwargs 透传给 MilvusClient
            db_name=Config.MILVUS_DATABASE,  # 指定数据库名，经 **kwargs 透传
            collection_name=collection,  # 集合名称
            dim=Config.MILVUS_VECTOR_DIM,    # 必须与嵌入模型维度一致
            # 关键：文本字段名必须是 "text"（即节点 node.dict() 的键），
            # MilvusVectorStore 内部用 node.dict()[text_key] 取文本；同时 BM25 的 input_field_names
            # 也指向它，才能对该字段开启分词器。这里通过枚举统一为 "text"。
            text_key=milvus_config.MilvusDocumentField.CONTENT.value,
            # 关键：告诉 LlamaIndex 使用哪个字段作为向量字段（默认 "embedding"）
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

        # 单独持有 cache 引用，供删除文档时清理 ingestion cache 使用
        self.ingestion_cache = IngestionCache(
            cache=RedisCache.from_host_and_port(Config.REDIS_HOST, Config.REDIS_PORT),
            collection=cache_col,
        )

    def _create_pipelines(self):
        """由子类实现：构建 self.text_pipeline 与 self.markdown_pipeline。"""
        raise NotImplementedError("子类必须实现 _create_pipelines()")


    def update_model_config(self, model_name: str, temperature: float, max_tokens: int):
        """动态更新 LLM 模型和温度"""
        Settings.llm = DashScope(
            api_key=Config.LLM_API_KEY,
            api_base=Config.LLM_BASE_URL,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info(f"模型已更新: model={model_name}, temperature={temperature}")


    # --------------------------- 摄取接口 ---------------------------
    def ingest_document(self, doc: Document) -> Tuple:
        """摄取文档并创建索引（公共骨架，子类无需关心）。"""
        try:
            if not doc:
                logger.info(f"文档摄取，入参文档为空")
                return "error", "入参文档为空", None

            file_name = doc.metadata.get("file_name")
            file_type = doc.metadata.get("file_type")
            if not file_type and file_name:
                file_type = os.path.splitext(os.path.basename(file_name))[1]

            if not file_type:
                logger.info(f"无法确定文件类型,file_name:{file_name}")
                return "error", "未知的文件类型", None

            logger.info(f"开始处理文档:{file_name}")

            if file_type in ('.markdown', '.md'):
                pipeline_nodes = self.markdown_pipeline.run(documents=[doc], show_progress=True)
            else:
                pipeline_nodes = self.text_pipeline.run(documents=[doc], show_progress=True)
            logger.info(f"文件[{file_name}]处理完成,生成了 {len(pipeline_nodes)} 个节点")

            # 检查是否有新节点生成（如果文档重复，pipeline_nodes 将为空）
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
        """批量摄取文档。"""
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


    def _delete_old_if_exists(self, doc: Document) -> None:
        """覆盖上传 / 重传保护：若同一 doc_id 的文档已存在于 docstore，先清理其旧向量数据。"""
        doc_id = doc.doc_id
        try:
            existing = self.redis_document_store.get_document(doc_id)
        except Exception:
            logger.exception(f"读取 docstore 判断文档是否已存在失败，跳过旧数据清理：{doc_id}")
            return

        if existing is None:
            return

        logger.info(f"文档已存在（覆盖上传/重传），先清理旧数据：{doc_id}")
        self.delete_document(doc_id)


    def delete_document(self, doc_path_name: str) -> None:
        """删除指定文档在向量库、docstore、index_store、ingestion cache 中的全部数据。"""
        doc_id = doc_path_name

        # 0. 删除前确保 Milvus 集合已加载到内存（delete 要求集合 loaded）
        try:
            self.milvus_vector_store.client.load_collection(
                self.milvus_vector_store.collection_name
            )
            logger.info(f"已加载集合 {self.milvus_vector_store.collection_name}")
        except Exception:
            logger.exception(f"加载 Milvus 集合失败（删除前）：{doc_path_name}")

        # 1. 删除 Milvus 向量数据（按 doc_id 过滤删除，不依赖 docstore 状态，最可靠）
        try:
            self.milvus_vector_store.delete(doc_id)
            logger.info(f"已从 Milvus 删除向量数据：{doc_path_name}")
        except Exception:
            logger.exception(f"从 Milvus 删除向量数据失败：{doc_path_name}")

        # 2. 删除 docstore 中的文档数据（含层次解析产生的父/子节点）
        #    说明：ingest_document 已把 HierarchicalNodeParser 产出的全部「父-子」节点写入
        #    docstore，每个节点以各自的 node_id 为 key、ref_doc_id 指向本文档 doc_id。
        #    - delete_document(doc_id) 仅删除「恰好以 doc_id 为 key」的那一条（原始 Document
        #      节点），无法清理以自身 node_id 存储的父/子节点，会留下孤儿节点；
        #    - delete_ref_doc(doc_id) 才会遍历 ref_doc_info.node_ids，级联删除所有父/子节点，
        #      并清除 ref_doc 记录（结尾还会顺手删掉 node_collection[doc_id]，即原始文档节点）。
        #    两者配合：
        #    * 新/修复后的文档：delete_ref_doc 已级联清理父/子 + 原始节点 + ref_doc 记录；
        #    * 旧版仅含原始 Document、无 ref_doc_info 的文档：delete_ref_doc 会提前返回，
        #      此时由 delete_document 兜底删除原始节点。两种情形都不会残留。
        try:
            self.redis_document_store.delete_ref_doc(doc_id, raise_error=False)
            self.redis_document_store.delete_document(doc_id, raise_error=False)
            logger.info(f"已从 docstore 删除文档引用（含父/子节点）：{doc_path_name}")
        except Exception:
            logger.exception(f"从 docstore 删除失败：{doc_path_name}")

        # 3. 清理 ingestion cache（缓存 key 为内容 hash，无法按文档精确删除，只能整体 clear）
        try:
            self.ingestion_cache.clear()
            logger.info(f"已清理 ingestion cache：{doc_path_name}")
        except Exception:
            logger.exception(f"清理 ingestion cache 失败：{doc_path_name}")

