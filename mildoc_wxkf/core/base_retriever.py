"""检索 / 对话引擎公共基类（取代原先散落在 core/llama_rag.py 的模块级函数）。

设计（对齐 mildoc_index 摄取侧的「基类 + 两个子类」结构）：
- BaseRetrievalPipeline 承载全部共享逻辑：模型初始化、提示词、重排后处理器、
  参考来源抽取、向量库/存储上下文构建、集合加载、查询引擎与对话引擎骨架。
- 两种检索模式（default 混合检索 / hierarchical AutoMergingRetriever）只差两处，
  因此抽象成两个 hook 由子类覆写：
  1) _resolve_storage_namespaces()：返回 (collection, doc_ns, index_ns)；
     default 模式无 docstore，后两者为 None；hierarchical 模式指向独立的
     HIER 集合与 Redis 命名空间（必须和摄取侧一致）。
  2) _build_retriever()：default 直接返回混合检索器；hierarchical 在外面再
     套一层 AutoMergingRetriever，借助 docstore 的父-子关系做节点合并。

子类只需聚焦这两处差异，其余（模型、重排、合成、对话引擎）全部复用基类。
"""
from typing import Any, List, Optional, Tuple

from pymilvus import DataType

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    get_response_synthesizer,
)
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore

from config.config import Config
from logger.logger import logger
from core.rerank import RerankService
from core.prompts import SYSTEM_PROMPT
from memory.service import MemoryService
from memory.llama_memory import MemoryServiceMemory


# ===================== 字段名（与 mildoc_index 摄取侧 milvus_config.MilvusDocumentField 同值） =====================
# 注意：这里直接写死字面量，避免从 mildoc_index 跨项目 import 其 milvus_config
# （会连带注入 sys.path，污染 wxkf 自身的 logger 包）。取值必须和摄取侧一致：
#   text / content_vector / content_sparse / file_path
TEXT_FIELD = "text"
EMBEDDING_FIELD = "content_vector"
SPARSE_FIELD = "content_sparse"
DOC_PATH_FIELD = "file_path"


# ===================== 提示词 =====================
QA_PROMPT = PromptTemplate(
    SYSTEM_PROMPT
    + "\n\n【参考资料】\n{context_str}\n\n"
    "【用户问题】\n{query_str}\n\n"
    "请基于以上要求作答："
)

CONDENSE_PROMPT = PromptTemplate(
    "给定一段对话记录（人类与助手之间）以及人类的最新追问，"
    "请将最新追问改写为一个独立、完整的问题，使其包含对话中所有必要的上下文信息。\n\n"
    "<对话记录>\n{chat_history}\n\n"
    "<最新追问>\n{question}\n\n"
    "<独立问题>\n"
)


# ===================== 模型初始化（全局单例） =====================
_settings_initialized = False


def ensure_settings() -> None:
    """懒初始化 LlamaIndex 全局 Settings（embedding / llm）。"""
    global _settings_initialized
    if _settings_initialized:
        return

    # 嵌入模型：走百炼 OpenAI 兼容接口，强制维度与 Milvus 集合一致
    Settings.embed_model = OpenAIEmbedding(
        model_name=Config.ENBEDDING_MODEL,
        dimensions=Config.MILVUS_VECTOR_DIM,
        api_key=Config.OPENAI_API_KEY,
        api_base=Config.OPENAI_BASE_URL,
        embed_batch_size=10,
    )

    # 生成模型：用 OpenAILike，避免 OpenAI 类对非官方模型名（如 qwen 系列）的白名单校验。
    # is_chat_model=True 让其走 /chat/completions 接口；
    # streaming=False（压缩步骤用 predict，必须非流式），真正的流式由响应合成器单独开启。
    Settings.llm = OpenAILike(
        model=Config.LLM_MODEL_NAME,
        api_key=Config.LLM_API_KEY,
        api_base=Config.LLM_BASE_URL,
        temperature=Config.TEMPERATURE,
        max_tokens=Config.MAX_TOKENS,
        is_chat_model=True,
        is_function_calling_model=False,
        context_window=Config.LLM_CONTEXT_WINDOW,  # 必须显式设置，否则默认 3900 令 COMPACT 合成器预算为负
    )

    _settings_initialized = True


# ===================== 重排后处理器 =====================
class _RerankAdapter:
    """把 NodeWithScore 适配成 RerankService 需要的「content/file_path/doc_name/score」接口。"""

    def __init__(self, node: NodeWithScore):
        self.node = node
        self.content = node.get_content()
        self.file_path = node.metadata.get("file_path", "")
        self.doc_name = node.metadata.get("doc_name", "")
        self.score = float(node.score or 0.0)


class RerankPostprocessor(BaseNodePostprocessor):
    """用现有 RerankService 对检索结果精排（DashScope rerank 或词法兜底）。"""

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[Any] = None,
    ) -> List[NodeWithScore]:
        if not nodes:
            return nodes

        query = query_bundle.query_str if query_bundle else ""

        adapters = [_RerankAdapter(n) for n in nodes]
        # RerankService.rerank 会就地修改 score 并按相关性返回重排后的 adapter 列表
        reranked = RerankService.rerank(query, adapters, top_n=Config.TOP_N)
        return [a.node for a in reranked]


def sources_from_nodes(nodes: List[NodeWithScore]) -> List[dict]:
    """从检索节点抽取前端「参考来源」（按 file_path 去重，仅保留轻量元数据）。"""
    seen = set()
    out = []
    for n in nodes:
        meta = n.node.metadata if hasattr(n, "node") else {}
        fp = meta.get("file_path", "")
        dn = meta.get("doc_name", "")
        key = fp or dn
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "file_path": fp,
            "doc_name": dn,
            "score": float(n.score or 0.0),
        })
    return out


# ===================== 基类 =====================
class BaseRetrievalPipeline:
    """检索 / 对话引擎基类。子类只需聚焦「向量库/存储上下文/检索器」差异，覆写两个 hook。

    公共骨架：
    - 懒加载并缓存 vector_store / query_engine（同一实例跨请求复用，避免重复构建）；
    - build_chat_engine() 构建每用户独立的 CondenseQuestionChatEngine（记忆按用户隔离）；
    - 集合加载、模型初始化、重排、响应合成为共享逻辑。
    """

    def __init__(self):
        self._vector_store: Optional[MilvusVectorStore] = None
        self._query_engine: Optional[RetrieverQueryEngine] = None
        # 存储命名空间由子类决定（default: 无 docstore；hierarchical: 独立命名空间）
        self.collection, self.doc_ns, self.index_ns = self._resolve_storage_namespaces()

    # --------------------------- 子类 hook ---------------------------
    def _resolve_storage_namespaces(self) -> Tuple[str, Optional[str], Optional[str]]:
        """返回 (collection, doc_ns, index_ns)。

        默认模式：只连向量库，不依赖 docstore，后两者为 None。
        子类（hierarchical）覆写此方法，指向独立的 HIER 集合与 Redis 命名空间。
        """
        return Config.MILVUS_COLLECTION, None, None

    def _build_retriever(self, index: VectorStoreIndex, storage_context: StorageContext) -> Any:
        """构建检索器（hook）。

        默认：稠密 + BM25 稀疏混合检索。
        子类（hierarchical）覆写：在混合检索之上再套 AutoMergingRetriever。
        """
        return index.as_retriever(
            similarity_top_k=Config.TOP_K,
            vector_store_query_mode="hybrid",
        )


    # --------------------------- 共享：存储组件 ---------------------------
    def _get_vector_store(self) -> MilvusVectorStore:
        """获取 MilvusVectorStore 单例（复用已有集合，overwrite=False）。"""
        if self._vector_store is None:
            # —— Milvus 索引 / 检索调优参数 ——
            # index_type / nlist / metric_type 属「建索引时」参数，仅集合首次创建时生效；
            #   若集合已存在（overwrite=False），改这些不会重建索引，必须删集合重摄取。
            # nprobe 属「查询时」参数，立即生效、无需重建。
            # 默认值（不配 .env 时）：FLAT + nlist=1024 + IP + nprobe=10，等价当前行为。
            # 推荐：embedding=text-embedding-v4 时把 metric_type 改成 COSINE；
            #       数据量大时把 index_type 改成 IVF_FLAT 并配合 nlist（经验 ≈ sqrt(N)）。
            # 检索侧只配nprobe就可以了，index_type和nlist可以不关注（在摄取侧关注）
            # dense_index_config = {"index_type": Config.MILVUS_INDEX_TYPE.upper()}
            # if Config.MILVUS_INDEX_TYPE.upper().startswith("IVF"):
            #     # nlist 仅 IVF_* 系列生效；FLAT 传 nlist 会被 Milvus 拒绝，故按需添加。
            #     dense_index_config["nlist"] = Config.MILVUS_NLIST
            dense_search_config = {"nprobe": Config.MILVUS_NPROBE}

            self._vector_store = MilvusVectorStore(
                uri=f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}",
                token=None,  # 使用 user/password 认证
                user=Config.MILVUS_USER,
                password=Config.MILVUS_PASSWORD,
                db_name=Config.MILVUS_DATABASE,
                collection_name=self.collection,
                dim=Config.MILVUS_VECTOR_DIM,
                # 文本字段名必须与摄取侧写入的节点键一致
                text_key=TEXT_FIELD,
                embedding_field=EMBEDDING_FIELD,
                enable_dense=True,
                enable_sparse=True,
                sparse_embedding_field=SPARSE_FIELD,
                sparse_embedding_function=BM25BuiltInFunction(
                    input_field_names=[TEXT_FIELD],
                    output_field_names=[SPARSE_FIELD],
                    analyzer_params={"tokenizer": "jieba"},  # 中文分词
                ),
                scalar_field_names=[DOC_PATH_FIELD],
                scalar_field_types=[DataType.VARCHAR],
                overwrite=False,  # 不覆盖已有数据
                # ===== Milvus 索引 / 检索调优（见上方 dense_index_config / dense_search_config）=====
                # index_config=dense_index_config,               # 建索引参数：index_type（+ IVF 时的 nlist）
                similarity_metric=Config.MILVUS_METRIC_TYPE.upper(),  # 建索引 + 查询共用的度量
                search_config=dense_search_config,             # 查询参数：nprobe（IVF 时生效）
            )
            logger.info(f"MilvusVectorStore 初始化完成（集合：{self.collection}）")
        return self._vector_store


    def _build_storage_context(self) -> StorageContext:
        """构建 StorageContext：默认只有向量库；hierarchical 额外挂载 docstore/index store。"""
        vector_store = self._get_vector_store()
        kwargs = {"vector_store": vector_store}
        if self.doc_ns:
            kwargs["docstore"] = RedisDocumentStore.from_host_and_port(
                host=Config.REDIS_HOST, port=Config.REDIS_PORT, namespace=self.doc_ns,
            )
        if self.index_ns:
            kwargs["index_store"] = RedisIndexStore.from_host_and_port(
                host=Config.REDIS_HOST, port=Config.REDIS_PORT, namespace=self.index_ns,
            )
        return StorageContext.from_defaults(**kwargs)


    def _load_collection(self) -> None:
        """确保集合已加载到内存（幂等；已加载时 Milvus 报错属正常，忽略）。"""
        try:
            self._get_vector_store().client.load_collection(self.collection)
        except Exception as e:
            logger.debug(f"加载集合提示(可忽略): {e}")


    # --------------------------- 共享：查询 / 对话引擎 ---------------------------
    def get_query_engine(self) -> RetrieverQueryEngine:
        """获取 RAG 查询引擎单例（retriever + 重排后处理器 + 流式响应合成器）。"""
        if self._query_engine is not None:
            return self._query_engine

        ensure_settings()
        self._load_collection()

        storage_context = self._build_storage_context()
        index = VectorStoreIndex.from_vector_store(
            vector_store=storage_context.vector_store,
            storage_context=storage_context,
        )

        # 检索器
        retriever = self._build_retriever(index, storage_context)

        # 响应合成器
        synthesizer = get_response_synthesizer(
            streaming=True,
            response_mode=ResponseMode.COMPACT,
            text_qa_template=QA_PROMPT,
        )

        # 查询引擎
        self._query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[RerankPostprocessor()],
            response_synthesizer=synthesizer,
        )

        logger.info("RAG 查询引擎初始化完成")
        return self._query_engine

    def build_chat_engine(self, username: str, app: Any = None) -> CondenseQuestionChatEngine:
        """构建每用户独立的 CondenseQuestionChatEngine（查询引擎与记忆分离）。

        - 查询引擎（retriever/合成器）全局缓存，构建一次；
        - 记忆按用户隔离，每次请求新建 MemoryServiceMemory 实例；
        - 每次请求都确保集合已加载（幂等安全网）。
        """
        self._load_collection()
        query_engine = self.get_query_engine()

        memory = MemoryServiceMemory.from_defaults(
            username=username,
            memory_service=MemoryService(),
            app=app,
        )

        return CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            memory=memory,
            llm=Settings.llm,
            condense_question_prompt=CONDENSE_PROMPT,
        )


# ===================== 工厂：按配置选择对话引擎 =====================
# 缓存各模式的 pipeline 实例，保证 query_engine 单例跨请求复用（避免每次请求重建索引/检索器）。
_pipeline_instances: dict = {}


def get_chat_engine(username: str, app: Any = None):
    """根据 RETRIEVER_MODE 返回对应模式的对话引擎。

    - 'hierarchical' -> HierarchicalRetrievalPipeline（AutoMergingRetriever）
    - 其它（默认）   -> DefaultRetrievalPipeline（原混合检索）

    两种模式共享代码骨架，仅在「存储命名空间」与「检索器是否套 AutoMerging」上不同；
    通过懒加载子类避免与 base_retriever 形成循环导入。
    """
    mode = (Config.RETRIEVER_MODE or "default").lower()

    # 懒加载子类，打破与 default_retriever / hierarchical_retriever 的循环导入
    if mode == "hierarchical":
        from core.hierarchical_retriever import HierarchicalRetrievalPipeline
        cls = HierarchicalRetrievalPipeline
        logger.info(">>> 检索模式：hierarchical（AutoMergingRetriever）")
    else:
        from core.default_retriever import DefaultRetrievalPipeline
        cls = DefaultRetrievalPipeline
        logger.info(">>> 检索模式：default（原混合检索）")

    if mode not in _pipeline_instances:
        _pipeline_instances[mode] = cls()
    return _pipeline_instances[mode].build_chat_engine(username, app)
