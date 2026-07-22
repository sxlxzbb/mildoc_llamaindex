"""基于 LlamaIndex 的 RAG 检索与对话引擎封装（取代原先的裸 MilvusClient 检索）

职责：
- MilvusVectorStore：复用 mildoc_index 已建集合，保留「稠密 + BM25 稀疏」混合检索。
- VectorStoreIndex.as_retriever：召回候选节点。
- RerankPostprocessor：包装现有 RerankService 做精排（保留 DashScope / 词法兜底）。
- CondenseQuestionChatEngine：多轮对话 + 响应合成（流式）。
- MemoryServiceMemory：实现 BaseMemory，把记忆委托给现有 MemoryService（Redis + MySQL）。

注意：Settings.llm 必须 streaming=False，因为 CondenseQuestionChatEngine 的压缩步骤
用 llm.predict（非流式）；真正的流式由响应合成器（streaming=True）单独开启。
"""
from typing import Any, List, Optional

from llama_index.core.response_synthesizers import ResponseMode
from pymilvus import DataType

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    get_response_synthesizer,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

from config.config import Config
from logger.logger import logger
from core.rerank import RerankService
from memory.service import MemoryService
from memory.llama_memory import MemoryServiceMemory
from core.prompts import SYSTEM_PROMPT


# ===================== 提示词 =====================
# 响应合成器使用的 QA 模板：把客服 SYSTEM_PROMPT 与检索上下文、问题拼在一起。
QA_PROMPT = PromptTemplate(
    SYSTEM_PROMPT
    + "\n\n【参考资料】\n{context_str}\n\n"
    "【用户问题】\n{query_str}\n\n"
    "请基于以上要求作答："
)

# 压缩问题模板（中文）：把多轮追问改写为独立、完整的检索问题。
CONDENSE_PROMPT = PromptTemplate(
    "给定一段对话记录（人类与助手之间）以及人类的最新追问，"
    "请将最新追问改写为一个独立、完整的问题，使其包含对话中所有必要的上下文信息。\n\n"
    "<对话记录>\n{chat_history}\n\n"
    "<最新追问>\n{question}\n\n"
    "<独立问题>\n"
)


# ===================== 全局单例 =====================
_settings_initialized = False
_vector_store = None
_query_engine = None


def _ensure_settings() -> None:
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


def _get_vector_store() -> MilvusVectorStore:
    """获取 MilvusVectorStore 单例，复用 mildoc_index 已建集合（overwrite=False）。"""
    global _vector_store
    if _vector_store is None:
        _vector_store = MilvusVectorStore(
            uri=f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}",
            token=None,  # 使用 user/password 认证
            user=Config.MILVUS_USER,
            password=Config.MILVUS_PASSWORD,
            db_name=Config.MILVUS_DATABASE,
            collection_name=Config.MILVUS_COLLECTION,
            dim=Config.MILVUS_VECTOR_DIM,
            text_key="text",
            embedding_field="content_vector",
            enable_dense=True,
            enable_sparse=True,
            sparse_embedding_field="content_sparse",
            sparse_embedding_function=BM25BuiltInFunction(
                input_field_names=["text"],
                output_field_names=["content_sparse"],
                analyzer_params={"tokenizer": "jieba"},
            ),
            scalar_field_names=["file_path"],
            scalar_field_types=[DataType.VARCHAR],
            overwrite=False,  # 不覆盖已有数据
        )
        logger.info("MilvusVectorStore 初始化完成（复用已有集合）")
    return _vector_store


def ensure_collection_loaded() -> None:
    """确保集合已加载到内存（幂等；已加载时 Milvus 报错属正常，忽略）。

    每次请求调用一次，等价于原 load_collection 行为，
    避免集合被外部释放后检索失败。
    """
    try:
        _get_vector_store().client.load_collection(Config.MILVUS_COLLECTION)
    except Exception as e:
        logger.debug(f"加载集合提示(可忽略): {e}")


def get_query_engine() -> RetrieverQueryEngine:
    """获取 RAG 查询引擎单例（retriever + 重排后处理器 + 流式响应合成器）。"""
    global _query_engine
    if _query_engine is not None:
        return _query_engine

    # 确保嵌入模型和大语言模型已初始化
    _ensure_settings()

    # 向量集合加载
    ensure_collection_loaded()

    # 构建检索器（这儿使用混合检索）
    index = VectorStoreIndex.from_vector_store(vector_store=_get_vector_store())
    retriever = index.as_retriever(
        similarity_top_k=Config.TOP_K,
        vector_store_query_mode="hybrid",  # 稠密 + BM25 稀疏混合检索
    )

    # 响应合成器
    synthesizer = get_response_synthesizer(
        streaming=True,
        response_mode=ResponseMode.COMPACT,
        text_qa_template=QA_PROMPT,
    )

    # 构建查询引擎
    _query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[RerankPostprocessor()],
        response_synthesizer=synthesizer,
    )

    logger.info("RAG 查询引擎初始化完成")
    return _query_engine



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

    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[Any] = None,) -> List[NodeWithScore]:
        if not nodes:
            return nodes

        query = query_bundle.query_str if query_bundle else ""

        adapters = [_RerankAdapter(n) for n in nodes]
        # RerankService.rerank 会就地修改 score 并按相关性返回重排后的 adapter 列表
        reranked = RerankService.rerank(query, adapters, top_n=Config.TOP_N)
        return [a.node for a in reranked]


# ===================== 对外入口 =====================
def build_chat_engine(username: str, app: Any = None) -> CondenseQuestionChatEngine:
    """构建每用户独立的 CondenseQuestionChatEngine（查询引擎与记忆分离）。

    - 查询引擎（retriever/合成器）全局缓存，构建一次；
    - 记忆按用户隔离，每次请求新建 MemoryServiceMemory 实例；
    - 每次请求都确保集合已加载（幂等安全网，等价于原 retrieval 的 load_collection）。
    """
    # 保证加载集合
    ensure_collection_loaded()

    # 查询引擎
    query_engine = get_query_engine()

    # 记忆模块
    memory = MemoryServiceMemory.from_defaults(
        username=username,
        memory_service=MemoryService(),
        app=app,
    )

    # 返回对话引擎
    return CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        memory=memory,
        llm=Settings.llm,
        condense_question_prompt=CONDENSE_PROMPT,
    )


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
