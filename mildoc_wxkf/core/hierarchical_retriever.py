"""层次节点检索（AutoMergingRetriever）—— HierarchicalRetrievalPipeline。

配合 mildoc_index 侧 NODE_PARSER_MODE=hierarchical 写入的「父-子」层级节点：
在混合检索召回若干「叶子节点」后，AutoMergingRetriever 会检查这些叶子是否
同属一个父节点、且召回比例超过阈值，若是则自动合并为更大的「父节点」文本，
从而用更完整、连贯的上下文去合成答案，提升长文档问答的精确度。

实现上只覆写 BaseRetrievalPipeline 的两个 hook：
- _resolve_storage_namespaces()：指向独立的 HIER 集合与 Redis 命名空间
  （必须与摄取侧一致，AutoMergingRetriever 才能通过节点关系找到父节点并合并）；
- _build_retriever()：在混合检索之上套一层 AutoMergingRetriever。

其余（模型、重排、合成、对话引擎）全部复用基类。
"""
from typing import Any, Optional, Tuple

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import AutoMergingRetriever

from config.config import Config
from core.base_retriever import BaseRetrievalPipeline


class HierarchicalRetrievalPipeline(BaseRetrievalPipeline):
    """层次节点检索（AutoMergingRetriever）。"""

    def _resolve_storage_namespaces(self) -> Tuple[str, Optional[str], Optional[str]]:
        """返回 (HIER 集合, HIER docstore 命名空间, HIER index store 命名空间)。

        必须与 mildoc_index 摄取侧写入层级节点时使用的集合 / 命名空间一致，
        否则无法解析「父-子」关系。未配置时报错提示。
        """
        doc_ns = Config.REDIS_DOC_NAME_SPACE_HIER
        index_ns = Config.REDIS_INDEX_NAME_SPACE_HIER
        if not doc_ns or not index_ns:
            raise ValueError(
                "使用层次检索(RETRIEVER_MODE=hierarchical)前，请在 .env 中配置 "
                "REDIS_DOC_NAME_SPACE_HIER / REDIS_INDEX_NAME_SPACE_HIER，"
                "且其值必须与 mildoc_index 侧摄取时的 REDIS_DOC_NAME_SPACE_HIER / "
                "REDIS_INDEX_NAME_SPACE_HIER 完全一致。"
            )

        # 层次解析集合
        collection = Config.MILVUS_COLLECTION_HIER
        return collection, doc_ns, index_ns

    def _build_retriever(self, index: VectorStoreIndex, storage_context: StorageContext) -> Any:
        """在混合检索之上套一层 AutoMergingRetriever。"""
        base_retriever = index.as_retriever(
            similarity_top_k=Config.TOP_K,
            vector_store_query_mode="hybrid",
        )
        return AutoMergingRetriever(
            base_retriever,
            storage_context,
            simple_ratio_thresh=0.5,  # 同父节点的叶子召回比例超过该阈值才合并
        )
