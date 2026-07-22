"""默认检索模式（原混合检索）：DefaultRetrievalPipeline。

行为与原 core/llama_rag.py 逐字节等价——它不覆写任何 hook，
完全复用 BaseRetrievalPipeline 的默认实现：
- _resolve_storage_namespaces() -> (MILVUS_COLLECTION, None, None)，即只连向量库；
- _build_retriever() -> 稠密 + BM25 稀疏混合检索。

这里单独成类只是为了与 hierarchical 模式对称、结构清晰，并非必须覆写。
"""
from core.base_retriever import BaseRetrievalPipeline


class DefaultRetrievalPipeline(BaseRetrievalPipeline):
    """默认混合检索（VectorStoreIndex.as_retriever + hybrid），零额外逻辑。"""
    pass
