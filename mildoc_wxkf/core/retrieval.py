from pymilvus import MilvusClient
from openai import OpenAI

from config.config import Config
from logger.logger import logger


# ===================== 单例客户端 =====================
_milvus_client = None
_embed_client = None


def get_milvus_client() -> MilvusClient:
    """获取 Milvus 客户端单例"""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient(
            uri=f"http://{Config.MILVUS_HOST}:{Config.MILVUS_PORT}",
            user=Config.MILVUS_USER,
            password=Config.MILVUS_PASSWORD,
            db_name=Config.MILVUS_DATABASE,
        )
        logger.info("Milvus 客户端初始化完成")
    return _milvus_client


def get_embed_client() -> OpenAI:
    """获取 embedding 的 OpenAI 兼容客户端单例（走 DashScope 兼容接口）"""
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL,
        )
        logger.info("Embedding 客户端初始化完成")
    return _embed_client


class RetrievedChunk:
    """检索召回的一个文档分片"""

    def __init__(self, content: str, file_path: str = '', doc_name: str = '', score: float = 0.0):
        self.content = content
        self.file_path = file_path
        self.doc_name = doc_name
        self.score = score

    def to_dict(self) -> dict:
        return {
            'content': self.content,
            'file_path': self.file_path,
            'doc_name': self.doc_name,
            'score': self.score,
        }

    def to_source_dict(self) -> dict:
        """用于前端「参考来源」展示的轻量字典（不包含大段文本内容）"""
        return {
            'file_path': self.file_path,
            'doc_name': self.doc_name,
            'score': self.score,
        }

# ===================== 核心方法 =====================
def embed_texts(texts: list) -> list:
    """批量生成文本向量。

    使用与 mildoc_index 相同的 text-embedding-v4 模型，
    并通过 dimensions 参数强制输出维度与 Milvus 集合一致。
    """
    client = get_embed_client()
    resp = client.embeddings.create(
        model=Config.ENBEDDING_MODEL,
        input=texts,
        extra_body={"dimensions": Config.MILVUS_VECTOR_DIM},
    )
    return [item.embedding for item in resp.data]


def retrieve(query: str) -> list:
    """向量检索：返回 top_k 个最相关的分片（RetrievedChunk 列表）"""
    if not query.strip():
        return []

    # 1. 问题向量化
    query_vec = embed_texts([query])[0]

    # 2. 检索
    client = get_milvus_client()
    try:
        client.load_collection(Config.MILVUS_COLLECTION)
    except Exception as e:
        # 已加载时 Milvus 会报错，属正常，忽略
        logger.debug(f"加载集合提示(可忽略): {e}")

    try:
        results = client.search(
            collection_name=Config.MILVUS_COLLECTION,
            data=[query_vec],
            anns_field="content_vector",
            limit=Config.TOP_K,
            output_fields=["text", "file_path", "doc_name"],
        )
    except Exception as e:
        logger.error(f"Milvus 检索失败: {e}")
        raise

    # 3. 组装结果
    chunks = []
    for hit in results[0]:
        entity = hit.get("entity", {}) or {}
        chunks.append(RetrievedChunk(
            content=entity.get("text", ""),
            file_path=entity.get("file_path", ""),
            doc_name=entity.get("doc_name", ""),
            score=float(hit.get("distance", 0.0)),
        ))

    logger.info(f"检索到 {len(chunks)} 个分片")
    return chunks


def dedupe_chunks(chunks: list) -> list:
    """按 file_path 对分片去重，同一文档只保留得分最高的一份。

    用于前端「参考来源」展示：避免同一文档的多个分片重复出现。
    注意：仅影响展示，不影响传入 LLM 的上下文（上下文仍用原始 chunks）。
    """
    best = {}
    for c in chunks:
        key = c.file_path or c.doc_name or c.content
        if key not in best or c.score > best[key].score:
            best[key] = c
    # 按得分降序返回，保证展示顺序稳定
    return sorted(best.values(), key=lambda c: c.score, reverse=True)


