import requests

from config.config import Config
from logger.logger import logger

class RerankService:
    """重排序服务"""

    @staticmethod
    def rerank(query: str, chunks: list, top_n: int = 3) -> list:
        """
        对候选分片重排序。

        Args:
            query: 用户问题
            chunks: RetrievedChunk 列表
            top_n: 返回数量

        Returns:
            重排后的 chunks（前 top_n 个）
        """
        if not chunks:
            return []

        if Config.RERANK_PROVIDER == 'dashscope' and Config.RERANK_API_KEY:
            return RerankService._rerank_dashscope(query, chunks, top_n)

        return chunks


    @staticmethod
    def _rerank_dashscope(query: str, chunks: list, top_n: int) -> list:
        """调用百炼 rerank 服务"""
        documents = [c.content for c in chunks]
        try:
            headers = {
                'Authorization': f'Bearer {Config.RERANK_API_KEY}',
                'Content-Type': 'application/json',
            }
            data = {
                'model': Config.RERANK_MODEL_NAME,
                'input': {
                    'query': query,
                    'documents': documents
                },
                'parameters': {
                    'return_documents': True,   # 显式设置返回文档内容
                    'top_n': top_n or len(documents)
                }
            }

            resp = requests.post(Config.RERANK_ENDPOINT, headers=headers, json = data, timeout=30,)
            resp.raise_for_status()
            results = resp.json()['output']['results']
            # results 已按相关性降序，每项含 index / relevance_score
            reranked = []
            for r in results:
                idx = r['index']
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx]
                    chunk.score = float(r.get('relevance_score', chunk.score))
                    reranked.append(chunk)
            return reranked
        except Exception as e:
            logger.exception(f"百炼重排序失败，回退关键词重排")
            return chunks

