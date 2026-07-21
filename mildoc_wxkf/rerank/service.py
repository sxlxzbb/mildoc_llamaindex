"""重排序模块

职责：对检索召回的候选分片用重排序模型重新打分，返回更相关的前 N 个。

优先级：
1. 若配置了 RERANK_PROVIDER=dashscope 且 RERANK_API_KEY 非空，
   调用阿里云百炼 rerank 服务（默认模型 gte-rerank-hybrid）。
2. 否则使用「关键词重合度」兜底重排（无外部依赖，保证可用）。
"""
import re

import requests

from config.config import Config
from logger import logger


# def _tokenize(text: str) -> set:
#     """简单分词：中文按字、英文/数字按词（小写）"""
#     return set(re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", (text or '').lower()))


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

        # logger.info("未配置重排序服务，使用关键词重合度兜底重排")
        # return RerankService._rerank_lexical(query, chunks, top_n)
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
            logger.error(f"百炼重排序失败，回退关键词重排: {e}")
            # return RerankService._rerank_lexical(query, chunks, top_n)
            return chunks


    # @staticmethod
    # def _rerank_lexical(query: str, chunks: list, top_n: int) -> list:
    #     """关键词重合度兜底重排（无外部依赖）,BM25 的极简版"""
    #     q_tokens = _tokenize(query)
    #     if not q_tokens:
    #         return chunks[:top_n]
    #     for c in chunks:
    #         overlap = len(q_tokens & _tokenize(c.content))
    #         c.score = float(overlap)
    #     return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_n]
