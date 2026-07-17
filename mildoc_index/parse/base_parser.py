import os
import hashlib
from typing import List

from llama_index.core import Document, SimpleDirectoryReader

from logger.logging import setup_logging

logger = setup_logging()

class BaseParser:

    @staticmethod
    def _calc_md5(text: str) -> str:
        """计算文本内容的 MD5，用于缓存失效判断（内容变化则重新向量化）。"""
        return hashlib.md5((text or "").encode("utf-8", errors="ignore")).hexdigest()

    def default_parse(self, file_path: str, doc_path_name: str) -> List[Document] | None:

        if not os.path.exists(file_path):
            logger.info(f"BaseParser.default_parse文件不存在:{file_path}")
            return None

        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()

        doc = docs[0]
        # 使用确定性的 doc_id（= minio 路径），与删除时保持一致，便于按路径删除
        doc.id_ = doc_path_name
        doc.metadata['file_path'] = doc_path_name
        # 内容 hash，供摄取时判断是否需要清除 ingestion cache（同路径覆盖上传场景）
        doc.metadata['doc_md5'] = self._calc_md5(doc.text)

        return docs


    def parse(self, file_path: str, doc_path_name: str, file_size: int = 0) -> List[Document] | None:
        """
        解析文档
        :param file_path:
        :param doc_path_name 文档在minio的路径
        :return:
        """
        pass

    def supports(self, content_type: str) -> bool:
        """
        当前解析器是否支持入参文档类型
        :param content_type:
        :return:
        """
        pass

if __name__ == '__main__':
    base_parse = BaseParser()
    file_path = r"D:\zbb\test\知识文档\nginx(已打印).docx"
    docs = base_parse.default_parse(file_path, "abc")
    doc = docs[0]
    print(doc.metadata)
    doc.metadata['file_path'] = os.path.join('abc', 'nginx(已打印).docx')
    print(doc.metadata)

    print(doc.text)
    print('=================================')
    doc.set_content('新内容')
    print(doc.text)
