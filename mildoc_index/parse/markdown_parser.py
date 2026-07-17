import os

from llama_index.core import Document
from llama_index.readers.file import MarkdownReader

from logger.logging import setup_logging
from parse.base_parser import BaseParser

logger = setup_logging()

class MarkdownParser(BaseParser):

    def parse(self, markdown_path: str, doc_path_name: str) -> list[Document] | None:
        if not os.path.exists(markdown_path):
            logger.info(f"MarkdownParser.parse文件不存在:{markdown_path}")
            return None

        docs = MarkdownReader().load_data(file=markdown_path)

        docs[0].metadata['file_path'] = doc_path_name

        return docs


    def supports(self, content_type: str) -> bool:
        """检查是否支持Markdown格式"""
        supported_types = [
            'text/markdown',
            'text/x-markdown',
            'application/markdown',
            'md'
        ]
        return content_type.lower() in [t.lower() for t in supported_types]