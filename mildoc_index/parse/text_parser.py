import os
from typing import List

from llama_index.core import Document

from logger.logging import setup_logging
from parse.base_parser import BaseParser
logger = setup_logging()

class TextParser(BaseParser):

    def parse(self, file_path: str, doc_path_name: str, file_size: int = 0) -> List[Document] | None:
        if not os.path.exists(file_path):
            logger.info(f"TextParser.parse文件不存在:{file_path}")
            return None

        return self.default_parse(file_path, doc_path_name)


    def supports(self, content_type: str) -> bool:
        """检查是否支持文本"""
        return content_type.lower() in ['text/plain', 'text/html', 'txt']