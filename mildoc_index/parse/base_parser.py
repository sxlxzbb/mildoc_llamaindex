import os
from typing import List

from llama_index.core import Document, SimpleDirectoryReader

from logger.logging import setup_logging

logger = setup_logging()

class BaseParser:

    def default_parse(self, file_path: str, doc_path_name: str) -> List[Document] | None:

        if not os.path.exists(file_path):
            logger.info(f"BaseParser.default_parse文件不存在:{file_path}")
            return None

        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()

        docs[0].metadata['file_path'] = doc_path_name

        return docs


    def parse(self, file_path: str, doc_path_name: str) -> List[Document] | None:
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
