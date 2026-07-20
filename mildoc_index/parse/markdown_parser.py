import os

from llama_index.core import Document
from llama_index.readers.file import MarkdownReader

from logger.logging import setup_logging
from parse.base_parser import BaseParser
from util.file_type_enum import FileType

logger = setup_logging()

class MarkdownParser(BaseParser):

    def parse(self, markdown_path: str, doc_path_name: str, file_size: int = 0) -> list[Document] | None:
        """
        直接使用MarkdownReader加载得到的文档的metadata是空的，可能是需要md文件开头的 YAML 风格键值对，并用 --- 包裹。就像这样：
        ---
        title: 我的文章标题
        author: 张三
        date: 2026-07-20
        ---

        # 正文开始...

        所以直接使用基础的目录读取器读取markdown文档

        :param markdown_path: 文档本地路径
        :param doc_path_name: 文本在对象存储器(这儿是minio)的路径
        :param file_size:
        :return:
        """

        return self.default_parse(markdown_path, doc_path_name)

        # if not os.path.exists(markdown_path):
        #     logger.info(f"MarkdownParser.parse文件不存在:{markdown_path}")
        #     return None
        #
        # docs = MarkdownReader().load_data(file=markdown_path)
        #
        # doc = docs[0]
        # # 使用确定性的 doc_id（= minio 路径），与删除时保持一致，便于按路径删除
        # doc.id_ = doc_path_name
        # doc.metadata['file_path'] = doc_path_name
        # doc.metadata['file_type'] = FileType.MARKDOWN.value
        #
        # doc.metadata['doc_md5'] = self._calc_md5(doc.text)
        #
        # return docs


    def supports(self, content_type: str) -> bool:
        """检查是否支持Markdown格式"""
        supported_types = [
            'text/markdown',
            'text/x-markdown',
            'application/markdown',
            'md'
        ]
        return content_type.lower() in [t.lower() for t in supported_types]