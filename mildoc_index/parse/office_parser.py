import os
from typing import List

from llama_index.core import Document
from markitdown import MarkItDown

from doc_convert.libre_office import LibreOffice
from logger.logging import setup_logging
from parse.base_parser import BaseParser
from parse.pdf_parser import PdfParser

logger = setup_logging()

class OfficeParser(BaseParser):
    def __init__(self, pdf_parser: PdfParser):
        """初始化markitdown实例"""
        # self.markitdown = MarkItDown()
        self.pdf_parser = pdf_parser
        self.libre_office = LibreOffice()


    def parse(self, file_path: str, doc_path_name: str) -> List[Document] | None:
        """
        先将office文档转为pdf，然后再走pdf的解析流程
        :param file_path:
        :return:
        """
        if not os.path.exists(file_path):
            logger.info(f"OfficeParser.parse文件不存在:{file_path}")
            return None

        try:
            file_name = os.path.basename(file_path)

            # office文档转为pdf
            pdf_file_path = self.libre_office.convert_doc_to_pdf(file_path)
            if not pdf_file_path or not os.path.exists(pdf_file_path):
                logger.info(f"OfficeParser.parse执行office转pdf返回结果为空或返回文件不存在,pdf_file_path:{pdf_file_path},使用默认解析器,{file_name}")
                # 使用默认解析器
                return self.default_parse(file_path, doc_path_name)

            pdf_parse_result = self.pdf_parser.parse(pdf_file_path, doc_path_name)
            if not pdf_parse_result:
                logger.info(f"OfficeParser.parse调pdf解析器解析返回结果为空,pdf_file_path:{pdf_file_path}")
                # 使用默认解析器
                return self.default_parse(file_path, doc_path_name)

            return pdf_parse_result

        except Exception:
            logger.exception(f"OfficeParser.parse异常")


    def supports(self, content_type: str) -> bool:
        "检查是否支持Office文档格式"
        supported_types = [
            # Word文档
            'application/msword', #.doc
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # .docx

            # Excel文档
            'application/vnd.ms-excel',  # .xls
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # .xlsx

            # PowerPoint文档
            'application/vnd.ms-powerpoint',  # .ppt
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',  # .pptx
        ]

        return content_type.lower() in [t.lower() for t in supported_types]