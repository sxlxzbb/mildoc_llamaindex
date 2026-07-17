import os
from datetime import date
from typing import List

from llama_index.core import Document

from config.config import Config
from doc_convert.base_pdf_processor import BasePdfProcessor
from doc_convert.mineru_pdf_processor import MineruPdfProcessor
from doc_convert.pymupdf4llm_processor import Pymupdf4llmProcessor
from logger.logging import setup_logging
from oss.upload_image_to_oss import UploadImageToOSS
from parse.base_parser import BaseParser
from util.file_type_enum import FileType

logger = setup_logging()

class PdfParser(BaseParser):
    """
    pdf文件解析器，将pdf文件转为markdown文件
    """
    def __init__(self):
        self.oss_upload = UploadImageToOSS()
        self.pdf_processor = BasePdfProcessor()

        self._set_pdf_processor()


    def _set_pdf_processor(self):
        """
        根据配置设置是否使用第三方工具解析pdf
        :return:
        """
        if Config.USE_MINREU:
            self.pdf_processor = MineruPdfProcessor()
        elif Config.USE_PYMUPDF4LLM:
            self.pdf_processor = Pymupdf4llmProcessor()


    def parse(self, pdf_path: str, doc_path_name: str) -> List[Document] | None:
        """
        将pdf文件转为markdown文件
        :param pdf_path:
        :return:
        """
        if not os.path.exists(pdf_path):
            logger.info(f"PdfParser.parse文件不存在:{pdf_path}")
            return None

        try:

            # 文件路径（不包含文件名）
            path_name = os.path.dirname(pdf_path)
            # 带后缀的文件名
            pdf_name = os.path.basename(pdf_path)

            # 图片临时目录
            image_dir = os.path.join(path_name, 'images')
            os.makedirs(image_dir)

            # 解析pdf文件
            content = self.pdf_processor.parse_pdf_to_markdown(pdf_path, image_dir)

            if not content:
                logger.info(f"PdfParser.parse文件解析结果为空:{pdf_name}")
                return None

            # content_type = FileType.MARKDOWN

            doc = Document(
                id_=doc_path_name,
                text=content,
                metadata={
                    'creation_date': date.today().strftime("%Y-%m-%d"),
                    "file_name": pdf_name,
                    "file_path": doc_path_name,
                    "file_type": FileType.MARKDOWN  # 标记类型，方便后续处理
                }
            )

            # 如果是通过第三方工具解析的，识别Markdown中的图片并上传OSS
            new_content = None
            if not isinstance(self.pdf_processor, BasePdfProcessor):
                new_content = self.oss_upload.upload_markdown_image_to_oss(content, pdf_path)
            else:
                doc.metadata['file_type'] = FileType.TEXT

            if not new_content:
                logger.info(f"PdfParser.parse替换Markdown文件中的图片返回结果为空:{pdf_name}")
                if not content:
                    doc.metadata['doc_md5'] = self._calc_md5(doc.text)
                    return [doc]
                else:
                    logger.info(f"PdfParser.parse替换markdown中图片使得原来的content内容也为空了:{pdf_name}")
                    return None

            doc.set_content(new_content)
            doc.metadata['doc_md5'] = self._calc_md5(doc.text)

            return [doc]
        except Exception:
            logger.exception("PdfParser.parse发生异常")


    def supports(self, content_type: str) -> bool:
        """检查是否支持PDF"""
        return content_type.lower() in ['application/pdf', 'pdf']

