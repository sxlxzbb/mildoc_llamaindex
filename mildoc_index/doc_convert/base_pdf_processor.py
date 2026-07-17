import os

from llama_index.readers.file import PDFReader

from logger.logging import setup_logging

logger = setup_logging()

class BasePdfProcessor:

    def parse_pdf_to_markdown(self, pdf_path: str, image_path: str) -> str | None:
        """
        默认的pdf解析器,直接读取PDF内容
        :param pdf_path:
        :param image_path:
        :return:
        """
        if not pdf_path:
            logger.info(f"BasePdfProcessor.parse_pdf_to_markdown入参pdf_path为空,pdf_path:{pdf_path}")
            return None

        if not os.path.exists(pdf_path):
            logger.info(f"BasePdfProcessor.parse_pdf_to_markdown入参pdf_path不存在,pdf_path:{pdf_path}")
            return None

        try:
            # 带后缀的文件名
            pdf_name = os.path.basename(pdf_path)

            docs = PDFReader.load_data(pdf_path)
            if not docs:
                logger.info(f"BasePdfProcessor.parse_pdf_to_markdown读取得到的PDF内容为空,{pdf_name}")
                return None

            return '/n'.join([doc.text for doc in docs])
        except Exception:
            logger.exception(f"BasePdfProcessor.parse_pdf_to_markdown异常")

