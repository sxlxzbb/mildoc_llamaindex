import os
import time
import uuid

import pymupdf4llm

from config.config import Config
from doc_convert.base_pdf_processor import BasePdfProcessor
from logger.logging import setup_logging

logger = setup_logging(__name__)


class Pymupdf4llmProcessor(BasePdfProcessor):
    """
    新版 PDF 处理器：PDF -> Markdown -> Document
    """

    # def __init__(self):
        # self.image_output_dir = image_output_dir
        # os.makedirs(image_output_dir, exist_ok=True)

    def parse_pdf_to_markdown(self, pdf_path: str, image_path: str) -> str | None:
        """
        将 PDF 转换为 Markdown 格式的 Document 对象
        通过pymupdf4llm转换得到的markdown中的表格是markdown格式的表格（mineru是html格式的）
        """

        if not pdf_path or not image_path:
            logger.info(f"Pymupdf4llmProcessor.parse_pdf_to_markdown入参pdf_path或image_path为空,pdf_path:{pdf_path},image_path:{image_path}")
            return None

        if not os.path.exists(pdf_path) or not os.path.exists(image_path):
            logger.info(f"Pymupdf4llmProcessor.parse_pdf_to_markdown入参pdf_path或image_path不存在,pdf_path:{pdf_path},image_path:{image_path}")
            return None

        try:
            # 带后缀的文件名
            pdf_name = os.path.basename(pdf_path)

            start_time = int(time.time() * 1000)
            logger.info(f"Pymupdf4llmProcessor.parse_pdf_to_markdown开始执行:{pdf_name}")

            md_text = pymupdf4llm.to_markdown(
                pdf_path,
                write_images=True,
                image_path=image_path,
                image_format="png"  # 可选：默认 'png' 或 'jpg'
            )

            logger.info(f"Pymupdf4llmProcessor.parse_pdf_to_markdown执行完成,{pdf_name},耗时:{int(time.time() * 1000) - start_time}ms")

            return md_text

        except Exception:
            logger.exception(f"Pymupdf4llmProcessor.parse_pdf_to_markdown异常")


if __name__ == '__main__':
    m = Pymupdf4llmProcessor()
    file_path = r"D:\zbb\test\知识文档\MongoDB-test.pdf"
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    temp_dir = os.path.join(Config.TMP_FILE_DIR, str(uuid.uuid4()))
    os.makedirs(temp_dir)

    image_path = os.path.join(temp_dir , Config.DOC_IMAGE_DIR)
    os.makedirs(image_path)

    doc = m.parse_pdf_to_markdown(file_path, image_path)
    md_path = os.path.join(temp_dir, f'{file_name}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(doc)
