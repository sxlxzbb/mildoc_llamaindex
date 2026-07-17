import os
import time

from mineru.cli.common import do_parse
from mineru.data.data_reader_writer import FileBasedDataWriter

from doc_convert.base_pdf_processor import BasePdfProcessor
from logger.logging import setup_logging

logger = setup_logging()

class MineruPdfProcessor(BasePdfProcessor):

    def parse_pdf_to_markdown(self, pdf_path: str, image_path: str) -> str | None:
        """
        通过mineru将pdf解析为markdown文件，并识别文件中的图片并上传到oss
        mineru转换结果中的表格是html格式
        :param pdf_path:
        :param image_path:
        :return:
        """
        if not pdf_path or not image_path:
            logger.info(f"MineruPdfProcessor.parse_pdf_to_markdown入参pdf_path或image_path为空,pdf_path:{pdf_path},image_path:{image_path}")
            return None

        if not os.path.exists(pdf_path) or not os.path.exists(image_path):
            logger.info(f"MineruPdfProcessor.parse_pdf_to_markdown入参pdf_path或image_path不存在,pdf_path:{pdf_path},image_path:{image_path}")
            return None

        try:
            start_time = int(time.time() * 1000)

            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            # 带后缀的文件名
            pdf_name = os.path.basename(pdf_path)
            # 不带后缀的文件名称
            pdf_name1 = os.path.splitext(pdf_name)[0]

            output_dir = os.path.dirname(pdf_path)

            logger.info(f"MineruPdfProcessor.parse_pdf_to_markdown开始将解析:{pdf_name}")
            img_writer = FileBasedDataWriter(image_path)

            do_parse(
                pdf_file_names=[pdf_name1],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=[""],
                output_dir=output_dir,
                img_writer=img_writer,
                parse_method="auto"
            )

            md_file_path = os.path.join(output_dir, pdf_name1, 'auto', f'{pdf_name1}.md')
            if not os.path.exists(md_file_path):
                logger.info(f"MineruPdfProcessor.parse_pdf_to_markdown执行以后，找不到markdown文件，fileName:{md_file_path}")
                return None

            md_content = None
            with open(md_file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()

            logger.info(f"MineruPdfProcessor.parse_pdf_to_markdown执行完成,{pdf_name},耗时:{int(time.time() * 1000) - start_time}ms")

            return md_content
        except Exception:
            logger.exception(f"MineruPdfProcessor.parse_pdf_to_markdown异常")