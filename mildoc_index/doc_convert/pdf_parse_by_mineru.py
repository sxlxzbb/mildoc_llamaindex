import os
import time

from mineru.cli.common import do_parse
from mineru.data.data_reader_writer import FileBasedDataWriter

from logger.logging import setup_logging
from oss.upload_image_to_oss import UploadImageToOSS

logger = setup_logging()

class PdfParseByMineru:
    def __init__(self):
        self.oss_tool = UploadImageToOSS()


    def parse_pdf_to_markdown(self, data: bytes, temp_dir: str, file_name: str) -> str | None:
        """
        通过mineru将pdf解析为markdown文件，并识别文件中的图片并上传到oss
        mineru转换结果中的表格是html格式
        :param data: 文档二进制数据
        :param temp_dir: 解析得到的markdown文件临时存储路径
        :param file_name: 不带后缀的文件名称
        :return:
        """
        try:
            start_time = int(time.time() * 1000)
            logger.info(f"开始将pdf解析为markdown:{file_name}.pdf")
            img_writer = FileBasedDataWriter(os.path.join(temp_dir, 'images'))

            do_parse(
                pdf_file_names=[file_name],
                pdf_bytes_list=[data],
                p_lang_list=[""],
                output_dir=temp_dir,
                img_writer=img_writer,
                parse_method="auto"
            )

            md_file_path = os.path.join(temp_dir, file_name, 'auto', f'{file_name}.md')
            if not os.path.exists(md_file_path):
                logger.info(f"pdf解析为markdown以后，找不到markdown文件，fileName:{md_file_path}")
                return None

            logger.info(f"pdf解析为markdown完成:{md_file_path},耗时:{int(time.time() * 1000) - start_time}ms")

            return self.oss_tool.process_markdown_with_threadpoll(md_file_path)
        except Exception:
            logger.exception("将PDF文件解析为markdown文件异常")