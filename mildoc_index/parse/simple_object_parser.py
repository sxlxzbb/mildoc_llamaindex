import os
import shutil
import uuid
from typing import Optional, Dict, Any, List

from llama_index.core import Document
from minio import Minio

from config.config import Config
from logger.logging import setup_logging
from parse.base_parser import BaseParser
from parse.markdown_parser import MarkdownParser
from parse.office_parser import OfficeParser
from parse.pdf_parser import PdfParser
from parse.text_parser import TextParser

logger = setup_logging()


class SimpleObjectParser:
    """简单对象解析器"""
    def __init__(self, minio_client:Minio):
        """
        初始化解析器
        :param minio_client:
        """
        # 临时文件目录
        self.temp_dir = Config.TMP_FILE_DIR

        # 初始化Minio客户端
        self.minio_client = minio_client

        # 注册解析器（按优先级排序）
        pdf_parser = PdfParser()
        self.parsers = [
            pdf_parser,
            OfficeParser(pdf_parser=pdf_parser),
            MarkdownParser(),
            TextParser()
        ]


    def _get_parser(self, content_type: str) -> Optional[BaseParser]:
        """
        根据内容类型获取合适的解析器
        :param content_type: 内容类型
        :return: 解析器实例，如果没有找到就返回None
        """
        for parser in self.parsers:
            if parser.supports(content_type):
                return parser
        return None


    def _extract_doc_type(self, content_type: str) -> str:
        """
        从content_type中提取文档类型
        :param content_type: 内容类型
        :return: 文档类型
        """
        if not content_type:
            return "unknown"

        # 提取主要类型
        main_type = content_type.split('/')[0].lower()
        sub_type = content_type.split('/')[-1].lower()

        # 映射常见类型
        type_mapping = {
            'application/pdf': 'pdf',
            'text/plain': 'txt',
            'text/html': 'html',
            'text/markdown': 'md',
            'text/x-markdown': 'md',
            'application/markdown': 'md',

            # Word文档
            'application/msword': 'doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',

            # Excel文档
            'application/vnd.ms-excel': 'xls',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',

            # PowerPoint文档
            'application/vnd.ms-powerpoint': 'ppt',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
        }

        return type_mapping.get(content_type.lower(), sub_type)


    def parse_object(self, bucket_name: str, object_name: str) -> None | list[Any] | list[Document]:
        """
        解析文件
        :param bucket_name:
        :param object_name:
        :return:
        """
        temp_dir = None
        try:
            # 先获取对象信息，检查文件大小
            logger.info(f"正在检查对象信息:{bucket_name}/{object_name}")
            stat = self.minio_client.stat_object(bucket_name, object_name)
            file_size = stat.size
            max_size = 512 * 1024 * 1024  # 512MB

            logger.info(f"{object_name}文件大小：{file_size}bytes, ({file_size/1024/1024:.2f}MB)")

            if file_size == 0:
                logger.info(f"空文件不处理,{bucket_name}/{object_name},file_size:{file_size}")
                return []

            if file_size > max_size:
                logger.info(f"文件过大 ({file_size / 1024 / 1024:.2f} MB > 512 MB)，跳过解析")
                return []

            # 从Minio获取对象
            logger.info(f"正在获取对象内容:{bucket_name}/{object_name}")
            response = self.minio_client.get_object(bucket_name, object_name)

            # 获取对象数据和元数据
            data = response.data
            headers = response.headers

            logger.info(f"对象大小:{len(data)}字节")
            # doc_name = os.path.basename(object_name)
            # doc_path_name = object_name
            content_type = headers.get('Content-Type', '')
            # doc_type = self._extract_doc_type(content_type)

            # doc_md5 = headers.get('ETag', '').strip('"')
            # # 如果ETag不是32位，则重新计算MD5，多部分上传：ETag={复合MD5}-{部分数量}（超过32字符）
            # if len(doc_md5) != 32:
            #     doc_md5 = self._calculate_md5(data)

            # 选择合适的解析器
            parser = self._get_parser(content_type)
            if not parser:
                logger.info(f"未找到适合 {content_type} 的解析器,{bucket_name}/{object_name}")
                return []

            # 保存临时文件
            temp_dir = os.path.join(self.temp_dir, str(uuid.uuid4()))
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, object_name)
            with open(file_path, 'wb') as f:
                f.write(data)

            # 解析文档内容
            logger.info(f"使用解析器:{parser.__class__.__name__}解析{object_name}")

            doc_path_name = f"{bucket_name}/{object_name}"
            documents = parser.parse(file_path, doc_path_name, file_size)
            if not documents:
                logger.info(f"文件解析结果为空:{bucket_name}/{object_name}")
                return []

            return documents
        except Exception:
            logger.exception(f"解析文件异常:{bucket_name}/{object_name}")
        finally:
            # 删除临时目录
            if temp_dir and os.path.exists(temp_dir):
                logger.info(f"删除临时目录：{temp_dir}")
                shutil.rmtree(temp_dir)



