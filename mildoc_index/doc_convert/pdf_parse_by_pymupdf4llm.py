import os
import pymupdf4llm
from llama_index.core.schema import Document

from config.config import Config
from logger.logging import setup_logging

logger = setup_logging(__name__)


class MultimodalPDFProcessor:
    """
    新版 PDF 处理器：PDF -> Markdown -> Document
    """

    def __init__(self, image_output_dir: str = Config.TMP_FILE_DIR):
        self.image_output_dir = image_output_dir
        os.makedirs(image_output_dir, exist_ok=True)

    def parse_pdf_to_markdown(self, pdf_path: str) -> Document:
        """
        将 PDF 转换为 Markdown 格式的 Document 对象
        通过pymupdf4llm转换得到的markdown中的表格是markdown格式的表格（mineru是html格式的）
        """
        file_name = os.path.basename(pdf_path)
        logger.info(f"开始将 PDF 转换为 Markdown: {file_name}")

        try:
            # 1. 使用 pymupdf4llm 提取 Markdown 和图片
            # write_images=True 会自动提取图片并保存到 image_path
            # image_path 是图片保存的文件夹
            # image_format 也就是在 markdown 里保存时使用的文件格式
            md_text = pymupdf4llm.to_markdown(
                pdf_path,
                write_images=True,
                image_path=self.image_output_dir,
                image_format="png"  # 可选：默认 'png' 或 'jpg'
            )

            # 2. 构造稳定的逻辑 ID (父文档 ID)
            stable_doc_id = f"knowledge_base/{file_name}"

            # 3. 封装成 LlamaIndex Document 对象
            # 注意：我们这里返回的是一个“大文档”，后续交给 Pipeline 的 MarkdownNodeParser 去切分
            doc = Document(
                id_=stable_doc_id,
                text=md_text,
                metadata={
                    "file_name": file_name,
                    "doc_id": stable_doc_id,
                    "file_path": pdf_path,
                    "content_type": "markdown"  # 标记类型，方便后续处理
                }
            )

            logger.info(f"PDF 转 Markdown 成功，长度: {len(md_text)} 字符")
            return doc

        except Exception as e:
            logger.error(f"PDF 转 Markdown 失败: {e}")
            raise e


if __name__ == '__main__':
    m = MultimodalPDFProcessor()
    file_path = r"D:\zbb\test\知识文档\MongoDB-test.pdf"
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    doc = m.parse_pdf_to_markdown(file_path)
    with open(f'{Config.TMP_FILE_DIR}/{file_name}.md', 'w', encoding='utf-8') as f:
        f.write(doc.text)
