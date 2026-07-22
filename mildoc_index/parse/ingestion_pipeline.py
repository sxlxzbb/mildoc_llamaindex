from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy

from config.config import Config
from logger.logging import setup_logging
from parse.base_ingestion import BaseDocumentIngestionPipeline

logger = setup_logging()


class DocumentIngestionPipeline(BaseDocumentIngestionPipeline):
    """原逻辑摄取管道：文本用 SentenceSplitter，Markdown 用 MarkdownNodeParser + SentenceSplitter。

    模型/存储/摄取/删除等公共逻辑全部继承自 BaseDocumentIngestionPipeline，
    本类只负责「如何把文档切成节点」这一步。
    """

    def _create_pipelines(self):
        """创建不同类型的摄取管道"""
        # 通用配置
        common_config = {
            "vector_store": self.milvus_vector_store,
            "docstore": self.redis_document_store,
            "cache": self.ingestion_cache,
            "docstore_strategy": DocstoreStrategy.UPSERTS  # 默认也是这个策略
        }

        # 1. 普通文本文件的管道 (使用 SentenceSplitter)
        self.text_pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=Config.CHUNK_SIZE,
                    chunk_overlap=Config.OVERLAP_SIZE
                ),
                TitleExtractor(nodes=Config.TITLE_EXTRACTOR_NODES),
                Settings.embed_model,
            ],
            **common_config
        )

        # 2. Markdown 文件的管道 (使用 MarkdownNodeParser)
        # 对于没有目录层级的文档，优化MarkdownNodeParser配置
        # MarkdownNodeParser 也会分隔文档，但只是粗略的分隔（粒度可能比较大），需要借助SentenceSplitter更精细的切割
        self.markdown_pipeline = IngestionPipeline(
            transformations=[
                MarkdownNodeParser(
                    include_metadata=True,       # 保留文档元数据
                    include_prev_next_rel=True,  # 保留节点之间的关系，有助于上下文理解
                ),
                # 使用更适合Markdown的分块策略，避免在图片标签中间切分
                SentenceSplitter(
                    chunk_size=Config.CHUNK_SIZE,
                    chunk_overlap=Config.OVERLAP_SIZE,
                    # 使用更安全的分隔符，避免在图片标签中间切分
                    separator="\n\n",
                    # 禁用默认的句子切分，使用段落级别切分
                    paragraph_separator="\n\n"
                ),
                TitleExtractor(nodes=Config.TITLE_EXTRACTOR_NODES),
                Settings.embed_model,
            ],
            **common_config
        )


if __name__ == '__main__':
    file_path = '../test_data/MongoDB-test.md'
    docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
    ingestion = DocumentIngestionPipeline()
    result = ingestion.ingest_document(docs[0])
    print(f"文档摄取结果:{result}")
