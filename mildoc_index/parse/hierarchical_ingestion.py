"""层次节点解析器摄取管道（继承 BaseDocumentIngestionPipeline，只聚焦解析器差异）。

区别：用 HierarchicalNodeParser 取代原有的 SentenceSplitter / MarkdownNodeParser，
把文档切成「父-子」层级节点（叶子是细粒度小块，父节点是若干叶子的合并）。
这样既保留稠密 + BM25 稀疏混合检索，又能在检索端用 AutoMergingRetriever
将召回的多个叶子节点自动合并为更大的父节点，提升长上下文问答的召回精确度。

模型/存储/摄取/删除等公共逻辑全部复用基类；本文件只实现：
- _resolve_storage_namespaces()：指向独立的集合 / Redis 命名空间（A/B 对比）
- _create_pipelines()：用 HierarchicalNodeParser 构建管道

默认行为不变：_get_ingestion_pipeline() 在 NODE_PARSER_MODE != 'hierarchical' 时
返回原始的 DocumentIngestionPipeline（基类默认命名空间），因此零改动。
"""
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import (
    SentenceSplitter,
    MarkdownNodeParser,
    HierarchicalNodeParser,
)
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy

from config.config import Config
from logger.logging import setup_logging
from parse.base_ingestion import BaseDocumentIngestionPipeline
# 复用原始管道类：当模式为 default 时，工厂函数直接返回它，保证原逻辑零改动。
from parse.ingestion_pipeline import DocumentIngestionPipeline

logger = setup_logging()


class HierarchicalDocumentIngestionPipeline(BaseDocumentIngestionPipeline):
    """层次节点解析器摄取管道（HierarchicalNodeParser 版）。"""

    def _resolve_storage_namespaces(self):
        """A/B 对比：返回独立集合 / Redis 命名空间"""
        return (
            Config.MILVUS_COLLECTION_HIER,
            Config.REDIS_DOC_NAME_SPACE_HIER,
            Config.REDIS_INDEX_NAME_SPACE_HIER,
            Config.REDIS_CACHE_HIER,
        )

    def _create_pipelines(self):
        """创建层次解析管道（适配新版 HierarchicalNodeParser API）。

        新版 from_defaults() 已无 node_parser 参数：
        - 只传 chunk_sizes -> 每一层自动用 SentenceSplitter 切分，形成「大块->小块」父子层级；
        - 若要某层用自定义 parser（如 Markdown 顶层保留标题），改用
          node_parser_ids + node_parser_map 显式指定每层解析器。
        """
        common_config = {
            "vector_store": self.milvus_vector_store,
            "docstore": self.redis_document_store,
            "cache": self.ingestion_cache,
            "docstore_strategy": DocstoreStrategy.UPSERTS,
        }

        # 文本类：直接用 chunk_sizes，由 HierarchicalNodeParser 在每层用 SentenceSplitter 切分，
        # 形成「大块 -> 小块」的父子层级节点。
        self.text_pipeline = IngestionPipeline(
            transformations=[
                HierarchicalNodeParser.from_defaults(
                    chunk_sizes=Config.HIERARCHICAL_CHUNK_SIZES,
                ),
                TitleExtractor(nodes=Config.TITLE_EXTRACTOR_NODES),
                Settings.embed_model,
            ],
            **common_config,
        )

        # Markdown 类：顶层用 MarkdownNodeParser 保留标题层级，更深层级再用 SentenceSplitter
        # 按配置的 chunk_sizes 继续切分（通过 node_parser_ids + node_parser_map 指定每层解析器）。
        sent_ids = [f"sent_{cs}" for cs in Config.HIERARCHICAL_CHUNK_SIZES]
        node_parser_map = {
            sid: SentenceSplitter(chunk_size=cs, chunk_overlap=Config.OVERLAP_SIZE)
            for sid, cs in zip(sent_ids, Config.HIERARCHICAL_CHUNK_SIZES)
        }
        md_id = "md"
        node_parser_ids = [md_id] + sent_ids
        node_parser_map[md_id] = MarkdownNodeParser(
            include_metadata=True,
            include_prev_next_rel=True,
        )
        self.markdown_pipeline = IngestionPipeline(
            transformations=[
                HierarchicalNodeParser.from_defaults(
                    node_parser_ids=node_parser_ids,
                    node_parser_map=node_parser_map,
                    include_metadata=True,
                    include_prev_next_rel=True,
                ),
                TitleExtractor(nodes=Config.TITLE_EXTRACTOR_NODES),
                Settings.embed_model,
            ],
            **common_config,
        )


if __name__ == '__main__':
    file_path = '../test_data/MongoDB-test.md'
    docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
    ingestion = HierarchicalDocumentIngestionPipeline()
    result = ingestion.ingest_document(docs[0])
    print(f"层次解析文档摄取结果:{result}")
