"""层次节点解析器摄取管道（继承 BaseDocumentIngestionPipeline，只聚焦解析器差异）。

区别：用 HierarchicalNodeParser 取代原有的 SentenceSplitter / MarkdownNodeParser，
把文档切成「父-子」层级节点（叶子是细粒度小块，父节点是若干叶子的合并）。

模型/存储/删除等公共逻辑全部复用基类；本文件只实现：
- _resolve_storage_namespaces()：指向独立的集合 / Redis 命名空间（A/B 对比）
- _create_pipelines()：用 HierarchicalNodeParser 构建「docstore 专用管道」
- ingest_document()：只把 leaf（叶子）节点写 Milvus，全部节点（父+子）写 docstore

为什么只把 leaf 写向量库：
HierarchicalNodeParser 返回的是「父+子」扁平节点列表，两者都会被 embedding 并写入
Milvus。但 AutoMergingRetriever 的工作方式是：先按相似度召回若干 leaf，再根据
leaf.parent_node 引用去 docstore 取父节点完整文本做合并（见 auto_merging_retriever.py）。
也就是说——检索只用 leaf 的向量，父节点的向量纯属冗余（占 Milvus 空间，还可能稀释精度）。
因此这里刻意拆开：
  1) docstore 专用管道只解析+落盘全部节点（父+子）到 Redis docstore，并保留
     IngestionCache 去重（重复文档该管道返回空，即「重复上传」）；
  2) 从同一批节点里筛出 leaf（无 CHILD 关系的节点），单独 embedding 后只写入 Milvus。
这样父子节点 ID 完全一致（同一次解析），且父节点向量不再进 Milvus。

默认行为不变：_get_ingestion_pipeline() 在 NODE_PARSER_MODE != 'hierarchical' 时
返回原始的 DocumentIngestionPipeline（基类默认命名空间），因此零改动。
"""
import os

from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import (
    SentenceSplitter,
    MarkdownNodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes,
)
from llama_index.core.schema import MetadataMode
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy

from config.config import Config
from logger.logging import setup_logging
from ingestion.base_ingestion import BaseDocumentIngestionPipeline
# 复用原始管道类：当模式为 default 时，工厂函数直接返回它，保证原逻辑零改动。
from ingestion.ingestion_pipeline import DocumentIngestionPipeline

logger = setup_logging()


class HierarchicalDocumentIngestionPipeline(BaseDocumentIngestionPipeline):
    """层次节点解析器摄取管道（HierarchicalNodeParser 版）。

    关键约定：同一次解析产出的父子节点共享一致 ID（llama_index 默认 id_func 是随机
    uuid，所以必须「只解析一次」，再从中分流到 docstore / 向量库，不能解析两遍）。
    """

    def _resolve_storage_namespaces(self):
        """A/B 对比：返回独立集合 / Redis 命名空间"""
        return (
            Config.MILVUS_COLLECTION_HIER,
            Config.REDIS_DOC_NAME_SPACE_HIER,
            Config.REDIS_INDEX_NAME_SPACE_HIER,
            Config.REDIS_CACHE_HIER,
        )

    def _create_pipelines(self):
        """创建「docstore 专用」解析管道。

        只解析 + 落盘全部节点（父+子）到 docstore，并保留 IngestionCache 去重。
        注意：这里【不传 vector_store、不加 embed_model】——
        - 不传 vector_store：全部节点只进 docstore，不进 Milvus（父节点向量不浪费）；
        - 不加 embed_model：父节点不需要 embedding（检索不靠它的向量），省一次嵌入调用。
        """
        # 该管道只写 docstore、不写 vector_store，故用 DUPLICATES_ONLY
        #（UPSERTS 需配合 vector_store，无 vector_store 时会被 llama_index 静默降级为
        #  duplicates_only，这里直接显式声明，避免告警）。去重真正依赖 IngestionCache。
        common_config = {
            "docstore": self.redis_document_store,
            "cache": self.ingestion_cache,
            "docstore_strategy": DocstoreStrategy.DUPLICATES_ONLY,
            # 刻意不传 vector_store -> 全部节点只写 docstore
        }

        # 文本类：直接用 chunk_sizes，由 HierarchicalNodeParser 在每层用 SentenceSplitter 切分，
        # 形成「大块 -> 小块」父子层级节点。
        self.docstore_text_pipeline = IngestionPipeline(
            transformations=[
                HierarchicalNodeParser.from_defaults(
                    chunk_sizes=Config.HIERARCHICAL_CHUNK_SIZES,
                ),
                TitleExtractor(nodes=Config.TITLE_EXTRACTOR_NODES),
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
        self.docstore_markdown_pipeline = IngestionPipeline(
            transformations=[
                HierarchicalNodeParser.from_defaults(
                    node_parser_ids=node_parser_ids,
                    node_parser_map=node_parser_map,
                    include_metadata=True,
                    include_prev_next_rel=True,
                ),
                TitleExtractor(nodes=Config.TITLE_EXTRACTOR_NODES),
            ],
            **common_config,
        )

        # 兼容基类属性名（ingest_document 已被本类覆写，此处仅为稳健）
        self.text_pipeline = self.docstore_text_pipeline
        self.markdown_pipeline = self.docstore_markdown_pipeline


    def ingest_document(self, doc):
        """摄取文档：解析一次 -> 全部节点写 docstore -> 仅 leaf 写 Milvus。

        相比基类：把「全部节点写 Milvus」改为「仅叶子节点写 Milvus」，父节点只留 docstore。
        去重信号保留：docstore 管道命中 IngestionCache 时返回空，即判定为重复上传。
        """
        try:
            if not doc:
                logger.info("文档摄取，入参文档为空")
                return "error", "入参文档为空", None

            file_name = doc.metadata.get("file_name")
            file_type = doc.metadata.get("file_type")
            if not file_type and file_name:
                file_type = os.path.splitext(os.path.basename(file_name))[1]

            if not file_type:
                logger.info(f"无法确定文件类型,file_name:{file_name}")
                return "error", "未知的文件类型", None

            logger.info(f"开始处理文档:{file_name}")

            # 0) 覆盖上传 / 内容变化保护：
            #    若同一 doc_id 的文档已存在于 docstore，且内容（hash）已变化，则先清理其旧
            #    索引（Milvus 向量 + docstore 父/子节点 + ingestion cache），再走正常摄取。
            #    这样「只改了文档一小部分后重传」不会再产生重复索引。
            #    若内容完全一致（hash 相同），则不清理，交给下方 IngestionCache 命中跳过，
            #    从而保留缓存去重能力（避免每次重传都清空缓存、全量重处理）。
            doc_id = doc.doc_id
            if doc_id:
                _changed = False
                try:
                    _existing = self.redis_document_store.get_document(doc_id)
                    if _existing is not None and _existing.hash != doc.hash:
                        _changed = True
                except Exception:
                    logger.exception(
                        f"判断文档内容是否变化失败，按「已变化」处理以规避重复索引：{doc_id}"
                    )
                    _changed = True
                if _changed:
                    logger.info(f"检测到文档内容变化（覆盖上传），先清理旧索引再重新摄取：{doc_id}")
                    self.delete_document(doc_id)

            # 1) 唯一一次解析：全部节点（父+子）写 docstore，并做缓存去重
            if file_type in ('.markdown', '.md'):
                all_nodes = self.docstore_markdown_pipeline.run(
                    documents=[doc], show_progress=True
                )
            else:
                all_nodes = self.docstore_text_pipeline.run(
                    documents=[doc], show_progress=True
                )

            # 命中缓存（文档重复上传）-> 管道返回空，跳过重传
            if not all_nodes:
                result = "文档重复上传，无需更新索引"
                logger.info(f"{result},file_name:{file_name}")
                return "success", result, all_nodes

            # IngestionPipeline 在「配置了 docstore 但没有 vector_store」时，
            # run() 内部 _update_docstore 收到的是 nodes_to_run（即转换前的原始 Document），
            # 而不是 HierarchicalNodeParser 产出的「父-子」层级节点。也就是说，管道只把
            # 原始 Document 写进了 docstore，父/子节点本身并未入库。
            # 而 AutoMergingRetriever 在检索时会按 leaf.parent_node / next_node 去 docstore
            # 取父节点，取不到就会抛 "doc_id ... not found"。
            # 这里把转换后的「全部（父+子）节点」显式写回 docstore，覆盖管道写入的原始
            # 文档，使父节点可被 docstore 检索并参与合并。
            self.redis_document_store.add_documents(all_nodes)

            # 2) 从同一批节点里筛出 leaf（无 CHILD 关系 = 叶子），只把叶子写向量库
            #    使用 llama_index 官方 helper，等价手写过滤，更地道
            leaf_nodes = get_leaf_nodes(all_nodes)
            for n in leaf_nodes:
                # 单独为叶子做 embedding（父节点不嵌入，省一次嵌入调用）
                n.embedding = Settings.embed_model.get_text_embedding(
                    n.get_content(metadata_mode=MetadataMode.NONE)
                )
            self.milvus_vector_store.add(leaf_nodes)

            logger.info(
                f"文件[{file_name}]处理完成, 生成 {len(all_nodes)} 个节点(父+子), "
                f"其中 {len(leaf_nodes)} 个叶子已写入向量库"
            )

            result = (
                f"文档摄取成功，生成了 {len(all_nodes)} 个节点"
                f"（仅 {len(leaf_nodes)} 个叶子写入向量库，父节点只留 docstore）"
            )
            return "success", result, all_nodes
        except Exception as e:
            logger.exception('文档摄取异常')
            error_msg = f"文档摄取失败: {str(e)}"
            return "error", error_msg, ""


if __name__ == '__main__':
    file_path = '../test_data/MongoDB-test.md'
    docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
    ingestion = HierarchicalDocumentIngestionPipeline()
    result = ingestion.ingest_document(docs[0])
    print(f"层次解析文档摄取结果:{result}")
