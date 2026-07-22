from dataclasses import dataclass
from enum import Enum

from pymilvus import DataType
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction

from logger.logging import setup_logging

logger = setup_logging()

@dataclass
class MilvusDocument:
    doc_name: str          # 文档名称
    doc_path_name: str     # 文档路径（含名字）
    doc_type: str          # 文档类型
    doc_md5:str            # 文档MD5
    doc_length: int        # 文档字节数
    content: str           # 文档分段内容
    content_vector: list   # 分段内容向量
    embedding_model: str   # embedding模型名称

class MilvusDocumentField(str, Enum):
    ID = "id"             # 主键ID
    DOC_NAME = "doc_name" # 文档名称
    DOC_PATH_NAME = "file_path" # 文档路径（含名字）
    DOC_TYPE = "doc_type" # 文档类型
    DOC_MD5 = "doc_md5"   # 文档MD5
    DOC_LENGTH = "doc_length" # 文档字节数
    CONTENT = "text"      # 文档分段内容（Milvus 文本字段名；必须等于节点 node.dict() 的键 "text"，
                           # 因为 MilvusVectorStore 内部用 node.dict()[text_key] 取文本，且 BM25 也依赖此字段名）
    CONTENT_VECTOR = "content_vector"   # 分段内容向量（dense，embedding模型生成）
    CONTENT_SPARSE = "content_sparse"   # 分段内容稀疏向量（BM25 Function服务端自动生成）
    EMBEDDING_MODEL = "embedding_model" # embedding模型名称


# 说明：不再显式声明业务标量字段（doc_name / doc_path_name / doc_type / doc_md5 /
# doc_length / embedding_model）。原因：
#   1) 新版本 MilvusVectorStore 建 schema 时 enable_dynamic_field=True，
#      节点的所有 metadata 会自动存进动态 JSON 字段，信息不丢失；
#   2) 当前 ingestion pipeline 并未给节点设置这些 metadata key，显式声明成
#      非 nullable 字段会导致插入时报 "Insert missed an field" 错误。
# 若后续需要按这些字段做高效过滤/索引，请在节点 metadata 中填充对应 key，
# 再在此处声明 scalar_field_names / scalar_field_types 即可。
SCALAR_FIELD_NAMES = [
    # MilvusDocumentField.DOC_NAME.value,
    MilvusDocumentField.DOC_PATH_NAME.value,
    # MilvusDocumentField.DOC_TYPE.value,
    # MilvusDocumentField.DOC_MD5.value,
    # MilvusDocumentField.DOC_LENGTH.value,
    # MilvusDocumentField.EMBEDDING_MODEL.value,
]
SCALAR_FIELD_TYPES = [
    # DataType.VARCHAR,   # doc_name
    DataType.VARCHAR,   # doc_path_name
    # DataType.VARCHAR,   # doc_type
    # DataType.VARCHAR,   # doc_md5
    # DataType.INT64,     # doc_length
    # DataType.VARCHAR,   # embedding_model
]

def build_bm25_function() -> BM25BuiltInFunction:
    """BM25 内置函数：服务端自动对 content 分词，生成稀疏向量到 content_sparse。"""
    return BM25BuiltInFunction(
        input_field_names=[MilvusDocumentField.CONTENT.value],
        output_field_names=[MilvusDocumentField.CONTENT_SPARSE.value],
        analyzer_params={"tokenizer": "jieba"},  # 中文分词
    )



