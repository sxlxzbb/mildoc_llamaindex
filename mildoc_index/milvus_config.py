from dataclasses import dataclass
from enum import Enum

from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import DataType, Function, FunctionType

from config.config import Config
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
    DOC_PATH_NAME = "doc_path_name" # 文档路径（含名字）
    DOC_TYPE = "doc_type" # 文档类型
    DOC_MD5 = "doc_md5"   # 文档MD5
    DOC_LENGTH = "doc_length" # 文档字节数
    CONTENT = "content"   # 文档分段内容
    CONTENT_VECTOR = "content_vector"   # 分段内容向量（dense，embedding模型生成）
    CONTENT_SPARSE = "content_sparse"   # 分段内容稀疏向量（BM25 Function服务端自动生成）
    EMBEDDING_MODEL = "embedding_model" # embedding模型名称


def initialize_milvus_schema():
    # 定义schema
    schema = MilvusVectorStore.create_schema(
        auto_id=True,  # 自动生生ID
        enable_dynamic_field=False
    )

    # 添加字段
    # 主键ID字段（自动生成）
    schema.add_field(
        field_name=MilvusDocumentField.ID.value,
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True
    )

    # 文档名称
    schema.add_field(
        field_name=MilvusDocumentField.DOC_NAME.value,
        datatype=DataType.VARCHAR,
        max_length=500
    )

    # 文档路径（含名字）
    schema.add_field(
        field_name=MilvusDocumentField.DOC_PATH_NAME.value,
        datatype=DataType.VARCHAR,
        max_length=1000
    )

    # 文档类型
    schema.add_field(
        field_name=MilvusDocumentField.DOC_TYPE.value,
        datatype=DataType.VARCHAR,
        max_length=50
    )

    # 文档MD5
    schema.add_field(
        field_name=MilvusDocumentField.DOC_MD5.value,
        datatype=DataType.VARCHAR,
        max_length=32
    )

    # 文档字节数
    schema.add_field(
        field_name=MilvusDocumentField.DOC_LENGTH.value,
        datatype=DataType.INT64,
    )

    # 文档内容（开启分词器，供BM25全文检索使用）
    schema.add_field(
        field_name=MilvusDocumentField.CONTENT.value,
        datatype=DataType.VARCHAR,
        max_length=65535,  # 最大长度
        enable_analyzer=True,  # 开启分词，BM25 Function依赖此项
        analyzer_params={"tokenizer": "jieba"}  # 中文分词
    )

    # 内容向量（text-embedding-v4的维度是1536）
    schema.add_field(
        field_name=MilvusDocumentField.CONTENT_VECTOR.value,
        datatype=DataType.FLOAT_VECTOR,
        dim=Config.MILVUS_VECTOR_DIM
    )

    # embedding模型名称
    schema.add_field(
        field_name=MilvusDocumentField.EMBEDDING_MODEL.value,
        datatype=DataType.VARCHAR,
        max_length=100
    )

    # 稀疏向量字段（BM25 Function的输出字段，由服务端自动生成，插入时无需提供）
    schema.add_field(
        field_name=MilvusDocumentField.CONTENT_SPARSE.value,
        datatype=DataType.SPARSE_FLOAT_VECTOR
    )

    # 定义BM25 Function：输入content文本，服务端自动分词并生成稀疏向量到content_sparse
    bm25_function = Function(
        name="content_bm25",
        input_field_names=[MilvusDocumentField.CONTENT.value],
        output_field_names=[MilvusDocumentField.CONTENT_SPARSE.value],
        function_type=FunctionType.BM25
    )
    schema.add_function(bm25_function)

    return schema


