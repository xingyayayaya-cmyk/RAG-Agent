#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: vector_store.py
作者: ZZS
项目: 3_代码
创建日期: 2026/8/24
描述: 
"""
# 以项目根目录优先级最高。
# todo 尽量放最上面。
import os
import sys
local_path = os.path.abspath(os.path.dirname(__file__))
rag_qa_path = os.path.abspath(os.path.dirname(local_path))
sys.path.insert(0, rag_qa_path)
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)

# 导包
# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
from milvus_model.hybrid import BGEM3EmbeddingFunction
# milvus相关的客户端、搜索函数、融合函数
from pymilvus import MilvusClient,DataType,AnnSearchRequest,WeightedRanker
from langchain_core.documents import Document
# 构建ranker
from sentence_transformers import CrossEncoder
# 生成唯一的 id
import hashlib
from base.config import config
from base.logger import logger


class VectorStore:
    """
    初始化向量库、reranker模型、embedding模型
    存储向量。
    混合检索和重排序
    """
    def __init__(self):

        # 初始化向量库：本类只服务远程 Milvus；本地降级见 create_vector_store() → LocalVectorStore
        self.client = MilvusClient(uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}", db_name=config.MILVUS_DATABASE_NAME)
        # reranker模型
        rerank_model_path = os.path.join(config.MODELS_DIR, "bge-reranker-large")
        self.reranker = CrossEncoder(rerank_model_path, device="cpu")
        # embedding模型
        bge_m3_model_path = os.path.join(config.MODELS_DIR, "bge-m3")
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path = bge_m3_model_path,
            use_fp16 = False,   # 不使用半精度的模型参数，会损失比较小的精度。这里默认为False，直接使用全精度的模型参数。
            device = "cpu"
        )
        # 稠密向量维度
        self.dense_dim = self.embedding_function.dim["dense"]
        self.collection_name = config.MILVUS_COLLECTION_NAME

        # todo 最后初始化创建数据库。否则拿不到集合和dense_dim参数。
        # 初始化mivlus数据库
        self._create_or_load_collection()


    # 定义私有方法，创建或加载 Milvus 集合
    def _create_or_load_collection(self):
        # 检查指定集合是否已存在
        # todo 这两种方式都可以
        if not self.client.has_collection(self.collection_name):
        # if not self.collection_name in self.client.list_collections():
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)


    def add_documents(self, documents):
        """
        将子块内容取出并向量化
        循环每一个子块文档，取出稀疏向量、稠密向量、生成子块内容hash值
        将所有数据加上元数据一起放到data列表中。
        传给milvus客户端存入到向量库中
        :param documents:
        :return:
        """
        # 1. 将子块内容取出，本身是documents对象
        texts = [doc.page_content for doc in documents]
        # 2. 向量化
        embeddings = self.embedding_function(texts)
        # 3. 循环每一个子块文档，取出稀疏向量、稠密向量、生成子块内容hash值
        data = []
        for i, doc in enumerate(documents):
            doc_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            # 取稀疏向量
            row = embeddings['sparse'][[i],:]
            indices = row.indices
            values = row.data
            # {0:0.1,1:0.2,,,,}
            sparse_vector = {}
            for idx, value in zip(indices, values):
                sparse_vector[idx] = value
            # 4.将所有数据加上元数据一起放到data列表中。
            data.append({
                "id": doc_hash,
                "text": doc.page_content,
                "dense_vector": embeddings["dense"][i],
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown")
            })

        # 5. 传给milvus客户端存入到向量库中
        self.client.insert(collection_name=self.collection_name, data=data)
        logger.info(f"已向集合 {self.collection_name} 添加 {len(data)} 条数据")

    # 定义方法，执行混合检索并重排序
    # todo conf -> config
    def hybrid_search_with_rerank(self, query, k=config.RETRIEVAL_K, source_filter=None):
        # 使用 BGE-M3 嵌入函数生成查询的嵌入
        query_embeddings = self.embedding_function([str(query)])
        # 获取查询的稠密向量
        dense_query_vector = query_embeddings["dense"][0]
        # 初始化查询的稀疏向量字典
        sparse_query_vector = {}
        # todo 获取稀疏向量的第0行数据
        row = query_embeddings["sparse"][[0], :]
        # 获取稀疏向量的非零值索引
        indices = row.indices
        # 获取稀疏向量的非零值
        values = row.data
        # 将索引和值配对，填充稀疏向量字典
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value

        # 初始化过滤表达式，默认不过滤
        filter_expr = f"source == '{source_filter}'" if source_filter else ""
        # 创建稠密向量搜索请求
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr
        )
        # 创建稀疏向量搜索请求
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr
        )

        # 创建加权排序器，稀疏向量权重 0.7，稠密向量权重 1.0
        ranker = WeightedRanker(0.7, 1.0)
        # 执行混合搜索，返回 Top-K 结果
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]
        )[0]

        # 将搜索结果转换为 Document 对象列表
        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]
        # 从子块中提取去重的父文档
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        # 如果只有1个文档，直接返回跳过重排序
        if len(parent_docs) < 2:
            # todo conf -> config
            return parent_docs[:config.CANDIDATE_M]
            # 如果有父文档，进行重排序
        if parent_docs:
            # 创建查询与文档内容的配对列表
            pairs = [[query, doc.page_content] for doc in parent_docs]
            # 使用 BGE-Reranker 计算每个配对的得分
            scores = self.reranker.predict(pairs)
            # 根据得分从高到低排序文档
            ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        # 如果没有父文档，返回空列表
        else:
            ranked_parent_docs = []

        # 返回前 k 个重排序后的文档
        # todo  conf -> config
        return ranked_parent_docs[:config.CANDIDATE_M]

    # 定义私有方法，从子块中提取去重的父文档
    def _get_unique_parent_docs(self, sub_chunks):
        # 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in sub_chunks:
            # 获取子块的父块内容，默认为子块内容
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 检查父块内容是否非空且未重复
            if parent_content and parent_content not in parent_contents:
                # 创建新的 Document 对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 将父块内容添加到去重集合
                parent_contents.add(parent_content)
        # 返回去重后的父文档列表
        return unique_docs

    # 定义私有方法，从 Milvus 查询结果创建 Document 对象
    def _doc_from_hit(self, hit):
        # 创建并返回 Document 对象，填充内容和元数据
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )



def create_vector_store():
    """按配置返回向量库后端：本地降级走 LocalVectorStore（numpy 暴力检索 + 持久化），远程走 Milvus。"""
    if config.LOCAL_MODE:
        from rag_qa.core.local_vector_store import LocalVectorStore
        logger.info("使用本地向量库（LocalVectorStore）")
        return LocalVectorStore()
    logger.info("使用远程 Milvus 向量库")
    return VectorStore()


if __name__ == '__main__':
    vector_store = VectorStore()
    print(vector_store.client)
    print(vector_store.client.list_databases())
    # <pymilvus.milvus_client.milvus_client.MilvusClient object at 0x000001B92B2E60B0>
    # ['itcast', 'milvus_demo1', 'milvus_demo', 'test', 'default', 'cd_db_test', 'cd_test']
    # ~~~~~~~~~~~~~~~~~~~~
    # 向量存储测试
    # from document_processor import process_documents
    # process_documents 是加载对应位置的文档，然后进行父子块切分，返回所有子块的document对象列表，包含一些父块的元数据
    # child_chunks = process_documents(r"D:\北京33期共享\3_代码\integrated_qa_system\rag_qa\data\ai_data")
    # 块的document对象列表向量化，存储到向量库中
    # vector_store.add_documents(documents=child_chunks)
    # ~~~~~~~~~~~~~~~~~~~~
    # 检索测试
    query = "如何使用Milvus进行向量检索？"
    results = vector_store.hybrid_search_with_rerank(query)
    print( results)