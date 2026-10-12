#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: local_vector_store.py
描述: 本地降级向量库 —— 无 Milvus/网络依赖，纯 numpy 暴力检索 + 磁盘持久化（pickle）。
     实现与 VectorStore 相同的接口（add_documents / hybrid_search_with_rerank），
     复用同一套 BGE-M3 嵌入（稠密+稀疏）与 bge-reranker 重排，保证本地离线可跑、可持久化。

     为什么不是 Milvus Lite：milvus-lite 需额外安装且本机无网；faiss/numpy 已随环境就位。
     混合打分沿用原 WeightedRanker(0.7 稀疏 + 1.0 稠密)，source_filter 做权限隔离。
"""
import os
import sys
import pickle

# 保证 base / rag_qa 可导入（与 vector_store.py 一致的路径处理）
local_path = os.path.abspath(os.path.dirname(__file__))
rag_qa_path = os.path.abspath(os.path.dirname(local_path))
sys.path.insert(0, rag_qa_path)
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)

import numpy as np
from milvus_model.hybrid import BGEM3EmbeddingFunction
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from base.config import config
from base.logger import logger


class LocalVectorStore:
    """本地暴力检索向量库：稠密+稀疏混合 + 父块去重 + 重排，接口对齐 VectorStore。"""

    def __init__(self):
        # reranker 模型（与远程一致）
        rerank_model_path = os.path.join(config.MODELS_DIR, "bge-reranker-large")
        self.reranker = CrossEncoder(rerank_model_path, device="cpu")
        # embedding 模型（BGE-M3：稠密 + 稀疏）
        bge_m3_model_path = os.path.join(config.MODELS_DIR, "bge-m3")
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=bge_m3_model_path, use_fp16=False, device="cpu"
        )
        self.dense_dim = self.embedding_function.dim["dense"]
        self.collection_name = config.MILVUS_COLLECTION_NAME

        # 持久化文件（与 milvus_lite.db 同目录）
        self._data_dir = os.path.dirname(config.MILVUS_LOCAL_DB)
        self._meta_path = os.path.join(self._data_dir, "local_vec_meta.pkl")

        # 内存态：文档 / 稠密向量(已L2归一化) / 稀疏向量 dict
        self._docs = []
        self._dense = np.empty((0, self.dense_dim), dtype="float32")
        self._sparse = []

        self._load()

    def _load(self):
        if os.path.exists(self._meta_path):
            try:
                with open(self._meta_path, "rb") as f:
                    data = pickle.load(f)
                self._docs = data.get("docs", [])
                self._dense = np.asarray(data.get("dense", []), dtype="float32").reshape(-1, self.dense_dim)
                self._sparse = data.get("sparse", [])
                logger.info(f"本地向量库已加载：{len(self._docs)} 个子块")
            except Exception as e:
                logger.error(f"本地向量库加载失败: {e}")

    def _save(self):
        try:
            os.makedirs(self._data_dir, exist_ok=True)
            with open(self._meta_path, "wb") as f:
                pickle.dump({"docs": self._docs, "dense": self._dense, "sparse": self._sparse}, f)
        except Exception as e:
            logger.error(f"本地向量库保存失败: {e}")

    def add_documents(self, documents):
        """向量化并追加子块，持久化到磁盘。"""
        if not documents:
            return
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_function(texts)

        # 稠密向量：stack + L2 归一化（IP == 余弦相似度）
        dense = np.vstack(embeddings["dense"]).astype("float32")
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dense = dense / norms

        # 稀疏向量：逐条转 dict[int, float]
        sparse_mat = embeddings["sparse"]
        for i in range(len(documents)):
            row = sparse_mat[[i], :]
            sparse_vec = {int(idx): float(val) for idx, val in zip(row.indices, row.data)}
            self._sparse.append(sparse_vec)

        self._docs.extend(documents)
        self._dense = np.vstack([self._dense, dense]) if self._dense.size else dense
        self._save()
        logger.info(f"本地向量库已添加 {len(documents)} 个子块（累计 {len(self._docs)}）")

    def _get_unique_parent_docs(self, sub_chunks):
        parent_contents = set()
        unique_docs = []
        for chunk in sub_chunks:
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            if parent_content and parent_content not in parent_contents:
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                parent_contents.add(parent_content)
        return unique_docs

    def hybrid_search_with_rerank(self, query, k=config.RETRIEVAL_K, source_filter=None):
        """稠密+稀疏混合检索 → 父块去重 → 重排。返回父文档列表（Document）。"""
        if len(self._docs) == 0:
            return []

        q_emb = self.embedding_function([str(query)])

        # 稠密打分（余弦）
        dense_q = np.asarray(q_emb["dense"], dtype="float32").reshape(-1)
        dense_q = dense_q / (np.linalg.norm(dense_q) or 1.0)
        dense_scores = self._dense @ dense_q  # (N,)

        # 稀疏打分（查询稀疏向量 → 对每文档做点积）
        q_row = q_emb["sparse"][[0], :]
        q_sparse = {int(idx): float(val) for idx, val in zip(q_row.indices, q_row.data)}
        sparse_scores = np.zeros(len(self._docs), dtype="float32")
        if q_sparse:
            for i, svec in enumerate(self._sparse):
                s = 0.0
                for idx, val in svec.items():
                    w = q_sparse.get(idx)
                    if w:
                        s += val * w
                sparse_scores[i] = s

        # 混合加权（与原 WeightedRanker(0.7 稀疏, 1.0 稠密) 一致）
        combined = 0.7 * sparse_scores + 1.0 * dense_scores

        # source 过滤（权限隔离）
        if source_filter:
            mask = np.array([d.metadata.get("source") == source_filter for d in self._docs])
            combined = np.where(mask, combined, -np.inf)

        top_k = min(int(k), len(self._docs))
        top_indices = np.argsort(combined)[::-1][:top_k]
        sub_chunks = [self._docs[i] for i in top_indices if combined[i] > -np.inf]

        parent_docs = self._get_unique_parent_docs(sub_chunks)
        if len(parent_docs) < 2:
            return parent_docs[:config.CANDIDATE_M]
        if parent_docs:
            pairs = [[query, doc.page_content] for doc in parent_docs]
            scores = self.reranker.predict(pairs)
            ranked = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        else:
            ranked = []
        return ranked[:config.CANDIDATE_M]
