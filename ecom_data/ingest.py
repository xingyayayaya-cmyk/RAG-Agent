#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: ingest.py
描述: 电商数据灌库脚本。读取 C:\\Users\\Administrator\\Desktop\\znkf 下的合成数据：
      - 商品 10008 / 售后政策 160+5  → Milvus（父子切分，source=product/policy）
      - FAQ 1808                      → Store（ecom_faq）
      - 订单 3000                     → Store（orders）

用法（EduRAG 环境）：
    python ecom_data/ingest.py                 # 全量灌入
    python ecom_data/ingest.py --limit 500     # 只灌 500 个商品（快速验证）
    ECOM_DATA_DIR=xxx python ecom_data/ingest.py
"""
import os
import sys
import json
import argparse
from datetime import datetime

# 项目根目录加入 sys.path，保证 base / rag_qa 可导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from base.config import config
from base.logger import logger
from base.store import create_store
from rag_qa.core.vector_store import create_vector_store
from rag_qa.edu_text_spliter.edu_chinese_recursive_text_splitter import ChineseRecursiveTextSplitter
from langchain_core.documents import Document

# 合成数据根目录（可用环境变量覆盖）
# 数据目录优先级：环境变量 ECOM_DATA_DIR > 仓库内 dataset/ > 旧版桌面路径（向后兼容）
_DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
if not os.path.isdir(_DEFAULT_DATA_DIR):
    _DEFAULT_DATA_DIR = r"C:\Users\Administrator\Desktop\znkf"
DATA_DIR = os.getenv("ECOM_DATA_DIR", _DEFAULT_DATA_DIR)

PRODUCTS_PATH = os.path.join(DATA_DIR, "01_商品库_products", "products.jsonl")
ORDERS_PATH = os.path.join(DATA_DIR, "02_订单_orders", "orders.jsonl")
POLICY_RULES_PATH = os.path.join(DATA_DIR, "03_售后政策_after_sales_policy", "policy_rules.jsonl")
POLICY_DOCS_DIR = os.path.join(DATA_DIR, "03_售后政策_after_sales_policy", "policy_docs")
FAQ_PATH = os.path.join(DATA_DIR, "04_FAQ问答对_faq", "faq.jsonl")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_child_chunks(parents, source):
    """
    父子切分：parent 用大块，child 用小块，子块携带 parent_id/parent_content/source 元数据。
    :param parents: [(parent_text, extra_metadata_dict), ...]
    :param source:   'product' 或 'policy'
    :return: list[Document]（子块）
    """
    parent_splitter = ChineseRecursiveTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    child_splitter = ChineseRecursiveTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    child_chunks = []
    ts = datetime.now().isoformat()
    for i, (text, meta) in enumerate(parents):
        if not text or not text.strip():
            continue
        parent_doc = Document(page_content=text, metadata=dict(meta))
        parent_docs = parent_splitter.split_documents([parent_doc])
        for j, pdoc in enumerate(parent_docs):
            parent_id = f"{source}_{i}_parent_{j}"
            subs = child_splitter.split_documents([pdoc])
            for k, sub in enumerate(subs):
                sub.metadata["parent_id"] = parent_id
                sub.metadata["parent_content"] = pdoc.page_content
                sub.metadata["source"] = source
                sub.metadata["timestamp"] = ts
                child_chunks.append(sub)
    return child_chunks


def add_in_batches(vector_store, chunks, batch=500):
    for i in range(0, len(chunks), batch):
        vector_store.add_documents(chunks[i:i + batch])
        logger.info(f"已灌入向量 {min(i + batch, len(chunks))}/{len(chunks)}")


def ingest_products(vector_store, limit=0):
    products = load_jsonl(PRODUCTS_PATH)
    if limit and limit < len(products):
        products = products[:limit]
    parents = [(p["text"], {"source": "product", "sku_id": p["sku_id"], "category": p["category"]}) for p in products]
    chunks = build_child_chunks(parents, "product")
    add_in_batches(vector_store, chunks)
    logger.info(f"商品灌库完成：{len(products)} 条 SKU → {len(chunks)} 个子块")


def ingest_policy(vector_store):
    rules = load_jsonl(POLICY_RULES_PATH)
    parents = [
        (r["content"], {"source": "policy", "rule_id": r["rule_id"], "platform": r["platform"], "category": r["category"]})
        for r in rules
    ]
    # 追加各平台 Markdown 政策文档
    for fn in sorted(os.listdir(POLICY_DOCS_DIR)) if os.path.isdir(POLICY_DOCS_DIR) else []:
        if fn.endswith(".md"):
            with open(os.path.join(POLICY_DOCS_DIR, fn), encoding="utf-8") as f:
                parents.append((f.read(), {"source": "policy", "file": fn}))
    chunks = build_child_chunks(parents, "policy")
    add_in_batches(vector_store, chunks)
    logger.info(f"政策灌库完成：{len(rules)} 条规则 + 平台文档 → {len(chunks)} 个子块")


def ingest_faq(store):
    faqs = load_jsonl(FAQ_PATH)
    rows = [(f["category"], f["question"], f["answer"]) for f in faqs]
    store.insert_faq_batch(rows)
    logger.info(f"FAQ 灌库完成：{len(rows)} 条")


def ingest_orders(store):
    orders = load_jsonl(ORDERS_PATH)
    store.insert_order_batch(orders)
    logger.info(f"订单灌库完成：{len(orders)} 条")


def main():
    parser = argparse.ArgumentParser(description="电商数据灌库")
    parser.add_argument("--limit", type=int, default=0, help="只灌入前 N 个商品（0=全部），用于快速验证")
    args = parser.parse_args()

    store = create_store()
    vector_store = create_vector_store()

    ingest_products(vector_store, limit=args.limit)
    ingest_policy(vector_store)
    ingest_faq(store)
    ingest_orders(store)

    store.close()
    logger.info("全部灌库完成")


if __name__ == "__main__":
    main()
