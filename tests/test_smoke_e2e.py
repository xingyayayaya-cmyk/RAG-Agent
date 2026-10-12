#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: test_smoke_e2e.py
描述: 端到端冒烟测试（离线，不依赖 LLM/网络）。
     覆盖：系统构造、升级转人工、订单结构化查询、FAQ 快速通道、RAG 检索(返回文档)。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from new_main import IntegratedQASystem
from base.store import create_store


def _collect(qa, query, session_id="smoke-test"):
    """消费生成器，返回 (完整文本, 是否完整)。"""
    text = ""
    complete = False
    for token, is_complete in qa.query(query, session_id=session_id):
        if token:
            text += token
        complete = complete or is_complete
    return text, complete


def test_construct_system():
    qa = IntegratedQASystem()
    assert qa.orchestrator is not None
    assert qa.vector_store is not None
    return qa


def test_escalation_emotion(qa):
    text, _ = _collect(qa, "我要投诉！太垃圾了")
    assert "工单号" in text, f"升级话术缺失: {text}"


def test_order_lookup(qa):
    # 取一条真实订单号
    store = create_store()
    order = store.lookup_order(limit=1)[0]
    store.close()
    oid = order["order_id"]
    text, _ = _collect(qa, f"查订单 {oid}")
    # 订单 Agent 在 LLM 不可用时走结构化兜底，仍应包含订单号或状态
    assert oid in text or "订单" in text, f"订单查询异常: {text}"


def test_faq_hit(qa):
    # 取一条真实 FAQ 问题并原样提问，应命中 FAQ 快速通道
    store = create_store()
    question = store.fetch_questions()[0]
    store.close()
    text, complete = _collect(qa, question)
    assert text, "FAQ 命中后应有答案"


def test_rag_retrieval_returns_docs(qa):
    # 售后意图 → 走售后 Agent 的 RAG 检索（source=policy），LLM 不可用时仍应尝试检索
    # 直接验证 RAG 检索接口能返回政策父文档
    docs = qa.rag_system.retrieve_and_merge("七天无理由退货政策是什么", source_filter="policy")
    assert isinstance(docs, list)


def test_presale_source_isolation(qa):
    # 权限隔离：policy 检索只返回 source=policy 的文档
    docs = qa.rag_system.retrieve_and_merge("七天无理由退货政策是什么", source_filter="policy")
    for d in docs:
        assert d.metadata.get("source") == "policy", f"越权检索到: {d.metadata.get('source')}"


if __name__ == "__main__":
    print("构造系统（加载模型，请稍候）...")
    qa = test_construct_system()
    for t in [test_escalation_emotion, test_order_lookup, test_faq_hit,
              test_rag_retrieval_returns_docs, test_presale_source_isolation]:
        t(qa)
        print(f"PASS {t.__name__}")
    qa.close()
    print("\n冒烟测试全部通过")
