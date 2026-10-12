#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: test_llm_e2e.py
描述: 真实 LLM 端到端验证（需代理）。覆盖：售前/售后/订单/闲聊/升级 五类路径。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from new_main import IntegratedQASystem
from base.store import create_store


def run(qa, query, session_id="llm-e2e"):
    text, complete = "", False
    for token, is_complete in qa.query(query, session_id=session_id):
        if token:
            text += token
        complete = complete or is_complete
    return text, complete


def main():
    qa = IntegratedQASystem()
    store = create_store()
    order = store.lookup_order(limit=1)[0]
    store.close()

    cases = [
        ("售前", "iPhone 和华为哪个续航好"),
        ("售后", "七天无理由退货政策是什么"),
        ("订单", f"查订单 {order['order_id']}"),
        ("闲聊", "给我讲个笑话"),
        ("升级", "我要投诉！太垃圾了"),
    ]
    for label, q in cases:
        text, complete = run(qa, q)
        preview = text.replace("\n", " ")[:90]
        print(f"[{label}] Q: {q}")
        print(f"    A: {preview}... (complete={complete})")
        print()
    qa.close()


if __name__ == "__main__":
    main()
