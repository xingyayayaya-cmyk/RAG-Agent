#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: test_comparison_e2e.py
描述: 组合对比问答端到端验证（需代理）。重点验证「跨品牌对比」在修复数据后的真实效果。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from new_main import IntegratedQASystem


def run(qa, query, session_id="compare-e2e"):
    text, complete = "", False
    for token, is_complete in qa.query(query, session_id=session_id):
        if token:
            text += token
        complete = complete or is_complete
    return text, complete


def main():
    qa = IntegratedQASystem()

    cases = [
        ("对比", "iPhone 和华为哪个续航好"),
        ("对比", "小米和苹果手机哪个好"),
        ("售后", "七天无理由退货政策是什么"),
        ("升级", "我要投诉！太垃圾了"),
    ]
    for label, q in cases:
        text, complete = run(qa, q)
        print(f"\n===== [{label}] {q} (complete={complete}) =====")
        print(text[:400])

    qa.close()


if __name__ == "__main__":
    main()
