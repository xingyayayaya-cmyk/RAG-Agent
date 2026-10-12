#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: test_router.py
描述: 意图路由单元测试（纯逻辑，不依赖网络/LLM）。
     覆盖：规则优先（实体覆盖 / 关键词首命中）+ LLM 兜底分类。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecom_agent.router import IntentRouter


def _dummy_llm(prompt):
    # 兜底 LLM：无论什么问题都返回「闲聊」，用于验证规则层已覆盖的场景
    return "闲聊"


def test_rule_entity_override_order():
    router = IntentRouter(_dummy_llm)
    assert router.route("查订单 ORD2026072800001 发货没") == "order"


def test_rule_keyword_presale():
    router = IntentRouter(_dummy_llm)
    assert router.route("iPhone 和华为哪个续航好") == "presale"


def test_rule_keyword_aftersale():
    router = IntentRouter(_dummy_llm)
    assert router.route("这个能 7 天无理由退吗") == "aftersale"


def test_rule_priority_aftersale_over_order():
    # 同时含「订单」（订单意图）与「退货」（售后意图）时，售后优先
    router = IntentRouter(_dummy_llm)
    assert router.route("我的订单要退货") == "aftersale"


def test_llm_fallback_chitchat():
    router = IntentRouter(_dummy_llm)
    # 无任何规则命中 → 走 LLM 兜底（dummy 返回 闲聊）
    assert router.route("给我讲个笑话") == "chitchat"


def test_llm_fallback_unknown():
    # dummy LLM 返回未知类别 → 兜底 chitchat
    def unknown_llm(prompt):
        return "不知道"
    router = IntentRouter(unknown_llm)
    assert router.route("随便说点什么") == "chitchat"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n共 {len(tests)} 个用例全部通过")
