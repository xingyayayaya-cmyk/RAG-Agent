#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: router.py
描述: 意图路由器 —— 规则优先（订单号正则 → 关键词首命中） + LLM 兜底分类。
     返回 intent ∈ {presale, order, aftersale, chitchat}。
"""
from base.llm_utils import llm_string
from ecom_agent.intent_rules import (
    ORDER_ID_RE, PHONE_RE, SKU_RE, INTENT_KEYWORDS, INTENT_PRIORITY,
)
from ecom_agent.prompts import INTENT_CLASSIFY_PROMPT

# LLM 返回的中文类别名 → 内部 intent key
_LABEL_TO_INTENT = {
    "售前": "presale",
    "订单": "order",
    "售后": "aftersale",
    "闲聊": "chitchat",
}


class IntentRouter:
    """规则优先 + LLM 兜底。llm 为流式生成器（call_dashscope）。"""

    def __init__(self, llm):
        self.llm = llm

    def _rule_route(self, query):
        # 1) 明确实体覆盖：订单号/手机号/SKU 任一命中 → 订单
        if ORDER_ID_RE.search(query) or PHONE_RE.search(query) or SKU_RE.search(query):
            return "order"
        # 2) 关键词首命中（售后 > 订单 > 售前）
        for intent in INTENT_PRIORITY:
            if any(kw in query for kw in INTENT_KEYWORDS[intent]):
                return intent
        return None

    def _llm_route(self, query):
        try:
            label = llm_string(self.llm, INTENT_CLASSIFY_PROMPT.format(question=query)).strip()
        except Exception:
            return None
        # 取返回文本中出现的第一个已知类别名
        for cn, key in _LABEL_TO_INTENT.items():
            if cn in label:
                return key
        return None

    def route(self, query):
        rule = self._rule_route(query)
        if rule:
            return rule
        return self._llm_route(query) or "chitchat"
