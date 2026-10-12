#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: test_escalation.py
描述: 升级 Agent 单元测试（纯逻辑，不依赖网络/LLM）。
     覆盖：强情绪关键词转人工、大额订单转人工、普通咨询不升级。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecom_agent.escalation_agent import EscalationAgent


class FakeStore:
    def __init__(self, orders=None):
        self.orders = orders or []

    def lookup_order(self, order_id=None, phone=None, sku_id=None, product_keyword=None, limit=20):
        return self.orders


def test_emotion_escalate():
    agent = EscalationAgent(FakeStore())
    escalated, reason = agent.detect("我要投诉！太垃圾了")
    assert escalated is True
    assert reason == "强情绪投诉"


def test_high_value_order_escalate():
    store = FakeStore([{"platform": "淘宝", "amount": 6000, "status": "已发货"}])
    agent = EscalationAgent(store)
    escalated, reason = agent.detect("订单 ORD2026072800001 退款")
    assert escalated is True
    assert "大额订单" in reason


def test_low_value_order_not_escalate():
    store = FakeStore([{"platform": "淘宝", "amount": 100, "status": "已发货"}])
    agent = EscalationAgent(store)
    escalated, reason = agent.detect("订单 ORD2026072800002 物流到哪了")
    assert escalated is False
    assert reason is None


def test_normal_consult_not_escalate():
    agent = EscalationAgent(FakeStore())
    escalated, reason = agent.detect("这个手机什么时候发货")
    assert escalated is False
    assert reason is None


def test_build_message_contains_ticket_and_phone():
    agent = EscalationAgent(FakeStore())
    msg = agent.build_message("强情绪投诉")
    assert "工单号" in msg
    assert "客服热线" in msg


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n共 {len(tests)} 个用例全部通过")
