#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: escalation_agent.py
描述: 升级 Agent —— 强情绪投诉 / 大额订单 转人工兜底。
     在意图路由之前独立执行，命中即返回转人工话术，不再进入子 Agent。
"""
import time
from base.config import config
from ecom_agent.intent_rules import EMOTION_KEYWORDS, ORDER_ID_RE, PHONE_RE


class EscalationAgent:
    """强情绪/大额订单 → 转人工。独立于意图路由，不参与业务问答。"""

    def __init__(self, store):
        self.store = store

    def detect(self, query, history=None):
        """返回 (是否升级, 原因)。"""
        # 1) 强情绪关键词
        if any(kw in query for kw in EMOTION_KEYWORDS):
            return True, "强情绪投诉"

        # 2) 大额订单：query 携带订单号/手机号，且查到金额 ≥ 平台阈值
        order_id = ORDER_ID_RE.search(query)
        phone = PHONE_RE.search(query)
        if not order_id and not phone:
            return False, None

        orders = self.store.lookup_order(
            order_id=order_id.group(0) if order_id else None,
            phone=phone.group(0) if phone else None,
        )
        for o in orders:
            platform = o.get("platform", "")
            threshold = config.ESCALATION_THRESHOLDS.get(platform)
            amount = float(o.get("amount") or 0)
            if threshold and amount >= threshold:
                return True, f"大额订单（{platform} {amount} 元 ≥ {threshold} 元）"

        return False, None

    def build_message(self, reason):
        ticket = f"T{int(time.time())}"
        return (
            f"非常抱歉给您带来不好的体验！您的问题已升级为人工优先处理。\n"
            f"【工单号】{ticket}\n"
            f"【升级原因】{reason}\n"
            f"专属客服将在工作时间内尽快与您联系，您也可直接拨打客服热线 {config.CUSTOMER_SERVICE_PHONE}。"
        )
