#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: order_agent.py
描述: 订单子 Agent —— 只能访问订单库（Store.lookup_order 结构化查询），负责订单状态/物流/退款进度。
"""
import json
from ecom_agent.base_agent import BaseAgent
from ecom_agent.prompts import ORDER_PROMPT
from ecom_agent.intent_rules import ORDER_ID_RE, PHONE_RE, SKU_RE


class OrderAgent(BaseAgent):
    def __init__(self, llm, store):
        super().__init__("订单Agent", llm)
        self.store = store
        self.prompt_template = ORDER_PROMPT

    def _extract_params(self, query):
        order_id = ORDER_ID_RE.search(query)
        phone = PHONE_RE.search(query)
        sku = SKU_RE.search(query)
        return (
            order_id.group(0) if order_id else None,
            phone.group(0) if phone else None,
            sku.group(0) if sku else None,
        )

    def _format_orders(self, orders):
        lines = [f"为您查到 {len(orders)} 笔订单："]
        for o in orders:
            lines.append(
                f"订单号 {o['order_id']}（{o['platform']}）：{o['product_name']} x{o['quantity']}，"
                f"金额 {o['amount']} 元，状态【{o['status']}】"
            )
        return "\n".join(lines)

    def handle(self, query, history=None):
        order_id, phone, sku = self._extract_params(query)
        # 只通过自己的工具（lookup_order）访问订单数据，不碰商品/政策
        orders = self.store.lookup_order(order_id=order_id, phone=phone, sku_id=sku)
        if not orders:
            yield "抱歉，没有查询到相关订单。请提供订单号（ORD 开头）或收货手机号，我会为您查询。"
            return

        context = json.dumps(orders, ensure_ascii=False)
        prompt = self.prompt_template.format(
            context=context, history=self._history_context(history), question=query
        )

        collected = ""
        for token in self.llm(prompt):
            if token:
                collected += token
                yield token
        if not collected:
            # LLM 不可用时的结构化兜底，保证订单查询在无 LLM 下也能回答
            yield self._format_orders(orders)
