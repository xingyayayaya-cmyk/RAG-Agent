#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: presale_agent.py
描述: 售前子 Agent —— 只能访问商品库（source="product"），负责商品咨询/对比/推荐。
"""
from ecom_agent.base_agent import RAGAgent
from ecom_agent.prompts import PRESALE_PROMPT


class PresaleAgent(RAGAgent):
    def __init__(self, llm, rag_system):
        super().__init__("售前Agent", llm, rag_system, source="product", prompt_template=PRESALE_PROMPT)
