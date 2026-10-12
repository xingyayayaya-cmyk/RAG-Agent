#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: aftersale_agent.py
描述: 售后子 Agent —— 只能访问售后政策库（source="policy"），负责退换货/保修/运费政策。
"""
from ecom_agent.base_agent import RAGAgent
from ecom_agent.prompts import AFTERSALE_PROMPT


class AftersaleAgent(RAGAgent):
    def __init__(self, llm, rag_system):
        super().__init__("售后Agent", llm, rag_system, source="policy", prompt_template=AFTERSALE_PROMPT)
