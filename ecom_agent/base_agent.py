#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: base_agent.py
描述: 子 Agent 基类 + 检索型 Agent（RAGAgent）。
     权限隔离：RAGAgent 的 source 在构造时固定，检索只能访问自己的 source_filter。
"""
from base.config import config


class BaseAgent:
    """子 Agent 基类。持有名称与 LLM 流式调用函数，子类实现 handle()。"""

    def __init__(self, name, llm):
        self.name = name
        self.llm = llm  # 流式生成器函数（如 call_dashscope）

    def _history_context(self, history):
        if not history:
            return "无"
        return "\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in history[-5:])

    def handle(self, query, history=None):
        raise NotImplementedError


class RAGAgent(BaseAgent):
    """
    检索型 Agent：走完整检索策略（直接检索 / HyDE / 子查询 / 回溯），
    source_filter 固定为自身数据源，实现权限隔离（无法跨域检索）。
    """

    def __init__(self, name, llm, rag_system, source, prompt_template):
        super().__init__(name, llm)
        self.rag_system = rag_system
        self.source = source  # 'product' / 'policy'，固定，不可变
        self.prompt_template = prompt_template

    def retrieve(self, query):
        # 复用 RAGSystem 的检索策略（含 HyDE/子查询），source_filter 做权限隔离
        return self.rag_system.retrieve_and_merge(query, source_filter=self.source)

    def handle(self, query, history=None):
        docs = self.retrieve(query)
        context = "\n\n".join(d.page_content for d in docs) if docs else ""
        history_ctx = self._history_context(history)
        prompt = self.prompt_template.format(
            context=context, history=history_ctx, question=query, phone=config.CUSTOMER_SERVICE_PHONE
        )

        collected = ""
        # self.llm 为流式生成器（call_dashscope 内部已捕获异常），逐 token 产出
        for token in self.llm(prompt):
            if token:
                collected += token
                yield token

        if not collected:
            yield f"抱歉，暂时无法回答您的问题，建议联系人工客服（电话 {config.CUSTOMER_SERVICE_PHONE}）。"
