#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: orchestrator.py
描述: 多 Agent 编排器 —— 按「升级 → FAQ → 意图路由 → 子 Agent」顺序编排，
     输出与 IntegratedQASystem.query() 完全一致的 (token, is_complete) 生成器协议。
"""
import time

from base.config import config
from base.logger import logger


class AgentOrchestrator:
    def __init__(self, llm, store, bm25_search, escalation_agent, router, agents):
        self.llm = llm          # 流式生成器（call_dashscope）
        self.store = store
        self.bm25_search = bm25_search
        self.escalation_agent = escalation_agent
        self.router = router
        self.agents = agents    # {"presale": ..., "order": ..., "aftersale": ...}

    def _chitchat(self, query, history=None):
        """闲聊：LLM 直答，无知识库依赖。"""
        collected = ""
        for token in self.llm(query):
            if token:
                collected += token
                yield token
        if not collected:
            yield f"您好，我是智能客服，可以帮您查询商品、订单和售后政策。如需人工服务请拨打 {config.CUSTOMER_SERVICE_PHONE}。"

    def handle(self, query, session_id=None):
        start = time.time()
        history = self.store.fetch_recent_conversations(session_id) if session_id else []
        logger.info(f"[编排] 处理查询: '{query}' (会话ID: {session_id})")

        # 1. 升级检测（强情绪 / 大额订单）→ 转人工
        escalated, reason = self.escalation_agent.detect(query, history)
        if escalated:
            msg = self.escalation_agent.build_message(reason)
            logger.info(f"[编排] 升级转人工: {reason}")
            if session_id:
                self.store.insert_conversation(session_id, query, msg)
            yield msg, True
            return

        # 2. FAQ 快速通道（BM25 ≥ 0.85）
        answer, _ = self.bm25_search.search(query, threshold=0.85)
        if answer:
            logger.info(f"[编排] FAQ 命中: {answer[:50]}...")
            if session_id:
                self.store.insert_conversation(session_id, query, answer)
            yield answer, True
            return

        # 3. 意图路由 + 分发给子 Agent
        intent = self.router.route(query)
        logger.info(f"[编排] 意图: {intent}")
        agent = self.agents.get(intent)

        collected = ""
        if agent:
            for token in agent.handle(query, history):
                if token:
                    collected += token
                    yield token, False
        else:
            # 闲聊 / 未识别 → LLM 直答
            for token in self._chitchat(query, history):
                collected += token
                yield token, False

        if session_id and collected:
            self.store.insert_conversation(session_id, query, collected)

        logger.info(f"[编排] 耗时 {time.time() - start:.2f}s")
        yield "", True
