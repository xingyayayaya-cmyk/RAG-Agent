#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: new_main.py
描述: 电商智能客服主系统 —— 集成 FAQ 快速通道、RAG 检索、多 Agent 编排（意图路由/子 Agent/升级）、流式输出。
     存储/缓存经抽象层（Store/Cache）接入，local_mode 下自动降级为 SQLite/Memory。
"""
# 导入 FAQ 快速通道（BM25）
from mysql_qa.retrieval.bm25_search import BM25Search
# 导入 RAG 系统组件，用于知识库检索和答案生成
from rag_qa.core.vector_store import create_vector_store
from rag_qa.core.new_rag_system import RAGSystem
# 导入配置、日志、存储/缓存抽象层
from base.config import config
from base.logger import logger
from base.store import create_store
from base.cache import create_cache
# 导入 OpenAI 客户端，用于调用 LLM API
from openai import OpenAI
import time
import uuid


class IntegratedQASystem:
    def __init__(self):
        # 初始化日志工具，用于记录系统运行信息
        self.logger = logger
        self.config = config
        # 存储 / 缓存抽象层（本地 SQLite/Memory 或远程 MySQL/Redis）
        self.store = create_store()
        self.cache = create_cache()
        # FAQ 快速通道（BM25 检索常见问题）
        self.bm25_search = BM25Search(self.cache, self.store)
        try:
            # 初始化 OpenAI 客户端，连接 LLM API
            self.client = OpenAI(api_key=self.config.DASHSCOPE_API_KEY, base_url=self.config.DASHSCOPE_BASE_URL)
        except Exception as e:
            self.logger.error(f"OpenAI 客户端初始化失败: {e}")
            raise
        # 初始化向量存储，用于 RAG 系统的知识库管理（本地降级 / 远程 Milvus）
        self.vector_store = create_vector_store()
        # 初始化 RAG 系统，传入向量存储和 LLM 调用函数
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)

        # 多 Agent 编排层（意图路由 / 子 Agent / 升级 Agent / 编排器）
        from ecom_agent.router import IntentRouter
        from ecom_agent.escalation_agent import EscalationAgent
        from ecom_agent.presale_agent import PresaleAgent
        from ecom_agent.aftersale_agent import AftersaleAgent
        from ecom_agent.order_agent import OrderAgent
        from ecom_agent.orchestrator import AgentOrchestrator

        router = IntentRouter(self.call_dashscope)
        escalation_agent = EscalationAgent(self.store)
        agents = {
            "presale": PresaleAgent(self.call_dashscope, self.rag_system),
            "aftersale": AftersaleAgent(self.call_dashscope, self.rag_system),
            "order": OrderAgent(self.call_dashscope, self.store),
        }
        self.orchestrator = AgentOrchestrator(
            self.call_dashscope, self.store, self.bm25_search, escalation_agent, router, agents
        )

    def call_dashscope(self, prompt):
        """调用 LLM API 生成答案（流式输出）"""
        try:
            # 创建聊天完成请求，启用流式输出
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个电商客服助手。"},
                    {"role": "user", "content": prompt},
                ],
                timeout=30,
                stream=True,
            )
            collected_content = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_content += content
                    yield content
            return collected_content
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            return f"错误：LLM调用失败 - {e}"

    def get_session_history(self, session_id: str) -> list:
        """获取最近 5 轮对话历史（按时间正序）"""
        return self.store.fetch_recent_conversations(session_id, limit=5)

    def clear_session_history(self, session_id: str) -> bool:
        """清除指定会话历史"""
        return self.store.clear_conversations(session_id)

    def query(self, query, source_filter=None, session_id=None):
        """查询集成系统，委托多 Agent 编排器处理，输出 (token, is_complete) 流。

        注：source_filter 仅为兼容旧接口保留；权限隔离由各子 Agent 内部固定 source 决定，
        外部传入的 source_filter 不再影响数据源。
        """
        start_time = time.time()
        self.logger.info(f"处理查询: '{query}' (会话ID: {session_id})")
        for token, is_complete in self.orchestrator.handle(query, session_id=session_id):
            yield token, is_complete
        self.logger.info(f"查询处理耗时 {time.time() - start_time:.2f}秒")

    def close(self):
        self.store.close()


def main():
    qa_system = IntegratedQASystem()
    session_id = str(uuid.uuid4())
    print("\n欢迎使用电商智能客服系统！")
    print(f"会话ID: {session_id}")
    print("输入查询进行问答，输入 'exit' 退出。")
    try:
        while True:
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                logger.info("退出系统")
                print("再见！")
                break
            print("\n答案: ", end="", flush=True)
            answer = ""
            for token, is_complete in qa_system.query(query, session_id=session_id):
                if token:
                    print(token, end="", flush=True)
                    answer += token
                if is_complete:
                    print()
                    break
    except Exception as e:
        logger.error(f"系统错误: {e}")
        print(f"发生错误: {e}")
    finally:
        qa_system.close()


if __name__ == "__main__":
    main()
