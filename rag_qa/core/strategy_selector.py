#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: strategy_selector.py
作者: ZZS
项目: 3_代码
创建日期: 2026/8/26
描述: 
"""
# 导包
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from base.config import config
from base.logger import logger


class StrategySelector:
    """
    将用户的输入传入给提示词模板，
    将模板给到大模型，
    生成四种检索策略。
    """
    def __init__(self):
        # 1. 创建大模型对象
        self.client = OpenAI(api_key=config.DASHSCOPE_API_KEY,
               base_url=config.DASHSCOPE_BASE_URL,
               timeout=10)  # 超时兜底，LLM 不可达时快速降级为直接检索
        # 2. 创建提示词模板
        self.strategy_template = self._get_strategy_prompt()

    def _get_strategy_prompt(self):
        """
        提示词：根据提示词输出检索策略。
        :param prompt:
        :return:
        """
        return PromptTemplate(
            template="""
                    你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

                    以下是几种检索增强策略及其适用场景：

                    1.  **直接检索：**
                        * 描述：对用户查询直接进行检索，不进行任何增强处理。
                        * 适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                            * 示例：
                                * 查询：AI 学科学费是多少？
                                * 策略：直接检索
                            * 查询：JAVA的课程大纲是什么？
                                * 策略：直接检索
                    2.  **假设问题检索（HyDE）：**
                        * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                        * 适用场景：适用于查询较为抽象，直接检索效果不佳的问题，例如：
                            * 示例：
                                * 查询：人工智能在教育领域的应用有哪些？
                                * 策略：假设问题检索
                    3.  **子查询检索：**
                        * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                        * 适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                            * 示例：
                                * 查询：比较 Milvus 和 Zilliz Cloud 的优缺点。
                                * 策略：子查询检索
                    4.  **回溯问题检索：**
                        * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                        * 适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题，例如：
                            * 示例：
                                * 查询：我有一个包含 100 亿条记录的数据集，想把它存储到 Milvus 中进行查询。可以吗？
                                * 策略：回溯问题检索

                    根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
                    """
            ,
            input_variables=["query"],
        )


    def call_dashscope(self, prompt):
        """
        将提示词构建完整，给到大模型做输出。
        :param query:
        :return:
        """
        try:
            completion = self.client.chat.completions.create(
                model = config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手，能够根据用户输入的Prompt严格执行并返回可靠的结果。"},
                    {"role": "user", "content": prompt},
                ],
                temperature = 0.0
            )
            # 返回结果
            # logger.info(f"DashScope API Response: {completion}")
            # todo 不知道返回啥，看一下。
            # 拿到到使用 直接检索兜底。
            return completion.choices[0].message.content if completion.choices else "直接检索"

        except Exception as e:
            logger.error(f"DashScope API Error: {e}")
            return "直接检索"


    def select_strategy(self, query):
        """
        根据用户输入，选择最合适的检索策略。
        :param query:
        :return:
        """
        strategy = self.call_dashscope(self.strategy_template.format(query=query))
        logger.info(f"Selected Strategy: {strategy}")
        return strategy

if __name__ == '__main__':
    selector = StrategySelector()
    selector.select_strategy("你好！")