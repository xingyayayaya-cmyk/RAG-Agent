#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: prompts.py
作者: ZZS
项目: 3_代码
创建日期: 2026/8/24
描述: 
"""
# 导入 PromptTemplate 类，用于创建 Prompt 模板
from langchain_core.prompts import PromptTemplate


# 定义 RAGPrompts 类，用于管理所有 Prompt 模板
class RAGPrompts:

    # 定义 RAG 提示模板
    # todo 如果rag系统是包含历史的，在最终生成答案的提示词中加入历史的参数。
    @staticmethod
    def rag_prompt():
        '''
        静态函数函数返回的好处
        每次调用都生成全新独立的模板对象，不会出现模板对象被意外修改、污染缓存的问题
        '''
        # 创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
                你是一个智能助手，帮助用户回答问题。请参考用户的助手的对话历史和上下文回答问题。
                如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。  
                如果答案来源于检索到的文档，请在回答中说明。

                对话历史: {history}
                上下文: {context}  
                问题: {question}  

                如果无法回答，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”  
                回答:  
                """,
            #   定义输入变量
            input_variables=["context", "history", "question", "phone"],
        )
    # 定义 RAG 提示模板
    # @staticmethod
    # def rag_prompt():
    #     # 创建并返回 PromptTemplate 对象
    #     return PromptTemplate(
    #         template="""
    #         你是一个智能助手，帮助用户回答问题。
    #         如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。
    #         如果答案来源于检索到的文档，请在回答中说明。
    #
    #         上下文: {context}
    #         问题: {question}
    #
    #         如果无法回答，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
    #         回答:
    #         """,
    #         #   定义输入变量
    #         input_variables=["context", "question", "phone"],
    #     )
        # @staticmethod
    # def rag_prompt():
    #     return PromptTemplate(
    #         template="""
    #     你是一个智能助手，负责帮助用户回答问题。请按照以下步骤处理：
    #
    #     1. **分析问题和上下文**：
    #        - 基于提供的上下文（如果有）和你的知识回答问题。
    #        - 如果答案来源于检索到的文档，请在回答中明确说明，例如：“根据提供的文档，……”。
    #
    #     2. **评估对话历史**：
    #        - 检查对话历史是否与当前问题相关（例如，是否涉及相同的话题、实体或问题背景）。
    #        - 如果对话历史与问题相关，请结合历史信息生成更准确的回答。
    #        - 如果对话历史无关（例如，仅包含问候或不相关的内容），忽略历史，仅基于上下文和问题回答。
    #
    #     3. **生成回答**：
    #        - 提供清晰、准确的回答，避免无关信息。
    #        - 如果上下文和历史消息均不足以回答问题，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
    #
    #     **上下文**: {context}
    #     **对话历史**:
    #     {history}
    #     **问题**: {question}
    #
    #     **回答**:
    #     """,
    #         input_variables=["context", "history", "question", "phone"],
    #     )


    @staticmethod
    def hyde_prompt():
        # 定义假设问题生成的 Prompt 模板
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
            问题: {query}  
            假设答案:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义子查询生成的 Prompt 模板
    @staticmethod
    def subquery_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""
            将用户的查询拆分为多个独立的检索子查询，每行一条，便于从商品知识库精确检索。

            拆分规则：
            1. 对比类问题（如「X 和 Y 哪个 Z 好」「X 和 Y 的区别」）：拆成分别检索 X、Y 的子查询，每条保留品牌名和对比属性词。
            2. 多条件问题：每个条件拆一条。
            3. 子查询必须保留原查询里的关键实体（品牌/商品名）和属性词，不得改变语义。

            示例：
            查询：iPhone 和华为哪个续航好
            子查询：
            iPhone 电池容量
            华为手机 电池容量

            查询: {query}
            子查询:
            """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义回溯问题生成的 Prompt 模板
    @staticmethod
    def backtracking_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            将以下复杂查询简化为一个更简单的问题：  
            查询: {query}  
            简化问题:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )