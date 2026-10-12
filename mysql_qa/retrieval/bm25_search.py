#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: bm25_search.py
作者: ZZS
项目: 3_代码
创建日期: 2026/8/21
描述: FAQ 快速通道 —— BM25 检索常见问题，命中（≥阈值）直接返回答案，否则回退 RAG。
     数据源统一走 Store（fetch_questions/fetch_answer），缓存统一走 Cache（get_answer/set_answer/get_data/set_data）。
"""
# 标准的BM25算法，标准原版 BM25（Okapi 版本），Elasticsearch 默认用的就是它
from rank_bm25 import BM25Okapi
import numpy as np
# jieba分词
from mysql_qa.utils.preprocess import preprocess_text
# 导入日志
from base.logger import logger

class BM25Search:
    """
    构建bm25算法，可以输入问题返回最相关的文档，还能缓存新问题与最相关问题的答案到缓存，方便重复问题的调用缓存。
    """
    def __init__(self, cache, store):
        self.logger = logger
        self.cache = cache          # get_answer/set_answer/get_data/set_data
        self.store = store          # fetch_questions() -> list[str] / fetch_answer(question) -> str|None
        self.bm25 = None
        self.questions = None  #存经过分词之后的问题列表BM25不能直接搜原始中文句子，必须先把每个问题切成词语
        self.original_questions = None #保存数据库读取出来没有做任何处理、原汁原味的原始问题字符串
        # 构建bm25模型
        self._load_data()

    # 私有加载函数，程序启动的时候自动跑一遍，把问题库加载到内存，建好 BM25 索引
    def _load_data(self):
       #定义 2 个缓存键名，缓存原始问题列表与分词后的问题列表
        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"
        # 从缓存中获取原始问题 和 分词问题
        self.original_questions = self.cache.get_data(original_key)
        self.tokenized_questions = self.cache.get_data(tokenized_key)

        # 如果缓存中没有，从 Store 读取所有问题
        if not self.original_questions or not self.tokenized_questions:
            self.original_questions = self.store.fetch_questions()
            # 分词
            self.tokenized_questions = [preprocess_text(q) for q in self.original_questions]

            # 缓存
            self.cache.set_data(original_key, self.original_questions)
            self.cache.set_data(tokenized_key, self.tokenized_questions)

        # 如果有的话。
        self.questions = self.tokenized_questions
        # 构建bm25模型
        self.bm25 = BM25Okapi(self.questions)
        self.logger.info("BM25模型构建完成")


    def _softmax(self, scores):
        # 计算 Softmax 分数
        exp_scores = np.exp(scores - np.max(scores))
        # 返回归一化分数
        return exp_scores / exp_scores.sum()
    #np.exp()：自然指数 \(e^x\)，拉大分数差距，高分变得更大,取最大值


    def search(self, query, threshold=0.85):
        """
        实现query最相关的文档返回
        :param query:
        :param threshold:
        :return: (answer, need_rag)
        """
        # 1. 输入校验
        if not query or not isinstance(query, str):
            return None, False

        # 2. 精确匹配（缓存）
        cached_answer = self.cache.get_answer(query)
        # 3. 如果有缓存，直接返回
        if cached_answer:
            self.logger.info(f"从缓存中获取答案: {query}")
            return cached_answer, False  # 不查询rag

        # 4. 如果没有，走bm25
        # 分词查询
        query_tokens = preprocess_text(query)
        # 计算 BM25 分数
        scores = self.bm25.get_scores(query_tokens)
        # 计算 Softmax 分数
        softmax_scores = self._softmax(scores)
        # 获取最高分索引
        best_idx = softmax_scores.argmax()
        # 获取最高分
        best_score = softmax_scores[best_idx]

        # 5. 判断最相关文档的得分概率是否大于阈值
        if best_score >= threshold:
            # 获取原始的问题
            original_question = self.original_questions[best_idx]
            # 缓存中做检索
            result = self.cache.get_answer(original_question)
            # 如果有缓存，返回缓存的答案，如果没有就走 Store 查询。
            answer = result or self.store.fetch_answer(original_question)

            if answer:
                # 缓存答案，不走rag
                self.cache.set_answer(query, answer)
                return answer, False

        # 没有相关问题，走rag
        return None, True

if __name__ == '__main__':
    from base.store import create_store
    from base.cache import create_cache

    bm25 = BM25Search(create_cache(), create_store())
    print(bm25.search("这个手机怎么退换货？", threshold=0.85))
