#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: new_rag_system.py
作者: ZZS
项目: 3_代码
创建日期: 2026/8/27
描述: 
"""
from rag_qa.core.prompts import RAGPrompts
#   导入 time 模块，用于计算时间
import time
from base.config import config
from base.logger import logger
from rag_qa.core.query_classifier import QueryClassifier  # 导入查询分类器
from rag_qa.core.strategy_selector import StrategySelector  # 导入策略选择器
from base.llm_utils import llm_string  # 将生成器消费为完整字符串


class RAGSystem:

    def __init__(self, vector_store, llm):
        #   设置向量数据库对象
        self.vector_store = vector_store
        #   设置大语言模型调用函数
        self.llm = llm
        #   获取 RAG 提示模板
        self.rag_prompt = RAGPrompts.rag_prompt()
        #   初始化查询分类器
        from base.config import config
        self.query_classifier = QueryClassifier(model_path=f'{config.MODELS_DIR}/bert_query_classifier')
        #   初始化策略选择器
        self.strategy_selector = StrategySelector()

    def _retrieve_with_hyde(self, query, source_filter=None):
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        #   获取假设问题生成的 Prompt 模板
        hyde_prompt_template = RAGPrompts.hyde_prompt() # 使用 template 后缀区分
        #   调用大语言模型生成假设答案
        try:
            hypo_answer = llm_string(self.llm, hyde_prompt_template.format(query=query))
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
            #   使用假设答案进行检索，并返回检索结果
            #   注意：HyDE 通常只用于生成检索向量，不一定需要 rerank 这一步，但这里复用了
            return self.vector_store.hybrid_search_with_rerank(
                hypo_answer, k=config.RETRIEVAL_K, source_filter=source_filter  # 使用 K 而非 M
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return []

    def _retrieve_with_subqueries(self, query, source_filter=None):
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        #   获取子查询生成的 Prompt 模板
        subquery_prompt_template = RAGPrompts.subquery_prompt() # 使用 template 后缀区分
        try:
            #   调用大语言模型生成子查询列表
            subqueries_text = llm_string(self.llm, subquery_prompt_template.format(query=query))
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return []

            #   每个子查询各自检索（内部已 hybrid search + rerank，返回 top-K 父文档），按子查询分组
            grouped = []
            for sub_q in subqueries:
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q, k=config.RETRIEVAL_K, source_filter=source_filter  # 使用 K
                )
                grouped.append(docs)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")

            #   跨子查询去重（按父文档内容），同时保持每个子查询内部的相关性顺序
            seen = set()
            unique_grouped = []
            for docs in grouped:
                uniq = []
                for d in docs:
                    if d.page_content not in seen:
                        seen.add(d.page_content)
                        uniq.append(d)
                unique_grouped.append(uniq)

            #   平衡轮询选取：轮流从每个子查询取一条，直到凑满 CANDIDATE_M。
            #   避免按子查询顺序截断导致对比类问题只保留一侧（如全是 iPhone 丢掉华为）。
            final_docs = []
            round_i = 0
            while len(final_docs) < config.CANDIDATE_M:
                advanced = False
                for grp in unique_grouped:
                    if round_i < len(grp):
                        final_docs.append(grp[round_i])
                        advanced = True
                        if len(final_docs) >= config.CANDIDATE_M:
                            break
                if not advanced:
                    break
                round_i += 1

            logger.info(f"所有子查询共检索到 {sum(len(g) for g in unique_grouped)} 个去重文档, 平衡选取 {len(final_docs)} 个")
            return final_docs

        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return []
    # todo 缩进问题，自己对齐下。
    def _retrieve_with_backtracking(self, query, source_filter=None):
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        #   获取回溯问题生成的 Prompt 模板
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()  # 使用 template 后缀区分
        try:
            #   调用大语言模型生成回溯问题
            simplified_query = llm_string(self.llm, backtrack_prompt_template.format(query=query)).strip()
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            #   使用回溯问题进行检索，并返回检索结果
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query, k=config.RETRIEVAL_K, source_filter=source_filter  # 使用 K
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []

    def retrieve_and_merge(self, query, source_filter=None, strategy=None):  # 新增 strategy 参数
        #   如果未指定检索策略，则使用策略选择器选择
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)

        #   根据检索策略选择不同的检索方式
        ranked_sub_chunks = []  # 初始化
        if strategy == "回溯问题检索":
            ranked_sub_chunks = self._retrieve_with_backtracking(query, source_filter=source_filter)
        elif strategy == "子查询检索":
            ranked_sub_chunks = self._retrieve_with_subqueries(query, source_filter=source_filter)  # 返回的是唯一文档列表
            # 注意：子查询返回的是已 rerank 过的父文档或子块列表，后续合并逻辑可能需要调整
            # 当前实现中，子查询返回的是初步检索（可能已rerank）的块，再进行合并
        elif strategy == "假设问题检索":
            ranked_sub_chunks = self._retrieve_with_hyde(query, source_filter=source_filter)
        else:  # 默认或“直接检索”
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(
                query, k=config.RETRIEVAL_K, source_filter=source_filter
            )  # 注意 hybrid_search_with_rerank 返回的是 rerank 后的父文档

        logger.info(f"策略 '{strategy}' 检索到 {len(ranked_sub_chunks)} 个候选文档 (可能已是父文档)")
        final_context_docs = ranked_sub_chunks[:config.CANDIDATE_M]
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")
        return final_context_docs

    def generate_answer(self, query, source_filter=None, history=None):
        """生成答案，支持对话历史，限制提示长度，并支持流式输出"""
        start_time = time.time()
        logger.info(f"开始处理查询: '{query}', 学科过滤: {source_filter}")

        # 验证历史格式
        # [{}]
        if history is not None and not isinstance(history, list):
            logger.warning(f"无效的历史格式: {type(history)}，忽略历史")
            history = []
        elif history:
            # history 是从mysql中拿到的session_id下的对应列表，按时间做了正排
            history = history[-5:]  # 限制最多5轮
            for h in history:
                if not (isinstance(h, dict) and "question" in h and "answer" in h):
                    logger.warning(f"无效的历史条目: {h}，忽略历史")
                    history = []
                    break

        # 构造历史上下文
        # TODO 这里构造了上下文
        # [{question: xxx,answer:xxx },{question:xxx,answer:xxx }]
        # Q：xxx
        # A: xxx
        # Q：xxx
        # A: xxx
        history_context = ""
        if history:
            history_context = "\n".join(
                [f"Q: {h['question']}\nA: {h['answer']}" for h in history]
            )
            logger.info(f"使用对话历史: {history_context[:100]}...")

        #   判断查询类型
        query_category = self.query_classifier.predict_category(query)
        logger.info(f"查询分类结果：{query_category} (查询: '{query}')")

        #   如果查询属于“通用知识”类别，则直接使用 LLM 回答
        if query_category == "通用知识":
            logger.info("查询为通用知识，直接调用 LLM")
            prompt_input = self.rag_prompt.format(
                context="", history=history_context, question=query, phone=config.CUSTOMER_SERVICE_PHONE
            )  # 不使用上下文
            try:
                answer = self.llm(prompt_input)
            except Exception as e:
                logger.error(f"直接调用 LLM 失败: {e}")
                answer = f"抱歉，处理您的通用知识问题时出错。请联系人工客服：{config.CUSTOMER_SERVICE_PHONE}"
            processing_time = time.time() - start_time
            logger.info(
                f"通用知识查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')"
            )
            return answer

        #   否则，进行 RAG 检索并生成答案
        logger.info("查询为专业咨询，执行 RAG 流程")
        #   选择检索策略
        strategy = self.strategy_selector.select_strategy(query)

        #   检索相关文档
        context_docs = self.retrieve_and_merge(
            query, source_filter=source_filter, strategy=strategy
        )  # 传递 strategy

        #   准备上下文
        if context_docs:
            context = "\n\n".join([doc.page_content for doc in context_docs])  # 使用换行符分隔文档
            logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
            # logger.debug(f"上下文内容:\n{context[:500]}...") # Debug 日志可以打印部分上下文
        else:
            context = ""
            logger.info("未检索到相关文档，上下文为空")

        #   构造 Prompt，调用大语言模型生成答案

        # 构造提示
        # TODO generate阶段，放到了提示词里面，让大模型进行整合
        prompt_input = self.rag_prompt.format(
            context=context,
            history=history_context,
            question=query,
            phone=config.CUSTOMER_SERVICE_PHONE
        )
        # logger.debug(f"最终生成的 Prompt:\n{prompt_input}") # Debug 日志

        try:
            answer = self.llm(prompt_input)
        except Exception as e:
            logger.error(f"调用 LLM 生成最终答案失败: {e}")
            answer = f"抱歉，处理您的专业咨询问题时出错。请联系人工客服：{config.CUSTOMER_SERVICE_PHONE}"

        #   记录查询处理完成的日志
        processing_time = time.time() - start_time
        logger.info(f"查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')")
        return answer