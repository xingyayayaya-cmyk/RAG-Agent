#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: ragas_evaluate.py
描述: 电商智能客服 RAGAS 评估。
      —— 用本地 BGE-M3 稠密嵌入（替代原 Ollama 教育域评估）+ DeepSeek，
         对电商客服系统的真实检索上下文与生成答案打分。
用法（需代理，LLM 需联网）:
    export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
    python rag_qa/rag_assessment/ragas_evaluate.py
"""
import json
import os
import sys

# 项目根目录加入 sys.path，保证 base / ecom_agent 可导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from base.config import config
from base.logger import logger
from ecom_agent.prompts import PRESALE_PROMPT, AFTERSALE_PROMPT
from new_main import IntegratedQASystem

from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # 忠实度：答案是否由上下文支撑
    answer_relevancy,    # 答案相关性：答案是否切题
    context_precision,   # 上下文精确度：检索到的上下文是否有用
    context_recall,      # 上下文召回率：ground_truth 是否被上下文覆盖
    ContextUtilization,  # 上下文利用率（0.2.x 名称；0.3.0 起改名为 ContextRelevance）
)
from datasets import Dataset
import pandas as pd

# 评估数据与结果文件（与本脚本同目录）
_HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DATA_FILE = os.path.join(_HERE, "ecom_eval_data.json")
OUTPUT_XLSX = os.path.join(_HERE, "ragas_evaluation_results.xlsx")


class LocalBGE3Embeddings:
    """LangChain 兼容的本地 BGE-M3 稠密嵌入封装（RAGAS 需要 embed_documents / embed_query）。"""

    def __init__(self, model_path):
        self._model = SentenceTransformer(model_path, device="cpu")

    def embed_documents(self, texts):
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


def build_dataset(qa, items):
    """跑真实电商系统：对每条问题检索上下文 + 生成答案，组装成 RAGAS 数据集。"""
    questions, contexts, answers, ground_truths = [], [], [], []
    for it in items:
        q, source, gt = it["question"], it["source"], it["ground_truth"]
        # 检索上下文（与子 Agent 相同的 source 权限隔离）
        docs = qa.rag_system.retrieve_and_merge(q, source_filter=source)
        ctx = [d.page_content for d in docs]
        # 生成答案（复刻 RAGAgent.handle：检索上下文 + 子 Agent prompt + 流式 LLM）
        prompt_tpl = PRESALE_PROMPT if source == "product" else AFTERSALE_PROMPT
        prompt = prompt_tpl.format(
            context="\n\n".join(ctx), history="无", question=q, phone=config.CUSTOMER_SERVICE_PHONE
        )
        ans = ""
        for token in qa.call_dashscope(prompt):
            if token:
                ans += token
        questions.append(q)
        contexts.append(ctx)
        answers.append(ans)
        ground_truths.append(gt)
        logger.info(f"已生成样本: {q}（上下文 {len(ctx)} 块, 答案 {len(ans)} 字）")
    return Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": ground_truths,
    })


def main():
    # 1. 加载电商评估数据
    with open(EVAL_DATA_FILE, "r", encoding="utf-8") as f:
        items = json.load(f)

    # 2. 初始化电商系统（local_mode）
    qa = IntegratedQASystem()

    # 3. 跑真实检索 + 生成，组装数据集
    dataset = build_dataset(qa, items)
    print(dataset)

    # 4. 模型：DeepSeek 做评判 LLM，本地 BGE-M3 做嵌入
    llm = ChatOpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.DASHSCOPE_BASE_URL,
        model_name=config.LLM_MODEL,
        temperature=0.0,
    )
    embedding = LocalBGE3Embeddings(os.path.join(config.MODELS_DIR, "bge-m3"))

    # 5. RAGAS 评估
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            ContextUtilization(),
        ],
        llm=llm,
        embeddings=embedding,
    )

    # 6. 打印并落盘
    print("RAGAS 评估结果：")
    print(result)
    result.to_pandas().to_excel(OUTPUT_XLSX, index=False)
    print(f"结果已写入: {OUTPUT_XLSX}")
    qa.close()


if __name__ == "__main__":
    main()
