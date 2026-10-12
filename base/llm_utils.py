#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: llm_utils.py
描述: LLM 调用工具。现有 call_dashscope 是「生成器函数」（逐 token yield + 末尾 return 全量），
     某些场景（HyDE/子查询/意图分类）需要一次性拿到完整字符串，本模块统一处理。
"""


def llm_string(llm, prompt):
    """
    调用 llm（生成器函数或普通函数），消费为完整字符串。
    - 若 llm 返回 str：直接返回
    - 若 llm 返回生成器：逐 token 累积；若提前 return 了错误信息（无 token），取 StopIteration.value
    """
    result = llm(prompt)
    if isinstance(result, str):
        return result

    collected = ""
    try:
        while True:
            token = next(result)
            if token:
                collected += token
    except StopIteration as e:
        if not collected and e.value:
            collected = str(e.value)
    return collected
