#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: preprocess.py
作者: ZZS
项目: 3_代码
创建日期: 2026/8/21
描述: 
"""

import jieba
from base.logger import logger

def preprocess_text(text):
    """
    使用jieba分词，将输入的文本分词后返回列表
    :param text:
    :return:
    """
    try:
        logger.info("开始分词...")
        # 将text转成小写然后在切分
        words = jieba.lcut(text.lower())
        return words
    except Exception as e:
        logger.error(f"分词失败: {e}")
        return []

#直接写在了模块的顶层（全局作用域）,所以这个函数能运行。
words = preprocess_text("这是一个测试文本，用于测试分词效果。")
print("分词结果：",words)
# 分词结果： ['这是', '一个', '测试', '文本', '，', '用于', '测试', '分词', '效果', '。']