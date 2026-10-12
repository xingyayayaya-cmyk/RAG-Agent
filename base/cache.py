#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: cache.py
描述: 缓存抽象层。远程走 Redis（复用现有 RedisClient），本地降级走内存字典。
     统一接口：get_answer / set_answer / get_data / set_data
"""
from base.config import config
from base.logger import logger


class RedisCache:
    """远程缓存：包装现有 RedisClient（Redis 键 answer:{query} 存原始字符串）。"""

    def __init__(self):
        # 延迟导入，避免本地模式也强依赖 redis 连接
        from mysql_qa.cache.redis_client import RedisClient
        self._rc = RedisClient()

    def get_answer(self, query):
        return self._rc.get_answer(query)

    def set_answer(self, query, answer):
        try:
            self._rc.client.set(f"answer:{query}", answer)
        except Exception as e:
            logger.error(f"Redis 缓存答案失败: {e}")

    def get_data(self, key):
        return self._rc.get_data(key)

    def set_data(self, key, value):
        self._rc.set_data(key, value)


class MemoryCache:
    """本地降级缓存：进程内字典，接口与 RedisCache 一致。"""

    def __init__(self):
        self._store = {}

    def get_answer(self, query):
        return self._store.get(f"answer:{query}")

    def set_answer(self, query, answer):
        self._store[f"answer:{query}"] = answer

    def get_data(self, key):
        return self._store.get(key)

    def set_data(self, key, value):
        self._store[key] = value


def create_cache():
    """按配置返回缓存后端。"""
    if config.LOCAL_MODE:
        logger.info("使用本地内存缓存（MemoryCache）")
        return MemoryCache()
    logger.info("使用远程 Redis 缓存")
    return RedisCache()
