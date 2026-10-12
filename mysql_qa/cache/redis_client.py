#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: redis_client.py
作者: ZZS
项目: 3_代码
创建日期: 2026/8/21
描述: 
"""
from idlelib.iomenu import encoding

# 导入 Redis 客户端
import redis
# 导入 JSON 处理
import json
# 导入配置和日志
# todo 将导包更改一下，单独导入配置或者logger
# from base import Config, logger
from base.config import config,Config
from base.logger import logger



class RedisClient:
    def __init__(self):
        # 初始化日志
        self.logger = logger
        try:
            # 连接 Redis
            self.client = redis.StrictRedis(
                host=Config().REDIS_HOST,
                port=Config().REDIS_PORT,
                password=Config().REDIS_PASSWORD,
                db=Config().REDIS_DB,
                decode_responses=True
            )
            #不加decode_responses=True，Redis 返回给你的字符串是 **bytes 字节串**
            # todo 测试连接是否成功。
            self.client.ping()
            # 记录连接成功
            self.logger.info("Redis 连接成功")
        except redis.RedisError as e:
            # 记录连接失败
            self.logger.error(f"Redis 连接失败: {e}")
            raise
            #`raise`：**把捕获到的异常，重新往外抛出去**

    def set_data(self, key, value):
        # 存储数据到 Redis
        try:
            # 存储 JSON 数据
            self.client.set(key, json.dumps(value))
            # 记录存储成功
            self.logger.info(f"存储数据到 Redis: {key}")
        except redis.RedisError as e:
            # 记录存储失败
            self.logger.error(f"Redis 存储失败: {e}")
        '''
        value可以是Python字典、列表。
        因为Redis的.set()不能直接存字典，所以先用json.dumps()把字典转成一段 JSON 字符串，再存入 Redis。
        '''

    def get_data(self, key):
        # 从 Redis 获取数据
        '''
        get_data()：通用的数据读取函数，你给它什么 key，它就直接查什么 key。
        '''
        try:
            # 获取数据
            data = self.client.get(key)
            # 返回解析后的 JSON 数据或 None
            return json.loads(data) if data else None
        except redis.RedisError as e:
            # 记录获取失败
            self.logger.error(f"Redis 获取失败: {e}")
            # 返回 None
            return None

    def get_answer(self, query):
        # 获取查询的缓存答案
        '''
        get_answer()：专门读取“问答缓存”，它不是让你随便传一个 Redis key。你传进去的是：query,也就是用户的问题。
        '''
        try:
            # 从 Redis 获取答案
            answer = self.client.get(f"answer:{query}")
            if answer:
                # 记录获取成功
                self.logger.info(f"从 Redis 获取答案: {query}")
                # 返回答案
                return answer
            # 返回 None
            return None
        except redis.RedisError as e:
            # 记录查询失败
            self.logger.error(f"Redis 查询失败: {e}")
            # 返回 None
            return None

if __name__ == '__main__':
    redcli = RedisClient()
    print(redcli)
    print(redcli.client.get("user:1"))
    print(redcli.client.smembers("set_test"))
    #smembers返回集合name的全部成员