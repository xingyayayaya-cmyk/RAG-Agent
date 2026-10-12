#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: logger.py
作者: ZZS
项目: 3_代码
创建日期: 2026/8/21
描述: 
"""
# 导入日志库
import logging
# 导入路径操作库
import os
# 导入配置类
# todo 在其他模块引用logger时，会报错No module named 'config'，可以在config前加入base.
# todo 如果还不行，直接不要到config ，log_file=写死的路径即可
# from base.config import Config
from base.config import config

def setup_logging(log_file=config.LOG_FILE):
    '''
    创建日志目录：操作硬盘（磁盘、文件文件夹）**，`os.makedirs` 在电脑上新建 `logs` 文件夹；
    创建日志器：操作内存（Python 程序里面的对象）**，`logging.getLogger` 在代码内存里新建一个打印日志的工具。
    '''
    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # 创建日志器
    #`getLogger()` 自带一条规则：`EduRAG`的日志器不存在直接建，存在就返回
    logger = logging.getLogger("EduRAG")
    # 设置日志级别
    logger.setLevel(logging.INFO)
    # 避免重复添加处理器
    if not logger.handlers:
        '''
        创建处理器：
        设置处理器格式(日期)
        添加处理器
        handler（处理器）负责把日志事件真正输出到目标位置。 
                   
        处理器分两种：文件/控制台
        '''
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        # 设置文件处理器级别
        file_handler.setLevel(logging.INFO)
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        # 设置控制台处理器级别
        console_handler.setLevel(logging.INFO)


        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # 为文件处理器设置格式
        file_handler.setFormatter(formatter)
        # 为控制台处理器设置格式
        console_handler.setFormatter(formatter)


        # 添加文件处理器
        logger.addHandler(file_handler)
        # 添加控制台处理器
        logger.addHandler(console_handler)
    # 返回日志器
    return logger

# 初始化日志器
logger = setup_logging()