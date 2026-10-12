#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: config.py
作者: ZZS
项目: 3_代码
创建日期: 2026/8/20
描述: 
"""

import configparser
import json
import os

class Config:
    """
    所有的配置项，通过configparser读取.ini文件，将其中的关于mysql和redis的配置取出来。
    """
    # todo 不要传 config_file = None
    def __init__(self, config_file = None):
        # 1. 实例化配置对象
        # 创建配置解析器，启用插值功能，ini 配置文件里面，实现「变量引用」
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())

        # os.path.dirname 获取当前文件所在的目录
        # print("*"*80)
        # print(__file__)
        # print(os.path.dirname(__file__))
        # print(os.path.dirname(os.path.dirname(__file__)))
        # print("*"*80)

        # todo 在这里，通过 os.path.dirname(os.path.dirname(__file__)) 找到项目的根目录
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

        # todo 打印下 self.log_file 看是不是斜杠有问题，logs/app.log 是错误的，改成 logs\app.log
        #`r` = **raw‑string 原始字符串**  作用：**关闭反斜杠 `\` 的转义功能**。因为本身window路径就是原始反斜杠
        #其实写不写r都可以，因为os.path.join 会自动帮你拼接分隔符，Windows 自动用`\`，Linux/Mac 自动用`/`。
        self.log_file = os.path.join(self.PROJECT_ROOT, r'logs\app.log')
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, r'rag_qa\data')
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, r'rag_qa\models')
        self.EDU_DOCUMENT_LOADERS_DIR = os.path.join(self.PROJECT_ROOT, r'rag_qa\edu_document_loaders')
        if config_file is None:
            config_file = os.path.join(self.PROJECT_ROOT, 'config.ini')

        # todo 新增的分类模型相关的参数。
        # 预分类模型目录
        self.pre_classify_model_dir = os.path.join(self.MODELS_DIR, 'bert-base-chinese')
        # 训练好的模型的路径
        self.trained_classify_model_dir = os.path.join(self.MODELS_DIR, 'bert_query_classifier')
        # 训练数据的位置
        self.train_data_dir = os.path.join(self.PROJECT_ROOT, r'rag_qa\classify_data\model_generic_5000.json')


        # 2. 读取配置文件
        # todo 增加编码参数即可
        self.config.read(config_file, encoding='utf-8')

        # 3. 获取mysql和redis的配置项和日志路径
        self.MYSQL_HOST = self.config.get("mysql","host", fallback="192.168.21.226")
        self.MYSQL_PORT = self.config.getint("mysql","port", fallback=3306)
        self.MYSQL_USER = self.config.get("mysql","user", fallback="root")
        self.MYSQL_PASSWORD = self.config.get("mysql","password", fallback="root")
        self.MYSQL_DATABASE = self.config.get("mysql","database", fallback="subjects_kg")
        '''
        config.get返回字符串 适合：IP 地址、账号、密码、集合名字，这些本来就是文本
        config.getint返回整数 适合：端口号、数据库编号，必须是数字
        '''
        # redis配置
        self.REDIS_HOST = self.config.get("redis","host", fallback="192.168.21.226")
        self.REDIS_PORT = self.config.getint("redis","port", fallback=6379)
        self.REDIS_PASSWORD = self.config.get("redis","password", fallback="")
        self.REDIS_DB = self.config.getint("redis","db", fallback=0)

        # Milvus 配置
        # Milvus 主机地址
        '''
        读取优先级:os.getenv>config.ini>fallback 默认值
        Milvus 属于向量数据库，线上部署、容器 Docker、云服务器环境最经常需要动态切换地址
        本地开发：直接写在`config.ini`里面
        服务器上线、Docker 部署：不想改代码 /ini 文件，直接设置系统环境变量`MILVUS_HOST`就可以覆盖地址
        '''
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='192.168.21.226'))
        # Milvus 端口
        self.MILVUS_PORT = os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback='19530'))
        # Milvus 数据库名
        self.MILVUS_DATABASE_NAME = os.getenv('MILVUS_DATABASE_NAME',
                                              self.config.get('milvus', 'database_name', fallback='milvus_xiaoya'))
        # Milvus 集合名
        self.MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME',
                                                self.config.get('milvus', 'collection_name', fallback='edurag'))

        # LLM 配置
        # LLM 模型名
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='deepseek-v4-pro')
        # API 密钥（环境变量可覆盖）
        self.DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY',
                                           self.config.get('llm', 'dashscope_api_key',
                                                           fallback=''))
        # API 地址（OpenAI 兼容接口）
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url',
                                                  fallback='https://api.deepseek.com/')

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 应用配置
        # 有效来源列表（电商域：商品/订单/政策）
        self.VALID_SOURCES = json.loads(
            self.config.get('app', 'valid_sources', fallback='["product", "order", "policy"]'))
        # 客服电话
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='12345678')
        # 本地降级开关
        self.LOCAL_MODE = self.config.getboolean('app', 'local_mode', fallback=True)
        # 本地文件路径（Milvus Lite / SQLite），相对项目根目录
        self.MILVUS_LOCAL_DB = os.path.join(self.PROJECT_ROOT,
                                            self.config.get('app', 'milvus_local_db', fallback='data/milvus_lite.db'))
        self.SQLITE_DB = os.path.join(self.PROJECT_ROOT,
                                      self.config.get('app', 'sqlite_db', fallback='data/ecom.db'))
        # 业务表名
        self.FAQ_TABLE = self.config.get('app', 'faq_table', fallback='ecom_faq')
        self.ORDERS_TABLE = self.config.get('app', 'orders_table', fallback='orders')
        self.CONVERSATION_TABLE = self.config.get('app', 'conversation_table', fallback='conversations')
        # 各平台大额订单阈值（元）
        self.ESCALATION_THRESHOLDS = json.loads(
            self.config.get('app', 'escalation_thresholds',
                            fallback='{"淘宝": 5000, "京东": 10000, "拼多多": 3000, "抖音": 8000, "1688": 20000}'))

        # 日志文件路径
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, 'logs')
        self.LOG_FILE = os.path.join(self.LOG_DIR, 'app.log')



# todo 如果找不到config可以导包时直接导入 from base.config import config  ，不要 from base.config import Config
config = Config()

if __name__ == '__main__':
    config = Config()
    # print(config.MYSQL_HOST)
    # print(config.MYSQL_PASSWORD)
    # print(config.MODELS_DIR)
    # print(config.log_file)
    # print(config.Milvus_HOST)
    print(config.pre_classify_model_dir)
    print(config.trained_classify_model_dir)
    print(config.train_data_dir)