#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: store.py
描述: 业务数据存储抽象层（FAQ / 订单 / 对话历史）。
     远程走 MySQL（pymysql），本地降级走 SQLite。二者接口完全一致，由 create_store() 工厂选择。

     统一接口：
       create_tables()
       insert_faq_batch(rows)            # rows = [(category, question, answer), ...]
       fetch_questions() -> list[str]
       fetch_answer(question) -> str|None
       insert_order_batch(orders)        # orders = [dict, ...]
       lookup_order(order_id, phone, sku_id, product_keyword, limit) -> list[dict]
       insert_conversation(session_id, question, answer)
       fetch_recent_conversations(session_id, limit) -> list[dict]   # 按时间正序
       clear_conversations(session_id)
       close()
"""
import sqlite3
from base.config import config
from base.logger import logger

# 订单表字段（与 ecom_data/ingest.py 生成的订单字段一一对应）
ORDER_FIELDS = [
    "order_id", "platform", "buyer_name", "buyer_phone_masked", "buyer_city",
    "sku_id", "product_name", "quantity", "unit_price", "amount",
    "is_high_value", "status", "order_time", "pay_time",
]


class SQLiteStore:
    """本地降级后端：SQLite，接口与 MySQLStore 一致。"""

    def __init__(self, db_path=None):
        db_path = db_path or config.SQLITE_DB
        import os
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()

    def create_tables(self):
        cur = self.conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {config.FAQ_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category VARCHAR(20),
                question TEXT,
                answer TEXT
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {config.ORDERS_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT,
                platform TEXT,
                buyer_name TEXT,
                buyer_phone_masked TEXT,
                buyer_city TEXT,
                sku_id TEXT,
                product_name TEXT,
                quantity INTEGER,
                unit_price REAL,
                amount REAL,
                is_high_value INTEGER,
                status TEXT,
                order_time TEXT,
                pay_time TEXT
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {config.CONVERSATION_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question TEXT,
                answer TEXT,
                timestamp TEXT
            )
        """)
        self.conn.commit()
        logger.info("SQLite 表初始化成功")

    def insert_faq_batch(self, rows):
        cur = self.conn.cursor()
        cur.executemany(
            f"INSERT INTO {config.FAQ_TABLE} (category, question, answer) VALUES (?, ?, ?)",
            rows,
        )
        self.conn.commit()
        logger.info(f"SQLite 插入 FAQ {len(rows)} 条")

    def fetch_questions(self):
        cur = self.conn.cursor()
        cur.execute(f"SELECT question FROM {config.FAQ_TABLE}")
        return [r["question"] for r in cur.fetchall()]

    def fetch_answer(self, question):
        cur = self.conn.cursor()
        cur.execute(f"SELECT answer FROM {config.FAQ_TABLE} WHERE question = ?", (question,))
        row = cur.fetchone()
        return row["answer"] if row else None

    def insert_order_batch(self, orders):
        cur = self.conn.cursor()
        placeholders = ", ".join(["?"] * len(ORDER_FIELDS))
        cols = ", ".join(ORDER_FIELDS)
        rows = [[o.get(f) for f in ORDER_FIELDS] for o in orders]
        cur.executemany(f"INSERT INTO {config.ORDERS_TABLE} ({cols}) VALUES ({placeholders})", rows)
        self.conn.commit()
        logger.info(f"SQLite 插入订单 {len(rows)} 条")

    def lookup_order(self, order_id=None, phone=None, sku_id=None, product_keyword=None, limit=20):
        where, params = [], []
        if order_id:
            where.append("order_id = ?"); params.append(order_id)
        if phone:
            where.append("buyer_phone_masked = ?"); params.append(phone)
        if sku_id:
            where.append("sku_id = ?"); params.append(sku_id)
        if product_keyword:
            where.append("product_name LIKE ?"); params.append(f"%{product_keyword}%")
        sql = f"SELECT {', '.join(ORDER_FIELDS)} FROM {config.ORDERS_TABLE}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY id DESC LIMIT {int(limit)}"
        cur = self.conn.cursor()
        cur.execute(sql, params)
        return [self._order_to_dict(r) for r in cur.fetchall()]

    def insert_conversation(self, session_id, question, answer):
        cur = self.conn.cursor()
        cur.execute(
            f"INSERT INTO {config.CONVERSATION_TABLE} (session_id, question, answer, timestamp) "
            "VALUES (?, ?, ?, datetime('now','localtime'))",
            (session_id, question, answer),
        )
        self.conn.commit()

    def fetch_recent_conversations(self, session_id, limit=5):
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT question, answer FROM {config.CONVERSATION_TABLE} "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, int(limit)),
        )
        rows = [{"question": r["question"], "answer": r["answer"]} for r in cur.fetchall()]
        return rows[::-1]

    def clear_conversations(self, session_id):
        cur = self.conn.cursor()
        cur.execute(f"DELETE FROM {config.CONVERSATION_TABLE} WHERE session_id = ?", (session_id,))
        self.conn.commit()
        return True

    @staticmethod
    def _order_to_dict(row):
        d = {f: row[f] for f in ORDER_FIELDS}
        d["is_high_value"] = bool(d.get("is_high_value"))
        return d

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


class MySQLStore:
    """远程后端：MySQL（pymysql），接口与 SQLiteStore 一致。"""

    def __init__(self):
        import pymysql
        self.conn = pymysql.connect(
            host=config.MYSQL_HOST,
            port=config.MYSQL_PORT,
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD,
            database=config.MYSQL_DATABASE,
        )
        self.create_tables()

    def create_tables(self):
        cur = self.conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {config.FAQ_TABLE} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                category VARCHAR(20),
                question VARCHAR(1000),
                answer VARCHAR(2000)
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {config.ORDERS_TABLE} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                order_id VARCHAR(40),
                platform VARCHAR(20),
                buyer_name VARCHAR(20),
                buyer_phone_masked VARCHAR(20),
                buyer_city VARCHAR(20),
                sku_id VARCHAR(40),
                product_name VARCHAR(200),
                quantity INT,
                unit_price DOUBLE,
                amount DOUBLE,
                is_high_value TINYINT,
                status VARCHAR(20),
                order_time VARCHAR(30),
                pay_time VARCHAR(30)
            )
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {config.CONVERSATION_TABLE} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(36),
                question TEXT,
                answer TEXT,
                timestamp DATETIME,
                INDEX idx_session_id (session_id)
            )
        """)
        self.conn.commit()
        logger.info("MySQL 表初始化成功")

    def insert_faq_batch(self, rows):
        cur = self.conn.cursor()
        cur.executemany(
            f"INSERT INTO {config.FAQ_TABLE} (category, question, answer) VALUES (%s, %s, %s)",
            rows,
        )
        self.conn.commit()
        logger.info(f"MySQL 插入 FAQ {len(rows)} 条")

    def fetch_questions(self):
        cur = self.conn.cursor()
        cur.execute(f"SELECT question FROM {config.FAQ_TABLE}")
        return [r[0] for r in cur.fetchall()]

    def fetch_answer(self, question):
        cur = self.conn.cursor()
        cur.execute(f"SELECT answer FROM {config.FAQ_TABLE} WHERE question = %s", (question,))
        row = cur.fetchone()
        return row[0] if row else None

    def insert_order_batch(self, orders):
        cur = self.conn.cursor()
        cols = ", ".join(ORDER_FIELDS)
        placeholders = ", ".join(["%s"] * len(ORDER_FIELDS))
        rows = [[o.get(f) for f in ORDER_FIELDS] for o in orders]
        cur.executemany(f"INSERT INTO {config.ORDERS_TABLE} ({cols}) VALUES ({placeholders})", rows)
        self.conn.commit()
        logger.info(f"MySQL 插入订单 {len(rows)} 条")

    def lookup_order(self, order_id=None, phone=None, sku_id=None, product_keyword=None, limit=20):
        where, params = [], []
        if order_id:
            where.append("order_id = %s"); params.append(order_id)
        if phone:
            where.append("buyer_phone_masked = %s"); params.append(phone)
        if sku_id:
            where.append("sku_id = %s"); params.append(sku_id)
        if product_keyword:
            where.append("product_name LIKE %s"); params.append(f"%{product_keyword}%")
        sql = f"SELECT {', '.join(ORDER_FIELDS)} FROM {config.ORDERS_TABLE}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY id DESC LIMIT {int(limit)}"
        cur = self.conn.cursor()
        cur.execute(sql, params)
        return [dict(zip(ORDER_FIELDS, row)) for row in cur.fetchall()]

    def insert_conversation(self, session_id, question, answer):
        cur = self.conn.cursor()
        cur.execute(
            f"INSERT INTO {config.CONVERSATION_TABLE} (session_id, question, answer, timestamp) "
            "VALUES (%s, %s, %s, NOW())",
            (session_id, question, answer),
        )
        self.conn.commit()

    def fetch_recent_conversations(self, session_id, limit=5):
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT question, answer FROM {config.CONVERSATION_TABLE} "
            "WHERE session_id = %s ORDER BY id DESC LIMIT %s",
            (session_id, int(limit)),
        )
        rows = [{"question": r[0], "answer": r[1]} for r in cur.fetchall()]
        return rows[::-1]

    def clear_conversations(self, session_id):
        cur = self.conn.cursor()
        cur.execute(f"DELETE FROM {config.CONVERSATION_TABLE} WHERE session_id = %s", (session_id,))
        self.conn.commit()
        return True

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


def create_store():
    """按配置返回存储后端。"""
    if config.LOCAL_MODE:
        logger.info("使用本地 SQLite 存储")
        return SQLiteStore(config.SQLITE_DB)
    logger.info("使用远程 MySQL 存储")
    return MySQLStore()
