#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: app.py
作者: ZZS
项目: 3_代码
创建日期: 2026/7/18
描述: 
"""
# 导入 FastAPI 相关模块，用于构建 API 和 WebSocket
from fastapi import FastAPI, WebSocket, HTTPException, Query, Depends
# 导入 FastAPI 响应类型，用于流式响应和文件服务
from fastapi.responses import StreamingResponse, FileResponse
# 导入 CORS 中间件，支持跨域请求
from fastapi.middleware.cors import CORSMiddleware
# 导入静态文件服务模块
from fastapi.staticfiles import StaticFiles
# 导入 WebSocket 断开异常
from starlette.websockets import WebSocketDisconnect
# 导入系统操作模块，用于文件目录管理
import os
# 导入 Pydantic 模型，用于请求验证
from pydantic import BaseModel
# 导入异步事件循环模块
import asyncio
# 导入 JSON 处理模块
import json
# 导入 UUID 模块，生成唯一会话 ID
import uuid
# 导入类型注解模块
from typing import Optional, List, Dict, Any
# 导入时间模块，记录处理时间
import time
# 导入正则表达式模块，用于匹配日常问候
import re
# 导入优化后的问答系统
from new_main import IntegratedQASystem

# =====================================================================
# 修复：Clash Verge 的 TUN 模式会拦截 127.0.0.1 回环流量，导致
# asyncio 的「自管道」(self-pipe) 无法建立 —— uvicorn 启动时永远卡死。
# 原因：Windows 下 socket.socketpair() 用 127.0.0.1 回环 TCP 实现，
# 回环被拦截后 connect/accept 超时。这里在回环不可用时，自动改用
# 一个「本机能连到自身」的地址来构造 socketpair，绕过坏掉的回环。
# （治本方法仍是关闭代理的 TUN 模式；此补丁让服务不依赖回环也能起。）
# =====================================================================
import socket as _socket


def _self_connect_ok(ip, timeout=1.5):
    """测试本机能否连到自己的 ip（能 = 该地址可用作 socketpair）。"""
    ls = cs = ss = None
    try:
        ls = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        ls.settimeout(timeout)
        ls.bind((ip, 0))
        ls.listen(1)
        cs = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        cs.settimeout(timeout)
        cs.connect(ls.getsockname())
        ss, _ = ls.accept()
        return True
    except Exception:
        return False
    finally:
        for s in (cs, ss, ls):
            if s:
                try:
                    s.close()
                except Exception:
                    pass


if not _self_connect_ok("127.0.0.1"):
    for _info in _socket.getaddrinfo(_socket.gethostname(), None, _socket.AF_INET):
        _ip = _info[4][0]
        if _ip != "127.0.0.1" and _self_connect_ok(_ip):
            # socket.py 的 _fallback_socketpair 引用模块级 _LOCALHOST
            _socket._LOCALHOST = _ip
            print(f"[app] 回环 127.0.0.1 不可用，已改用本机地址 {_ip} 构造事件循环自管道")
            break

# 创建 FastAPI 应用实例，设置标题和描述
app = FastAPI(title="问答系统API", description="集成MySQL和RAG的智能问答系统")

# 配置 CORS 中间件，允许跨域请求
# 需要增加一下add_middleware，否则前后端不能链接。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（生产环境需限制）
    allow_credentials=True,  # 允许凭证
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有头部
)

# 创建静态文件目录（如果不存在）
os.makedirs("static", exist_ok=True)

# 创建全局问答系统实例
qa_system = IntegratedQASystem()

# 定义日常问候用语模式和回复（电商客服文案）
GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|hello)",  # 匹配问候语
        "response": "您好！我是平台智能客服，可以帮您查询商品、订单和售后政策，很高兴为您服务！"
    },
    {
        "pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)",  # 匹配身份询问
        "response": "我是平台智能客服，为您提供商品咨询、订单查询和售后服务解答。"
    },
    {
        "pattern": r"^(在吗|在不在|有人吗)",  # 匹配在线确认
        "response": "我在！我是平台智能客服，随时为您解答商品、订单和售后问题！"
    },
    {
        "pattern": r"^(干嘛呢|你在干嘛|做什么)",  # 匹配状态询问
        "response": "我正在待命，随时为您解答商品咨询、订单查询和售后政策相关的问题！有什么我可以帮您的？"
    }
]

# 定义查询请求模型
class QueryRequest(BaseModel):
    query: str  # 查询内容，必填
    source_filter: Optional[str] = None  # 学科过滤，可选
    session_id: Optional[str] = None  # 会话 ID，可选

# 定义查询响应模型
class QueryResponse(BaseModel):
    answer: str  # 答案内容
    is_streaming: bool  # 是否流式响应
    session_id: str  # 会话 ID
    processing_time: float  # 处理时间


# 挂载静态文件目录，服务前端文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 根路径重定向到 index.html
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

# 创建新会话接口
@app.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())  # 生成唯一会话 ID
    return {"session_id": session_id}  # 返回会话 ID

# 查询历史消息接口
@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    try:
        # 获取指定会话的历史记录
        history = qa_system.get_session_history(session_id)
        # 返回会话 ID 和历史记录
        return {"session_id": session_id, "history": history}
    except Exception as e:
        # 抛出 HTTP 异常，包含错误信息
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {str(e)}")

# 清除历史消息接口
@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    # 清除指定会话的历史记录
    success = qa_system.clear_session_history(session_id)
    if success:
        # 返回成功状态
        return {"status": "success", "message": "历史记录已清除"}
    else:
        # 抛出 HTTP 异常
        raise HTTPException(status_code=500, detail="清除历史记录失败")

# 检查是否为日常问候用语并返回模板回复
def check_greeting(query: str) -> Optional[str]:
    query_text = query.strip()  # 去除首尾空格
    for pattern_info in GREETING_PATTERNS:
        # 使用正则匹配，忽略大小写
        if re.match(pattern_info["pattern"], query_text, re.IGNORECASE):
            return pattern_info["response"]  # 返回匹配的回复
    return None  # 无匹配返回 None

# 非流式查询接口
@app.post("/api/query")
async def query(request: QueryRequest):
    start_time = time.time()  # 记录开始时间
    # 使用请求中的 session_id 或生成新 ID
    session_id = request.session_id or str(uuid.uuid4())
    # 检查是否为日常问候
    greeting_response = check_greeting(request.query)
    if greeting_response:
        # 返回问候回复
        return {
            "answer": greeting_response,
            "is_streaming": False,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }
    # 执行 BM25 搜索
    #  todo FQA 缓存检索。
    answer, need_rag = qa_system.bm25_search.search(request.query, threshold=0.85)
    if need_rag:
        # 需要 RAG，提示使用 WebSocket
        # todo 如果需要rag，就会返回 "is_streaming": True，前端就调用 /api/stream 接口
        #  如果不需要rag，直接返回答案。
        return {
            "answer": "请使用WebSocket接口获取流式响应",
            "is_streaming": True,
            "session_id": session_id,
            "processing_time": time.time() - start_time
        }
    # 返回 MySQL 答案
    return {
        "answer": answer,
        "is_streaming": False,
        "session_id": session_id,
        "processing_time": time.time() - start_time
    }

# 流式查询 WebSocket 接口
@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # 接受 WebSocket 连接
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            request_data = json.loads(data)  # 解析 JSON 数据
            # 获取查询参数
            query = request_data.get("query")
            source_filter = request_data.get("source_filter")
            session_id = request_data.get("session_id", str(uuid.uuid4()))
            start_time = time.time()  # 记录开始时间
            # 发送开始标志
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.send_json({
                    "type": "start",
                    "session_id": session_id
                })
            # 检查是否为日常问候
            greeting_response = check_greeting(query)
            if greeting_response:
                if websocket.client_state == websocket.client_state.CONNECTED:
                    # 发送问候回复
                    await websocket.send_json({
                        "type": "token",
                        "token": greeting_response,
                        "session_id": session_id
                    })
                    # 发送结束标志
                    await websocket.send_json({
                        "type": "end",
                        "session_id": session_id,
                        "is_complete": True,
                        "processing_time": time.time() - start_time
                    })
                break
            # 调用问答系统，流式处理查询
            collected_answer = ""
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                collected_answer += token  # 累积答案
                if is_complete and not collected_answer:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        # 发送结束标志
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break
                if token and websocket.client_state == websocket.client_state.CONNECTED:
                    # 发送 token 数据
                    await websocket.send_json({
                        "type": "token",
                        "token": token,
                        "session_id": session_id
                    })
                if is_complete:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        # 发送结束标志
                        await websocket.send_json({
                            "type": "end",
                            "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time
                        })
                    break
                await asyncio.sleep(0.01)  # 控制流式输出的速度
    except WebSocketDisconnect as e:
        # 记录 WebSocket 断开信息
        print(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    except Exception as e:
        # 记录错误信息
        print(f"WebSocket error: {str(e)}")
        if websocket.client_state == websocket.client_state.CONNECTED:
            # 发送错误消息
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
    finally:
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                # 关闭 WebSocket 连接
                await websocket.close()
        except Exception as e:
            # 记录关闭连接时的错误
            print(f"Error closing WebSocket: {str(e)}")

# 健康检查接口
@app.get("/health")
async def health_check():
    return {"status": "healthy"}  # 返回健康状态

# 获取有效学科类别接口
@app.get("/api/sources")
async def get_sources():
    return {"sources": qa_system.config.VALID_SOURCES}  # 返回学科类别列表

# 主程序入口
if __name__ == "__main__":

    import uvicorn

    import os
    # 从环境变量获取主机和端口，默认值为 0.0.0.0:8080
    # todo 如果你系统中没有环境变量，HOST，PORT,默认就行。
    #  如果有的话直接写死。
    # host = os.getenv('HOST', '0.0.0.0')
    # port = int(os.getenv('PORT', 18080))
    host = '0.0.0.0'
    port = 18080

    # 运行 FastAPI 应用，监听指定的主机和端口
    # 注意：直接传 app 对象（而非 "app:app" 字符串），避免 uvicorn 二次导入模块、
    # 触发 IntegratedQASystem 重复初始化（BGE-M3/重排器/BERT 加载两遍导致卡死）。
    uvicorn.run(app, host=host, port=port, reload=False)