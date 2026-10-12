# syntax=docker/dockerfile:1
# 电商智能客服一体化镜像：FastAPI + BM25 + RAG（BGE-M3 稠密/稀疏混合检索 + bge-reranker 重排）+ 多 Agent 编排
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

# 基础工具 + CPU 版 PyTorch（避免默认拉 CUDA 版导致镜像膨胀数 GB）
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

# 先装依赖（充分利用镜像层缓存）
COPY requirements.txt ./
RUN pip install -r requirements.txt

# 拷贝项目代码与种子数据
COPY . .

# 模型目录 / 日志目录 / 数据目录（模型首次启动自动下载，见 scripts/entrypoint.sh）
RUN mkdir -p /app/rag_qa/models /app/logs /app/data

EXPOSE 18080

ENTRYPOINT ["bash", "scripts/entrypoint.sh"]
