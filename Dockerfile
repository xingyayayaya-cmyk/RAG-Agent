FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

# CPU 版 torch（体积远小于默认 CUDA 版，容器内推理走 CPU）
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p logs data

EXPOSE 18080

# 首次启动自动下载 BGE-M3 / bge-reranker-large / bert-base-chinese 到 rag_qa/models
ENTRYPOINT ["python", "scripts/entrypoint.py"]