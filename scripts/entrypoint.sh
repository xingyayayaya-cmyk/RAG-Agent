#!/usr/bin/env bash
# 容器入口：首次启动自动下载缺失模型，然后启动 FastAPI 服务
set -e
cd /app

need_model=0
for m in bge-m3 bge-reranker-large bert-base-chinese; do
  if [ ! -d "rag_qa/models/$m" ]; then
    need_model=1
    break
  fi
done

if [ "$need_model" -eq 1 ]; then
  echo "[entrypoint] 首次启动：下载 RAG 模型（BGE-M3 / bge-reranker-large / bert-base-chinese，约 4-5GB）..."
  echo "[entrypoint] 当前 HF_ENDPOINT=${HF_ENDPOINT:-<unset>}，国内可用 https://hf-mirror.com"
  python scripts/download_models.py || echo "[entrypoint] ⚠️ 模型下载失败：服务仍会启动，但向量检索功能不可用（可重启容器重试）"
else
  echo "[entrypoint] 模型已就绪"
fi

if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
  echo "[entrypoint] ⚠️ 未设置 DASHSCOPE_API_KEY 环境变量，LLM 调用将失败（docker-compose.yml 中配置）"
fi

exec python app.py
