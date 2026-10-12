#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 RAG 所需模型到 rag_qa/models/，与代码中的本地模型路径一一对应。

支持 HF_ENDPOINT 环境变量切换镜像（如 https://hf-mirror.com）。
"""
import os

from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "rag_qa", "models")

MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "bge-reranker-large": "BAAI/bge-reranker-large",
    "bert-base-chinese": "google-bert/bert-base-chinese",
}


def main():
    for local_name, repo_id in MODELS.items():
        target = os.path.join(MODELS_DIR, local_name)
        if os.path.isdir(target) and os.listdir(target):
            print(f"[skip] {local_name} 已存在", flush=True)
            continue
        print(f"[download] {repo_id} -> {target}", flush=True)
        snapshot_download(repo_id=repo_id, local_dir=target)
    print("[done] 全部模型就绪", flush=True)


if __name__ == "__main__":
    main()