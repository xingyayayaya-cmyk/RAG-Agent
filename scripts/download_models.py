#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: download_models.py
描述: 下载 RAG 检索所需的 HuggingFace 模型快照到 rag_qa/models/ 下。
     对应代码中的加载路径（见 rag_qa/core/local_vector_store.py / vector_store.py / query_classifier.py）：

       rag_qa/models/bge-m3              <- BAAI/bge-m3               （稠密+稀疏 embedding，排除 onnx 省 2GB+）
       rag_qa/models/bge-reranker-large  <- BAAI/bge-reranker-large   （重排）
       rag_qa/models/bert-base-chinese   <- google-bert/bert-base-chinese（查询分类微调底座）

     bert_query_classifier 为本地微调产物，无法从 Hub 下载；
     缺失时代码自动回退到 bert-base-chinese（见 query_classifier.load_model）。

用法:
    python scripts/download_models.py
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_models.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "rag_qa", "models")

# repo_id -> (本地目录名, ignore_patterns)
MODELS = {
    "BAAI/bge-m3": ("bge-m3", ["onnx/*", "*.onnx", "*.onnx_data"]),
    "BAAI/bge-reranker-large": ("bge-reranker-large", ["onnx/*", "*.onnx"]),
    "google-bert/bert-base-chinese": ("bert-base-chinese", ["tf_model.h5", "flax_model.msgpack"]),
}


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("缺少 huggingface_hub，请先: pip install huggingface_hub")

    os.makedirs(MODELS_DIR, exist_ok=True)
    total_ok = True
    for repo_id, (local_name, ignores) in MODELS.items():
        local_path = os.path.join(MODELS_DIR, local_name)
        if os.path.isdir(local_path) and any(
            f.endswith((".bin", ".safetensors")) for f in os.listdir(local_path)
        ):
            print(f"[skip] {local_name} 已存在")
            continue
        print(f"[下载] {repo_id} -> {local_path} （忽略: {ignores}）")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_path,
                ignore_patterns=ignores,
            )
            print(f"[完成] {local_name}")
        except Exception as e:
            total_ok = False
            print(f"[失败] {repo_id}: {e}", file=sys.stderr)

    sys.exit(0 if total_ok else 1)


if __name__ == "__main__":
    main()
