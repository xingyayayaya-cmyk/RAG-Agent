#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""容器入口：首次启动时补全本地配置并下载模型，然后拉起 FastAPI 服务。"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_config():
    cfg = os.path.join(ROOT, "config.ini")
    example = os.path.join(ROOT, "config.example.ini")
    if not os.path.exists(cfg) and os.path.exists(example):
        shutil.copyfile(example, cfg)
        print("[entrypoint] config.ini 已由 config.example.ini 生成", flush=True)


def ensure_models():
    models_dir = os.path.join(ROOT, "rag_qa", "models")
    flag = os.path.join(models_dir, ".ready")
    if os.path.exists(flag):
        return
    print("[entrypoint] 首次启动：下载 BGE-M3 / bge-reranker-large / bert-base-chinese（约 4GB，仅需一次）...", flush=True)
    subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts", "download_models.py")])
    os.makedirs(models_dir, exist_ok=True)
    with open(flag, "w") as f:
        f.write("ok")


if __name__ == "__main__":
    ensure_config()
    ensure_models()
    os.chdir(ROOT)
    os.execvp(sys.executable, [sys.executable, "app.py"])