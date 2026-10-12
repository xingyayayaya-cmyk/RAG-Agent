# 电商智能客服系统（E-Commerce Smart Customer Service）

基于 **FastAPI + RAG 双引擎 + 多 Agent 编排** 的电商智能客服：BM25 FAQ 快速通道、BGE-M3 混合检索 + bge-reranker-large 重排、意图路由 + 售前/售后/订单子 Agent + 大额订单升级人工，支持流式输出与本地降级（SQLite + 本地向量库，无外部依赖可跑）。

![客服 Copilot 工作台](docs/images/copilot-workbench.png)

## 技术栈

| 层 | 技术 |
| --- | --- |
| Web 框架 | FastAPI + WebSocket + SSE 流式输出 |
| LLM | DeepSeek（OpenAI 兼容接口） |
| RAG | BGE-M3（稠密+稀疏混合检索）+ bge-reranker-large 重排 + 父子分块 |
| FAQ 通道 | BM25（rank-bm25 + jieba），命中阈值直返，未命中回退 RAG |
| 多 Agent | 意图路由 Router + 售前/售后/订单子 Agent + 升级 Agent + Orchestrator |
| 存储 | 远程：MySQL + Milvus + Redis；本地降级：SQLite + numpy 本地向量库 + 内存缓存 |
| 查询分类 | BERT 二分类（通用知识 / 专业咨询），支持微调训练与原始模型降级 |
| 评估 | RAGAS 评估脚本 + 端到端对比测试 |

## 目录结构

```
├── app.py / new_main.py      # FastAPI 入口 / 问答主系统（IntegratedQASystem）
├── base/                     # 配置、日志、存储/缓存抽象层（远程/本地可切换）
├── ecom_agent/               # 多 Agent：router / presale / aftersale / order / escalation / orchestrator
├── mysql_qa/                 # BM25 FAQ 快速通道
├── rag_qa/                   # RAG：向量库（Milvus/本地降级）、父子分块、查询分类、RAGAS 评估
├── ecom_data/ingest.py       # 数据灌库脚本（商品/政策/FAQ -> 向量库）
├── dataset/                  # 电商客服开源数据集（商品库/订单/售后政策/FAQ/多轮对话/意图情绪）
├── data/ecom.db              # 种子 SQLite 库（订单 + FAQ，开箱即用）
├── static/index.html         # 客服对话前端
├── tests/                    # 冒烟 / 路由 / 升级 / LLM 端到端测试
└── docs/                     # 改造方案、数据源调研、演示页面、产品截图
```

## 快速开始（Docker，推荐）

前置：Docker 已安装；一个 DeepSeek API Key。

```bash
# 方式一：docker compose
export DASHSCOPE_API_KEY=sk-your-deepseek-api-key
docker compose up -d --build
# 打开 http://localhost:18080

# 方式二：单容器
docker build -t ecom-cs .
docker run -d -p 18080:18080 -e DASHSCOPE_API_KEY=sk-your-deepseek-api-key \
  -v ecom-models:/app/rag_qa/models -v ecom-data:/app/data ecom-cs
```

首次启动会自动从 HuggingFace 下载 BGE-M3 / bge-reranker-large / bert-base-chinese（约 4GB，仅需一次，持久化在 volume 中）。国内网络已在 Dockerfile 内置 `HF_ENDPOINT=https://hf-mirror.com` 镜像。

### 灌库（构建向量索引）

种子库已带订单与 FAQ（BM25 通道开箱可用）。若要启用 RAG 知识库检索：

```bash
docker compose exec app python ecom_data/ingest.py --limit 500   # 先小批量验证
docker compose exec app python ecom_data/ingest.py               # 全量灌入
```

## 本地运行

```bash
pip install -r requirements.txt   # torch 建议装 CPU 版：pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/download_models.py # 下载模型到 rag_qa/models/
cp config.example.ini config.ini  # 填入 DASHSCOPE_API_KEY（或用环境变量同名覆盖）
python app.py                     # http://localhost:18080
```

## 配置说明

- `local_mode = true`（默认）：SQLite + 本地向量库 + 内存缓存，无需 MySQL/Milvus/Redis。
- `local_mode = false`：连接远程 MySQL + Milvus + Redis（config.ini 中配置）。
- 敏感项（API Key 等）优先从环境变量读取：`DASHSCOPE_API_KEY`、`MILVUS_HOST`、`MILVUS_PORT` 等。

## 数据说明

`dataset/` 收录了五类电商客服开源数据（详见 `dataset/README_数据清单.md`）：商品库、订单、各平台售后政策、FAQ 问答对（EcomGPT 等）、多轮客服对话（dianjin / full-ecom-chatbot）、意图情绪标注（ChineseEcomQA）。

## 文档

- `docs/plans/`：改造方案、数据源调研与采集计划、数据下载报告
- `docs/智能客服最终形态.html`：产品形态演示页（浏览器直接打开）
- `docs/project-notes/新人培训指南.md`