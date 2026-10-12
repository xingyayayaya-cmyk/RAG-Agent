# INTEGRATED_QA_SYSTEM 改造工作方案

**——Plan Mode 探查计划与四块工作拆解**

| 项目 | 内容 |
|---|---|
| 文档版本 | v1.0 |
| 日期 | 2026-09-17 |
| 前置状态 | 业务数据闭环完成（4 个开源数据集 + 913 条大额订单已生成） |
| 本轮目标 | 进入系统本体改造：意图路由、多 Agent 编排、升级 Agent、数据接入 |
| 当前状态 | **建议先进 Plan Mode 探查**；本文档为探查计划与方案框架 |
| 关联文档 | 改造方案 / 数据源调研 / 数据下载完成报告（均在桌面） |

---

## 0. 结论：同意进 Plan Mode，但范围要收窄

改造涉及多文件、有架构取舍，**先探再动是对的**。但 Plan Mode 不必把整个项目读完——`integrated_qa_system` 里大量是教育域数据和已废弃的旧版代码（`旧版old_rag_system.py`、`旧版query_classifier_origin.py`、`old_main.py`），读它们不产生决策价值。

**建议探查范围收窄到 8 个文件**（见第 4 节），读完即可出改造方案。

---

## 1. 现有代码结构勘察结果

> 这一节是我实际扫过项目目录后记下的，不是推测。后续讨论以这里的实测值为准。

### 1.1 项目位置与副本问题

```
C:\Users\Administrator\Desktop\项目介绍\RAG&Agent项目代码\
├── integrated_qa_system8月29\     ← 副本 A
├── integrated_qa_system8月30\     ← 副本 B（较新，建议作基线）
├── SmartVoyage9月4日\             ← 另一个项目
└── SmartV_外部工具APIKEY.txt
```

⚠️ **第一个待确认项**：存在两份副本，且仅靠目录名的日期无法判断内容差异。需确认以 `8月30` 为基线，还是另有指定版本。**改造前建议先拷一份新目录（如 `ecs_qa_system`），不动原目录**——教育域那套还能跑，改坏了没得回退。

### 1.2 目录结构（`integrated_qa_system8月30`）

```
├── app.py                    # Flask + WebSocket /api/stream（11.6 KB）
├── new_main.py               # ★ 主流程：BM25 → RAG 两段式（15.8 KB）
├── old_main.py               # 旧版，可忽略
├── main.py                   # 入口占位
├── config.ini                # ★ 配置（Milvus/MySQL/Redis/LLM）
├── base/
│   ├── config.py             # ★ 配置读取（含 Key 读取逻辑，有已知问题）
│   └── logger.py
├── mysql_qa/
│   ├── db/mysql_client.py
│   ├── retrieval/bm25_search.py     # ★ BM25 快速通道（阈值 0.85）
│   ├── cache/                        # Redis 客户端
│   ├── main.py / utils/
├── rag_qa/
│   ├── core/
│   │   ├── vector_store.py          # ★ Milvus 混合检索 + Rerank
│   │   ├── new_rag_system.py        # ★ HyDE / 子查询 / 回溯
│   │   ├── strategy_selector.py     # ★ 检索策略选择
│   │   ├── query_classifier.py      # ★ BERT 二分类（需扩为四分类）
│   │   ├── document_processor.py    # ★ 文档解析 + 父子切分
│   │   ├── prompts.py               # ★ 提示词（需重写为电商域）
│   │   ├── bert_results/ tmp_trainer/
│   │   ├── 旧版old_rag_system.py / 旧版query_classifier_origin.py / 旧版rag_system.py   # 废弃
│   ├── models/                      # 本地模型（离线可用，很关键）
│   │   ├── bert-base-chinese
│   │   ├── bge-m3
│   │   ├── bge-reranker-large
│   │   ├── nlp_bert_document-segmentation_chinese-base
│   │   └── bert_query_classifier     # 已训好的二分类模型
│   ├── classify_data/
│   │   ├── model_generic_5000.json   # ★ BERT 训练数据 5000 条（2 分类）
│   │   └── 提示词模块.txt
│   ├── data/ai_data java_data        # 教育域语料，不涉及
│   ├── edu_document_loaders/ edu_text_spliter/
│   ├── rag_assessment/
│   │   ├── ragas_evaluate.py
│   │   ├── rag_evaluate_data.json / _small.json   # ★ 需换域重建
│   │   └── ragas_evaluation_results.xlsx           # 教育域基线，保留作对照
│   └── main.py
├── data/JP学科知识问答.csv
└── static/index.html                # ★ 前端（需新增 Agent 身份/订单卡片/转人工）
```

### 1.3 config.ini 实测值 —— 发现 4 个坑

| 项 | 实测值 | 问题 |
|---|---|---|
| MySQL | `192.168.21.226`，库 `subjects_kg` | 库名是教育域，电商需新库或新表前缀 |
| Redis 端口 | **16379**（不是 6379） | 之前文档里写的 6379 是错的，连通性检测要改 |
| Milvus | `192.168.21.226:19530`，库 `milvus_demo`，**collection `edurag`** | 务必新建 collection，不可复用 `edurag` |
| LLM | model `deepseek-v4-pro`，base_url `https://api.deepseek.com/anthropic` | ⚠️ **`/anthropic` 路径可疑**，DeepSeek 官方是 `https://api.deepseek.com` 或 `/v1`。这个地址大概率是走 Anthropic 兼容网关，需实测能否调通 |
| API Key | 明文写死在 `config.ini`（`sk-dd6e…44`） | ⚠️ **不能入 git**；建议改环境变量优先读取 |
| 日志路径 | 写死 `D:\heima123\3_代码\...` | 本机无此盘符则日志直接报错，需改 |
| `customer_service_phone` | `10086` | 就是那个"静态客服电话"，升级 Agent 要替换它 |
| `valid_sources` | `["ai","java","test","ops","bigdata"]` | 教育域学科过滤，需换成电商维度（平台 / 类目） |

> **好消息**：`vector_store.py` 的 collection 名是从 `config.MILVUS_COLLECTION_NAME` 读的（`vector_store.py:56`）。所以换 collection **不用改代码，只改 config.ini**——数据隔离成本几乎为零。

### 1.4 主流程现状（`new_main.py`）

`IntegratedQASystem.query()`（215–266 行）是标准两段式：

```
query()
  ├─ bm25_search.search(query, threshold=0.85)   → (answer, need_rag)
  ├─ answer 命中    → yield(answer, True)          一次性返回
  ├─ need_rag      → rag_system.generate_answer(...) → yield(token, False) 流式
  └─ 都无          → yield("未找到答案", True)
```

**改造要点**：这是 `yield (token, is_complete)` 的生成器协议，`app.py` 的 WebSocket 依赖它。多 Agent 改造必须**保持这个协议不变**，只在内部插入路由层和风控层——否则前端和 `app.py` 全要重写。

---

## 2. 四块工作拆解

### 2.1 意图路由扩展

| 项 | 现状 | 目标 |
|---|---|---|
| 分类器 | `query_classifier.py`，BERT 二分类（通用知识 / 专业咨询），训练数据 `model_generic_5000.json` | 三/四路：订单 / 售前 / 售后（+ 通用） |
| 主流程 | `new_main.py` 的 BM25 → RAG | 规则优先 → BERT 分类 → LLM fallback 三级 |
| 基座 | `bert-base-chinese` 已在 `models/` | 可直接微调，无需下载 |

**三级路由设计**

1. **规则层**（零延迟）：订单号正则 `\d{15,20}`、"我的订单/物流/退款"、金额+商品组合 → 订单意图
2. **BERT 层**（毫秒级）：微调 `bert-base-chinese`，输出 4 类概率；低置信度（<0.6）转 LLM
3. **LLM fallback**：用 DeepSeek 做最终判定，兜住规则与 BERT 都拿不准的长尾

**待确认**：是三分类（订单/售前/售后）还是四分类（+ 通用）？我倾向**四分类**——"你好"、"你是谁"这类必须有个去处，否则会被硬塞进业务 Agent。

### 2.2 多 Agent 编排 + 权限隔离

**新增目录**

```
agent/
  base.py             # 子 Agent 基类，构造时静态绑定权限
  orchestrator.py     # 主 Agent：只做路由，不持有检索器
  order_agent.py      # 订单
  presale_agent.py    # 售前
  aftersale_agent.py  # 售后
  general_agent.py    # 通用
  escalation_agent.py # 升级
registry/
  permissions.py      # agent_id → {retrievers, tools, visible_fields, mask_rules}
adapters/
  product.py order.py policy.py    # mock / real 同接口，DATA_SOURCE 切换
```

**权限矩阵**

| Agent | 数据源 | 工具 | 禁止 |
|---|---|---|---|
| 订单 | 订单库 | `order.get_by_id` `order.list_by_user` `order.get_logistics` | 商品成本、政策裁决、**非本人订单** |
| 售前 | 商品库 | `product.search` `product.get_price_stock` | 任何订单数据、售后政策 |
| 售后 | 政策库 + 工单 | `policy.match` `ticket.create` `refund.calc` | 其他用户订单、成本利润 |
| 通用 | FAQ | `faq.match` `kb.search` | 任何用户级数据 |
| 主 Agent | — | — | **不持有任何检索器** |

**两条硬约束**
1. 权限在 Agent **构造时静态注入**（白名单），不做运行时 if-else 判断
2. 出口统一过 `field_mask` 脱敏（手机号 `138****1234`、姓名 `张**`、地址到区县）

### 2.3 升级 Agent

独立于意图识别，**与路由并行**跑，命中即短路到转人工。

| 信号 | 来源 |
|---|---|
| 强情绪 | 规则关键词表 + LLM 分级（0–3，≥2 触发）；用 **DianJin 18 种情绪策略语料**校准，其中「情绪安抚」「转移升级」两类标签可直接映射升级触发 |
| 大额 | **913 条大额订单**已生成，用于确定阈值分布与验证命中率 |
| 显式 | `转人工` / `人工客服` 直通 |

**落地**：写 `tickets` 表 + 前端提示工单号（MVP）；替换掉 `config.ini` 里写死的 `customer_service_phone = 10086`。

**待确认**：金额阈值数字（默认 ¥2000）；转人工是仅工单表还是对接现有系统。

### 2.4 数据接入

| 数据 | 落点 | 动作 |
|---|---|---|
| 商品库 | Milvus 新 collection（建议 `ecommerce_kb`）+ `adapters/product.py` | 走 `document_processor.py` 父子切分入库 |
| 售后政策 | Milvus 同 collection 独立 partition | 同上 |
| 订单 | MySQL 新库/新表 + `adapters/order.py` | 建 `orders` 表，含金额字段 |
| FAQ | MySQL `faq` 表 + `bm25_search.py` | 从商品/政策派生，保证与知识库自洽 |
| 评测集 | `rag_assessment/rag_evaluate_data.json` | 换域重建；**保留教育域 xlsx 作对照基线** |

**注意**：`config.ini` 里 MySQL 库名是 `subjects_kg`（教育域）。建议**新建库 `ecommerce_kb`**，而不是往教育库里塞表。

---

## 3. 需要你拍板的 5 个架构取舍

| # | 取舍 | 选项 A | 选项 B | 我的倾向 |
|---|---|---|---|---|
| 1 | 代码基线 | 在 `8月30` 上直接改 | **拷新目录 `ecs_qa_system` 再改** | B（保住可跑的教育域版本） |
| 2 | 意图分类数 | 三分类 | **四分类（+通用）** | B |
| 3 | 权限实现 | 运行时 if-else 判断 | **构造时静态注入白名单** | B（可审计、防越权） |
| 4 | MySQL 库 | 复用 `subjects_kg` 加表 | **新建 `ecommerce_kb` 库** | B |
| 5 | 改造深度 | 最小改动跑通 | **分层改造，保留生成器协议** | B（前端与 app.py 不重写） |

---

## 4. Plan Mode 探查清单（建议只读这 8 个文件）

| 优先级 | 文件 | 要确认什么 |
|---|---|---|
| P0 | `new_main.py` | 已读。`query()` 生成器协议、BM25→RAG 接缝在哪插路由 |
| P0 | `base/config.py` | Key 读取逻辑（已知问题在哪）、如何加新配置项 |
| P0 | `rag_qa/core/query_classifier.py` | 二分类模型加载方式、能否复用基座扩为四分类 |
| P0 | `mysql_qa/retrieval/bm25_search.py` | 表结构依赖、如何加新 FAQ 源 |
| P1 | `rag_qa/core/vector_store.py` | 已扫。collection 从 config 读，切换成本低；混合检索入参 |
| P1 | `rag_qa/core/new_rag_system.py` | HyDE/子查询入口，子 Agent 如何复用 |
| P1 | `app.py` | WebSocket 协议字段、如何扩展 `agent_id`/`card`/`handoff` |
| P1 | `static/index.html` | 现有消息渲染逻辑，改造量评估 |

**不需要读**：`old_main.py`、`旧版*.py`、`data/`、`rag_qa/data/`、`edu_*` 旧模块、`logs/`、`__pycache__`。

**探查期需实测的 3 件事**
1. `192.168.21.226` 的 3306 / **16379** / 19530 是否通
2. `https://api.deepseek.com/anthropic` 这个 base_url 能否调通（我怀疑不行）
3. `subjects_kg` 库、`edurag` collection 当前数据量（评估迁移影响）

---

## 5. 分阶段实施与验收标准

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| **P1 探查** | 读 8 个文件 + 3 项实测 | 出一份改造方案文档，含文件级改动清单 |
| **P2 骨架** | 新建 `agent/` `registry/` `adapters/`，空跑通 | `python -c "from agent.orchestrator import Orchestrator"` 不报错 |
| **P3 路由** | 三级路由接入 `new_main.py` | 规则/BERT/LLM 三级可单独开关，日志能看到路由决策 |
| **P4 子 Agent** | 3 个子 Agent + 权限白名单 | 越权调用抛异常且有日志；脱敏字段正确 |
| **P5 升级 Agent** | 情绪 + 金额检测 | 913 条大额订单跑一遍，输出命中率与误报率 |
| **P6 数据接入** | 4 份数据入 Milvus/MySQL/BM25 | 各源可检索，`edurag` 未被污染 |
| **P7 前端** | Agent 身份 / 订单卡片 / 转人工 | WebSocket 端到端可演示 |
| **P8 评测** | 换域 RAGAS + EcomGPT 9 个中文任务 | 出评测报告，与教育域基线对比 |

---

## 6. 风险

| 风险 | 应对 |
|---|---|
| 两份副本，改错目录 | 先确认基线；改造副本不动原目录 |
| 误污染 `edurag` | 新建 collection，旧库只读；`config.ini` 加注释警示 |
| API Key 明文写死 | 改环境变量优先读取；`.gitignore` 拦截 |
| base_url `/anthropic` 调不通 | P1 阶段实测，不通则换 `https://api.deepseek.com` |
| 改造破坏生成器协议，前端全崩 | 保持 `yield (token, is_complete)` 不变，内部插层 |
| 日志路径 `D:\heima123` 本机不存在 | 改为相对路径或环境变量 |
| 四分类训练数据不足 | `model_generic_5000.json` 作参考，每类 ≥500 条；冷启动用规则+LLM 兜底 |

---

## 7. 待确认回执

| # | 事项 | 你的决定 |
|---|---|---|
| 1 | 代码基线 | ☐ `8月30` 直接改 ☐ **拷新目录再改** |
| 2 | 意图分类 | ☐ 三分类 ☐ **四分类** |
| 3 | 环境 `192.168.21.226` | ☐ 可用 ☐ 不可用（新地址：______） |
| 4 | DeepSeek base_url | ☐ 用现有 `/anthropic` ☐ 换成官方地址 |
| 5 | 金额阈值 | `¥______`（默认 2000） |
| 6 | 是否进 Plan Mode | ☐ 开始探查 ☐ 先看某份生成数据 |
