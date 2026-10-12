# 智能客服 — 数据清单

> 生成时间：2026-09-17
> 说明：本目录存放「智能客服」项目的所有业务数据。开源数据已下载并校验，剩余三类业务数据（商品库/订单/售后政策）由代码合成/构造。

## 目录结构

```
znkf/
├── 00_原始下载_raw/                 # （暂空）预留：需手动下载的原始数据
├── 01_商品库_products/              # 商品 SKU 知识库（✅ 已合成 10,008 条）
├── 02_订单_orders/                  # 订单数据（✅ 已合成 3,000 条，大额 913 条）
├── 03_售后政策_after_sales_policy/  # 各平台退换货/保修政策（✅ 已构造 160 规则 + 5 份文档）
├── 04_FAQ问答对_faq/                # FAQ 问答对（✅ 已合成 1,808 条 + 开源样例）
├── 05_客服对话_多轮_dialog/         # 多轮客服对话（开源，已下载）
└── 06_意图情绪标注_intent_emotion/  # 意图分类 / 情绪策略标注（开源，已下载）
```

## 已下载的开源数据

### 1. DianJin/DianJin-CSC-Data（中文客服对话，带情绪策略标注）✅ 核心
- 来源：https://huggingface.co/datasets/DianJin/DianJin-CSC-Data
- 文件：
  - `data/CSConv.json` — 1,855 条对话（客服/客户角色 + 18 种情绪/服务策略标注）
  - `data/RoleCS.json` — 11,232 条对话（同上，规模更大）
- 用途：多轮对话语料；「强情绪检测」训练/规则参考（策略标注含礼貌问候、确认身份、情绪安抚、转移升级等）

### 2. rescommons/Full-Ecom-Chatbot-Dataset（电商智能体数据，含工具调用与意图）✅ 核心
- 来源：https://huggingface.co/datasets/rescommons/Full-Ecom-Chatbot-Dataset
- 文件：
  - `data/train-00000-of-00001.parquet` — 35,213 条
  - `data/test-00000-of-00001.parquet` — 约 9 千条
- 字段亮点：`prompt`、`history`、`tools`(函数定义)、`response_type`(tool_call)、
  `intent_category`(如 returns_exchanges)、`intent`、`sub_intent`、`domain`、`context`(含 order_details)
- 用途：**多 Agent 编排 / 工具调用结构的现成参考**；意图分类（订单/售后/退换货）路由的标注参考
- 注意：为英文，主要借鉴其「工具定义 + 意图 + 订单上下文字段」的结构设计，中文业务数据用合成补齐

### 3. OpenStellarTeam/Chinese-EcomQA（中文电商品牌/类目问答）✅
- 来源：https://huggingface.co/datasets/OpenStellarTeam/Chinese-EcomQA
- 文件：`ChineseEcomQA.jsonl` — 1,800 条（品牌知识问答、类目问答、意图问答）
- 用途：商品/品牌知识库的问答样式参考；FAQ 样例

### 4. Andyrasika/Ecommerce_FAQ（电商 FAQ 小样本）
- 来源：https://huggingface.co/datasets/Andyrasika/Ecommerce_FAQ
- 文件：`train.json` — 79 条（账户/订单/支付/物流/退货 FAQ）
- 用途：FAQ 快速通道的字段格式参考（量小，仅作样例）

## 未能直接下载的数据（供记录，不影响进度）

| 数据集 | 原因 |
|---|---|
| THU-StarLab/CustomerService（8.7万条客服QA） | HF gated 仓库，需登录授权；且为电信域，非电商 |
| Mix-Ecom（售前/物流/售后 + 82条规则） | 论文称"将公开"，实际尚未发布下载链接 |
| CCSE-CS（9478条情绪对话） | 仓库只含代码，数据未公开 |
| EcomGPT/EcomInstruct（250万条指令） | 仓库只含 12 个评测集（各500条），全量数据在 ModelScope |
| 淘宝客服对话（100万） | heywhale 需登录 |
| JDDC（京东102万对话） | 需签署协议，仅限学术 |

## 已合成/构造的数据（✅ 完成）

1. **商品库**（01_商品库_products/products.jsonl）：10,008 条 SKU，字段：sku_id/商品名/品牌/类目/规格/价格/平台/参数/卖点/保修/是否7天无理由/库存/text（序列化描述，可直接做父子切分）
2. **订单数据**（02_订单_orders/orders.jsonl）：3,000 条，字段：订单号/平台/买家(脱敏)/城市/sku_id/商品名/数量/单价/金额/是否大额/状态/下单时间/支付时间；大额订单 913 条（金额 5005~277,646 元，中位 11,192）
3. **售后政策**（03_售后政策_after_sales_policy/）：policy_rules.jsonl 160 条规则（5 平台 × 8 类通用规则 + 25 类目 × 5 平台差异化规则）+ policy_docs/ 5 份平台 Markdown 文档，可直接切分到 ~2000 chunk
4. **FAQ 问答对**（04_FAQ问答对_faq/faq.jsonl）：1,808 条（商品 FAQ + 政策 FAQ + 订单 FAQ），带 sku_id/类目标签，供 BM25 快速通道用

> 生成脚本：scripts/generate_all_data.py（deterministic，random.seed 固定，可复现）
