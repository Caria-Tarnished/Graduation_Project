# 毕设开发计划（PLAN）

版本：0.1  
日期：2025-11-15  
范围：BERT 快讯情感 + 财报 RAG + Agent 调度 + Streamlit UI；云端 LLM 推理，训练在 Colab，本地 CPU 推理。

---

## 1. 目标与成功指标（Acceptance Criteria）

- 业务目标
  - 快讯情感分类（利好/中性/利空），用事件窗收益验证有效性。
  - 财报检索问答（RAG），返回引用与页码定位。
  - UI 集成对话、K线联动与报告预览。
- 技术约束
  - MX570 2GB：训练上云，推理本地 CPU；LLM 用云端 API。
- 成功指标（首版阈值，可迭代）
  - 分类：F1-macro ≥ 0.70；CPU 推理 ≤ 500ms/条；内存 ≤ 1GB。
  - 事件窗：30/60/120 分钟超额收益显著（t 检验 p < 0.05，至少一窗）。
  - RAG：Recall@5 ≥ 0.70；ragas ≥ 0.60；答案含引用页码。
  - UI：关键交互 ≤ 1.5s；首屏 ≤ 3s。

---

## 2. 架构与关键设计

- Engine A（快讯分类）
  - 数据：快讯文本 + 分钟级 K 线，交易日历与时区统一。
  - 代理标注：事件窗收益阈值上/下界；中性带避免噪声。
  - 模型：TF-IDF 基线 → BERT 微调（Colab）；本地量化推理。
- Engine B（RAG）
  - 解析：PyMuPDF 文本/页码；可选表格抽取（Camelot/Tabula）。
  - 切片/嵌入：Recursive Splitter（chunk_size≈500, overlap≈50）；`bge-m3`/`m3e-base`（CPU）。
  - 向量库：Chroma（可选 BM25 混合）；可接 reranker `bge-reranker-base`。
- Agent
  - 工具：行情、新闻、分类、检索、作图、LLM 调用；超时/重试/熔断与日志。
- 部署
  - 本地 Streamlit + 本地 BERT/RAG + 云端 LLM API；训练与 7B QLoRA 在 Colab。

---

## 3. 里程碑与任务（精要）

- M0：范围与脚手架（1–2 天）
  - 确认标的池与时间范围；事件窗与阈值。
  - 目录/配置/SQLite 空库与表；脚本骨架。
- M1：数据与代理标注 + Baseline（3–5 天）
  - OHLCV + 交易日历入库；单一快讯源采集与去重。
  - 事件对齐与收益计算；生成训练/验证集。
  - TF-IDF + 线性模型基线与报告。
- M2：BERT 微调与本地推理（3–5 天）
  - Colab 训练、最佳权重导出；本地量化推理；误差分析。
- M3：RAG 管线（3–5 天）
  - PDF 解析、切片、嵌入、Chroma 索引；检索 API；评测（Recall@k、ragas）。
- M4：Agent + UI 集成（3–5 天）
  - 工具封装/函数调用；K 线联动新闻；PDF 页定位；设置页。
- M5：评测与答辩包（2–3 天）
  - 时间切分与消融；指标脚本与看板；一键运行脚本与最小样本。

---

## 4. 目录结构（建议）

```text
app/
  ui/          # Streamlit pages
  agents/      # 工具与路由
  services/    # 行情/新闻/RAG 服务
configs/
  config.yaml
scripts/
  fetch_prices.py
  fetch_news.py
  label_events.py
  train_baseline.py
  train_bert_colab.ipynb
  build_vector_store.py
  infer_sentiment.py
  eval_*.py
data/{raw,interim,processed}/
models/{bert,rag}/
reports/ logs/
finance.db  .env.example  requirements.txt  PLAN.md
```

---

## 5. 数据库与向量库（最小集）

```sql
-- prices(ticker, ts, open, high, low, close, volume, adj_close, trade_date)
-- news(id, ts, source, title, content, ticker_hint, url, hash)
-- labels(news_id, ticker, window_min, ret, label in {bullish,neutral,bearish})
-- reports(id, ticker, period, source, file_path, page_idx, section)
```

Chroma Collection：`reports_chunks`，元数据：`{ticker, period, section, page_idx}`。

---

## 6. 实施细节与验收要点

- 代理标注
  - 时区统一 Asia/Shanghai；非交易时段滚动至下一开盘。
  - 窗口：{30,60,120} 分钟；中性带 ±0.3%（可按波动率自适应）。
  - 泄露防护：时间切分（训练≤T，验证>T）。
- Baseline 与 BERT
  - Baseline：TF-IDF + LogReg/SVM，形成地板指标。
  - BERT：`bert-base-chinese`/`macbert-base`；早停；导出权重；本地量化推理。
- RAG
  - Splitter：chunk_size≈500；嵌入：`bge-m3`；检索：Chroma（可混合 BM25）。
  - 评测：Recall@k、ragas；人工抽检 Top-50 并记录失效原因。
- Agent 与 UI
  - 工具签名稳定（行情/分类/检索/作图/LLM）；函数调用 JSON schema；日志记录耗时与摘要。
  - UI：Chat、K 线（事件标注与点击联动）、财报检索（列表+PDF页跳转）、设置。

---

## 7. 立即下一步（需确认）

- 标的与市场（示例：上证指数 + 两只沪深300成分股）。
- 时间范围（示例：近 18 个月）。
- 事件窗与阈值（±0.3%，30/60/120 分钟）。
- 嵌入模型（`bge-m3`）与向量库（Chroma）。

确认后将按 M0 执行：创建脚手架、配置与 SQLite 基表，并补充 `requirements.txt` 与 `.env.example`。
