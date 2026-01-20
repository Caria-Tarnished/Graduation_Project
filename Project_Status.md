# Project Status

更新时间：2026-01-20
负责人：Caria-Tarnished

---

## 1) 概览

- 目标：财经快讯情感分类（代理标注验证）+ 财报 RAG（后续里程碑）+ Streamlit UI。
- 时间范围：2024-01-01 至 2026-01-01（Asia/Shanghai）。
- 标的池：000001.SH、300750.SZ、600036.SH、XAUUSD=X、NVDA。
- 数据存储：SQLite（finance.db），向量库：Chroma（后续）。

## 2) 已完成（Done）

- 计划文档：`PLAN.md`（目标、架构、里程碑、验收标准）。
- 配置与目录：`configs/config.yaml`、`.env.example`、标准目录与 `.gitkeep`。
- 依赖：`requirements.txt`。
- 数据库初始化：`scripts/init_db.py`（prices/news/labels/reports/trading_calendar）。
- 行情抓取：`scripts/fetch_prices.py`（A 股指数/个股：Tushare；美股/黄金：yfinance）。
- 新闻导入：`scripts/fetch_news.py`（CSV 入库与去重）。
- 代理标注（“日级窗”）：`scripts/label_events.py`（CSV 或 DB → labels）。
- 代码风格：已进行首次 PEP8 整理（长行/空行）。

## 3) 进行中 / 下一步（Next）

- M1 数据与代理标注：
  - [ ] 拉取 2024–2026 全部标的日线（需 `.env` 中设置 `TUSHARE_TOKEN` 才能抓取 A 股）。
  - [ ] 准备 2024–2026 快讯 CSV（列：`ts/Title/Content/...`；可含 `ticker_hint`）。
  - [ ] 运行 `label_events.py` 生成 labels（窗口 `1/3/5` 天，`neutral_band=±0.3%`）。
  - [ ] 基线：`train_baseline.py`（TF-IDF + 线性模型；时间切分避免泄露）。
- 项目周边：
  - [ ] 建立 GitHub 远程仓库并推送（本文件与 `.gitignore` 已生成）。

## 4) 备忘 / 风险（Memos / Risks）

- Tushare 令牌：无则无法抓取 A 股（指数与个股）。
- yfinance 速率限制：必要时分批或加入退避。
- 时区与交易日：新闻对齐到下一开盘日，节假日需注意（SQLite 可扩展交易日历）。
- 分钟级窗：首版以“日级窗”为主，分钟级将在小样本验证后补充。
- 隐私与密钥：使用 `.env`，严禁将密钥提交到 Git。

## 5) 变更记录（Changelog）

- 2026-01-20
  - 更新 `configs/config.yaml`：设置标的池与 `windows_days=[1,3,5]`。
  - 实现 `scripts/label_events.py`（CSV/DB → labels，中文日期解析）。
  - 整理 `scripts/fetch_prices.py` 与 `scripts/fetch_news.py` 的 PEP8。
  - 增加本文件 `Project_Status.md` 与 `.gitignore`（初版）。
- 2025-11-15
  - 创建 `PLAN.md`（首版开发计划）。

## 6) 快速指引（Quick Start）

- 创建与填写 `.env`（复制 `.env.example`）：
  - `TUSHARE_TOKEN=...`
- 初始化数据库
  ```powershell
  python scripts/init_db.py
  ```
- 抓取行情（使用 config 中的时间与标的）
  ```powershell
  python scripts/fetch_prices.py
  ```
- 从 CSV 入库新闻
  ```powershell
  python scripts/fetch_news.py --csv path\to\news.csv
  ```
- 生成“日级窗”代理标注
  ```powershell
  python scripts/label_events.py --csv path\to\news.csv --out-csv data\processed\labels.csv
  ```

## 7) 关键文件

- 计划：`PLAN.md`
- 状态：`Project_Status.md`
- 配置：`configs/config.yaml`
- 环境：`.env.example`
- 数据库：`finance.db`
- 脚本：`scripts/`
