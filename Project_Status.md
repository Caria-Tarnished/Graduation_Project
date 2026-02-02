# Project Status

更新时间：2026-02-02
负责人：Caria-Tarnished

---

## 1) 概览

- 目标：财经快讯情感分类（代理标注验证）+ 财报 RAG（后续里程碑）+ Streamlit UI。
- 时间范围：2024-01-01 至 2026-01-21（Asia/Shanghai）。
- 标的池：000001.SH、300750.SZ、600036.SH、XAUUSD=X、NVDA。
- 数据存储：SQLite（finance.db），向量库：Chroma（后续）。
- 新增（分钟级冲击分析）：基于 MT5 XAUUSD M1 + 金十（flash/calendar），统一至 finance_analysis.db；价格覆盖 2024-01-02 09:00 至 2026-01-31 07:59（Asia/Shanghai）。

## 2) 已完成（Done）

- 计划文档：`PLAN.md`（目标、架构、里程碑、验收标准）。
- 配置与目录：`configs/config.yaml`、`.env.example`、标准目录与 `.gitkeep`。
- 依赖：`requirements.txt`。
- 数据库初始化：`scripts/init_db.py`（prices/news/labels/reports/trading_calendar）。
- 行情抓取：`scripts/fetch_prices.py`（A 股指数/个股：Tushare；美股/黄金：yfinance）。
- 新闻导入：`scripts/fetch_news.py`（CSV 入库与去重）。
- 代理标注（“日级窗”）：`scripts/label_events.py`（CSV 或 DB → labels）。
- 代码风格：已进行首次 PEP8 整理（长行/空行）。
- 金十爬虫（当前在用）与入库：
  - `scripts/crawlers/jin10_dynamic.py`：日历模式已稳定可用，支持“只看重要”开关、DB 入库与 CSV 增量写入；接口与参考脚本保持一致（`--months/--start/--end/--output/--db/...`）。
  - `scripts/crawlers/jin10_flash_api.py`：快讯 API 模式（支持接口发现/过滤/CSV 流式/SQLite 入库）。
- SQLite 存储：`scripts/crawlers/storage.py`（upsert、URL/内容哈希去重、索引）。

- 分钟级冲击分析（MT5 + 金十，已联通）
  - MT5 分钟价抓取：`scripts/fetch_intraday_xauusd_mt5.py`（自动符号选择、分片抓取、时区标准化、UTF-8 CSV）。
  - 冲击计算与入库：`scripts/build_finance_analysis.py`（prices_m1 / events / event_impacts 三表，时间索引、asof 对齐、UPSERT）。
  - 数据库：`finance_analysis.db`（prices_m1=736,304 行；覆盖 2024-01-02 09:00 至 2026-01-31 07:59，Asia/Shanghai）。
  - 覆盖率验证（自 2026-01-27 起）：events_total=1475、events_with_impacts=1475、coverage=100%；四窗口均为 1475（5/10/15/30 分钟）。
  - 导出训练集：
    - `data/processed/training_event_impacts_xauusd_2025Q4_2026Jan.csv`（35808×27）。
    - 全量：`data/processed/training_event_impacts_xauusd_all.csv` 与 `.parquet`（104,728×27）。
  - 30 分钟窗口打标与切分：
    - `data/processed/train_30m_labeled.csv`、`val_30m_labeled.csv`、`test_30m_labeled.csv`。
    - `data/processed/labeling_thresholds.json`（q_low=-0.0003825, q_high=0.0004687；train=15071、val=3122、test=7989）。
    - 标注规则（30 分钟窗口）：
      - 时间切分：按 `event_ts_local` 做时间切分（train/val/test），保证同一事件不跨集合，避免泄漏。
      - 收益定义：`ret = (price_future - price_event) / price_event`；`price_event`/`price_future` 均由分钟价在北京时间对齐得到（asof 对齐；窗口 30 分钟）。
      - 阈值估计：仅在训练集上计算 `ret` 的分位阈值（默认 30%/70%），得到 `q_low` 与 `q_high`，写入 `labeling_thresholds.json`。
      - 标签映射：`-1` 若 `ret <= q_low`；`1` 若 `ret >= q_high`；否则 `0`。
      - 一致性：验证/测试集沿用训练集阈值（固定阈值），避免信息泄漏。

## 3) 进行中 / 下一步（Next）

- 金十（优先）：
  - [X] 使用 `jin10_flash_api.py`（API 模式，支持 `--db` 入库）作为快讯主抓取；未知接口时使用“接口发现”。
  - [X] 使用 `jin10_dynamic.py` 日历模式逐日抓取（直达 URL，支持“只看重要”），可直接入库 `finance.db` 并同步写 CSV（已完成初版联调）。
  - [X] 如需更多来源，后续再引入（现阶段聚焦金十）。
- 东方财富：
  - [ ] 暂缓（清理阶段不保留相关爬虫与脚本）。
- Yahoo Finance：
  - [ ] 暂缓。
- 代理标注与基线：
  - [ ] 使用入库新闻运行 `scripts/label_events.py` 生成 labels（窗口 1/3/5 天、`neutral_band=±0.3%`）。
  - [ ] 训练 `train_baseline.py` 并产出报告与预测明细（reports/）。
- 脚本整理：

- [X] 非爬虫脚本（`scripts/fetch_news.py`、`scripts/fetch_prices.py`、`scripts/ingest_listing_csv.py`、`scripts/label_events.py`、`scripts/uplift_articles_to_news.py`）如近期不用，可归档至 `archive/unused_scripts_YYYYMMDD/`。

- 分钟级冲击建模（新增）
  - [ ] 文本清洗与特征工程脚本化（保留前缀特征，如 `[SRC=]`/`[STAR=]`/`[IMP=]`/`[CTRY=]`，正则去噪）。
  - [X] 快速基线：TF-IDF + LinearSVC，输出指标与预测明细（完成多组对比）。
    - 实验组合与指标（macro_f1，val/test）：
      - 默认（char 2-4，含前缀，未加权）：0.3659 / 0.3458。
      - char 1-3 + balanced：0.3667 / 0.3376。
      - char 1-3 + balanced + C=2.0：0.3565 / 0.3394。
      - char 2-4 + balanced：0.3707 / 0.3422。
      - 无前缀：0.3666 / 0.3378。
    - 推荐基线：char 2-4，保留前缀，不加权（测试集最优）。
    - 增强：新增 `--sublinear_tf/--norm/--dual` 参数，便于正则化与适配样本-特征维度关系。
  - [ ] BERT 微调（中文 RoBERTa-wwm-ext，时间切分不泄漏），保存最优 checkpoint 与推理脚本。
  - [ ] 多窗口样本导出（5/10/15/30 作为样本，事件×4 行），统一标签或多任务设置。
  - [ ] 阈值与标签策略调参（如 25/75 或固定阈值），并固化至 JSON。

  - [X] 多维复合标签数据集（15 分钟基础 + 前 120 分钟趋势对照）
    - 生成脚本：`scripts/modeling/prepare_multilabel_dataset.py`
    - 标签体系：
      - 基础方向（label_base）：基于 15 分钟收益 `ret_post` 的训练集分位阈值，映射为 `-1/0/1`（bearish/neutral/bullish，固定阈值避免泄漏）。
      - 预期兑现（label_priced_in）：若前 120 分钟存在显著单边（|pre_ret|≥阈），且“基本面方向”（`actual-consensus`）与发布后 15 分钟方向相反，则标注，区分 `bullish_priced_in` 与 `bearish_priced_in`。
      - 观望（label_watch）：发布后窗口内 Range 显著放大但 |ret_post| 极小（高波动低净变动）。
      - 组合多类（label_multi_cls）：优先级 观望(5) > 兑现(3/4) > 基础方向(1/2) > 中性(0)。
    - 指标输出：`ret_post/pre_ret/range_ratio/abs_ret_post/surprise` 等；包含异常值裁剪与分位阈值鲁棒化。

- 数据 QA 与归档
  - [X] 新增脚本：`scripts/qa/validate_datasets.py`，输出 `data/processed/qa_dataset_report.json`。
  - [ ] 运行 QA：确认多类数据集时间跨度（期望约 2024-01-01 至 2026-02-01）、事件不跨集合（any_overlap=false）、`event_id` 去重无异常、标签分布合理，并校验 DB 的 `events/event_impacts/prices_m1` 覆盖范围。
  - [ ] 基于报告的 `archive_suggestions` 制定归档清单（可迁移至 `archive/` 或 Git LFS/Release）。

- 数据扩展与校验
  - [ ] 如需：补齐其余时间段或其他 MT5 符号（如 XAUUSD.i、GOLD）并复检覆盖率。
  - [ ] 交易时段/夏令时敏感性检查与说明。

- 交付与文档
  - [ ] 训练集字段字典与处理流程文档。
  - [ ] 训练/验证/测试集统计报告与可视化。

## 4) 备忘 / 风险（Memos / Risks）

- 登录与反爬：部分站点需登录，建议用 `--user-data-dir` 或 `--storage` 持久化登录态。
- DOM 变更：动态站点结构变化频繁，需通过 `--debug-dir` 反馈 HTML 以快速适配。
- 运行稳定性：请勿手动关闭浏览器；长时运行建议无头并控制 `--max-loads`、`--delay`。
- 速率与封禁：必要时添加随机延迟/代理池；尊重 robots.txt。
- 时区与交易日：新闻对齐到下一开盘日，节假日需注意（可扩展交易日历）。
- 数据质量：快讯页部分条目无明确时间戳，可用 `--allow-undated` 暂保留，事后人工补齐。
- 后续任务（抓取与入库规范）：

  - [X] 日历爬虫效果验证：对照网页可见项与解析结果，抽样核对时间/标题；必要时补充选择器回退策略与星级识别鲁棒性。
  - [ ] 入库结构核对：验证两条管线入库字段一致性（site/source/title/content/published_at/url/extra_json），统一 `extra_json` 键名（如 date_text/hot_levels 等）。
  - [ ] 规范化与多表对齐：如有必要，调整 `articles` 表或新增规范化表（如 indicators/events），明确主键（内容/URL 哈希）与外键（交易日/标的），保证“快讯/日历/行情”多表对齐逻辑。
  - [ ] 测试与样例：为日历解析与 API 解析准备最小样例（页面快照/接口 JSON 片段）和断言，确保升级后解析/入库不回退。

## 5) 变更记录（Changelog）

- 2026-02-01

  - 新增：分钟级冲击分析管线（MT5 + 金十）：
    - `scripts/fetch_intraday_xauusd_mt5.py`（分钟价抓取）。
    - `scripts/build_finance_analysis.py`（构建 finance_analysis.db，计算事件冲击）。
  - 回填与重建：抓取 2024–2025 M1 并重建 `finance_analysis.db`；`prices_m1` 共 736,304 行（2024-01-02 09:00 至 2026-01-31 07:59）。
  - 验证：自 2026-01-27 起，事件覆盖率 100%，四窗口统计各 1475。
  - 导出：全量与区间训练集（CSV/Parquet），并生成 30 分钟窗口打标与时间切分数据集与阈值 JSON。
  - 基线：`scripts/modeling/baseline_tfidf_svm.py` 现支持 `--class_weight/--C/--sublinear_tf/--norm/--dual`；完成 char 1-3/2-4 与加权对比，锁定推荐基线配置。
  - 新增：`scripts/modeling/prepare_multilabel_dataset.py`，输出 `train/val/test_multi_labeled.csv` 与 `labeling_thresholds_multilabel.json`，实现“基础方向/预期兑现/建议观望”的复合标签。

- 2026-01-29

  - 调整：`scripts/crawlers/jin10_dynamic.py` 与参考脚本对齐，采用统一 CLI：
    - 新增参数：`--months/--start/--end/--output/--db/--source/--headed/--debug/--important-only/--user-data-dir/--recheck-important-every/--use-slider/--setup-seconds`。
    - 支持边爬边入库（`Article`），并按参考列顺序增量写 CSV；长行（flake8 E501）分行处理。
    - 保留请求拦截（禁 image/media/font）提速，确保“只看重要”开关与星级兜底过滤。
  - 文档：更新本页“金十日历/数据（逐日回填）”使用指令为新 CLI。
- 2026-01-27

  - 新增：`scripts/crawlers/jin10_flash_api.py`（快讯 API 爬虫，支持接口发现、筛选与 CSV 流式写入、SQLite 入库 `--db`）。
  - 增强：`scripts/crawlers/jin10_dynamic.py` 日历模式：
    - 直达 `https://rili.jin10.com/day/YYYY-MM-DD`，按日期倒序抓取；
    - 开启资源拦截（屏蔽图片/字体/媒体）提速；
    - 解析重构：在页面端 `page.evaluate` 快速提取，仅统计“可见”行；
    - 支持星级识别并在客户端兜底过滤“只看重要”（≥3 星）；
    - 尝试确保“经济数据”页签与“只看重要”开关；
    - 预览阶段 `storage_state` 导出、正式阶段导入，避免 Windows 上 `user_data_dir` 锁导致卡顿；
    - 边爬边入库（`upsert_many`），输出每日提取与累计入库计数；
    - 修复 0 条/卡住问题的若干鲁棒性细节与调试日志。
  - 增强：`scripts/crawlers/jin10_flash_api.py` 接口发现与入库：
    - 同时监听 request/response，仅在 JSON 响应与包含 `params=` 的 URL 时确认；
    - 收紧主机/路径匹配（排除主站/日历/热榜等非列表接口）；
    - 时间字段规范化、最后一条时间调试输出、完整请求头透传；
    - 支持 `--stream` 边抓边写 CSV 与 `--db` 入库（URL/内容哈希去重）。
  - 文档：补充“快讯 API 模式”和“数据库常用指令（只读/删除）”。
  - 清理：归档未使用爬虫脚本至 `archive/unused_crawlers_20260127/`：
    - `scripts/crawlers/providers/` 全部
    - `scripts/crawlers/list_crawl.py`
    - `scripts/crawlers/parse_listing.py`
    - `scripts/crawlers/fetch_from_urls.py`
  - 代码风格：补充 flake8 清理（E501 行宽、空白行 W293/E306）于 `jin10_dynamic.py` / `jin10_flash_api.py`，不改动业务逻辑。
  - Git：`.gitignore` 新增忽略 `archive/` 与 `参考代码和文档/`。
- 2026-01-23

  - 新增：`scripts/crawlers/jin10_dynamic.py`（Playwright 动态抓取：快讯倒序回溯与日历模式；登录持久化、跨 frame、滚动/加载更多、调试快照、入库）。
  - 新增：`scripts/crawlers/ingest_listing_csv.py`（listing CSV 入库）。
  - 新增：`scripts/crawlers/providers/jin10_events.py`（金十重要事件列表）。
  - 增强：`scripts/crawlers/list_crawl.py`（英文相对时间解析、时间窗过滤、SQLite 入库）。
  - 增强：`scripts/crawlers/parse_listing.py`（统一 `content_text` 与 provider 注册）。
  - 增强：`scripts/crawlers/fetch_from_urls.py`（加入 SQLite 入库与去重）。
  - 修复：`scripts/crawlers/storage.py` 长行与稳健性（upsert/去重）。
- 2026-01-20

  - 更新 `configs/config.yaml`：设置标的池与 `windows_days=[1,3,5]`。
  - 实现 `scripts/label_events.py`（CSV/DB → labels，中文日期解析）。
  - 整理 `scripts/fetch_prices.py` 与 `scripts/fetch_news.py` 的 PEP8。
  - 增加本文件 `Project_Status.md` 与 `.gitignore`（初版）。
  - 创建本地 Conda `.venv` 环境并安装依赖。
  - 完成并规范化 `scripts/train_baseline.py`（TF-IDF+LinearSVC，时间切分；支持 DB/CSV；生成报告与预测；flake8 通过）。
  - `.gitignore` 新增 `.windsurf/` 与 `文字材料文档/`，并将后者从 Git 索引中移除。
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
- 安装 Playwright（首次）
  ```powershell
  pip install playwright
  python -m playwright install chromium
  ```
- 金十快讯（API 模式，推荐，入库）
  ```powershell
  # 已知接口直连（示例：请替换为你的真实 API 基址）
  python -m scripts.crawlers.jin10_flash_api \
    --months 12 \
    --output data/raw/flash_last_12m.csv \
    --api-base https://your-api.example.com/flash/get_flash_list \
    --stream --important-only \
    --sleep 1.8 \
    --db finance.db \
    --source flash_api

  # 接口发现（首次使用/未知接口时）：
  python -m scripts.crawlers.jin10_flash_api \
    --months 12 \
    --output data/raw/flash_last_12m.csv \
    --page-url https://www.jin10.com/flash \
    --headed --user-data-dir .pw_jin10 --setup-seconds 12 \
    --stream --important-only \
    --sleep 1.8 \
    --db finance.db \
    --source flash_api
  ```
- 金十日历/数据（逐日回填）
  ```powershell
  # 最近 3 个月（仅 CSV）
  python -m scripts.crawlers.jin10_dynamic \
    --months 3 \
    --output data/raw/jin10_calendar_last3m.csv \
    --important-only \
    --user-data-dir .pw_jin10

  # 指定区间 + 入库（推荐）
  python -m scripts.crawlers.jin10_dynamic \
    --start 2024-01-01 \
    --end 2026-01-21 \
    --output data/raw/jin10_calendar_2024_2026.csv \
    --db finance.db \
    --source listing_data \
    --important-only \
    --user-data-dir .pw_jin10

  # 单日验证（start=end）
  python -m scripts.crawlers.jin10_dynamic \
    --start 2024-12-07 \
    --end 2024-12-07 \
    --output data/raw/jin10_calendar_20241207.csv \
    --db finance.db \
    --source listing_data \
    --important-only \
    --user-data-dir .pw_jin10
  ```
- 从 CSV 入库新闻
  ```powershell
  python scripts/fetch_news.py --csv path\to\news.csv
  ```
- 生成“日级窗”代理标注
  ```powershell
  python scripts/label_events.py --csv path\to\news.csv --out-csv data\processed\labels.csv
  ```

- 分钟级冲击分析（MT5 + 金十）
  - 抓取分钟价（示例：2024–2025 全量）
    ```powershell
    python scripts\fetch_intraday_xauusd_mt5.py `
      --timeframe M1 `
      --start "2024-01-01 00:00:00" `
      --end   "2025-12-31 23:59:59" `
      --chunk_days 15 `
      --tz_out "Asia/Shanghai" `
      --label_ticker "XAUUSD" `
      --symbol "XAUUSD" `
      --out "data\processed\xauusd_m1_mt5_2024_2025.csv"
    ```
  - 数据 QA 校验与归档建议（输出 JSON 报告）
    ```powershell
    python scripts/qa/validate_datasets.py `
      --processed_dir data/processed `
      --raw_dir data/raw `
      --db finance_analysis.db `
      --ticker XAUUSD
    ```
    - 报告路径：`data/processed/qa_dataset_report.json`。请检查：
      - 三个多类集合的时间范围是否覆盖 2024-01-01 至 2026-02-01（或你期望的区间）。
      - `cross_split_event_id_intersections.any_overlap` 应为 false（无事件跨集合）。
      - `dups_event_id.dups` 应为 0（`event_id` 无重复）。
      - 标签分布是否合理（少数类比例）。
      - 数据库 `events/event_impacts/prices_m1` 的时间覆盖是否与预期一致。

  - Colab 训练（推荐，GPU）：
    1) 打开 https://colab.research.google.com 并选择 GPU（Runtime → Change runtime type → GPU）。
    2) 打开本仓库中的笔记本：`notebooks/bert_multilabel_colab.ipynb`（Colab 文件 → GitHub 选项卡搜索仓库名）。
    3) 依次执行单元：安装依赖 → 克隆仓库 → 上传三份 CSV → 启动训练 → 查看指标 → 打包下载模型输出。
  - Colab 训练（多标签 6 类 + Drive，FinBERT/中文 BERT）
    - 挂载 Drive 并设置目录：`DATA_DIR=/content/drive/MyDrive/datasets/xauusd_multilabel`，`MODEL_DIR=/content/drive/MyDrive/models/bert_xauusd_multilabel_6cls`。
    - 确认 Drive 下存在三份 CSV：`train_multi_labeled.csv`、`val_multi_labeled.csv`、`test_multi_labeled.csv`。若缺失，可运行笔记本中的“从仓库复制到 Drive”或手动上传。
    - 设置预训练模型 `MODEL_NAME`：
      - 中文数据：优先使用中文预训练模型（推荐 `hfl/chinese-roberta-wwm-ext` 作为强基线）。
      - 英文文本：可用 `ProsusAI/finbert`（英文金融领域）。
      - 说明：若数据为中文，不建议使用英文 FinBERT；其分词词表对中文支持有限，效果通常显著劣于中文模型。
    - 运行“6 分类训练（使用原始 CSV 与 `label_multi_cls`）”单元。其等价 CLI 示例（PowerShell）：
      ```powershell
      python scripts/modeling/bert_finetune_cls.py `
        --train_csv data/processed/train_multi_labeled.csv `
        --val_csv   data/processed/val_multi_labeled.csv `
        --test_csv  data/processed/test_multi_labeled.csv `
        --output_dir models/bert_xauusd_multilabel_6cls `
        --label_col label_multi_cls `
        --model_name hfl/chinese-roberta-wwm-ext `
        --train_bs 16 --eval_bs 32 --epochs 2 --lr 2e-5 --max_length 256
      ```
    - 产出：`metrics_val.json`、`metrics_test.json`、`report_test.txt`、`pred_test.csv` 与 `best/`（最优模型）。
  - 6 类 BERT 首次评估（中文 RoBERTa-wwm-ext）
    - val：accuracy=0.4321，macro_f1=0.1822。
    - test：accuracy=0.4251，macro_f1=0.1631。
    - 按类简述（test 支持数：0/1/2/3/4/5 = 4030/1654/1417/15/5/868）：
      - 0（中性）：F1=0.6013（主导类）。
      - 1/2（方向）：F1 偏低。
      - 3/4（兑现）：样本极少，F1=0。
      - 5（观望）：F1=0。
    - 改进方向：更久训练（epochs↑）、更长序列（max_length↑）、类权重（加权交叉熵）、早停（EarlyStopping）、warmup 与 weight_decay、按 steps 验证与保存。
  - 复合标签训练集导出（15 分钟基础 + 前 120 分钟趋势对照）
    ```powershell
    python scripts/modeling/prepare_multilabel_dataset.py `
      --db finance_analysis.db `
      --ticker XAUUSD `
      --window_post 15 `
      --pre_minutes 120 `
      --out_dir data\processed
    ```
  - 构建与计算冲击（UPSERT 幂等）
    ```powershell
    python scripts\build_finance_analysis.py `
      --prices_csv "data\processed\xauusd_m1_mt5_2024_2025.csv" `
      --flash_csv "data\raw\flash_last_14m.csv" `
      --calendar_csv "data\raw\jin10_calendar_q4.csv" `
      --db "finance_analysis.db" `
      --price_tz "Asia/Shanghai" --flash_tz "Asia/Shanghai" --calendar_tz "Asia/Shanghai" `
      --ticker "XAUUSD" `
      --windows 5 10 15 30
    ```
  - 覆盖率复检（PowerShell here-string → Python 标准输入，稳定免转义）
    ```powershell
    $code = @'
    import sqlite3, pandas as pd
    c = sqlite3.connect("finance_analysis.db")
    q = "select (select count(*) from events where ts_local>='2026-01-27 00:00:00') as events_total, (select count(distinct e.event_id) from events e join event_impacts ei on ei.event_id=e.event_id and ei.ticker='XAUUSD' where e.ts_local>='2026-01-27 00:00:00') as events_with_impacts, round(1.0*(select count(distinct e.event_id) from events e join event_impacts ei on ei.event_id=e.event_id and ei.ticker='XAUUSD' where e.ts_local>='2026-01-27 00:00:00')/(select count(*) from events where ts_local>='2026-01-27 00:00:00'),4) as coverage"
    print(pd.read_sql_query(q, c))
    c.close()
    '@
    $code | python -
    ```

  - 建模脚本（Baseline / BERT）
    ```powershell
    # 基线：TF-IDF + LinearSVC（CPU 即可）
    python scripts/modeling/baseline_tfidf_svm.py \
      --train_csv data/processed/train_30m_labeled.csv \
      --val_csv   data/processed/val_30m_labeled.csv \
      --test_csv  data/processed/test_30m_labeled.csv \
      --output_dir models/baseline_tfidf_svm \
      --sublinear_tf --norm l2 --dual auto

    # BERT 中文微调（RoBERTa-wwm-ext）
    # 依赖：pip install -U transformers datasets accelerate evaluate; 如无 GPU，可安装 CPU 版 torch
    python scripts/modeling/bert_finetune_cls.py \
      --train_csv data/processed/train_30m_labeled.csv \
      --val_csv   data/processed/val_30m_labeled.csv \
      --test_csv  data/processed/test_30m_labeled.csv \
      --output_dir models/bert_xauusd_cls \
      --epochs 2 --lr 2e-5 --max_length 256
    ```
  - 导出训练集（全量 CSV + Parquet）
    ```powershell
    $code = @'
    import sqlite3, pandas as pd, os
    c = sqlite3.connect("finance_analysis.db")
    q = """
    select
      ei.event_id,
      e.source,
      e.ts_local as event_ts_local,
      e.ts_utc   as event_ts_utc,
      e.country, e.name, e.content, e.star, e.previous, e.consensus, e.actual,
      e.affect, e.detail_url, e.important, e.hot, e.indicator_name, e.unit,
      ei.ticker, ei.window_min,
      ei.price_event, ei.price_future, ei.delta, ei.ret,
      ei.price_event_ts_local, ei.price_future_ts_local,
      ei.price_event_ts_utc,   ei.price_future_ts_utc
    from event_impacts ei
    join events e on e.event_id = ei.event_id
    where ei.ticker='XAUUSD'
    order by e.ts_local asc, ei.window_min asc
    """
    df = pd.read_sql_query(q, c)
    c.close()
    os.makedirs(r"data\processed", exist_ok=True)
    out_csv = r"data\processed\training_event_impacts_xauusd_all.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(r"data\processed\training_event_impacts_xauusd_all.parquet", engine="pyarrow", index=False)
    except Exception:
        pass
    print(out_csv, df.shape)
    '@
    $code | python -
    ```
  - 30 分钟窗口打标与切分（时间切分避免泄漏）
    ```powershell
    $code = @'
    import sqlite3, pandas as pd, os, json
    c = sqlite3.connect("finance_analysis.db")
    q = """
    select
      ei.event_id, e.source, e.ts_local as event_ts_local, e.ts_utc as event_ts_utc,
      e.country, e.name, e.content, e.star, e.previous, e.consensus, e.actual,
      e.affect, e.detail_url, e.important, e.hot, e.indicator_name, e.unit,
      ei.ticker, ei.window_min, ei.price_event, ei.price_future, ei.delta, ei.ret
    from event_impacts ei
    join events e on e.event_id = ei.event_id
    where ei.ticker='XAUUSD' and ei.window_min=30
    order by e.ts_local asc
    """
    df = pd.read_sql_query(q, c)
    c.close()
    df["text"] = df["content"].fillna("").astype(str)
    m = df["text"].str.len()==0
    df.loc[m, "text"] = df.loc[m, "name"].fillna("").astype(str)
    df["event_ts_local"] = pd.to_datetime(df["event_ts_local"], errors="coerce")
    t1, t2, t3 = pd.Timestamp("2025-08-01 00:00:00"), pd.Timestamp("2025-11-01 00:00:00"), pd.Timestamp("2026-02-01 00:00:00")
    train = df[df["event_ts_local"] < t1].copy()
    val   = df[(df["event_ts_local"] >= t1) & (df["event_ts_local"] < t2)].copy()
    test  = df[(df["event_ts_local"] >= t2) & (df["event_ts_local"] < t3)].copy()
    train = train.dropna(subset=["ret"]).copy()
    ql, qh = (float(train["ret"].quantile(0.30)) if len(train) else -0.001), (float(train["ret"].quantile(0.70)) if len(train) else 0.001)
    def lab(x, lo, hi):
        import math
        return 0 if pd.isna(x) else (-1 if x<=lo else (1 if x>=hi else 0))
    for part in (train,val,test):
        part["label"] = part["ret"].apply(lambda x: lab(x, ql, qh))
    keep=["event_id","event_ts_local","event_ts_utc","source","country","name","content","text","star","previous","consensus","actual","affect","detail_url","important","hot","indicator_name","unit","ticker","window_min","price_event","price_future","delta","ret","label"]
    os.makedirs(r"data\processed", exist_ok=True)
    train[keep].to_csv(r"data\processed\train_30m_labeled.csv", index=False, encoding="utf-8")
    val[keep].to_csv(  r"data\processed\val_30m_labeled.csv",   index=False, encoding="utf-8")
    test[keep].to_csv( r"data\processed\test_30m_labeled.csv",  index=False, encoding="utf-8")
    with open(r"data\processed\labeling_thresholds.json","w",encoding="utf-8") as f:
        json.dump({"window_min":30,"q_low":ql,"q_high":qh,"sizes":{"train":len(train),"val":len(val),"test":len(test)},"splits":{"train_end":str(t1),"val_end":str(t2),"test_end":str(t3)}}, f, ensure_ascii=False, indent=2)
    print("done")
    '@
    $code | python -
    ```

## 7) 关键文件

- 计划：`PLAN.md`
- 状态：`Project_Status.md`
- 配置：`configs/config.yaml`
- 环境：`.env.example`
- 数据库：`finance.db`
- 分钟级冲击数据库：`finance_analysis.db`
- 脚本：`scripts/`
  - `scripts/crawlers/jin10_dynamic.py`
  - `scripts/crawlers/jin10_flash_api.py`
  - `scripts/crawlers/storage.py`
  - `scripts/fetch_intraday_xauusd_mt5.py`
  - `scripts/build_finance_analysis.py`
  - `scripts/modeling/baseline_tfidf_svm.py`
  - `scripts/modeling/bert_finetune_cls.py`

## 8) 常用指令（数据库）

- 只读：列出所有表
  ```powershell
  python -c "import sqlite3; conn=sqlite3.connect('finance.db'); c=conn.cursor(); c.execute('SELECT name FROM sqlite_master WHERE type=? ORDER BY name', ('table',)); print([r[0] for r in c.fetchall()]); conn.close()"
  ```
- 只读：统计 `articles` 总数与某日（日历来源）条数
  ```powershell
  python -c "import sqlite3; conn=sqlite3.connect('finance.db'); c=conn.cursor(); tbl='articles'; c.execute('SELECT COUNT(*) FROM '+tbl); total=c.fetchone()[0]; c.execute('SELECT COUNT(*) FROM '+tbl+' WHERE site=? AND source=? AND substr(published_at,1,10)=?', ('rili.jin10.com','listing_data','2024-12-07')); day=c.fetchone()[0]; print('total:', total, ' 2024-12-07:', day); conn.close()"
  ```
- 只读：查看当日详细（时间+标题）
  ```powershell
  python -c "import sqlite3, json; conn=sqlite3.connect('finance.db'); c=conn.cursor(); q='SELECT published_at, title FROM articles WHERE site=? AND source=? AND substr(published_at,1,10)=? ORDER BY published_at ASC, id ASC'; rows=c.execute(q, ('rili.jin10.com','listing_data','2024-12-07')).fetchall(); print(json.dumps({'rows': len(rows), 'items': rows}, ensure_ascii=False, indent=2)); conn.close()"
  ```
- 删除：按日删除“日历/只看重要”当天数据（危险操作，谨慎执行）
  ```powershell
  python -c "import sqlite3; conn=sqlite3.connect('finance.db'); c=conn.cursor(); c.execute('DELETE FROM articles WHERE site=? AND source=? AND substr(published_at,1,10)=?', ('rili.jin10.com','listing_data','2024-12-07')); conn.commit(); conn.close()"
  ```
- 删除：清空所有“日历/listing_data”数据（危险）
  ```powershell
  python -c "import sqlite3; conn=sqlite3.connect('finance.db'); c=conn.cursor(); c.execute('DELETE FROM articles WHERE site=? AND source=?', ('rili.jin10.com','listing_data')); conn.commit(); conn.close()"
  ```
- 删除：清空所有“快讯/flash_api”数据（危险）
  ```powershell
  python -c "import sqlite3; conn=sqlite3.connect('finance.db'); c=conn.cursor(); c.execute('DELETE FROM articles WHERE site=? AND source=?', ('www.jin10.com','flash_api')); conn.commit(); conn.close()"
  ```
- 维护：回收空间（VACUUM）
  ```powershell
  python -c "import sqlite3; conn=sqlite3.connect('finance.db'); conn.execute('VACUUM'); conn.close()"
  ```
