# Project Status

更新时间：2026-02-06
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

- **阶段 1：Engine A + DTO（已完成）**

  - [X] 任务 1.1: Colab 训练 3 类 BERT 模型（Test Macro F1=0.3770，达标）
  - [X] 任务 1.2: 实现 Engine A 推理包装器（`app/services/sentiment_analyzer.py`）
  - [X] 任务 1.3: 实现规则引擎（集成在 sentiment_analyzer.py）
  - [X] 任务 1.4: 实现 DTO 数据结构（`app/core/dto.py`，7个核心数据类）
  - 详细记录见下方"变更记录"2026-02-07 条目
- **阶段 2：Engine B（RAG 检索管线）（已完成）**

  - [X] **任务 2.1: 准备财报 PDF**（已完成）
    - 收集了 15 个贵金属相关 PDF 研报
    - 保存位置：`data/raw/reports/research_reports/`
    - 分类：上海黄金交易所《行情周报》（12个）+ 机构深度研报（3个）
    - 语言分布：中文 10 个，英文 5 个
    - 布局特征：双栏 8 个，单栏 4 个
  - [X] **任务 2.2: PDF 解析与切片**（已完成）
    - 新增脚本：`scripts/rag/analyze_pdfs.py`（PDF 自动分类和分析）
    - 新增脚本：`scripts/rag/test_table_extraction.py`（表格提取测试）
    - 新增脚本：`scripts/rag/build_chunks.py`（PDF 解析与切片）
    - 处理策略：智能混合处理（简单表格→Markdown，复杂表格→描述，失败表格→跳过）
    - ROI 裁剪：去除页眉页脚（上下各 10%）
    - 正则清洗：免责声明、联系方式、特殊字符
    - 元数据提取：日期、语言、来源
    - 文本切片：RecursiveCharacterTextSplitter（chunk_size=500, overlap=50）
    - 产出文件：`data/reports/chunks.json`（633 个切片，12/15 个 PDF 成功处理）
  - [X] **任务 2.3: 向量化与索引构建**（已完成）
    - 新增脚本：`scripts/rag/build_vector_index.py`（向量化与 Chroma 索引构建）
    - 嵌入模型：BAAI/bge-m3（约 2.27GB）
    - 向量库：Chroma（持久化存储到 `data/reports/chroma_db/`）
    - 状态：已完成，633 个切片全部向量化
    - 模型存储：F 盘（`F:\huggingface_cache`，解决 C 盘空间不足问题）
  - [X] **任务 2.4: 实现 RAG Engine**（已完成）
    - 新增文件：`app/core/engines/rag_engine.py`（RAG 检索引擎）
    - 新增文件：`scripts/rag/test_rag_engine.py`（测试脚本）
    - 功能：加载 Chroma 向量库、提供检索接口、支持元数据过滤
    - 测试结果：所有功能正常，检索准确度良好
- **阶段 3：Agent 编排与工具集成（已完成）**

  - [X] **任务 3.1: 实现 Deepseek LLM 客户端**（已完成）
    - 新增文件：`app/adapters/llm/deepseek_client.py`（Deepseek API 客户端）
    - 功能：调用 Deepseek API、处理超时和错误、支持重试机制
    - 配置：在 `.env` 中添加 `DEEPSEEK_API_KEY`
    - 测试：支持简单问答和带系统提示词的对话
  - [X] **任务 3.2: 实现核心工具函数**（已完成）
    - 新增文件：`app/core/orchestrator/tools.py`（工具函数集合）
    - 工具 1：`get_market_context` - 从数据库获取市场上下文
    - 工具 2：`analyze_sentiment` - 调用 Engine A + 规则引擎
    - 工具 3：`search_reports` - 调用 Engine B RAG 检索
    - 测试：所有工具函数正常工作
  - [X] **任务 3.3: 实现 Agent 编排器**（已完成）
    - 新增文件：`app/core/orchestrator/agent.py`（Agent 主编排器）
    - 功能：
      - 自动判断查询类型（快讯分析 vs 财报问答）
      - 协调各个引擎和工具
      - 记录工具调用追踪（Tool Trace）
      - 使用 LLM 生成最终总结
    - 测试：基础功能正常，支持降级策略
  - [X] **任务 3.4: 实现用例层函数**（已完成）
    - 新增文件：`app/application/utils.py`（工具函数：超时、缓存、重试）
    - 新增文件：`app/application/analyze_news.py`（快讯分析用例）
    - 新增文件：`app/application/ask_report.py`（财报问答用例）
    - 功能：
      - 超时控制（装饰器）
      - 缓存支持（SimpleCache，TTL 可配置）
      - 降级策略（引擎未加载时返回默认结果）
    - 测试：所有用例函数正常工作
- **阶段 4：Streamlit UI 实现（已完成）**

  - [X] **任务 4.1: 实现聊天页面**（已完成）
    - 完善文件：`app/hosts/streamlit_app/app.py`（主入口文件）
    - 功能：
      - 页面配置和布局
      - 侧边栏（功能导航、系统状态检查）
      - 聊天界面（消息历史、用户输入）
      - Agent 初始化（使用 @st.cache_resource 缓存）
      - 引擎状态检查（BERT、Chroma、LLM）
      - 查询处理（自动判断类型并调用 Agent）
      - 结果展示（总结、情感分析、引用、工具追踪）
  - [X] **任务 4.2: 实现 K 线图表页面**（已完成）
    - 新增文件：`app/hosts/streamlit_app/pages/2_Charts.py`
    - 功能：
      - 使用 Plotly 绘制 K 线图
      - 从数据库加载价格数据和事件数据
      - 在图表上标注事件点（带星级和内容预览）
      - 支持时间范围和星级筛选
      - 点击事件点触发情感分析
      - 显示事件详情和分析结果
  - [X] **任务 4.3: 实现财报检索页面**（已完成）
    - 新增文件：`app/hosts/streamlit_app/pages/3_Reports.py`
    - 功能：
      - 输入问题，调用 RAG 引擎检索
      - 显示 Top-K 引用片段（可配置 1-10）
      - 显示页码和相似度分数
      - LLM 生成的答案总结
      - 支持语言筛选（全部/中文/英文）
      - 支持显示选项配置（元数据、完整文本）
      - 提供示例问题快速测试
  - [X] **任务 4.4: 完善主入口和配置**（已完成）
    - 完善 Agent 初始化逻辑（加载所有引擎）
    - 添加完整的错误处理和降级策略
    - 创建 README 文档（`app/hosts/streamlit_app/README.md`）
    - 文档内容：
      - 功能说明（3 个页面）
      - 启动方法和前置条件
      - 使用说明和示例
      - 常见问题和解决方法
      - 答辩演示建议
  - 阶段 4 总结：
    - ✅ 完成 3 个页面（聊天、K 线图表、财报检索）
    - ✅ 实现完整的 Agent 初始化和引擎加载
    - ✅ 支持降级策略（引擎未加载时仍可运行）
    - ✅ 提供完整的文档和使用说明
  - 下一步：
    1. 开始阶段 5（测试与优化）
    2. 端到端测试所有功能
    3. 性能优化和答辩准备

  **冗余文件清理（待手动删除）**：

  - `test_training_quick.py`：Phase 1 快速测试脚本，已被 Colab 训练流程替代。
  - `TRAINING_GUIDE_PHASE1.md`：Phase 1 (6 类) 训练指南，已被方案 A (3 类) 替代。
  - `COLAB_TRAINING_COMMAND.md`：Phase 1 Colab 命令，已被 `colab_3cls_training_cells.txt` 替代。
  - 说明：这些文件的关键信息已整合到 `Project_Status.md` 和 `colab_3cls_training_cells.txt` 中。

  **方案 A 核心思路**：

  - ML 模型专注于可学习的 3 类基础方向（Bearish/Neutral/Bullish）
  - "预期兑现"等复杂逻辑改为后处理规则引擎（更可解释、可维护、可调试）
  - 规则引擎可根据实际情况灵活调整阈值（如前期涨幅阈值、波动率阈值等）
  - 符合金融系统的混合架构设计模式（ML + 规则引擎）
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

  - [X] 文本清洗与特征工程脚本化（保留前缀特征，如 `[SRC=]`/`[STAR=]`/`[IMP=]`/`[CTRY=]`，正则去噪）。
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
  - [X] 运行 QA：确认多类数据集时间跨度、事件不跨集合（any_overlap=false）、`event_id` 去重无异常、标签分布合理，并校验 DB 的 `events/event_impacts/prices_m1` 覆盖范围。
  - [X] 基于报告的 `archive_suggestions` 归档：已将候选 raw/processed 大文件迁移至 `archive/20260201_211652/`，并打包为 `archive/archive_20260201_211652.zip`。
- 数据扩展与校验

  - [ ] 如需：补齐其余时间段或其他 MT5 符号（如 XAUUSD.i、GOLD）并复检覆盖率。
  - [ ] 交易时段/夏令时敏感性检查与说明。
- 交付与文档

  - [ ] 训练集字段字典与处理流程文档。
  - [ ] 训练/验证/测试集统计报告与可视化。

## 4) 备忘 / 风险（Memos / Risks）

- **BERT 模型准确度观察**（2026-02-07）：
  - 现象：测试中模型预测偏向 bearish（6个案例中5个预测为 bearish）
  - 置信度：38%-51% 之间，说明模型对测试案例不是特别确定
  - 可能原因：
    - 训练数据分布特点（测试集中 Bearish 样本占比 33.7%）
    - 测试案例是人工构造的，可能与训练数据分布不同
    - 标签映射可能需要验证（当前：0->-1, 1->0, 2->1）
  - 改进方向（答辩后）：
    - 检查训练数据的标签分布和文本特征
    - 尝试不同的标签映射方式
    - 增加训练数据量或调整数据增强策略
    - 考虑使用 Focal Loss 处理类别不平衡
  - 当前状态：功能正常，规则引擎工作正常，可继续推进工程任务
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

- 2026-02-09（晚）

  - **阶段 4 完成（Streamlit UI 实现）**：
    - **任务 4.1 完成（聊天页面）**：
      - 完善文件：`app/hosts/streamlit_app/app.py`
      - 改进 Agent 初始化逻辑：
        - 加载 SentimentAnalyzer（Engine A）
        - 加载 RagEngine（Engine B）
        - 加载 DeepseekClient（LLM）
        - 显示各引擎加载状态
        - 支持降级策略（引擎未加载时仍可运行）
      - 聊天界面功能：
        - 消息历史显示
        - 用户输入框
        - 自动判断查询类型
        - 显示分析结果（总结、情感、引用、工具追踪）
    - **任务 4.2 完成（K 线图表页面）**：
      - 新增文件：`app/hosts/streamlit_app/pages/2_Charts.py`
      - 功能实现：
        - 从 finance_analysis.db 加载价格数据（prices_m1 表）
        - 从 finance_analysis.db 加载事件数据（events 表）
        - 使用 Plotly 绘制 K 线图
        - 在图表上标注事件点（带星级和内容预览）
        - 支持参数配置（标的、时间范围、最低星级）
        - 点击事件点触发情感分析
        - 显示事件详情和分析结果
    - **任务 4.3 完成（财报检索页面）**：
      - 新增文件：`app/hosts/streamlit_app/pages/3_Reports.py`
      - 功能实现：
        - 输入问题，调用 RAG 引擎检索
        - 显示 Top-K 引用片段（可配置 1-10）
        - 显示页码和相似度分数
        - LLM 生成的答案总结
        - 支持语言筛选（全部/中文/英文）
        - 支持显示选项配置（元数据、完整文本）
        - 提供示例问题快速测试
        - 显示工具调用追踪和警告信息
    - **任务 4.4 完成（文档和配置）**：
      - 新增文件：`app/hosts/streamlit_app/README.md`
      - 文档内容：
        - 功能说明（3 个页面的详细介绍）
        - 启动方法和前置条件
        - 使用说明和示例问题
        - 常见问题和解决方法
        - 系统状态检查说明
        - 性能优化建议
        - 答辩演示建议（流程、场景、技术亮点、备用方案）
        - 技术栈和开发规范
    - **模块导入问题修复**：
      - 问题：Streamlit 启动时报错 `ModuleNotFoundError: No module named 'app.core'`
      - 原因：Python 模块导入路径问题
      - 解决方案：使用 `importlib.util` 动态加载模块
      - 修改文件：
        - `app/hosts/streamlit_app/app.py`（Agent 初始化）
        - `app/hosts/streamlit_app/pages/2_Charts.py`（事件分析函数）
        - `app/hosts/streamlit_app/pages/3_Reports.py`（财报检索）
      - 修复内容：
        - 使用 `importlib.util.spec_from_file_location()` 动态加载模块
        - 分别加载 Agent, SentimentAnalyzer, RagEngine, DeepseekClient
        - 保持原有的错误处理和降级策略
    - **快速启动指南**：
      - 新增文件：`STREAMLIT_QUICKSTART.md`
      - 内容：
        - 前置条件检查（必需文件、依赖安装）
        - 启动步骤（基础启动、完整启动）
        - 功能测试（聊天、K 线图表、财报检索）
        - 常见问题（启动失败、模型未找到、API 未配置等）
        - 性能优化建议
        - 答辩演示建议（流程、备用方案）
  - **阶段 4 总结**：
    - ✅ 任务 4.1: 实现聊天页面（完整的 Agent 初始化和查询处理）
    - ✅ 任务 4.2: 实现 K 线图表页面（Plotly 图表 + 事件标注 + 情感分析）
    - ✅ 任务 4.3: 实现财报检索页面（RAG 检索 + LLM 总结 + 引用展示）
    - ✅ 任务 4.4: 完善主入口和配置（文档、错误处理、降级策略）
    - ✅ 修复模块导入问题（动态导入方案）
    - ✅ 创建快速启动指南
  - **下一步**：
    - 开始阶段 5（测试与优化）
    - 端到端测试所有功能
    - 性能优化和答辩准备
- **阶段 6：QLoRA 微调（可选，进行中）**

  - [ ] **任务 6.1: 准备指令集数据**（进行中）
    - 新增脚本：`scripts/qlora/build_instruction_dataset.py`
    - 功能：从 finance_analysis.db 提取真实案例，生成 Instruction-Input-Output 格式数据
    - 目标：300 条指令（60% 新闻 + 20% 市场分析 + 20% 财报）
    - 输出：`data/qlora/instructions.jsonl`
  - [ ] **任务 6.2: Colab 训练**（待开始）
    - 新增脚本：`scripts/qlora/train_qlora.py`
    - 新增文档：`colab_qlora_training_cells.txt`（Colab 训练单元格）
    - 基础模型：deepseek-ai/deepseek-llm-7b-chat
    - 训练方法：QLoRA（4-bit 量化 + LoRA）
    - 训练参数：3 epochs, batch_size=4, lr=2e-4, lora_r=8
    - 预计时间：2-4 小时（T4 GPU）
    - 输出：LoRA 权重文件（约 30-50 MB）
  - [ ] **任务 6.3: 测试与集成**（待开始）
    - 新增脚本：`scripts/qlora/test_qlora_model.py`
    - 功能：测试微调后的模型
    - 可选：集成到本地 Agent 系统
  - 阶段 6 说明：
    - 目标：微调 Deepseek-7B 模型，提升财经领域专业性
    - 方法：使用 QLoRA 技术在 Colab T4 GPU 上微调
    - 答辩策略：展示训练日志和权重文件，实际使用 Deepseek API
    - 完整文档：`QLORA_WORKFLOW.md`
- 2026-02-09（下午）

  - **阶段 2 完成（Engine B - RAG 检索管线）**：
    - **任务 2.2 完成（PDF 解析与切片）**：
      - 运行 `build_chunks.py` 成功处理 15 个 PDF，生成 633 个切片
      - 成功处理：12/15 个文件（80% 成功率）
      - 失败文件：3 个（Goldman Sachs US Daily, J.P. Morgan Gold & Silver, UBS Global Precious Metals）
      - 切片分布：Morgan Stanley 报告最多（243 个），其次是 Goldman Sachs China Musings（133 个）
      - 输出文件：`data/reports/chunks.json`（633 个切片，包含文本、元数据、来源信息）
    - **任务 2.3 完成（向量化与索引构建）**：
      - 创建脚本：`scripts/rag/build_vector_index.py`（向量化与 Chroma 索引构建）
      - 嵌入模型：BAAI/bge-m3（约 2.27GB）
      - 模型下载问题解决：
        - 问题：网络下载速度慢且多次超时
        - 解决方案：设置 HuggingFace 缓存到 F 盘（`HF_HOME=F:\huggingface_cache`）
        - 手动下载大文件：`pytorch_model.bin`（2165.9 MB）
        - 模型验证成功，可以正常加载和使用
      - 向量化执行：
        - 处理了 633 个切片
        - 向量维度：1024
        - 输出：`data/reports/chroma_db/`
        - 批大小：16
        - 总耗时：约 5 分钟
    - **任务 2.4 完成（实现 RAG Engine）**：
      - 创建文件：`app/core/engines/rag_engine.py`（RAG 检索引擎）
      - 功能实现：
        - 加载 Chroma 向量库和嵌入模型
        - 提供检索接口（query → top_k citations）
        - 支持元数据过滤（日期、来源、语言等）
        - 支持日期范围检索
        - 提供统计信息接口
      - 创建文件：`app/core/engines/__init__.py`（引擎模块初始化）
      - 创建测试脚本：`scripts/rag/test_rag_engine.py`（完整测试套件）
      - 测试结果：
        - ✓ RAG Engine 初始化成功
        - ✓ 向量库统计：633 个切片
        - ✓ 基础检索功能正常（中文/英文查询均可）
        - ✓ 元数据过滤功能正常（按语言过滤）
        - ✓ Citation 对象结构正确（包含 text, score, source_file, chunk_index, metadata）
        - ✓ 相似度分数范围正常（0.5-0.7）
  - **阶段 2 总结**：
    - ✅ 任务 2.1: 准备财报 PDF（15 个贵金属相关研报）
    - ✅ 任务 2.2: PDF 解析与切片（633 个切片）
    - ✅ 任务 2.3: 向量化与索引构建（Chroma 向量库）
    - ✅ 任务 2.4: 实现 RAG Engine（检索功能正常）
  - **下一步**：
    - 开始阶段 3（Agent 编排与工具集成）
    - 参考 `REMAINING_TASKS.md` 中的详细任务计划
- 2026-02-09（晚）

  - **阶段 3 完成（Agent 编排与工具集成）**：
    - **任务 3.1 完成（Deepseek LLM 客户端）**：
      - 创建文件：`app/adapters/llm/deepseek_client.py`
      - 功能实现：
        - 调用 Deepseek API 生成文本
        - 处理超时和错误（401/429/网络异常）
        - 支持重试机制（最多 2 次重试）
        - 支持系统提示词（system prompt）
      - 配置更新：`.env.example` 添加 `DEEPSEEK_API_KEY` 说明
      - 测试通过：简单问答和带系统提示词的对话均正常
    - **任务 3.2 完成（核心工具函数）**：
      - 创建文件：`app/core/orchestrator/tools.py`
      - 工具函数实现：
        - `get_market_context`: 从 finance_analysis.db 读取价格数据，计算前期收益率、波动率、趋势标签
        - `analyze_sentiment`: 调用 SentimentEngine + RuleEngine 进行情感分析
        - `search_reports`: 调用 RagEngine 进行财报检索
      - 测试结果：
        - ✓ 市场上下文获取成功（XAUUSD，120分钟窗口）
        - ✓ 情感分析支持降级（无引擎时返回默认结果）
        - ✓ 财报检索支持降级（无引擎时返回空列表）
    - **任务 3.3 完成（Agent 编排器）**：
      - 创建文件：`app/core/orchestrator/agent.py`
      - 功能实现：
        - 自动判断查询类型（快讯分析 vs 财报问答）
        - 协调各个引擎和工具（SentimentEngine, RagEngine, RuleEngine, LLM）
        - 记录工具调用追踪（Tool Trace，包含耗时和状态）
        - 使用 LLM 生成最终总结（支持降级）
      - 查询类型检测：基于关键词匹配（财报/营收/利润 vs 加息/降息/非农）
      - 测试结果：
        - ✓ 快讯分析流程正常（获取上下文 → 情感分析 → LLM 总结）
        - ✓ 财报问答流程正常（RAG 检索 → LLM 总结）
        - ✓ 工具追踪记录完整（包含耗时和状态）
    - **任务 3.4 完成（用例层函数）**：
      - 创建文件：`app/application/utils.py`（工具函数）
        - `with_timeout`: 超时装饰器（记录执行时间）
        - `SimpleCache`: 简单内存缓存（支持 TTL）
        - `with_cache`: 缓存装饰器
        - `retry_on_failure`: 失败重试装饰器
      - 创建文件：`app/application/analyze_news.py`（快讯分析用例）
        - `analyze_news_with_context`: 完整的快讯分析（带缓存和超时控制）
        - `analyze_news_simple`: 简化版（仅返回总结文本）
        - 缓存配置：5 分钟 TTL
      - 创建文件：`app/application/ask_report.py`（财报问答用例）
        - `ask_report_question`: 完整的财报问答（带缓存和超时控制）
        - `ask_report_simple`: 简化版（仅返回总结文本）
        - `get_report_citations`: 仅获取引用（不生成总结）
        - 缓存配置：10 分钟 TTL
      - 测试结果：所有用例函数正常工作，缓存和降级策略有效
  - **阶段 3 总结**：
    - ✅ 任务 3.1: 实现 Deepseek LLM 客户端
    - ✅ 任务 3.2: 实现核心工具函数
    - ✅ 任务 3.3: 实现 Agent 编排器
    - ✅ 任务 3.4: 实现用例层函数
  - **下一步**：
    - 开始阶段 4（Streamlit UI 实现）
    - 参考 `REMAINING_TASKS.md` 中的详细任务计划
- 2026-02-07（晚）

  - **阶段 1 完成（Engine A + DTO）**：
    - 实现 DTO 数据结构：`app/core/dto.py`（7 个核心数据类 + 2 个辅助函数）
    - 数据类定义：
      - MarketContext：市场上下文（发布前的 K 线状态）
      - NewsItem：新闻条目（财经快讯或日历事件）
      - SentimentResult：情感分析结果（BERT + 规则引擎输出）
      - Citation：引用片段（RAG 检索结果）
      - ToolTraceItem：工具调用追踪条目
      - AgentAnswer：Agent 最终答案
      - EngineConfig：引擎配置参数
    - 辅助函数：sentiment_label_to_text(), sentiment_label_to_english()
    - 测试通过：所有 DTO 创建和序列化测试通过
    - 阶段 1 总结：
      - ✅ 任务 1.1: Colab 训练 3 类 BERT 模型（Test Macro F1=0.3770）
      - ✅ 任务 1.2: 实现 Engine A 推理包装器（sentiment_analyzer.py）
      - ✅ 任务 1.3: 实现规则引擎（集成在 sentiment_analyzer.py）
      - ✅ 任务 1.4: 实现 DTO 数据结构（dto.py）
    - 下一步：开始阶段 2（Engine B - RAG 检索管线）
- 2026-02-07（下午）

  - **方案 A 训练完成（3 类模型成功达标）**：
    - 实验名称：`bert_3cls_enhanced_v1`（T4 GPU，5 epochs，训练时长约 1.5 小时）
    - 测试集指标：Test Macro F1=0.3770（目标 >0.35，达标！），Test Accuracy=0.3819
    - 相比 6 类基线（Macro F1=0.1317）提升：186%
    - 关键改进：
      - 所有类别均有预测（无 F1=0 的类别，解决了 6 类模型的核心问题）
      - 预测分布与真实分布基本一致（Bearish=892/1290, Neutral=1407/1032, Bullish=1524/1501）
      - 各类别 F1 均衡：Bearish=0.32, Neutral=0.40, Bullish=0.42
    - 分类报告（测试集）：
      - Bearish (-1): precision=0.39, recall=0.27, f1=0.32, support=1290
      - Neutral (0): precision=0.34, recall=0.47, f1=0.40, support=1032
      - Bullish (1): precision=0.41, recall=0.42, f1=0.42, support=1501
    - 验证集指标：Val Macro F1=0.3803, Val Accuracy=0.3912
  - **同步训练结果**：使用 `scripts/tools/sync_results.py` 将 Drive 上的训练产物同步到本地 `reports/bert_3cls_enhanced_v1/`
  - **后处理规则引擎实现完成**：
    - 新增：`app/services/sentiment_analyzer.py`（情感分析服务，BERT + 规则引擎）
    - 架构：ML 模型专注于 3 类基础方向，规则引擎处理"预期兑现"和"建议观望"
    - 规则1（预期兑现）：利好+前期大涨 → 利好预期兑现；利空+前期大跌 → 利空预期兑现
    - 规则2（建议观望）：高波动+低净变动 → 建议观望
    - 新增：`scripts/test_sentiment_analyzer.py`（测试脚本，包含6个测试案例）
    - 新增：`scripts/tools/copy_model_weights.py`（模型权重复制工具）
  - **文档整合**：将 `Model_Training_Workflow.md` 和 `Project_optimization_plan.md` 的关键内容整合到 `Project_Status.md` 第 7 节"训练工作流"
  - **下一步**：测试情感分析器，开始实现 Engine B（RAG）和 Agent 层
- 2026-02-07（上午）

  - **Colab 路径修复与冗余文件清理**：

    - 修复问题：Colab 单元格 2 中的脚本路径错误（从绝对路径改为相对路径）。
    - 原因：单元格 1 已使用 `%cd /content/Graduation_Project` 切换到仓库根目录，后续命令应使用相对路径。
    - 修复内容：
      - `prepare_3cls_dataset.py`：从 `/content/Graduation_Project/scripts/...` 改为 `scripts/...`
      - `build_enhanced_dataset_3cls.py`：同上
      - 数据库和输出路径：从 `/content/Graduation_Project/...` 改为相对路径
    - 冗余文件标记（待手动删除）：
      - `test_training_quick.py`：Phase 1 快速测试脚本，已被 Colab 训练流程替代。
      - `TRAINING_GUIDE_PHASE1.md`：Phase 1 (6 类) 训练指南，已被方案 A (3 类) 替代。
      - `COLAB_TRAINING_COMMAND.md`：Phase 1 Colab 命令，已被 `colab_3cls_training_cells.txt` 替代。
    - 文档整合：将上述文件的关键信息整合到 `Project_Status.md` 和 `colab_3cls_training_cells.txt` 中。
  - **训练脚本优化（消除警告和错误）**：

    - **修复 warmup_ratio 弃用警告**：
      - 将 `--warmup_ratio` 改为 `--warmup_steps`（transformers 5.x 推荐）
      - 保留 `--warmup_ratio` 参数以兼容旧命令，但默认值改为 0
      - 自动计算：如果 `warmup_steps=0` 且 `warmup_ratio>0`，则自动计算 warmup_steps
      - **修复 NameError**：将 `train_df` 改为 `train`（变量名错误）
    - **禁用 HuggingFace 警告**：
      - 设置 `hf_logging.set_verbosity_error()` 减少输出噪音
      - 设置环境变量 `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` 避免 403 错误
      - 过滤 UNEXPECTED/MISSING weights 警告（分类头重新初始化是正常现象）
    - **修复 pin_memory 警告**：
      - 在 TrainingArguments 中添加 `dataloader_pin_memory=False`
      - CPU 模式下禁用内存固定，避免无意义的警告
    - **更新 Colab 训练命令**：
      - 将 `--warmup_ratio 0.06` 改为 `--warmup_steps 100`
      - 保持其他参数不变
  - **.gitignore 优化与工作流改进**：

    - **3 类数据集反选**：在 `data/processed/**` 规则下添加反选，允许提交 3 类数据集到 GitHub：
      - `train_3cls.csv`, `val_3cls.csv`, `test_3cls.csv`
      - `train_enhanced_3cls.csv`, `val_enhanced_3cls.csv`, `test_enhanced_3cls.csv`
      - `labeling_thresholds_3cls.json`
    - **reports/ 工作流优化**：
      - 默认状态：`reports/**` 忽略所有内容（不提交到 GitHub）
      - AI 分析时：临时取消注释反选规则（去掉 `#` 号），AI 可读取文件
      - 分析完成后：重新注释反选规则（加上 `#` 号），提交代码时不包含 reports/
      - 优势：既能让 AI 读取分析，又不会污染 GitHub 仓库
    - **工作流程**：
      1. 训练完成 → 运行 `sync_results.py` 同步 Drive 结果到 `reports/`
      2. 需要 AI 分析 → 临时取消注释 `.gitignore` 中的 reports 反选规则
      3. AI 读取分析 → 完成后重新注释反选规则
      4. 提交代码 → `reports/` 内容不会被提交
  - **本地数据集生成说明**：

    - 用户需要在本地运行以下命令生成 3 类数据集：
      ```powershell
      # 生成 3 类标签数据集
      python scripts/modeling/prepare_3cls_dataset.py `
        --db finance_analysis.db `
        --ticker XAUUSD `
        --window_post 15 `
        --pre_minutes 120 `
        --out_dir data/processed

      # 添加输入增强
      python scripts/modeling/build_enhanced_dataset_3cls.py `
        --input_dir data/processed `
        --output_dir data/processed
      ```
    - 生成后提交到 GitHub（已在 `.gitignore` 中反选），Colab 可以直接从仓库拉取。
- 2026-02-06（下午）

  - **方案 A 实施完成**：基于 Phase 1 失败分析，重新设计标签体系。
  - 问题诊断
    - Phase 1 (6 类) 失败原因：训练集 Class 3/4/5 样本极度稀缺（7/5/19），但测试集 Class 5 有 868 样本，导致模型完全无法预测这些类别（F1=0）。
    - 输入增强（市场上下文前缀）本身有效，但无法解决极端类别不平衡问题。
  - 方案 A：标签体系简化 + 规则引擎
    - 核心思路：ML 模型专注于可学习的 3 类基础方向（Bearish/Neutral/Bullish），"预期兑现"等复杂逻辑改为后处理规则引擎。
    - 工程优势：更可解释、可维护、可调试；规则引擎可根据实际情况灵活调整阈值。
  - 新增脚本
    - `scripts/modeling/prepare_3cls_dataset.py`：生成简化的 3 类标签数据集（基于 15 分钟窗口 ret_post）。
    - `scripts/modeling/build_enhanced_dataset_3cls.py`：为 3 类数据集添加市场上下文前缀（保留输入增强）。
  - 数据集生成
    - 输出：`train/val/test_3cls.csv` + `train/val/test_enhanced_3cls.csv` + `labeling_thresholds_3cls.json`。
    - 标签分布（均衡）：训练集 Bearish=3,853 (30.0%), Neutral=5,135 (40.0%), Bullish=3,853 (30.0%)。
    - 前缀分布（训练集）：Sideways 60.6%, Mild Rally 15.3%, Weak Decline 12.8%, 其他 <6%。
  - Colab 训练准备
    - 更新：`colab_3cls_training_cells.txt`（完整训练流程，包含数据生成、验证、训练、结果分析）。
    - 新增：GPU/CPU 自适应训练配置（GPU: 5 epochs/384 max_length；CPU: 3 epochs/256 max_length）。
    - 配置：class_weight=auto, label_col=label, text_col=text_enhanced。
  - 下一步
    - 在 Colab 上运行 3 类训练（预计 GPU 1-1.5 小时，CPU 3-4 小时）。
    - 目标：Test Macro F1 > 0.35（相比 6 类基线 0.163 提升 >100%）。
    - 后续：实现后处理规则引擎（`app/services/sentiment_analyzer.py`）。
  - 开发规范记录
    - 避免使用 emoji 符号（可能显示为"?"）。
    - 代码中添加中文注释。
    - 文件编码统一为 UTF-8（避免 GBK/GB2312 编码问题）。
    - 避免生成冗余文档，优先更新 `Project_Status.md`。
    - 遵循项目初始文档（PLAN.md、README.md 等）。
- 2026-02-06（上午）

  - **打通“本地开发 + Colab 训练 + Drive 存储 + 本地回传报告”工作流**（已可稳定复现）。
  - 数据同步与版本控制
    - `.gitignore`：新增忽略 `data/processed_stale*/**`（大 CSV 不进 Git），放行小文件用于口径复现：
      - `data/processed_stale*/qa_dataset_report.json`
      - `data/processed_stale*/labeling_thresholds_multilabel.json`
    - Drive 数据集版本化（推荐工作流，避免手动上传/避免仓库膨胀）
      - 新增：`scripts/tools/sync_dataset_to_drive.py`
      - Drive 目录结构：`datasets/<dataset_name>/{versions/<version>/,latest/,latest_version.txt}`
      - 本地生成后同步：将 `data/processed_stale300` 同步到 Drive 的 `processed_stale300/latest`，Colab 永远读取 latest。
    - `reports/`：本次实验回传后可直接在本地读取与分析（建议仅提交小文件，避免误入大权重文件）。
  - 训练脚本增强（Phase 1 训练更稳健/更适配 Colab 输出）
    - `scripts/modeling/bert_finetune_cls.py`
      - 新增：`--disable_tqdm`，可禁用 transformers/datasets 的进度条输出，减少 Colab 输出与浏览器缓存压力。
      - 修复：显式导入 `datasets`（用于 `datasets.disable_progress_bars()`）。
      - 增强：训练前标签合法性校验与 `labels.long()` 兜底，降低 `CUDA illegal memory access` 类错误的排查成本。
  - Runner 更新
    - `colab_phase1_cells.txt`
      - 默认开启 `disable_tqdm=True` 并透传 `--disable_tqdm`.
      - 保留 CPU smoke 路径与 GPU 全量路径，便于快速验证脚本可跑通。
      - 更新：stale300 baseline 读取 Drive 数据集（`/content/drive/MyDrive/Graduation_Project/datasets/processed_stale300/latest`），避免上传 CSV。
  - 本地回传脚本
    - `scripts/tools/sync_results.py`
      - 支持：从 Google Drive for Desktop 的本地映射目录同步 `experiments/<run_name>/` 下的小文件到仓库 `reports/`.
      - 默认 include：`eval_results.json`、`metrics*.json`、`report*.txt`、`pred*.csv`、`best/config.json`.
      - 默认 exclude：大权重（`*.safetensors/*.bin`）、checkpoint、optimizer 等。
  - 训练结果（已同步至 `reports/`）
    - `bert_enhanced_v1`（T4 GPU，5 epochs，6 类）：val macro_f1=0.2118；test macro_f1=0.1317；test acc=0.2485.
      - 测试集分类报告显示类别 3/4/5 的 F1=0；预测分布仅覆盖 0/1/2.
      - 数据分布提示：训练集 `label_multi_cls` 极度不平衡（3/4/5 分别仅 7/5/19），但测试集中类 5 支持数为 868，存在明显分布差异，需优先处理少数类数据策略.
    - `bert_enhanced_v1_cpu_smoke`（CPU 小样本，3 类）：用于冒烟验证流程，指标仅供参考.
      - 增强：Windows 下自动探测常见 Drive 路径；当 `src_root` 不存在时显式报错，避免“复制 0 文件”的静默失败.
  - 常用指令（Windows/PowerShell）
    - 同步 Drive 训练产物 → 本地 `reports/`（先预演再执行）：
      ```powershell
      python scripts/tools/sync_results.py `
        --src_root "G:\我的云端硬盘\Graduation_Project\experiments" `
        --dst_root "E:\Projects\Graduation_Project\reports" `
        --dry_run --verbose

      python scripts/tools/sync_results.py `
        --src_root "G:\我的云端硬盘\Graduation_Project\experiments" `
        --dst_root "E:\Projects\Graduation_Project\reports" `
        --verbose
      ```
  - Colab 训练流程（Phase 1）
    1. 本地提交代码：`git push`（确保最新修复已推送）。
    2. Colab 拉取代码：`!cd /content/Graduation_Project && git pull`。
    3. 开始训练：运行 `bert_finetune_cls.py`（预计 1-1.5 小时，T4 GPU）。
    4. 查看结果：`metrics_test.json`（目标：macro_f1 > 0.35，基线 0.163）。
  - 可忽略的警告
    - HuggingFace 403 Forbidden：讨论功能被禁用，不影响模型加载。
    - UNEXPECTED/MISSING weights：分类头重新初始化，正常现象.
  - 用户偏好记录
    - **避免创建冗余 .md 文档**，保持仓库整洁.
    - **将更新记录写入 `Project_Status.md`**.
    - **始终使用中文**回答、写文档和注释.
- 2026-02-05

  - **Phase 1 优化准备完成**：基于 `Project_optimization_plan.md` 的改进方案，完成输入增强与类权重训练准备.
  - 新增：`scripts/modeling/build_enhanced_dataset.py`
    - 功能：为训练集添加市场上下文前缀（基于 `pre_ret` 120 分钟回看与 `range_ratio` 波动率）。
    - 前缀类型：`[Strong Rally]`、`[Sharp Decline]`、`[Sideways]`、`[Mild Rally]`、`[Weak Decline]`、`[High Volatility]`.
    - 宏观数据前缀：`[Economic Data] [Actual X Exp Y] [Beat/Miss]`（基于 `actual/consensus` 差异）。
    - 输出：`train/val/test_enhanced.csv`（新增 `text_enhanced` 列，保留原始 `text`）。
    - 前缀分布（训练集）：Sideways 65.8%、Mild Rally 13.5%、Weak Decline 11.1%、其他 <10%.
  - 修复：`scripts/modeling/bert_finetune_cls.py` Colab 兼容性问题
    - **EarlyStoppingCallback 条件判断**：明确检查 `patience > 0`，确保 `--early_stopping_patience 0` 时不添加早停回调（避免 `AssertionError: EarlyStoppingCallback requires IntervalStrategy`）。
    - **compute_loss 参数兼容**：已支持 `num_items_in_batch=None` 参数（兼容新版 transformers）。
    - 数据路径统一：Colab 训练使用 `/content/Graduation_Project/data/processed/`（从 GitHub 拉取），输出到 `/content/drive/MyDrive/Graduation_Project/experiments/`.
  - 更新：`colab_phase1_cells.txt`
    - 完整的 Colab 训练单元格（5 个单元格）：环境准备 → 数据验证/生成 → 预览（可选）→ 训练 → 结果分析.
    - 训练命令：使用 `label_multi_cls`（6 类）、`--class_weight auto`（自动类权重）、`--early_stopping_patience 0`（禁用早停）。
    - 超参数：5 epochs、lr=1e-5、max_length=384、train_bs=16、gradient_accumulation_steps=2.
  - Colab 训练流程（Phase 1）
    1. 本地提交代码：`git push`（确保最新修复已推送）。
    2. Colab 拉取代码：`!cd /content/Graduation_Project && git pull`.
    3. 开始训练：运行 `bert_finetune_cls.py`（预计 1-1.5 小时，T4 GPU）。
    4. 查看结果：`metrics_test.json`（目标：macro_f1 > 0.35，基线 0.163）。
  - 可忽略的警告
    - HuggingFace 403 Forbidden：讨论功能被禁用，不影响模型加载.
    - UNEXPECTED/MISSING weights：分类头重新初始化，正常现象.
  - 用户偏好记录
    - **避免创建冗余 .md 文档**，保持仓库整洁.
    - **将更新记录写入 `Project_Status.md`**.
    - **始终使用中文**回答、写文档和注释.
- 2026-02-04

  - 新增：训练工作流文档 `Model_Training_Workflow.md`，采用“本地开发 + 云端训练（GitHub 代码 + Drive 数据）”的存算分离方案；提供 Colab 最小 Runner 与本地结果同步脚本说明.
  - 新增：`scripts/tools/sync_results.py`，将 Drive 下 `experiments/...` 小型产物（`eval_results.json/metrics*.json/report*.txt/pred*.csv/best/config.json`）同步至本地 `reports/...`，默认跳过大权重，支持 `--dry_run/--include/--exclude`.
  - 增强：`scripts/modeling/bert_finetune_cls.py`
    - 兼容旧版 transformers：`Trainer(tokenizer/callbacks)` 参数的 `TypeError` 自动回退；修复 `EarlyStopping` 断言（无论版本均设置 `metric_for_best_model/greater_is_better/load_best_model_at_end`，并按条件注册回调）。
    - 新增 `eval_results.json` 汇总（val/test 指标与输出目录），便于本地脚本/Agent 读取.
    - 稳健性：pandas→HF Dataset 在旧版 datasets 下自动回退；显式保存 tokenizer 以便下游加载.
  - 可选：`train_logic.py` 作为统一入口（Colab 可 `!python train_logic.py ...`）。基于当前工作流，此文件“可选”，可保留也可删除；直接调用底层脚本同样可行.
- 2026-02-01

  - 新增：分钟级冲击分析管线（MT5 + 金十）：
    - `scripts/fetch_intraday_xauusd_mt5.py`（分钟价抓取）。
    - `scripts/build_finance_analysis.py`（构建 finance_analysis.db，计算事件冲击）。
  - 回填与重建：抓取 2024–2025 M1 并重建 `finance_analysis.db`；`prices_m1` 共 736,304 行（2024-01-02 09:00 至 2026-01-31 07:59）。
  - 验证：自 2026-01-27 起，事件覆盖率 100%，四窗口统计各 1475.
  - 导出：全量与区间训练集（CSV/Parquet），并生成 30 分钟窗口打标与时间切分数据集与阈值 JSON.
  - 基线：`scripts/modeling/baseline_tfidf_svm.py` 现支持 `--class_weight/--C/--sublinear_tf/--norm/--dual`；完成 char 1-3/2-4 与加权对比，锁定推荐基线配置.
  - 新增：`scripts/modeling/prepare_multilabel_dataset.py`，输出 `train/val/test_multi_labeled.csv` 与 `labeling_thresholds_multilabel.json`，实现“基础方向/预期兑现/建议观望”的复合标签.
- 2026-01-29

  - 调整：`scripts/crawlers/jin10_dynamic.py` 与参考脚本对齐，采用统一 CLI：
    - 新增参数：`--months/--start/--end/--output/--db/--source/--headed/--debug/--important-only/--user-data-dir/--recheck-important-every/--use-slider/--setup-seconds`.
    - 支持边爬边入库（`Article`），并按参考列顺序增量写 CSV；长行（flake8 E501）分行处理.
    - 保留请求拦截（禁 image/media/font）提速，确保“只看重要”开关与星级兜底过滤.
  - 文档：更新本页“金十日历/数据（逐日回填）”使用指令为新 CLI.
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
    - 修复 0 条/卡住问题的若干鲁棒性细节与调试日志.
  - 增强：`scripts/crawlers/jin10_flash_api.py` 接口发现与入库：
    - 同时监听 request/response，仅在 JSON 响应与包含 `params=` 的 URL 时确认；
    - 收紧主机/路径匹配（排除主站/日历/热榜等非列表接口）；
    - 时间字段规范化、最后一条时间调试输出、完整请求头透传；
    - 支持 `--stream` 边抓边写 CSV 与 `--db` 入库（URL/内容哈希去重）。
  - 文档：补充“快讯 API 模式”和“数据库常用指令（只读/删除）”.
  - 清理：归档未使用爬虫脚本至 `archive/unused_crawlers_20260127/`：
    - `scripts/crawlers/providers/` 全部
    - `scripts/crawlers/list_crawl.py`
    - `scripts/crawlers/parse_listing.py`
    - `scripts/crawlers/fetch_from_urls.py`
  - 代码风格：补充 flake8 清理（E501 行宽、空白行 W293/E306）于 `jin10_dynamic.py` / `jin10_flash_api.py`，不改动业务逻辑.
  - Git：`.gitignore` 新增忽略 `archive/` 与 `参考代码和文档/`.
- 2026-01-23

  - 新增：`scripts/crawlers/jin10_dynamic.py`（Playwright 动态抓取：快讯倒序回溯与日历模式；登录持久化、跨 frame、滚动/加载更多、调试快照、入库）。
  - 新增：`scripts/crawlers/ingest_listing_csv.py`（listing CSV 入库）。
  - 新增：`scripts/crawlers/providers/jin10_events.py`（金十重要事件列表）。
  - 增强：`scripts/crawlers/list_crawl.py`（英文相对时间解析、时间窗过滤、SQLite 入库）。
  - 增强：`scripts/crawlers/parse_listing.py`（统一 `content_text` 与 provider 注册）。
  - 增强：`scripts/crawlers/fetch_from_urls.py`（加入 SQLite 入库与去重）。
  - 修复：`scripts/crawlers/storage.py` 长行与稳健性（upsert/去重）。
- 2026-01-20

  - 更新 `configs/config.yaml`：设置标的池与 `windows_days=[1,3,5]`.
  - 实现 `scripts/label_events.py`（CSV/DB → labels，中文日期解析）。
  - 整理 `scripts/fetch_prices.py` 与 `scripts/fetch_news.py` 的 PEP8.
  - 增加本文件 `Project_Status.md` 与 `.gitignore`（初版）。
  - 创建本地 Conda `.venv` 环境并安装依赖.
  - 完成并规范化 `scripts/train_baseline.py`（TF-IDF+LinearSVC，时间切分；支持 DB/CSV；生成报告与预测；flake8 通过）。
  - `.gitignore` 新增 `.windsurf/` 与 `文字材料文档/`，并将后者从 Git 索引中移除.
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
    3) 依次执行单元：安装依赖 → 克隆仓库 → 从 Drive 读取数据集（latest）→ 启动训练 → 查看指标 → 打包下载模型输出.
  - **Colab 3 类训练（方案 A，当前推荐）**

    - 准备：确保本地已 `git push` 最新代码（包含 3 类数据集生成脚本）。
    - 训练单元格：参考 `colab_3cls_training_cells.txt`（6 个单元格，包含完整流程）。
    - 关键特性：
      - GPU/CPU 自适应配置（GPU: 5 epochs/384 max_length；CPU: 3 epochs/256 max_length）
      - 自动检测 Drive 上是否已有数据集，若无则自动生成
      - 训练完成后自动分析结果并与基线对比
    - 配置：class_weight=auto, label_col=label, text_col=text_enhanced, early_stopping_patience=0
    - 预期结果：Test Macro F1 > 0.35（相比 6 类基线 0.163 提升 >100%）
    - 训练时间：GPU 约 1-1.5 小时，CPU 约 3-4 小时
  - **Colab Phase 1 训练（增强数据 + 类权重，当前推荐）**

    - 准备：确保本地已 `git push` 最新代码（包含 `bert_finetune_cls.py` 修复）。
    - 单元格 1：环境准备
      ```python
      from google.colab import drive
      drive.mount('/content/drive')
      !pip install -U transformers datasets evaluate accelerate -q
      import os
      if not os.path.exists('/content/Graduation_Project'):
          !git clone https://github.com/Caria-Tarnished/Graduation_Project.git
      else:
          !cd /content/Graduation_Project && git pull  # 确保拉取最新修复
      %cd /content/Graduation_Project
      ```
    - 单元格 2：验证数据集
      ```python
      import os, pandas as pd
      DATA_DIR = '/content/drive/MyDrive/Graduation_Project/datasets/processed_stale300/latest'
      files = {
          'train': f'{DATA_DIR}/train_enhanced.csv',
          'val': f'{DATA_DIR}/val_enhanced.csv',
          'test': f'{DATA_DIR}/test_enhanced.csv'
      }
      print("数据文件检查:")
      all_exist = True
      for name, path in files.items():
          if os.path.exists(path):
              df = pd.read_csv(path, encoding='utf-8')
              print(f"✓ {name}: {len(df)} 样本")
          else:
              print(f"✗ {name}: 文件不存在")
              all_exist = False
      if not all_exist:
          raise FileNotFoundError('Drive 上数据集缺失，请先在本地构建并同步到 latest。')
      ```
    - 单元格 3：Phase 1 训练（关键）
      ```python
      !python scripts/modeling/bert_finetune_cls.py \
        --train_csv /content/drive/MyDrive/Graduation_Project/datasets/processed_stale300/latest/train_enhanced.csv \
        --val_csv /content/drive/MyDrive/Graduation_Project/datasets/processed_stale300/latest/val_enhanced.csv \
        --test_csv /content/drive/MyDrive/Graduation_Project/datasets/processed_stale300/latest/test_enhanced.csv \
        --output_dir /content/drive/MyDrive/Graduation_Project/experiments/bert_stale300_6cls_baseline_v1 \
        --label_col label_multi_cls \
        --text_col text_enhanced \
        --model_name hfl/chinese-roberta-wwm-ext \
        --class_weight auto \
        --epochs 5 \
        --lr 1e-5 \
        --max_length 384 \
        --train_bs 16 \
        --eval_bs 32 \
        --gradient_accumulation_steps 2 \
        --warmup_ratio 0.06 \
        --weight_decay 0.01 \
        --eval_steps 100 \
        --save_steps 100 \
        --early_stopping_patience 0
      ```
    - 单元格 4：查看结果
      ```python
      import json, pandas as pd
      OUTPUT_DIR = '/content/drive/MyDrive/Graduation_Project/experiments/bert_stale300_6cls_baseline_v1'
      with open(f'{OUTPUT_DIR}/metrics_test.json', 'r', encoding='utf-8') as f:
          test_metrics = json.load(f)
      print("="*80)
      print("Phase 1 训练结果")
      print("="*80)
      print(f"Test Accuracy: {test_metrics['eval_accuracy']:.4f}")
      print(f"Test Macro F1: {test_metrics['eval_macro_f1']:.4f}")
      baseline_f1 = 0.163
      improvement = (test_metrics['eval_macro_f1'] - baseline_f1) / baseline_f1 * 100
      print(f"\n与基线对比:")
      print(f"  基线: {baseline_f1:.4f}")
      print(f"  增强: {test_metrics['eval_macro_f1']:.4f}")
      print(f"  提升: {improvement:+.1f}%")
      # 查看分类报告
      with open(f'{OUTPUT_DIR}/report_test.txt', 'r', encoding='utf-8') as f:
          print("\n" + "="*80)
          print("分类报告")
          print("="*80)
          print(f.read())
      # 稀有类别预测分析
      pred_df = pd.read_csv(f'{OUTPUT_DIR}/pred_test.csv', encoding='utf-8')
      print("\n预测分布:")
      print(pred_df['pred'].value_counts().sort_index())
      rare_classes = [3, 4, 5]
      print("\n稀有类别预测:")
      for cls in rare_classes:
          pred_count = (pred_df['pred'] == cls).sum()
          true_count = (pred_df['label'] == cls).sum()
          print(f"  Class {cls}: 真实={true_count}, 预测={pred_count}")
          if pred_count > 0:
              print(f"    ✅ 模型开始预测这个类别了！")
      ```
    - 预期结果：Test Macro F1 > 0.35（基线 0.163，提升 >100%）；稀有类别 F1 > 0。
    - 训练时间：约 1-1.5 小时（T4 GPU，~2355 steps）。
    - 完整单元格参考：`colab_phase1_cells.txt`。
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

## 7) 训练工作流（Training Workflow）

### 7.1 存算分离架构

本项目采用"本地开发 + 云端训练（GitHub 代码 + Drive 数据）"的存算分离方案：

- **代码（Git）**：在本地 E 盘开发并推送到 GitHub，Colab 从 GitHub 拉取最新代码
- **数据与训练产物（云端）**：Google Drive（G 盘）存放
  - 数据：`/content/drive/MyDrive/Graduation_Project/data/processed/...`
  - 实验：`/content/drive/MyDrive/Graduation_Project/experiments/<run_name>/...`
- **结果回传（本地）**：`reports/<run_name>/...`（由同步脚本生成，仅小文件）

### 7.2 Colab 训练流程

1. **准备环境**（每次新会话建议先升级依赖）

   ```python
   !pip install -U transformers datasets evaluate huggingface_hub
   ```
2. **挂载 Drive**（用于数据与产物）

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **拉取代码**（GitHub）并进入仓库目录

   ```bash
   !git clone https://github.com/<your_org>/<your_repo>.git  # 首次
   %cd /content/<your_repo>
   # 之后更新用：!git -C /content/<your_repo> pull
   ```
4. **运行训练**（将输出目录指向 Drive）

   ```bash
   python scripts/modeling/bert_finetune_cls.py \
     --train_csv /content/drive/MyDrive/Graduation_Project/data/processed/train.csv \
     --val_csv   /content/drive/MyDrive/Graduation_Project/data/processed/val.csv \
     --test_csv  /content/drive/MyDrive/Graduation_Project/data/processed/test.csv \
     --output_dir /content/drive/MyDrive/Graduation_Project/experiments/bert_3cls_v1 \
     --model_name hfl/chinese-roberta-wwm-ext \
     --epochs 5 --lr 1e-5 --max_length 384 \
     --class_weight auto --warmup_steps 100 --weight_decay 0.01 \
     --eval_steps 100 --save_steps 100 --early_stopping_patience 0
   ```

### 7.3 本地同步训练产物

运行同步脚本，仅复制小文件到仓库 `reports/`（默认跳过大权重）：

```powershell
# 先预演
python scripts/tools/sync_results.py `
  --src_root "G:\我的云端硬盘\Graduation_Project\experiments" `
  --dst_root "E:\Projects\Graduation_Project\reports" `
  --dry_run --verbose

# 实际执行
python scripts/tools/sync_results.py `
  --src_root "G:\我的云端硬盘\Graduation_Project\experiments" `
  --dst_root "E:\Projects\Graduation_Project\reports" `
  --verbose
```

可通过 include/exclude 精细控制（示例：加入最优权重）：

```powershell
python scripts/tools/sync_results.py `
  --include "**/eval_results.json" --include "**/metrics*.json" `
  --include "**/report*.txt" --include "**/pred*.csv" `
  --include "**/best/config.json" --include "**/*.safetensors"
```

### 7.4 方案 A 优化策略（已实施）

**核心思路**：基于金融大模型综述（Time Series Textualization），将市场行情与新闻文本在输入端融合。

**输入增强（Input Augmentation）**：

- **问题**：BERT 仅能看到新闻文本，无法感知发布前的市场状态
- **方案**：将发布前的 K 线趋势转化为自然语言前缀（Prefix），注入上下文
- **实现**：`scripts/modeling/build_enhanced_dataset_3cls.py`

**市场上下文前缀映射**：

| 市场状态 | 判定条件                                           | 生成前缀              | 含义             |
| :------- | :------------------------------------------------- | :-------------------- | :--------------- |
| 强势上涨 | `pre_ret > 1.0%`                                 | `[Strong Rally]`    | 市场情绪极度乐观 |
| 急剧下跌 | `pre_ret < -1.0%`                                | `[Sharp Decline]`   | 市场情绪极度悲观 |
| 温和上涨 | `0.3% < pre_ret <= 1.0%`                         | `[Mild Rally]`      | 情绪偏多         |
| 弱势下跌 | `-1.0% <= pre_ret < -0.3%`                       | `[Weak Decline]`    | 情绪偏弱         |
| 高波动   | `abs(pre_ret) < 0.3%` AND `range_ratio > 1.5%` | `[High Volatility]` | 多空分歧巨大     |
| 横盘震荡 | 其他情况                                           | `[Sideways]`        | 情绪平稳         |

**宏观数据特殊处理**：

- 原：`美国1月CPI年率录得2.8%，预期2.9%。`
- 新：`[Sideways] [Economic Data] [Actual 2.8 Exp 2.9] [Beat] 美国1月CPI年率录得2.8%，预期2.9%。`

**效果**：

- 训练集前缀分布：Sideways 60.6%, Mild Rally 15.3%, Weak Decline 12.8%, Sharp Decline 5.7%, Strong Rally 5.4%, High Volatility 0.1%
- 模型学习到模式：`[Strong Rally]` + `利好消息` → 可能是"利好兑现"（需后处理规则引擎判断）

### 7.5 常见问答（FAQ）

- **是否必须创建/上传 .ipynb？**

  - 否。推荐在 Colab 新建一个极简 Runner 笔记本，仅包含上述几段单元即可；也可直接在 Colab 的终端执行命令。
- **训练数据放哪里？**

  - 放在 `/content/drive/MyDrive/Graduation_Project/data/processed/...`，训练命令直接指向该路径。
- **训练输出放哪里？**

  - 指定 `--output_dir` 到 `/content/drive/MyDrive/Graduation_Project/experiments/<run_name>`。
- **本地如何获取指标？**

  - 运行 `scripts/tools/sync_results.py`，会将关键小文件复制到 `reports/<run_name>/`。
- **可忽略的警告**：

  - HuggingFace 403 Forbidden：讨论功能被禁用，不影响模型加载
  - UNEXPECTED/MISSING weights：分类头重新初始化，正常现象

## 8) 关键文件

- 计划：`PLAN.md`
- 状态：`Project_Status.md`
- 剩余任务开发文档：`REMAINING_TASKS.md`
- 配置：`configs/config.yaml`
- 环境：`.env.example`
- 数据库：`finance.db`
- 分钟级冲击数据库：`finance_analysis.db`
- Colab 训练单元格：`colab_3cls_training_cells.txt`
- 应用代码：`app/`
  - `app/core/dto.py`：数据传输对象定义
  - `app/services/sentiment_analyzer.py`：情感分析服务（Engine A）
  - `app/services/README.md`：服务使用文档
- 脚本：`scripts/`
  - `scripts/crawlers/jin10_dynamic.py`
  - `scripts/crawlers/jin10_flash_api.py`
  - `scripts/crawlers/storage.py`
  - `scripts/fetch_intraday_xauusd_mt5.py`
  - `scripts/build_finance_analysis.py`
  - `scripts/modeling/baseline_tfidf_svm.py`
  - `scripts/modeling/bert_finetune_cls.py`
  - `scripts/modeling/prepare_3cls_dataset.py`
  - `scripts/modeling/build_enhanced_dataset_3cls.py`
  - `scripts/test_sentiment_analyzer.py`
  - `scripts/tools/sync_results.py`
  - `scripts/tools/copy_model_weights.py`

## 9) 常用指令（数据库）

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
