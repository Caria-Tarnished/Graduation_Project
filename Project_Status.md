# Project Status

更新时间：2026-01-29
负责人：Caria-Tarnished

---

## 1) 概览

- 目标：财经快讯情感分类（代理标注验证）+ 财报 RAG（后续里程碑）+ Streamlit UI。
- 时间范围：2024-01-01 至 2026-01-21（Asia/Shanghai）。
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
- 金十爬虫（当前在用）与入库：
  - `scripts/crawlers/jin10_dynamic.py`：日历模式已稳定可用，支持“只看重要”开关、DB 入库与 CSV 增量写入；接口与参考脚本保持一致（`--months/--start/--end/--output/--db/...`）。
  - `scripts/crawlers/jin10_flash_api.py`：快讯 API 模式（支持接口发现/过滤/CSV 流式/SQLite 入库）。
- SQLite 存储：`scripts/crawlers/storage.py`（upsert、URL/内容哈希去重、索引）。

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

## 7) 关键文件

- 计划：`PLAN.md`
- 状态：`Project_Status.md`
- 配置：`configs/config.yaml`
- 环境：`.env.example`
- 数据库：`finance.db`
- 脚本：`scripts/`
  - `scripts/crawlers/jin10_dynamic.py`
  - `scripts/crawlers/jin10_flash_api.py`
  - `scripts/crawlers/storage.py`

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
