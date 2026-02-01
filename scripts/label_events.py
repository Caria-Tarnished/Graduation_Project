# -*- coding: utf-8 -*-
"""
基于日级价格数据为新闻/日历事件生成“事件窗收益”代理标注（bullish/neutral/bearish）。

- 输入：SQLite 数据库（finance.db），Article 表与 prices 表。
- 输出：SQLite labels 表（若不存在自动创建），可选导出 CSV。
- 当前版本：日级窗口（例如 1d/2d/5d）。
- 用法示例：
  python scripts/label_events.py \
    --db finance.db \
    --tickers "GC=F" \
    --windows "1d,2d,5d" \
    --neutral 0.003 \
    --tz "Asia/Shanghai" \
    --out data/processed/labels_gc_daily.csv
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from typing import Dict, List, Optional, Tuple

# 依赖
try:
    import pandas as pd  # type: ignore
except Exception as e:  # noqa: BLE001
    raise SystemExit("缺少 pandas，请先安装：pip install pandas") from e

CSV_ENCODING = "utf-8"


def _ensure_labels_schema(conn: sqlite3.Connection) -> None:
    """确保 labels 表存在。
    说明：
    - 主键使用 (news_id, ticker, window)，避免重复写入。
    - window 采用文本格式，如 "1d"、"5d"；ret 为对数或算术收益？此处采用算术收益。
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS labels (
            news_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            window TEXT NOT NULL,
            ret REAL,
            label TEXT,
            PRIMARY KEY (news_id, ticker, window)
        )
        """
    )
    conn.commit()


def _parse_windows(s: str) -> List[str]:
    """解析窗口字符串，如 "1d,2d,5d" -> ["1d","2d","5d"]。"""
    out: List[str] = []
    for x in (s or "").split(","):
        x2 = x.strip()
        if not x2:
            continue
        # 简单校验格式；仅支持以 d 结尾的日级窗口
        if not x2.endswith("d"):
            raise ValueError(f"仅支持日级窗口：{x2}")
        try:
            _ = int(x2[:-1])
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"非法窗口：{x2}") from e
        out.append(x2)
    if not out:
        out = ["1d", "2d", "5d"]
    return out


def _load_prices(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    """读取某个标的的日级收盘价，返回按 trade_date 升序的 DataFrame。
    需要字段：trade_date, close。若 trade_date 为空，则回退使用 ts 字段推导日期。
    """
    # 首选使用已规范的 trade_date
    sql = (
        "SELECT trade_date, close FROM prices WHERE ticker=? "
        "AND trade_date IS NOT NULL "
        "AND trim(trade_date) <> '' "
        "ORDER BY trade_date ASC"
    )
    df = pd.read_sql_query(sql, conn, params=(ticker,))
    if df is None or df.empty:
        # 回退：使用 ts 推导 trade_date（兼容历史数据）
        sql2 = (
            "SELECT ts, close FROM prices WHERE ticker=? "
            "AND ts IS NOT NULL "
            "AND trim(ts) <> '' "
            "ORDER BY ts ASC"
        )
        df2 = pd.read_sql_query(sql2, conn, params=(ticker,))
        if df2 is None or df2.empty:
            return pd.DataFrame(columns=["trade_date", "close"])  # 空
        dtmp = df2.copy()
        try:
            dtmp["trade_date"] = (
                pd.to_datetime(dtmp["ts"], errors="coerce")
                .dt.strftime("%Y-%m-%d")
            )
        except Exception:
            dtmp["trade_date"] = None
        dtmp = dtmp.dropna(subset=["trade_date"]).reset_index(drop=True)
        dtmp = dtmp.drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
        return dtmp[["trade_date", "close"]]
    # 去重取最后一条（如有重复）
    df = df.drop_duplicates(subset=["trade_date"], keep="last").reset_index(drop=True)
    return df


def _next_trading_index(trading_dates: List[str], idx: int, ndays: int) -> Optional[int]:
    """基于交易日序列，返回 idx 往后 ndays 的索引位置；若越界则返回 None。"""
    j = idx + ndays
    if j < 0 or j >= len(trading_dates):
        return None
    return j


def _label_for_ret(ret: float, neutral: float) -> str:
    if ret is None:
        return "neutral"
    if ret >= neutral:
        return "bullish"
    if ret <= -neutral:
        return "bearish"
    return "neutral"


def _align_anchor_date(trading_dates: List[str], pub_date: str) -> Optional[int]:
    """将事件的日期对齐到交易日日历上的锚点索引。
    逻辑：找到第一个 >= pub_date 的交易日索引。
    """
    # 线性扫描（交易日日志数量通常不大，简单实现）
    for i, d in enumerate(trading_dates):
        if d >= pub_date:
            return i
    return None


def _load_articles(conn: sqlite3.Connection) -> pd.DataFrame:
    """读取 articles 基本字段（与爬虫 storage.ensure_schema 保持一致的表名）。
    需要字段：id, published_at。其余字段作为参考。
    """
    # 首选表名 articles；若历史环境为 Article，做一次兼容兜底
    try:
        sql = (
            "SELECT id, published_at, title, content, source, site "
            "FROM articles "
            "WHERE published_at IS NOT NULL"
        )
        df = pd.read_sql_query(sql, conn)
        return df
    except Exception:
        try:
            # 最小兜底，仅取 id 与 published_at（旧表名）
            df = pd.read_sql_query(
                "SELECT id, published_at "
                "FROM Article "
                "WHERE published_at IS NOT NULL",
                conn,
            )
            return df
        except Exception:
            # 再次尝试仅取 id 与 published_at（新表名）
            df = pd.read_sql_query(
                "SELECT id, published_at "
                "FROM articles "
                "WHERE published_at IS NOT NULL",
                conn,
            )
            return df


def _to_date_str_shanghai(ts: str, tz: str = "Asia/Shanghai") -> Optional[str]:
    """将时间戳字符串转换为日期字符串（YYYY-MM-DD）。
    这里不做复杂时区处理，默认按本地/给定时区视作同一天。
    """
    if not ts:
        return None
    try:
        # 尽量解析常见格式
        dt = pd.to_datetime(ts, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="基于日级价格的事件窗代理标注")
    parser.add_argument(
        "--db", type=str, default="finance.db", help="SQLite 数据库路径"
    )
    parser.add_argument(
        "--tickers", type=str, default="GC=F", help="以逗号分隔的标的列表，例如 GC=F"
    )
    parser.add_argument(
        "--windows", type=str, default="1d,2d,5d", help="窗口集合，如 1d,2d,5d"
    )
    parser.add_argument(
        "--neutral", type=float, default=0.003, help="中性带阈值（算术收益）"
    )
    parser.add_argument(
        "--tz", type=str, default="Asia/Shanghai", help="时区名，仅用于日期简化处理"
    )
    parser.add_argument(
        "--out", type=str, default="", help="可选：导出 CSV 路径（如 data/processed/labels_gc_daily.csv）"
    )
    args = parser.parse_args()

    windows = _parse_windows(args.windows)

    conn = sqlite3.connect(args.db)
    _ensure_labels_schema(conn)

    # 读取事件（Article）
    arts = _load_articles(conn)
    if arts is None or arts.empty:
        print("[WARN] Article 表为空或无 published_at")
        return

    # 预处理事件日期
    arts = arts.copy()
    arts["pub_date"] = arts["published_at"].apply(_to_date_str_shanghai)
    arts = arts.dropna(subset=["pub_date"]).reset_index(drop=True)

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    inserts: List[Tuple[int, str, str, float, str]] = []

    for t in tickers:
        # 读取该标的价格
        px = _load_prices(conn, t)
        if px is None or px.empty:
            print(f"[WARN] 价格数据为空：{t}")
            continue
        trading_dates = px["trade_date"].astype(str).tolist()
        closes = px["close"].astype(float).tolist()
        # 构建日期 -> 索引
        idx_map: Dict[str, int] = {d: i for i, d in enumerate(trading_dates)}

        for _, row in arts.iterrows():
            nid = int(row["id"])
            pub_date: str = str(row["pub_date"])  # YYYY-MM-DD
            # 找到锚点索引（>= pub_date 的首个交易日）
            i = idx_map.get(pub_date)
            if i is None:
                i = _align_anchor_date(trading_dates, pub_date)
            if i is None:
                continue
            # 计算各窗口收益并打标
            for w in windows:
                nd = int(w[:-1])  # 去掉尾部的 'd'
                j = _next_trading_index(trading_dates, i, nd)
                if j is None:
                    continue
                c0 = closes[i]
                c1 = closes[j]
                if c0 is None or c1 is None:
                    continue
                try:
                    ret = float(c1) / float(c0) - 1.0
                except Exception:
                    continue
                lb = _label_for_ret(ret, args.neutral)
                inserts.append((nid, t, w, ret, lb))

    # 入库（去重写入）
    if inserts:
        conn.executemany(
            """
            INSERT OR REPLACE INTO labels (news_id, ticker, window, ret, label)
            VALUES (?, ?, ?, ?, ?)
            """,
            inserts,
        )
        conn.commit()
        print(f"[INFO] 写入 labels 条数：{len(inserts)}")

    # 可选：导出 CSV
    if args.out:
        try:
            out_dir = os.path.dirname(args.out)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            df_out = pd.DataFrame(
                inserts,
                columns=[
                    "news_id",
                    "ticker",
                    "window",
                    "ret",
                    "label",
                ],
            )  # type: ignore
            df_out.to_csv(args.out, index=False, encoding=CSV_ENCODING)
            print(f"[INFO] 已导出 CSV：{args.out}")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] 导出 CSV 失败：{e}")

    conn.close()


if __name__ == "__main__":
    main()
