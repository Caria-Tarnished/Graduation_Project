# -*- coding: utf-8 -*-
"""
构建全新 finance_analysis.db：
- 读取分钟级价格 CSV（由 fetch_intraday_xauusd.py 产出）。
- 读取本地金十快讯与日历 CSV，时间统一、分钟对齐（向下取整）。
- 计算事件发布后 5/10/15/30 分钟收益与价差，写入 SQLite，并为所有时间戳建索引。

注意：
- 价格 CSV 的 ts 列代表在 price_tz 下的“墙上时间”，脚本会转换为 UTC 存储；
  同时保留本地时区字符串。
- 快讯与日历默认认为时间为 Asia/Shanghai，可通过参数调整。
- 若未来分钟无精确对应报价，将回退到该时刻之前最近一个报价（asof 逻辑）。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
from typing import Iterable, List, Optional, Tuple

import pandas as pd  # type: ignore


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _read_prices_csv(path: str, price_tz: str) -> pd.DataFrame:
    """读取分钟级价格 CSV，并生成 ts_local、ts_utc 两列。
    要求 CSV 至少包含：ticker, ts, open, high, low, close, volume。
    """
    df = pd.read_csv(path, encoding="utf-8")
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "ticker", "ts_local", "ts_utc", "open", "high", "low", "close", "volume",
        ])
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "ts" not in df.columns:
        for cand in ["time", "datetime", "date_time"]:
            if cand in df.columns:
                df.rename(columns={cand: "ts"}, inplace=True)
                break
    # 解析本地时间（price_tz）
    ts_local = pd.to_datetime(df["ts"], errors="coerce")
    ts_local = ts_local.dt.floor("min")
    # 本地→UTC
    ts_aware = ts_local.dt.tz_localize(price_tz, nonexistent="shift_forward", ambiguous="NaT")
    ts_utc = ts_aware.dt.tz_convert("UTC")
    out = pd.DataFrame()
    out["ticker"] = df.get("ticker", "XAUUSD=X")
    out["ts_local"] = ts_local.dt.strftime("%Y-%m-%d %H:%M:%S")
    out["ts_utc"] = ts_utc.dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(df.get(col, pd.NA), errors="coerce")
    # 去重并排序
    out = (
        out.dropna(subset=["ts_utc"])  # 确保有时间
           .drop_duplicates(subset=["ticker", "ts_utc"], keep="last")
           .sort_values(["ticker", "ts_utc"]).reset_index(drop=True)
    )
    return out


def _read_flash_csv(path: str, flash_tz: str) -> pd.DataFrame:
    """读取快讯 CSV，生成统一事件表字段。
    输入列（尽可能兼容）：
    id,time,type,content,important,hot,indicator_name,previous,consensus,actual,star,country,unit
    输出字段：
    event_id, source, ts_local, ts_utc, country, name, content, star, previous,
    consensus, actual, affect, detail_url, important, hot, indicator_name, unit
    其中 name 对于快讯为空。
    """
    cols = [
        "id", "time", "type", "content", "important", "hot", "indicator_name",
        "previous", "consensus", "actual", "star", "country", "unit",
    ]
    df = pd.read_csv(path, encoding="utf-8")
    # 只保留需要列（不存在则补空）
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    # 解析时间并向下取整到分钟
    ts_local = pd.to_datetime(df["time"], errors="coerce").dt.floor("min")
    # 本地→UTC
    ts_aware = ts_local.dt.tz_localize(flash_tz, nonexistent="shift_forward", ambiguous="NaT")
    ts_utc = ts_aware.dt.tz_convert("UTC")
    out = pd.DataFrame()
    out["event_id"] = df["id"].astype(str).map(lambda x: f"flash:{x}")
    out["source"] = "flash"
    out["ts_local"] = ts_local.dt.strftime("%Y-%m-%d %H:%M:%S")
    out["ts_utc"] = ts_utc.dt.strftime("%Y-%m-%d %H:%M:%S")
    out["country"] = df["country"].astype(str)
    out["name"] = pd.NA  # 快讯无 name
    out["content"] = df["content"].astype(str)
    out["star"] = pd.to_numeric(df["star"], errors="coerce")
    out["previous"] = df["previous"].astype(str)
    out["consensus"] = df["consensus"].astype(str)
    out["actual"] = df["actual"].astype(str)
    out["affect"] = pd.NA
    out["detail_url"] = pd.NA
    out["important"] = pd.to_numeric(df["important"], errors="coerce")
    out["hot"] = df["hot"].astype(str)
    out["indicator_name"] = df["indicator_name"].astype(str)
    out["unit"] = df["unit"].astype(str)
    # 丢弃没有有效时间的事件
    out = out.dropna(subset=["ts_utc"]).reset_index(drop=True)
    return out


def _calendar_event_id(row: pd.Series) -> str:
    key = f"{row.get('date','')} {row.get('time','')}|{row.get('name','')}|{row.get('country','')}"
    # 稳定短ID（避免 detail_url 缺失时的重复）
    hid = hashlib.md5(key.encode("utf-8")).hexdigest()  # nosec - 仅用于生成稳定ID
    return f"calendar:{hid}"


def _read_calendar_csv(path: str, cal_tz: str) -> pd.DataFrame:
    """读取日历 CSV，生成统一事件表字段。
    输入列：date,time,country,name,star,previous,consensus,actual,affect,detail_url
    输出字段同 _read_flash_csv。
    """
    base_cols = [
        "date", "time", "country", "name", "star", "previous", "consensus", "actual", "affect", "detail_url",
    ]
    df = pd.read_csv(path, encoding="utf-8")
    for c in base_cols:
        if c not in df.columns:
            df[c] = pd.NA
    # 跳过缺少 time 的行（无法对齐到分钟）
    df = df[df["time"].astype(str).str.len() > 0].copy()
    # 合成本地时间字符串，并下取整分钟
    ts_local = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce",
    ).dt.floor("min")
    ts_aware = ts_local.dt.tz_localize(cal_tz, nonexistent="shift_forward", ambiguous="NaT")
    ts_utc = ts_aware.dt.tz_convert("UTC")

    out = pd.DataFrame()
    out["event_id"] = df.apply(_calendar_event_id, axis=1)
    out["source"] = "calendar"
    out["ts_local"] = ts_local.dt.strftime("%Y-%m-%d %H:%M:%S")
    out["ts_utc"] = ts_utc.dt.strftime("%Y-%m-%d %H:%M:%S")
    out["country"] = df["country"].astype(str)
    out["name"] = df["name"].astype(str)
    out["content"] = pd.NA
    out["star"] = pd.to_numeric(df["star"], errors="coerce")
    out["previous"] = df["previous"].astype(str)
    out["consensus"] = df["consensus"].astype(str)
    out["actual"] = df["actual"].astype(str)
    out["affect"] = df["affect"].astype(str)
    out["detail_url"] = df["detail_url"].astype(str)
    out["important"] = pd.NA
    out["hot"] = pd.NA
    out["indicator_name"] = pd.NA
    out["unit"] = pd.NA
    out = out.dropna(subset=["ts_utc"]).reset_index(drop=True)
    return out


def _prices_indexed(prices: pd.DataFrame) -> pd.DataFrame:
    """将价格表转为按 UTC 时间排序的索引，便于 asof 查找。"""
    df = prices.copy()
    ts = pd.to_datetime(df["ts_utc"], errors="coerce")
    df["ts_utc_dt"] = ts
    df = df.dropna(subset=["ts_utc_dt"]).sort_values(["ticker", "ts_utc_dt"]).reset_index(drop=True)
    return df.set_index("ts_utc_dt")


def _asof_close(df_idx: pd.DataFrame, ticker: str, ts_utc: pd.Timestamp) -> Optional[Tuple[pd.Timestamp, float]]:
    """返回该标的在 ts_utc 时刻“之前或等于”的最近一分钟收盘价。
    若不存在任何历史则返回 None。
    返回 (实际价格时间戳, close)。
    """
    # 过滤同标的视图（避免多标的交叉干扰）
    sub = df_idx[df_idx["ticker"] == ticker]
    if sub.empty:
        return None
    # asof：通过 searchsorted 寻找位置
    pos = sub.index.searchsorted(ts_utc, side="right") - 1
    if pos < 0:
        return None
    ts_found = sub.index[pos]
    close = sub.iloc[pos]["close"]
    try:
        close = float(close)
    except Exception:
        close = float("nan")
    return ts_found, close


def _compute_impacts(
    events: pd.DataFrame,
    prices_idx: pd.DataFrame,
    ticker: str,
    windows: Iterable[int],
    local_tz: str,
) -> pd.DataFrame:
    """为事件计算多窗口收益与价差。
    返回列：event_id, ticker, window_min, price_event, price_future, delta, ret, price_event_ts_utc, price_future_ts_utc
    """
    rows: List[dict] = []
    for _, r in events.iterrows():
        eid = str(r["event_id"])
        ts0 = pd.to_datetime(r["ts_utc"], errors="coerce")
        if pd.isna(ts0):
            continue
        base = _asof_close(prices_idx, ticker, ts0)
        if base is None:
            continue
        ts_e, p0 = base
        if not (p0 == p0) or p0 == 0:  # NaN 或 0
            continue
        for w in windows:
            ts_f = ts0 + pd.Timedelta(minutes=int(w))
            fut = _asof_close(prices_idx, ticker, ts_f)
            if fut is None:
                continue
            ts_f2, p1 = fut
            if not (p1 == p1):
                continue
            delta = p1 - p0
            ret = delta / p0 if p0 else None
            # 生成本地时间（北京时间等）
            try:
                ts_e_local = (
                    pd.Timestamp(ts_e).tz_localize("UTC").tz_convert(local_tz).strftime("%Y-%m-%d %H:%M:%S")
                )
            except Exception:
                ts_e_local = pd.Timestamp(ts_e).strftime("%Y-%m-%d %H:%M:%S")
            try:
                ts_f_local = (
                    pd.Timestamp(ts_f2).tz_localize("UTC").tz_convert(local_tz).strftime("%Y-%m-%d %H:%M:%S")
                )
            except Exception:
                ts_f_local = pd.Timestamp(ts_f2).strftime("%Y-%m-%d %H:%M:%S")
            rows.append({
                "event_id": eid,
                "ticker": ticker,
                "window_min": int(w),
                "price_event": float(p0),
                "price_future": float(p1),
                "delta": float(delta),
                "ret": float(ret) if ret is not None else None,
                "price_event_ts_utc": ts_e.strftime("%Y-%m-%d %H:%M:%S"),
                "price_future_ts_utc": ts_f2.strftime("%Y-%m-%d %H:%M:%S"),
                "price_event_ts_local": ts_e_local,
                "price_future_ts_local": ts_f_local,
            })
    return pd.DataFrame(rows)


def _init_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    # prices_m1
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prices_m1 (
          ticker TEXT NOT NULL,
          ts_local TEXT NOT NULL,
          ts_utc   TEXT NOT NULL,
          open REAL, high REAL, low REAL, close REAL, volume REAL,
          PRIMARY KEY (ticker, ts_utc)
        )
        """
    )
    # events 统一表
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
          event_id TEXT PRIMARY KEY,
          source TEXT NOT NULL, -- flash / calendar
          ts_local TEXT NOT NULL,
          ts_utc   TEXT NOT NULL,
          country TEXT,
          name TEXT,
          content TEXT,
          star REAL,
          previous TEXT,
          consensus TEXT,
          actual TEXT,
          affect TEXT,
          detail_url TEXT,
          important REAL,
          hot TEXT,
          indicator_name TEXT,
          unit TEXT
        )
        """
    )
    # impacts
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS event_impacts (
          event_id TEXT NOT NULL,
          ticker TEXT NOT NULL,
          window_min INTEGER NOT NULL,
          price_event REAL,
          price_future REAL,
          delta REAL,
          ret REAL,
          price_event_ts_utc TEXT,
          price_future_ts_utc TEXT,
          price_event_ts_local TEXT,
          price_future_ts_local TEXT,
          PRIMARY KEY (event_id, ticker, window_min),
          FOREIGN KEY (event_id) REFERENCES events(event_id)
        )
        """
    )
    # 索引（所有时间戳字段）
    cur.execute("CREATE INDEX IF NOT EXISTS idx_prices_m1_ts_utc ON prices_m1(ts_utc)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_prices_m1_ts_local ON prices_m1(ts_local)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_ts_utc ON events(ts_utc)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_ts_local ON events(ts_local)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_impacts_evt_ts ON event_impacts(price_event_ts_utc)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_impacts_evt_ts_local ON event_impacts(price_event_ts_local)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_impacts_ticker ON event_impacts(ticker)")
    conn.commit()


def _insert_or_ignore_events(conn: sqlite3.Connection, events: pd.DataFrame) -> None:
    if events is None or events.empty:
        return
    df = events.drop_duplicates(subset=["event_id"], keep="last").reset_index(drop=True)
    conn.execute("DROP TABLE IF EXISTS _tmp_events")
    df.to_sql("_tmp_events", conn, if_exists="replace", index=False)
    cols = [
        "event_id", "source", "ts_local", "ts_utc", "country", "name", "content", "star",
        "previous", "consensus", "actual", "affect", "detail_url", "important", "hot",
        "indicator_name", "unit"
    ]
    col_list = ", ".join(cols)
    sql = f"INSERT OR IGNORE INTO events ({col_list}) SELECT {col_list} FROM _tmp_events"
    conn.execute(sql)
    conn.execute("DROP TABLE IF EXISTS _tmp_events")


def _insert_or_ignore_impacts(conn: sqlite3.Connection, impacts: pd.DataFrame) -> None:
    if impacts is None or impacts.empty:
        return
    conn.execute("DROP TABLE IF EXISTS _tmp_impacts")
    impacts.to_sql("_tmp_impacts", conn, if_exists="replace", index=False)
    cols = [
        "event_id", "ticker", "window_min", "price_event", "price_future", "delta", "ret",
        "price_event_ts_utc", "price_future_ts_utc", "price_event_ts_local", "price_future_ts_local"
    ]
    col_list = ", ".join(cols)
    sql = f"INSERT OR IGNORE INTO event_impacts ({col_list}) SELECT {col_list} FROM _tmp_impacts"
    conn.execute(sql)
    conn.execute("DROP TABLE IF EXISTS _tmp_impacts")


def _insert_or_ignore_prices(conn: sqlite3.Connection, prices: pd.DataFrame) -> None:
    if prices is None or prices.empty:
        return
    df = (
        prices.dropna(subset=["ts_utc"])
              .drop_duplicates(subset=["ticker", "ts_utc"], keep="last")
              .reset_index(drop=True)
    )
    conn.execute("DROP TABLE IF EXISTS _tmp_prices")
    df.to_sql("_tmp_prices", conn, if_exists="replace", index=False)
    cols = [
        "ticker", "ts_local", "ts_utc", "open", "high", "low", "close", "volume",
    ]
    col_list = ", ".join(cols)
    sql = f"INSERT OR IGNORE INTO prices_m1 ({col_list}) SELECT {col_list} FROM _tmp_prices"
    conn.execute(sql)
    conn.execute("DROP TABLE IF EXISTS _tmp_prices")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 finance_analysis.db")
    parser.add_argument("--prices_csv", type=str, required=True, help="价格 CSV 路径（含列 ts/ticker/close 等）")
    parser.add_argument("--flash_csv", type=str, required=True, help="快讯 CSV 路径")
    parser.add_argument("--calendar_csv", type=str, required=True, help="日历 CSV 路径")
    parser.add_argument("--db", type=str, default="finance_analysis.db", help="输出 SQLite DB 路径")
    parser.add_argument(
        "--price_tz",
        type=str,
        default="Asia/Shanghai",
        help="价格 ts 所在时区，默认 Asia/Shanghai（北京时间）",
    )
    parser.add_argument("--flash_tz", type=str, default="Asia/Shanghai", help="快讯时间所在时区")
    parser.add_argument("--calendar_tz", type=str, default="Asia/Shanghai", help="日历时间所在时区")
    parser.add_argument(
        "--ticker",
        type=str,
        default="XAUUSD",
        help="标的，默认 XAUUSD；需与价格 CSV 中的 ticker 完全一致",
    )
    parser.add_argument("--windows", type=int, nargs="+", default=[5, 10, 15, 30], help="分钟窗口列表")
    args = parser.parse_args()

    # 1) 读取与标准化价格
    prices = _read_prices_csv(args.prices_csv, args.price_tz)
    if prices is None or prices.empty:
        raise SystemExit("价格 CSV 为空或无有效数据，请先使用 fetch_intraday_xauusd.py 生成并检查。")

    # 2) 读取事件源
    flash = _read_flash_csv(args.flash_csv, args.flash_tz)
    calendar = _read_calendar_csv(args.calendar_csv, args.calendar_tz)
    # 合并事件（统一表）
    events = pd.concat([flash, calendar], ignore_index=True)
    if events.empty:
        raise SystemExit("快讯与日历事件均为空或无有效时间，无法计算冲击。")

    # 3) 计算冲击
    prices_idx = _prices_indexed(prices)
    impacts = _compute_impacts(
        events,
        prices_idx,
        args.ticker,
        args.windows,
        args.price_tz,
    )

    # 4) 写入 SQLite
    _ensure_dir(args.db)
    conn = sqlite3.connect(args.db)
    try:
        _init_schema(conn)
        _insert_or_ignore_prices(conn, prices)
        _insert_or_ignore_events(conn, events)
        _insert_or_ignore_impacts(conn, impacts)
        conn.commit()
    finally:
        conn.close()

    print(f"写库完成：{args.db}")
    print(f"prices_m1 行数：{len(prices)}; events 行数：{len(events)}; event_impacts 行数：{len(impacts)}")


if __name__ == "__main__":
    main()
