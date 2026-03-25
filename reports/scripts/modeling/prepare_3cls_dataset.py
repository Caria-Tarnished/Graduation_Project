# -*- coding: utf-8 -*-
"""
从 finance_analysis.db 生成简化的 3 类标签训练集（方案 A）：
- 仅保留基础方向标签：Bearish (-1) / Neutral (0) / Bullish (1)
- 基于事件发布后 15 分钟收益（ret_post）分位阈值（train 上估计）
- 移除"预期兑现"和"观望"标签（改为后处理规则引擎）

输出：按时间切分 train/val/test 三个 CSV 与一个元数据 JSON（阈值与标签映射）。
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


def _ensure_dir(path: str) -> None:
    p = os.path.abspath(path)
    _, ext = os.path.splitext(p)
    d = os.path.dirname(p) if ext else p
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _read_prices(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    """读取分钟价，并构建 UTC 时间索引（asof 查价用）。"""
    q = (
        "select ts_utc, close, high, low from prices_m1 "
        "where ticker=? order by ts_utc asc"
    )
    df = pd.read_sql_query(q, conn, params=(ticker,))
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts_utc_dt", "close", "high", "low"]).set_index(
            "ts_utc_dt"
        )
    ts = pd.to_datetime(df["ts_utc"], errors="coerce")
    df["ts_utc_dt"] = ts
    df = df.dropna(subset=["ts_utc_dt"]).sort_values("ts_utc_dt").reset_index(drop=True)
    return df.set_index("ts_utc_dt")


def _asof_close(df_idx: pd.DataFrame, ts_utc: pd.Timestamp) -> Optional[Tuple[pd.Timestamp, float]]:
    """返回 ts_utc 之前或等于的最近一分钟收盘价。(ts_found, close)
    若历史为空或无效返回 None。
    """
    if df_idx is None or len(df_idx) == 0:
        return None
    pos = df_idx.index.searchsorted(ts_utc, side="right") - 1
    if pos < 0:
        return None
    ts_found = df_idx.index[pos]
    try:
        p = float(df_idx.iloc[pos]["close"])  # type: ignore[index]
    except Exception:
        p = float("nan")
    if not (p == p):
        return None
    return ts_found, p


def _range_hl(df_idx: pd.DataFrame, ts0: pd.Timestamp, ts1: pd.Timestamp) -> Tuple[float, float]:
    """统计窗口内的最高价与最低价（含端点），若为空返回 (nan, nan)。"""
    if df_idx is None or len(df_idx) == 0:
        return float("nan"), float("nan")
    sub = df_idx[(df_idx.index >= ts0) & (df_idx.index <= ts1)]
    if sub.empty:
        return float("nan"), float("nan")
    try:
        hi = float(np.nanmax(sub["high"].values))
        lo = float(np.nanmin(sub["low"].values))
    except Exception:
        return float("nan"), float("nan")
    return hi, lo


def _compute_quantiles(train: pd.DataFrame, q_low: float, q_high: float) -> Tuple[float, float]:
    """在训练集上估计 ret_post 的分位阈值（基础方向标签）。"""
    ser = train["ret_post"].dropna().astype(float)
    if len(ser) == 0:
        return -0.001, 0.001
    return float(ser.quantile(q_low)), float(ser.quantile(q_high))


def main() -> None:
    parser = argparse.ArgumentParser(description="生成简化的 3 类标签训练集（方案 A）")
    parser.add_argument("--db", type=str, default="finance_analysis.db")
    parser.add_argument("--ticker", type=str, default="XAUUSD")
    parser.add_argument("--window_post", type=int, default=15)
    parser.add_argument("--pre_minutes", type=int, default=120)
    parser.add_argument(
        "--max_stale_sec",
        type=float,
        default=300.0,
        help="过滤 impacts 的时间戳回退样本：impact_ts 与事件 ts 偏差超过该秒数则丢弃（默认 300 秒）",
    )
    # 时间切分（北京时间字符串，与入库保持一致）
    parser.add_argument("--train_end", type=str, default="2025-08-01 00:00:00")
    parser.add_argument("--val_end", type=str, default="2025-11-01 00:00:00")
    parser.add_argument("--test_end", type=str, default="2026-02-01 00:00:00")
    # 分位阈值
    parser.add_argument("--neutral_q_low", type=float, default=0.30)
    parser.add_argument("--neutral_q_high", type=float, default=0.70)
    parser.add_argument("--clip_ret", type=float, default=0.05, help="收益裁剪阈值（绝对值）")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("data", "processed"),
        help="输出目录（CSV 与 JSON）",
    )
    args = parser.parse_args()

    _ensure_dir(args.out_dir)

    # 1) 读库：事件（带文本）与 15 分钟冲击（ret_post）
    conn = sqlite3.connect(args.db)
    try:
        q_evt = (
            "select event_id, ts_local as event_ts_local, ts_utc as event_ts_utc, "
            "source, country, name, content, star, previous, consensus, actual, "
            "indicator_name, unit, important, hot "
            "from events order by ts_local asc"
        )
        events = pd.read_sql_query(q_evt, conn)
        q_imp = (
            "select event_id, price_event, price_future, delta, ret as ret_post, "
            "price_event_ts_utc, price_future_ts_utc "
            "from event_impacts where ticker=? and window_min=?"
        )
        impacts = pd.read_sql_query(q_imp, conn, params=(args.ticker, args.window_post))
        df = events.merge(impacts, on="event_id", how="inner")
        # 文本优先使用 content，缺失时回退 name
        df["text"] = df["content"].fillna("").astype(str)
        m = df["text"].str.len() == 0
        df.loc[m, "text"] = df.loc[m, "name"].fillna("").astype(str)

        # 时间解析（本地与 UTC）
        df["event_ts_local"] = pd.to_datetime(df["event_ts_local"], errors="coerce")
        df["event_ts_utc"] = pd.to_datetime(df["event_ts_utc"], errors="coerce")

        # stale 样本过滤：避免 prices_m1 缺口导致的 asof 回退过远
        df["impact_event_ts_utc"] = pd.to_datetime(df.get("price_event_ts_utc"), errors="coerce")
        df["impact_future_ts_utc"] = pd.to_datetime(df.get("price_future_ts_utc"), errors="coerce")
        expected_future = df["event_ts_utc"] + pd.Timedelta(minutes=int(args.window_post))
        df["delta_event_sec"] = (
            (df["impact_event_ts_utc"] - df["event_ts_utc"]).dt.total_seconds().abs()
        )
        df["delta_future_sec"] = (
            (df["impact_future_ts_utc"] - expected_future).dt.total_seconds().abs()
        )
        thr = float(args.max_stale_sec)
        before = int(len(df))
        mask = (df["delta_event_sec"] <= thr) & (df["delta_future_sec"] <= thr)
        mask = mask.fillna(False)
        df = df.loc[mask].copy()
        after = int(len(df))
        print(
            f"stale_filter applied: before={before}, after={after}, dropped={before - after}, max_stale_sec={thr}"
        )

        # 2) 读分钟价索引，计算 pre_ret 和 range_ratio（用于后续增强）
        prices = _read_prices(conn, args.ticker)
    finally:
        conn.close()

    # 预处理收益：裁剪极端值，提升稳健性
    df["ret_post"] = pd.to_numeric(df["ret_post"], errors="coerce")
    clip = float(abs(args.clip_ret))
    df.loc[:, "ret_post"] = df["ret_post"].clip(lower=-clip, upper=clip)

    # 逐事件计算 pre_ret 和 range_ratio（保留用于输入增强）
    pre_list: List[Optional[float]] = []
    rng_list: List[Optional[float]] = []
    for _, r in df.iterrows():
        ts_evt: pd.Timestamp = r["event_ts_utc"]
        if pd.isna(ts_evt):
            pre_list.append(None)
            rng_list.append(None)
            continue
        # pre: ts_evt - pre_minutes asof 查价
        ts_pre = ts_evt - pd.Timedelta(minutes=int(args.pre_minutes))
        base = _asof_close(prices, ts_pre)
        pevt = float(r.get("price_event", float("nan")))
        if base is None or not (pevt == pevt):
            pre_list.append(None)
        else:
            ppre = base[1]
            try:
                pre_ret = (pevt - ppre) / ppre if ppre else None
            except Exception:
                pre_ret = None
            if pre_ret is not None:
                pre_ret = float(np.clip(pre_ret, -clip, clip))
            pre_list.append(pre_ret)
        # range: [ts_evt, ts_evt + window_post]
        ts_end = ts_evt + pd.Timedelta(minutes=int(args.window_post))
        hi, lo = _range_hl(prices, ts_evt, ts_end)
        if not (pevt == pevt) or not (hi == hi) or not (lo == lo):
            rng = None
        else:
            try:
                rng = (hi - lo) / pevt if pevt else None
            except Exception:
                rng = None
        rng_list.append(rng)

    df["pre_ret"] = pre_list
    df["range_ratio"] = rng_list

    # 时间切分（仅按本地时间）
    t1 = pd.Timestamp(args.train_end)
    t2 = pd.Timestamp(args.val_end)
    t3 = pd.Timestamp(args.test_end)
    train = df[df["event_ts_local"] < t1].copy()
    val = df[(df["event_ts_local"] >= t1) & (df["event_ts_local"] < t2)].copy()
    test = df[(df["event_ts_local"] >= t2) & (df["event_ts_local"] < t3)].copy()

    # 3) 估计阈值（仅使用训练集）
    ql, qh = _compute_quantiles(train, args.neutral_q_low, args.neutral_q_high)

    # 4) 打标函数（3 类：-1/0/1）
    def base_label(x: Optional[float]) -> int:
        if x is None or not (x == x):
            return 0
        if x <= ql:
            return -1
        if x >= qh:
            return 1
        return 0

    for part in (train, val, test):
        part["label"] = part["ret_post"].apply(base_label)

    # 5) 输出
    keep_cols = [
        "event_id",
        "event_ts_local",
        "event_ts_utc",
        "source",
        "country",
        "name",
        "content",
        "text",
        "star",
        "previous",
        "consensus",
        "actual",
        "indicator_name",
        "unit",
        "important",
        "hot",
        "price_event",
        "price_event_ts_utc",
        "price_future",
        "price_future_ts_utc",
        "delta",
        "delta_event_sec",
        "delta_future_sec",
        "ret_post",
        "pre_ret",
        "range_ratio",
        "label",
    ]

    out_train = os.path.join(args.out_dir, "train_3cls.csv")
    out_val = os.path.join(args.out_dir, "val_3cls.csv")
    out_test = os.path.join(args.out_dir, "test_3cls.csv")
    train[keep_cols].to_csv(out_train, index=False, encoding="utf-8")
    val[keep_cols].to_csv(out_val, index=False, encoding="utf-8")
    test[keep_cols].to_csv(out_test, index=False, encoding="utf-8")

    # 阈值与映射元数据
    meta: Dict[str, object] = {
        "window_post": int(args.window_post),
        "pre_minutes": int(args.pre_minutes),
        "clip_ret": float(clip),
        "splits": {
            "train_end": args.train_end,
            "val_end": args.val_end,
            "test_end": args.test_end,
        },
        "thresholds": {
            "q_low": ql,
            "q_high": qh,
        },
        "label_mapping": {
            "-1": "bearish",
            "0": "neutral",
            "1": "bullish",
        },
        "sizes": {
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
        },
        "label_distribution": {
            "train": {
                "bearish": int((train["label"] == -1).sum()),
                "neutral": int((train["label"] == 0).sum()),
                "bullish": int((train["label"] == 1).sum()),
            },
            "val": {
                "bearish": int((val["label"] == -1).sum()),
                "neutral": int((val["label"] == 0).sum()),
                "bullish": int((val["label"] == 1).sum()),
            },
            "test": {
                "bearish": int((test["label"] == -1).sum()),
                "neutral": int((test["label"] == 0).sum()),
                "bullish": int((test["label"] == 1).sum()),
            },
        },
    }
    meta_path = os.path.join(args.out_dir, "labeling_thresholds_3cls.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        "已生成:",
        os.path.abspath(out_train),
        os.path.abspath(out_val),
        os.path.abspath(out_test),
        "并保存阈值到",
        os.path.abspath(meta_path),
    )
    print("\n标签分布:")
    print(f"训练集: Bearish={meta['label_distribution']['train']['bearish']}, "
          f"Neutral={meta['label_distribution']['train']['neutral']}, "
          f"Bullish={meta['label_distribution']['train']['bullish']}")
    print(f"验证集: Bearish={meta['label_distribution']['val']['bearish']}, "
          f"Neutral={meta['label_distribution']['val']['neutral']}, "
          f"Bullish={meta['label_distribution']['val']['bullish']}")
    print(f"测试集: Bearish={meta['label_distribution']['test']['bearish']}, "
          f"Neutral={meta['label_distribution']['test']['neutral']}, "
          f"Bullish={meta['label_distribution']['test']['bullish']}")


if __name__ == "__main__":
    main()
