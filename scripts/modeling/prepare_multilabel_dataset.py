# -*- coding: utf-8 -*-
"""
从 finance_analysis.db 生成“多维复合标签”训练集：
- 基础方向标签：基于事件发布后 15 分钟收益（ret_post）分位阈值（train 上估计）。
- “预期兑现”标签：比较发布前 120 分钟趋势与发布后 15 分钟方向，结合指标“预期差”（actual-consensus）。
- “建议观望”标签：高波动（Range/p0 高分位）+ 低净变化（|ret_post| 低分位）的十字星态势。

输出：按时间切分 train/val/test 三个 CSV 与一个元数据 JSON（阈值与标签映射）。

注意：
- 采用 asof 逻辑查价（<= 该时间的最近分钟），与构库脚本保持一致；
- 使用分位阈值，避免被极端值支配；可通过 CLI 指定固定阈值；
- 中文注释仅解释关键逻辑；其余保持简洁以便审阅。
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
    d = os.path.dirname(os.path.abspath(path))
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


def _to_float(x: object) -> Optional[float]:
    """将字符串中的数值解析为 float；无法解析返回 None。支持百分号与逗号。"""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-", "na"}:
        return None
    # 去除中文符号、百分号、逗号等
    s = s.replace("%", "").replace(",", "").replace("\u200b", "")
    try:
        return float(s)
    except Exception:
        return None


def _sign(x: Optional[float]) -> int:
    if x is None or not (x == x):
        return 0
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _compute_quantiles(train: pd.DataFrame, q_low: float, q_high: float) -> Tuple[float, float]:
    """在训练集上估计 ret_post 的分位阈值（基础方向标签）。"""
    ser = train["ret_post"].dropna().astype(float)
    if len(ser) == 0:
        return -0.001, 0.001
    return float(ser.quantile(q_low)), float(ser.quantile(q_high))


def main() -> None:
    parser = argparse.ArgumentParser(description="生成多维复合标签训练集")
    parser.add_argument("--db", type=str, default="finance_analysis.db")
    parser.add_argument("--ticker", type=str, default="XAUUSD")
    parser.add_argument("--window_post", type=int, default=15)
    parser.add_argument("--pre_minutes", type=int, default=120)
    # 时间切分（北京时间字符串，与入库保持一致）
    parser.add_argument("--train_end", type=str, default="2025-08-01 00:00:00")
    parser.add_argument("--val_end", type=str, default="2025-11-01 00:00:00")
    parser.add_argument("--test_end", type=str, default="2026-02-01 00:00:00")
    # 分位阈值（可选固定阈值覆盖）
    parser.add_argument("--neutral_q_low", type=float, default=0.30)
    parser.add_argument("--neutral_q_high", type=float, default=0.70)
    parser.add_argument("--pricedin_pre_abs_q", type=float, default=0.80)
    parser.add_argument("--pricedin_post_abs_q", type=float, default=0.70)
    parser.add_argument("--watch_range_q", type=float, default=0.85)
    parser.add_argument("--watch_absret_q", type=float, default=0.20)
    parser.add_argument("--fixed_pre_abs", type=float, default=None)
    parser.add_argument("--fixed_post_abs", type=float, default=None)
    parser.add_argument("--fixed_watch_range", type=float, default=None)
    parser.add_argument("--fixed_watch_absret", type=float, default=None)
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
            "select event_id, price_event, price_future, delta, ret as ret_post "
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

        # 2) 读分钟价索引，计算 pre 返回与 Range（post 窗）
        prices = _read_prices(conn, args.ticker)
    finally:
        conn.close()

    # 预处理收益：裁剪极端值，提升稳健性
    df["ret_post"] = pd.to_numeric(df["ret_post"], errors="coerce")
    clip = float(abs(args.clip_ret))
    df.loc[:, "ret_post"] = df["ret_post"].clip(lower=-clip, upper=clip)

    # 逐事件计算 pre_ret、range_ratio、abs_ret
    pre_list: List[Optional[float]] = []
    rng_list: List[Optional[float]] = []
    absret_list: List[Optional[float]] = []
    for _, r in df.iterrows():
        ts_evt: pd.Timestamp = r["event_ts_utc"]
        if pd.isna(ts_evt):
            pre_list.append(None)
            rng_list.append(None)
            absret_list.append(None)
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
        # abs(ret_post)
        rp = r.get("ret_post", None)
        try:
            abs_rp = abs(float(rp)) if rp is not None else None
        except Exception:
            abs_rp = None
        absret_list.append(abs_rp)

    df["pre_ret"] = pre_list
    df["range_ratio"] = rng_list
    df["abs_ret_post"] = absret_list

    # 预期差与方向（仅对可解析的指标行生效）
    act = df["actual"].apply(_to_float)
    con = df["consensus"].apply(_to_float)
    surp: List[Optional[float]] = []
    for a, c in zip(act, con):
        if a is None or c is None:
            surp.append(None)
        else:
            surp.append(a - c)
    df["surprise"] = surp
    df["fundamental_sign"] = df["surprise"].apply(_sign)

    # 时间切分（仅按本地时间）
    t1 = pd.Timestamp(args.train_end)
    t2 = pd.Timestamp(args.val_end)
    t3 = pd.Timestamp(args.test_end)
    train = df[df["event_ts_local"] < t1].copy()
    val = df[(df["event_ts_local"] >= t1) & (df["event_ts_local"] < t2)].copy()
    test = df[(df["event_ts_local"] >= t2) & (df["event_ts_local"] < t3)].copy()

    # 3) 估计阈值（仅使用训练集）
    ql, qh = _compute_quantiles(train, args.neutral_q_low, args.neutral_q_high)
    # 预期兑现：pre 单边与 post 反向的幅度阈值
    pre_abs_thr = (
        float(train["pre_ret"].abs().quantile(args.pricedin_pre_abs_q))
        if args.fixed_pre_abs is None
        else float(abs(args.fixed_pre_abs))
    )
    post_abs_thr = (
        float(train["ret_post"].abs().quantile(args.pricedin_post_abs_q))
        if args.fixed_post_abs is None
        else float(abs(args.fixed_post_abs))
    )
    # 观望：高波动 + 低净变化
    range_thr = (
        float(train["range_ratio"].quantile(args.watch_range_q))
        if args.fixed_watch_range is None
        else float(abs(args.fixed_watch_range))
    )
    absret_thr = (
        float(train["abs_ret_post"].quantile(args.watch_absret_q))
        if args.fixed_watch_absret is None
        else float(abs(args.fixed_watch_absret))
    )

    # 4) 打标函数
    def base_label(x: Optional[float]) -> int:
        if x is None or not (x == x):
            return 0
        if x <= ql:
            return -1
        if x >= qh:
            return 1
        return 0

    def priced_in(row: pd.Series) -> int:
        fs = int(row.get("fundamental_sign", 0))
        pre = row.get("pre_ret", None)
        post = row.get("ret_post", None)
        try:
            pre_ok = pre is not None and (abs(float(pre)) >= pre_abs_thr)
            post_ok = post is not None and (abs(float(post)) >= post_abs_thr)
        except Exception:
            pre_ok, post_ok = False, False
        if not (fs != 0 and pre_ok and post_ok):
            return 0
        # 方向：发布后与基本面相反，且 pre 与基本面同向
        try:
            if fs > 0 and float(pre) > 0 and float(post) < 0:
                return 1  # 利好兑现
            if fs < 0 and float(pre) < 0 and float(post) > 0:
                return -1  # 利空兑现
        except Exception:
            return 0
        return 0

    def watch_flag(row: pd.Series) -> int:
        rr = row.get("range_ratio", None)
        ap = row.get("abs_ret_post", None)
        try:
            if rr is None or ap is None:
                return 0
            return 1 if (float(rr) >= range_thr and float(ap) <= absret_thr) else 0
        except Exception:
            return 0

    for part in (train, val, test):
        part["label_base"] = part["ret_post"].apply(base_label)
        part["label_priced_in"] = part.apply(priced_in, axis=1)
        part["label_watch"] = part.apply(watch_flag, axis=1)
        # 合成单列多类标签（优先级：观望 > 兑现 > 基础方向）
        multi: List[int] = []
        for lb, pi, wf in zip(
            part["label_base"].values, part["label_priced_in"].values, part["label_watch"].values
        ):
            if int(wf) == 1:
                multi.append(5)
            elif int(pi) == 1:
                multi.append(3)
            elif int(pi) == -1:
                multi.append(4)
            elif int(lb) == 1:
                multi.append(1)
            elif int(lb) == -1:
                multi.append(2)
            else:
                multi.append(0)
        part["label_multi_cls"] = multi

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
        "price_future",
        "delta",
        "ret_post",
        "pre_ret",
        "range_ratio",
        "abs_ret_post",
        "surprise",
        "fundamental_sign",
        "label_base",
        "label_priced_in",
        "label_watch",
        "label_multi_cls",
    ]

    out_train = os.path.join(args.out_dir, "train_multi_labeled.csv")
    out_val = os.path.join(args.out_dir, "val_multi_labeled.csv")
    out_test = os.path.join(args.out_dir, "test_multi_labeled.csv")
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
            "base": {"q_low": ql, "q_high": qh},
            "priced_in": {
                "pre_abs": pre_abs_thr,
                "post_abs": post_abs_thr,
                "q_pre_abs": float(args.pricedin_pre_abs_q),
                "q_post_abs": float(args.pricedin_post_abs_q),
            },
            "watch": {
                "range_thr": range_thr,
                "absret_thr": absret_thr,
                "q_range": float(args.watch_range_q),
                "q_absret": float(args.watch_absret_q),
            },
        },
        "label_multi_mapping": {
            "0": "neutral",
            "1": "bullish",
            "2": "bearish",
            "3": "bullish_priced_in",
            "4": "bearish_priced_in",
            "5": "watch",
        },
        "sizes": {
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
        },
    }
    meta_path = os.path.join(args.out_dir, "labeling_thresholds_multilabel.json")
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


if __name__ == "__main__":
    main()
