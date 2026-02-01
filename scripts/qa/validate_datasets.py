# -*- coding: utf-8 -*-
r"""
数据与标签 QA 校验脚本：
- 校验多维复合标签数据集（train/val/test_multi_labeled.csv）的时间跨度、去重、交叉污染（event_id 不跨集合）、标签分布。
- 可选：校验 finance_analysis.db（events/event_impacts/prices_m1）的时间范围与重复项。
- 输出汇总 JSON 报告到 data/processed/qa_dataset_report.json，便于归档与留档。

使用示例：
  python scripts/qa/validate_datasets.py \
    --processed_dir data/processed \
    --raw_dir data/raw \
    --db finance_analysis.db \
    --ticker XAUUSD
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import pandas as pd  # type: ignore


@dataclass
class SpanReport:
    min_ts: Optional[str]
    max_ts: Optional[str]
    rows: int


def _stat_time_span(df: pd.DataFrame, ts_col: str) -> SpanReport:
    """统计时间范围（遇到空表/无时间列时做兜底）。"""
    if df is None or df.empty or ts_col not in df.columns:
        return SpanReport(None, None, 0)
    s = pd.to_datetime(df[ts_col], errors="coerce").dropna()
    if len(s) == 0:
        return SpanReport(None, None, int(len(df)))
    return SpanReport(str(s.min()), str(s.max()), int(len(df)))


def _label_dist(df: pd.DataFrame, col: str) -> Dict[str, int]:
    """标签分布统计（字符串键，稳定 JSON）。"""
    if df is None or df.empty or col not in df.columns:
        return {}
    vc = df[col].astype("Int64").value_counts(dropna=False).sort_index()
    return {str(int(k)) if pd.notna(k) else "NaN": int(v) for k, v in vc.items()}


def _read_csv_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()


def _dup_report_by_keys(df: pd.DataFrame, keys: List[str]) -> Dict[str, int]:
    if df is None or df.empty:
        return {"dups": 0}
    dups = df.duplicated(subset=keys, keep=False).sum()
    uniq = df.drop_duplicates(subset=keys)
    return {"dups": int(dups), "unique_rows": int(len(uniq))}


def _db_exists(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        return False


def _table_cols(conn: sqlite3.Connection, table: str) -> List[str]:
    """获取指定表的列名列表。"""
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in cur.fetchall()]
        return cols
    except Exception:
        return []


def _select_safe(conn: sqlite3.Connection, table: str, want_cols: List[str]) -> pd.DataFrame:
    """按存在列构造 SELECT，避免缺列报错。"""
    cols = _table_cols(conn, table)
    keep = [c for c in want_cols if c in cols]
    if not keep:
        return pd.DataFrame()
    sql = f"select {', '.join(keep)} from {table}"
    try:
        return pd.read_sql_query(sql, conn)
    except Exception:
        return pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description="数据与标签 QA 校验脚本")
    ap.add_argument(
        "--processed_dir",
        type=str,
        default=os.path.join("data", "processed"),
    )
    ap.add_argument(
        "--raw_dir",
        type=str,
        default=os.path.join("data", "raw"),
    )
    ap.add_argument(
        "--db",
        type=str,
        default="finance_analysis.db",
    )
    ap.add_argument(
        "--ticker",
        type=str,
        default="XAUUSD",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default=os.path.join("data", "processed", "qa_dataset_report.json"),
    )
    args = ap.parse_args()

    os.makedirs(args.processed_dir, exist_ok=True)

    report: Dict[str, object] = {
        "processed_dir": os.path.abspath(args.processed_dir),
        "raw_dir": os.path.abspath(args.raw_dir),
        "db": os.path.abspath(args.db),
        "ticker": args.ticker,
    }

    # 1) 校验多维复合标签数据集
    p_train = os.path.join(args.processed_dir, "train_multi_labeled.csv")
    p_val = os.path.join(args.processed_dir, "val_multi_labeled.csv")
    p_test = os.path.join(args.processed_dir, "test_multi_labeled.csv")

    df_train = _read_csv_safe(p_train)
    df_val = _read_csv_safe(p_val)
    df_test = _read_csv_safe(p_test)

    for df in (df_train, df_val, df_test):
        # 基础兜底：text 列与 label 列存在性
        if "text" not in df.columns and "content" in df.columns:
            df["text"] = df["content"].fillna("").astype(str)
        elif "text" not in df.columns:
            df["text"] = ""

    # 交叉污染检查：event_id 不跨集合
    ids_train = set(
        df_train.get("event_id", pd.Series(dtype=str)).astype(str)
    )
    ids_val = set(
        df_val.get("event_id", pd.Series(dtype=str)).astype(str)
    )
    ids_test = set(
        df_test.get("event_id", pd.Series(dtype=str)).astype(str)
    )
    inter_tv = sorted(list(ids_train.intersection(ids_val)))[:10]
    inter_tt = sorted(list(ids_train.intersection(ids_test)))[:10]
    inter_vt = sorted(list(ids_val.intersection(ids_test)))[:10]

    processed_summary = {
        "train": {
            "span": asdict(_stat_time_span(df_train, "event_ts_local")),
            "rows": int(len(df_train)),
            "dups_event_id": _dup_report_by_keys(df_train, ["event_id"]),
            "label_multi_dist": _label_dist(df_train, "label_multi_cls"),
        },
        "val": {
            "span": asdict(_stat_time_span(df_val, "event_ts_local")),
            "rows": int(len(df_val)),
            "dups_event_id": _dup_report_by_keys(df_val, ["event_id"]),
            "label_multi_dist": _label_dist(df_val, "label_multi_cls"),
        },
        "test": {
            "span": asdict(_stat_time_span(df_test, "event_ts_local")),
            "rows": int(len(df_test)),
            "dups_event_id": _dup_report_by_keys(df_test, ["event_id"]),
            "label_multi_dist": _label_dist(df_test, "label_multi_cls"),
        },
        "cross_split_event_id_intersections": {
            "train_val_first10": inter_tv,
            "train_test_first10": inter_tt,
            "val_test_first10": inter_vt,
            "any_overlap": bool(inter_tv or inter_tt or inter_vt),
        },
    }
    report["processed_multi_labeled"] = processed_summary

    # 2) 数据库校验（可选）
    db_info: Dict[str, object] = {}
    if _db_exists(args.db):
        try:
            conn = sqlite3.connect(args.db)
            # events
            q_events = (
                "select count(*) as cnt, min(ts_local) as min_ts, "
                "max(ts_local) as max_ts from events"
            )
            dfe = pd.read_sql_query(q_events, conn)
            db_info["events"] = {
                "rows": int(dfe["cnt"].iloc[0]) if len(dfe) else 0,
                "span": {
                    "min_ts": None if dfe.empty else str(dfe["min_ts"].iloc[0]),
                    "max_ts": None if dfe.empty else str(dfe["max_ts"].iloc[0]),
                },
            }
            # event_impacts（window 15）
            q_imp = (
                "select count(*) as cnt from event_impacts where ticker=? and window_min=15"
            )
            dfi = pd.read_sql_query(q_imp, conn, params=(args.ticker,))
            db_info["event_impacts_w15"] = {"rows": int(dfi["cnt"].iloc[0]) if len(dfi) else 0}
            # prices_m1
            q_p = (
                "select count(*) as cnt, min(ts_utc) as min_ts, "
                "max(ts_utc) as max_ts from prices_m1 where ticker=?"
            )
            dfp = pd.read_sql_query(q_p, conn, params=(args.ticker,))
            db_info["prices_m1"] = {
                "rows": int(dfp["cnt"].iloc[0]) if len(dfp) else 0,
                "span": {
                    "min_ts": None if dfp.empty else str(dfp["min_ts"].iloc[0]),
                    "max_ts": None if dfp.empty else str(dfp["max_ts"].iloc[0]),
                },
            }
            conn.close()
        except Exception as e:
            db_info["error"] = str(e)
    report["database"] = db_info

    # 3) 列出 raw/processed 目录下的文件（大小与修改时间），便于归档决策
    def list_dir_info(d: str) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        if not os.path.isdir(d):
            return out
        for root, _, files in os.walk(d):
            for fn in files:
                path = os.path.join(root, fn)
                try:
                    st = os.stat(path)
                    out.append(
                        {
                            "path": os.path.relpath(path, start=d),
                            "size_bytes": int(st.st_size),
                            "mtime": str(pd.Timestamp(st.st_mtime, unit="s")),
                        }
                    )
                except Exception:
                    pass
        out.sort(key=lambda x: x["size_bytes"], reverse=True)
        return out

    report["raw_files"] = list_dir_info(args.raw_dir)
    report["processed_files"] = list_dir_info(args.processed_dir)

    # 4) 归档建议（启发式）：
    # - raw/ 下超过 100MB 的大文件优先归档；
    # - processed/ 下历史窗口或旧命名（如包含 2025Q4、last3m 等）的 CSV 可归档；
    def suggest_archive(entries: List[Dict[str, object]], keywords: List[str], min_size: int) -> List[str]:
        sugg: List[str] = []
        for it in entries:
            size = int(it.get("size_bytes", 0))
            p = str(it.get("path", ""))
            if size >= min_size:
                sugg.append(p)
                continue
            for kw in keywords:
                if kw in p:
                    sugg.append(p)
                    break
        return sugg[:50]

    report["archive_suggestions"] = {
        "raw_candidates": suggest_archive(
            report["raw_files"],
            ["flash", "calendar", "mt5", "xauusd"],
            100 * 1024 * 1024,
        ),
        "processed_candidates": suggest_archive(
            report["processed_files"],
            ["2025Q4", "last3m", "_30m_"],
            50 * 1024 * 1024,
        ),
    }

    # 5) 写出报告
    out_path = os.path.abspath(args.out_json)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("QA 报告已写出：", out_path)
    except Exception as e:
        print("写出失败：", e)


if __name__ == "__main__":
    main()
