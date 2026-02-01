# -*- coding: utf-8 -*-
"""
使用 MetaTrader5 抓取黄金分钟级行情，按北京时间输出为 CSV。

- 默认按候选符号列表自动匹配你账户可用的黄金合约（如 XAUUSD、XAUUSD.i、GOLD 等）。
- 支持 M1/M5/M15 等时间框架，支持按日期范围分片抓取并合并。
- 输出列：ticker, ts(北京时间), open, high, low, close, volume。

用法示例（先做小范围连通性测试）：
  python scripts/fetch_intraday_xauusd_mt5.py \
    --timeframe M1 \
    --start 2025-12-01 \
    --end 2026-01-30 \
    --out "data/processed/xauusd_m1_mt5_test.csv"

如需全量 2024-2025：
  python scripts/fetch_intraday_xauusd_mt5.py \
    --timeframe M1 \
    --start 2024-01-01 \
    --end 2025-12-31 \
    --chunk_days 15 \
    --out "data/processed/xauusd_m1_mt5_2024_2025.csv"

可选：若需要显式指定登录或终端路径：
  python scripts/fetch_intraday_xauusd_mt5.py \
    --login 12345678 \
    --password ****** \
    --server "YourBroker-Server" \
    --mt5_path "C:/Program Files/MetaTrader 5/terminal64.exe" \
    --timeframe M1 \
    --start 2025-12-01 \
    --end 2026-01-30 \
    --out "data/processed/xauusd_m1_mt5.csv"
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from typing import List, Optional

import MetaTrader5 as mt5  # type: ignore
import pandas as pd  # type: ignore


def _ensure_dir(path: str) -> None:
    """确保输出目录存在。"""
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _parse_date(s: str) -> datetime:
    """解析日期字符串，支持 YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS（返回 naive datetime）。"""
    s = s.strip()
    if len(s) > 10:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    return datetime.strptime(s, "%Y-%m-%d")


def _map_timeframe(tf: str) -> int:
    """将 M1/M5/M15/M30/H1 映射为 MT5 时间框架常量。"""
    tf = tf.upper()
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
    }
    if tf not in mapping:
        raise SystemExit(f"不支持的 timeframe: {tf}")
    return mapping[tf]


def _select_symbol(candidates: List[str]) -> Optional[str]:
    """在当前账户中选出第一个可用且可见的符号；必要时调用 symbol_select 使其可见。"""
    for name in candidates:
        info = mt5.symbol_info(name)
        if info is None:
            continue
        if not info.visible:
            mt5.symbol_select(name, True)
        # 再查一次，确保已可用
        info = mt5.symbol_info(name)
        if info is not None and info.visible:
            return name
    # 兜底：尝试扫描 XAU/GOLD 关键词
    for pattern in ("*XAU*", "*GOLD*"):
        symbols = mt5.symbols_get(pattern) or []
        for s in symbols:
            name = s.name
            if not s.visible:
                mt5.symbol_select(name, True)
            info2 = mt5.symbol_info(name)
            if info2 is not None and info2.visible:
                return name
    return None


def _fetch_range(
    symbol: str,
    tf_code: int,
    start_dt: datetime,
    end_dt: datetime,
    chunk_days: int,
) -> pd.DataFrame:
    """按日期范围分片抓取，避免一次性数据量过大导致失败。"""
    dfs: List[pd.DataFrame] = []
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + timedelta(days=chunk_days), end_dt)
        rates = mt5.copy_rates_range(symbol, tf_code, cur, nxt)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            dfs.append(df)
        cur = nxt
    if not dfs:
        return pd.DataFrame()
    df_all = pd.concat(dfs, ignore_index=True)
    # 去重并按时间排序
    if "time" in df_all.columns:
        df_all = (
            df_all.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        )
    return df_all


def _normalize_mt5_df(
    df: pd.DataFrame, tz_out: str, label_ticker: str
) -> pd.DataFrame:
    """将 MT5 返回的 rates 数据规范为统一格式。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "ts", "open", "high", "low", "close", "volume"])

    # MT5 的 time 为从 1970-01-01 起的秒数（通常以 UTC 解释）。
    # 注意：这里返回的是 Series，需要通过 .dt 进行时区转换。
    t_col = "time"
    if t_col not in df.columns:
        return pd.DataFrame(columns=["ticker", "ts", "open", "high", "low", "close", "volume"])
    t_utc = pd.to_datetime(df[t_col], unit="s", utc=True, errors="coerce")
    t_local = t_utc.dt.tz_convert(tz_out)

    out = pd.DataFrame()
    out["ticker"] = [label_ticker] * len(df)
    out["ts"] = t_local.dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
    out["open"] = pd.to_numeric(df.get("open", pd.NA), errors="coerce")
    out["high"] = pd.to_numeric(df.get("high", pd.NA), errors="coerce")
    out["low"] = pd.to_numeric(df.get("low", pd.NA), errors="coerce")
    out["close"] = pd.to_numeric(df.get("close", pd.NA), errors="coerce")
    # 优先使用真实成交量 real_volume，不存在时回退 tick_volume
    vol = (
        df["real_volume"]
        if "real_volume" in df.columns
        else df.get("tick_volume", 0)
    )
    out["volume"] = pd.to_numeric(vol, errors="coerce").fillna(0)

    out = (
        out.dropna(subset=["ts"]).drop_duplicates(subset=["ticker", "ts"], keep="last").reset_index(drop=True)
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MT5 分钟级黄金行情抓取并导出 CSV（北京时间）"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="XAUUSD",
        help="优先尝试的符号名，如 XAUUSD；留空则自动匹配",
    )
    parser.add_argument(
        "--symbol_candidates",
        type=str,
        default="XAUUSD,XAUUSD.i,XAUUSDm,GOLD,XAUUSD.a",
        help="候选符号名，逗号分隔，将按顺序尝试",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="M1",
        choices=["M1", "M5", "M15", "M30", "H1"],
        help="时间框架",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2024-01-01",
        help="开始日期，YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS",
    )
    parser.add_argument(
        "--end", type=str, default="", help="结束日期，留空=当前时间"
    )
    parser.add_argument(
        "--chunk_days",
        type=int,
        default=15,
        help="按天分片抓取的分片大小，避免一次性过大",
    )
    parser.add_argument(
        "--tz_out",
        type=str,
        default="Asia/Shanghai",
        help="输出时区（ts 将为该时区的字符串）",
    )
    parser.add_argument(
        "--label_ticker",
        type=str,
        default="XAUUSD",
        help="输出 CSV 中的 ticker 标签名",
    )
    parser.add_argument("--out", type=str, required=True, help="输出 CSV 路径")
    parser.add_argument(
        "--mt5_path",
        type=str,
        default="",
        help=(
            "可选：MT5 终端路径，如 C:/Program Files/MetaTrader 5/terminal64.exe"
        ),
    )
    parser.add_argument(
        "--login",
        type=int,
        default=0,
        help="可选：账户登录号；不填则使用当前终端的已登录账户",
    )
    parser.add_argument("--password", type=str, default="", help="可选：账户密码")
    parser.add_argument(
        "--server", type=str, default="", help="可选：交易服务器名，如 YourBroker-Server"
    )

    args = parser.parse_args()

    # 初始化 MT5 终端
    if args.mt5_path:
        ok = mt5.initialize(path=args.mt5_path)
    else:
        ok = mt5.initialize()
    if not ok:
        code, msg = mt5.last_error()
        raise SystemExit(f"MT5 初始化失败：{code}, {msg}")

    try:
        # 可选登录
        if args.login:
            if not mt5.login(args.login, password=args.password or "", server=args.server or ""):
                code, msg = mt5.last_error()
                raise SystemExit(f"MT5 登录失败：{code}, {msg}")

        # 解析时间范围
        start_dt = _parse_date(args.start)
        end_dt = _parse_date(args.end) if args.end.strip() else datetime.now()
        if end_dt <= start_dt:
            raise SystemExit("结束时间必须晚于开始时间")

        # 选择符号
        candidates: List[str] = [s.strip() for s in args.symbol_candidates.split(",") if s.strip()]
        if args.symbol.strip():
            # 将 --symbol 放到候选列表最前
            sym_first = args.symbol.strip()
            candidates = [sym_first] + [s for s in candidates if s != sym_first]
        symbol = _select_symbol(candidates)
        if not symbol:
            raise SystemExit(
                "未找到可用的黄金符号；请使用 --symbol 指定你账户中的具体符号名"
            )

        tf_code = _map_timeframe(args.timeframe)

        # 抓取数据
        raw = _fetch_range(symbol, tf_code, start_dt, end_dt, args.chunk_days)
        if raw is None or raw.empty:
            raise SystemExit(
                "MT5 返回数据为空；请检查符号是否有历史数据、账户是否已登录、或缩小时间范围重试"
            )

        out_df = _normalize_mt5_df(raw, args.tz_out, args.label_ticker)
        if out_df.empty:
            raise SystemExit("标准化后的数据为空，请检查原始数据结构是否变更。")

        _ensure_dir(args.out)
        out_df.to_csv(args.out, index=False, encoding="utf-8")
        print(f"已写出 {len(out_df)} 行到 {args.out}")
        print(out_df.head(5).to_string())

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
