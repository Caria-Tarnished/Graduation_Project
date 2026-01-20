import argparse
import hashlib
import os
import sqlite3
from typing import List, Dict, Optional

import pandas as pd
import yaml


def _load_cfg() -> dict:
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _db_path(cfg: dict) -> str:
    return cfg.get('storage', {}).get('sqlite_path', 'finance.db')


def _mk_id(title: str, content: str, ts: pd.Timestamp) -> str:
    base = (str(title) + str(content) + str(ts)).encode('utf-8')
    return hashlib.sha1(base).hexdigest()


def _parse_date(s: str) -> Optional[pd.Timestamp]:
    if pd.isna(s):
        return None
    s = str(s).strip()
    # Try Chinese date like 2016年5月12日
    try:
        import re
        m = re.match(r'\s*(\d{4})年(\d{1,2})月(\d{1,2})日', s)
        if m:
            y, mo, d = map(int, m.groups())
            return pd.Timestamp(year=y, month=mo, day=d)
    except Exception:
        pass
    # Fallback to pandas parser
    try:
        return pd.to_datetime(s, errors='coerce')
    except Exception:
        return None


def _get_universe(cfg: dict) -> List[str]:
    u = cfg.get('universe', {})
    t: List[str] = []
    t += u.get('a_index', []) or []
    t += u.get('hs300_components', []) or []
    t += u.get('gold', []) or []
    t += u.get('us_equities', []) or []
    return [x for x in t if x]


def _load_prices(
    conn: sqlite3.Connection,
    tickers: List[str],
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        q = (
            "SELECT ts, adj_close FROM prices WHERE ticker = ? ORDER BY ts ASC"
        )
        df = pd.read_sql_query(q, conn, params=(t,))
        if df is None or df.empty:
            out[t] = pd.DataFrame(columns=['date', 'adj_close'])
            continue
        df['date'] = pd.to_datetime(df['ts']).dt.normalize()
        df = df[['date', 'adj_close']].dropna().reset_index(drop=True)
        out[t] = df
    return out


def _label_returns(ret: float, neutral_band: float) -> str:
    if ret > neutral_band:
        return 'bullish'
    if ret < -neutral_band:
        return 'bearish'
    return 'neutral'


def _candidate_tickers(row: pd.Series, args, cfg: dict) -> List[str]:
    if args.tickers:
        return args.tickers
    if args.ticker:
        return [args.ticker]
    # From CSV or DB row
    hint = row.get('ticker_hint')
    if isinstance(hint, str) and hint.strip():
        parts = [
            x.strip()
            for x in hint.replace(';', ',').split(',')
            if x.strip()
        ]
        if parts:
            return parts
    return _get_universe(cfg)


def label_from_csv(args) -> pd.DataFrame:
    cfg = _load_cfg()
    conn = sqlite3.connect(_db_path(cfg))
    windows = cfg.get('events', {}).get('windows_days', [1, 3, 5])
    neutral_band = float(cfg.get('events', {}).get('neutral_band', 0.003))

    use_cols = [args.date_col, args.title_col, args.content_col]
    opt_cols = ['ticker_hint']
    df = pd.read_csv(args.csv)
    # Normalize column names if possible
    for col in use_cols:
        if col not in df.columns:
            raise ValueError(f'Column {col} not found in CSV')
    for c in opt_cols:
        if c not in df.columns:
            df[c] = ''

    df['_ts'] = df[args.date_col].apply(_parse_date)
    df = df[df['_ts'].notna()].copy()
    df['_ts'] = pd.to_datetime(df['_ts']).dt.normalize()

    # Preload prices
    universe = set()
    # Tentative: if a default ticker is given, preload that, else all
    if args.tickers:
        universe.update(args.tickers)
    elif args.ticker:
        universe.add(args.ticker)
    else:
        universe.update(_get_universe(cfg))
    prices = _load_prices(conn, sorted(universe))

    rows = []
    for _, r in df.iterrows():
        ts0 = r['_ts']
        title = r.get(args.title_col, '')
        content = r.get(args.content_col, '')
        nid = _mk_id(title, content, ts0)
        tickers = _candidate_tickers(r, args, cfg)
        for t in tickers:
            p = prices.get(t)
            if p is None:
                # lazy load if not preloaded
                p = _load_prices(conn, [t]).get(t)
                prices[t] = p
            if p is None or p.empty:
                continue
            dates = p['date'].values
            # find next trading day index
            idx = dates.searchsorted(ts0, side='left')
            if idx >= len(p):
                continue
            for d in windows:
                idx2 = idx + int(d)
                if idx2 >= len(p):
                    continue
                p0 = float(p.iloc[idx]['adj_close'])
                p1 = float(p.iloc[idx2]['adj_close'])
                if p0 == 0:
                    continue
                ret = p1 / p0 - 1.0
                label = _label_returns(ret, neutral_band)
                rows.append({
                    'news_id': nid,
                    'ticker': t,
                    'window_min': int(d) * 1440,
                    'ret': ret,
                    'label': label,
                    'ts': ts0,
                    'title': title,
                })

    out_df = pd.DataFrame(rows)
    if not args.dry_run and not out_df.empty:
        to_insert = [
            (
                r['news_id'],
                r['ticker'],
                int(r['window_min']),
                float(r['ret']),
                r['label'],
            )
            for _, r in out_df.iterrows()
        ]
        cur = conn.cursor()
        cur.executemany(
            'INSERT OR REPLACE INTO labels '
            '(news_id, ticker, window_min, ret, label) VALUES (?, ?, ?, ?, ?)',
            to_insert,
        )
        conn.commit()
    conn.close()
    return out_df


def label_from_db(args) -> pd.DataFrame:
    cfg = _load_cfg()
    conn = sqlite3.connect(_db_path(cfg))
    windows = cfg.get('events', {}).get('windows_days', [1, 3, 5])
    neutral_band = float(cfg.get('events', {}).get('neutral_band', 0.003))

    news = pd.read_sql_query(
        'SELECT id as news_id, ts, title, content, ticker_hint FROM news', conn
    )
    if news.empty:
        conn.close()
        return pd.DataFrame()
    news['ts'] = pd.to_datetime(news['ts'], errors='coerce').dt.normalize()
    news = news[news['ts'].notna()].copy()

    # Universe and prices
    candidates = set(_get_universe(cfg))
    if args.tickers:
        candidates = set(args.tickers)
    elif args.ticker:
        candidates = {args.ticker}
    prices = _load_prices(conn, sorted(candidates))

    rows = []
    for _, r in news.iterrows():
        nid = r['news_id']
        ts0 = r['ts']
        tickers = _candidate_tickers(r, args, cfg)
        for t in tickers:
            p = prices.get(t)
            if p is None or p.empty:
                continue
            dates = p['date'].values
            idx = dates.searchsorted(ts0, side='left')
            if idx >= len(p):
                continue
            for d in windows:
                idx2 = idx + int(d)
                if idx2 >= len(p):
                    continue
                p0 = float(p.iloc[idx]['adj_close'])
                p1 = float(p.iloc[idx2]['adj_close'])
                if p0 == 0:
                    continue
                ret = p1 / p0 - 1.0
                label = _label_returns(ret, neutral_band)
                rows.append({
                    'news_id': nid,
                    'ticker': t,
                    'window_min': int(d) * 1440,
                    'ret': ret,
                    'label': label,
                })

    out_df = pd.DataFrame(rows)
    if not args.dry_run and not out_df.empty:
        to_insert = [
            (
                r['news_id'],
                r['ticker'],
                int(r['window_min']),
                float(r['ret']),
                r['label'],
            )
            for _, r in out_df.iterrows()
        ]
        cur = conn.cursor()
        cur.executemany(
            'INSERT OR REPLACE INTO labels '
            '(news_id, ticker, window_min, ret, label) VALUES (?, ?, ?, ?, ?)',
            to_insert,
        )
        conn.commit()
    conn.close()
    return out_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', help='Optional: path to CSV (Title/Date/Content)')
    p.add_argument('--date-col', default='Date')
    p.add_argument('--title-col', default='Title')
    p.add_argument('--content-col', default='Content')
    p.add_argument(
        '--ticker',
        help='Default ticker for all rows (e.g., XAUUSD=X)',
    )
    p.add_argument('--tickers', nargs='*', help='Override list of tickers')
    p.add_argument('--out-csv', help='Optional: export labeled rows to CSV')
    p.add_argument('--dry-run', action='store_true', help='Do not write to DB')
    args = p.parse_args()

    if args.csv:
        out = label_from_csv(args)
    else:
        out = label_from_db(args)

    print(f'labeled rows: {len(out)}')
    if args.out_csv and not out.empty:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print(f'exported to {args.out_csv}')


if __name__ == '__main__':
    main()
