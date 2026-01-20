import os
import argparse
import sqlite3
import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

with open('configs/config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

DB_PATH = cfg.get('storage', {}).get('sqlite_path', 'finance.db')


def is_a_share(ticker: str) -> bool:
    return ticker.endswith('.SH') or ticker.endswith('.SZ')


def is_a_index(ticker: str) -> bool:
    a_idx = cfg.get('universe', {}).get('a_index', []) or []
    return ticker in a_idx


def fetch_yf(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval='1d',
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={
        'Date': 'ts',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close',
        'Volume': 'volume',
    })
    if 'adj_close' not in df.columns:
        df['adj_close'] = df['close']
    df['trade_date'] = pd.to_datetime(df['ts']).dt.date.astype(str)
    df['ticker'] = ticker
    cols = [
        'ticker', 'ts', 'open', 'high', 'low',
        'close', 'volume', 'adj_close', 'trade_date',
    ]
    df = df[cols]
    return df


def fetch_ts_stock(ticker: str, start: str, end: str) -> pd.DataFrame:
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        return pd.DataFrame()
    import tushare as ts
    pro = ts.pro_api(token)
    st = start.replace('-', '')
    ed = end.replace('-', '')
    df = pro.daily(ts_code=ticker, start_date=st, end_date=ed)
    if df is None or df.empty:
        return pd.DataFrame()
    df['ts'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.rename(columns={'vol': 'volume'})
    df['adj_close'] = df['close']
    df['trade_date'] = df['ts'].dt.date.astype(str)
    df['ticker'] = ticker
    cols = [
        'ticker', 'ts', 'open', 'high', 'low',
        'close', 'volume', 'adj_close', 'trade_date',
    ]
    df = df[cols].sort_values('ts')
    return df


def fetch_ts_index(ticker: str, start: str, end: str) -> pd.DataFrame:
    token = os.getenv('TUSHARE_TOKEN')
    if not token:
        return pd.DataFrame()
    import tushare as ts
    pro = ts.pro_api(token)
    st = start.replace('-', '')
    ed = end.replace('-', '')
    df = pro.index_daily(ts_code=ticker, start_date=st, end_date=ed)
    if df is None or df.empty:
        return pd.DataFrame()
    df['ts'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.rename(columns={'vol': 'volume'})
    # Index has no adj_close; mirror close
    df['adj_close'] = df['close']
    df['trade_date'] = df['ts'].dt.date.astype(str)
    df['ticker'] = ticker
    cols = [
        'ticker', 'ts', 'open', 'high', 'low',
        'close', 'volume', 'adj_close', 'trade_date',
    ]
    df = df[cols].sort_values('ts')
    return df


def upsert(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    rows = [(
        str(r['ticker']),
        pd.to_datetime(r['ts']).strftime('%Y-%m-%d %H:%M:%S'),
        float(r['open']),
        float(r['high']),
        float(r['low']),
        float(r['close']),
        float(r['volume']) if pd.notna(r['volume']) else None,
        float(r['adj_close']) if pd.notna(r['adj_close']) else None,
        str(r['trade_date'])
    ) for _, r in df.iterrows()]
    sql = (
        'INSERT OR REPLACE INTO prices '
        '(ticker, ts, open, high, low, close, volume, adj_close, trade_date) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'
    )
    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def get_universe() -> list:
    u = cfg.get('universe', {})
    t = []
    t += u.get('a_index', []) or []
    t += u.get('hs300_components', []) or []
    t += u.get('gold', []) or []
    t += u.get('us_equities', []) or []
    return [i for i in t if i and not str(i).startswith('PLACEHOLDER')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', default=cfg['universe']['start_date'])
    p.add_argument('--end', default=cfg['universe']['end_date'])
    p.add_argument('--tickers', nargs='*')
    args = p.parse_args()
    tickers = args.tickers or get_universe()
    conn = sqlite3.connect(DB_PATH)
    total = 0
    for t in tickers:
        if is_a_share(t):
            # route A-share index vs stock
            if is_a_index(t):
                df = fetch_ts_index(t, args.start, args.end)
            else:
                df = fetch_ts_stock(t, args.start, args.end)
            if df.empty:
                print(f'skip {t}: tushare unavailable or no data')
        else:
            df = fetch_yf(t, args.start, args.end)
            if df.empty:
                print(f'skip {t}: yfinance no data')
        n = upsert(conn, df)
        total += n
        print(f'{t}: {n} rows')
    conn.close()
    print(f'done: {total} rows')


if __name__ == '__main__':
    main()
