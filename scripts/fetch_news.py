import argparse
import sqlite3
import pandas as pd
import yaml
import hashlib

with open('configs/config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

DB_PATH = cfg.get('storage', {}).get('sqlite_path', 'finance.db')


def _mk_id(row) -> str:
    base = (
        str(row.get('title', ''))
        + str(row.get('content', ''))
        + str(row.get('ts', ''))
    ).encode('utf-8')
    return hashlib.sha1(base).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--csv',
        required=False,
        help=(
            'path to CSV with columns: '
            'ts,source,title,content,url,ticker_hint'
        ),
    )
    args = p.parse_args()
    if not args.csv:
        print('no csv provided; noop')
        return
    df = pd.read_csv(args.csv)
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'])
    else:
        df['ts'] = pd.Timestamp.utcnow()
    if 'source' not in df.columns:
        df['source'] = 'csv'
    if 'ticker_hint' not in df.columns:
        df['ticker_hint'] = ''
    if 'url' not in df.columns:
        df['url'] = ''
    if 'title' not in df.columns:
        df['title'] = ''
    if 'content' not in df.columns:
        df['content'] = ''
    df['id'] = df.apply(_mk_id, axis=1)
    df['hash'] = df['id']
    cols = [
        'id', 'ts', 'source', 'title', 'content',
        'ticker_hint', 'url', 'hash',
    ]
    df = df[cols]
    conn = sqlite3.connect(DB_PATH)
    rows = [
        (
            r['id'],
            pd.to_datetime(r['ts']).strftime('%Y-%m-%d %H:%M:%S'),
            r['source'],
            r['title'],
            r['content'],
            r['ticker_hint'],
            r['url'],
            r['hash'],
        )
        for _, r in df.iterrows()
    ]
    sql = (
        'INSERT OR REPLACE INTO news '
        '(id, ts, source, title, content, ticker_hint, url, hash) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
    )
    conn.executemany(sql, rows)
    conn.commit()
    conn.close()
    print(f'inserted {len(rows)} rows')


if __name__ == '__main__':
    main()
