# -*- coding: utf-8 -*-
"""
为 SQLite 创建常用索引（可重复执行、幂等）。
- labels: (news_id), (ticker, window_min)
- news: (ts)
注意：prices(ticker, ts) 为主键；
init_db.py 已创建 (ticker, trade_date) 索引。
"""
import os
import sqlite3
import sys
import yaml


def _load_cfg() -> dict:
    # 从配置读取数据库路径
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _db_path(cfg: dict) -> str:
    return cfg.get('storage', {}).get('sqlite_path', 'finance.db')


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main() -> None:
    cfg = _load_cfg()
    db = _db_path(cfg)
    _ensure_dir(db)

    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        # 仅在目标表存在时才创建索引，避免 "no such table" 错误
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        existing = {r[0] for r in cur.fetchall()}

        if 'labels' in existing:
            cur.execute(
                'CREATE INDEX IF NOT EXISTS idx_labels_news_id '
                'ON labels(news_id);'
            )
            cur.execute(
                'CREATE INDEX IF NOT EXISTS idx_labels_ticker_window '
                'ON labels(ticker, window_min);'
            )
        if 'news' in existing:
            cur.execute(
                'CREATE INDEX IF NOT EXISTS idx_news_ts ON news(ts);'
            )
        conn.commit()
    except Exception as e:
        print(f'[indexes] failed: {e}', file=sys.stderr)
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

    print('[indexes] created or already existed.')


if __name__ == '__main__':
    main()
