import sqlite3
import yaml

with open('configs/config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

schema = """
CREATE TABLE IF NOT EXISTS prices (
  ticker TEXT,
  ts DATETIME,
  open REAL,
  high REAL,
  low REAL,
  close REAL,
  volume REAL,
  adj_close REAL,
  trade_date DATE,
  PRIMARY KEY (ticker, ts)
);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(ticker, trade_date);

CREATE TABLE IF NOT EXISTS news (
  id TEXT PRIMARY KEY,
  ts DATETIME,
  source TEXT,
  title TEXT,
  content TEXT,
  ticker_hint TEXT,
  url TEXT,
  hash TEXT
);
CREATE INDEX IF NOT EXISTS idx_news_ts ON news(ts);

CREATE TABLE IF NOT EXISTS labels (
  news_id TEXT,
  ticker TEXT,
  window_min INTEGER,
  ret REAL,
  label TEXT CHECK(label IN ('bullish','neutral','bearish')),
  PRIMARY KEY (news_id, ticker, window_min)
);

CREATE TABLE IF NOT EXISTS reports (
  id TEXT,
  ticker TEXT,
  period TEXT,
  source TEXT,
  file_path TEXT,
  page_idx INTEGER,
  section TEXT,
  PRIMARY KEY (id, page_idx)
);

CREATE TABLE IF NOT EXISTS trading_calendar (
  market TEXT,
  trade_date DATE,
  is_open INTEGER,
  PRIMARY KEY (market, trade_date)
);
"""

db_path = cfg.get('storage', {}).get('sqlite_path', 'finance.db')
conn = sqlite3.connect(db_path)
conn.executescript(schema)
conn.commit()
conn.close()
print('initialized', db_path)
