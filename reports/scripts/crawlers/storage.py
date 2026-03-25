# -*- coding: utf-8 -*-
"""
SQLite 持久化工具：用于存储抓取到的新闻/快讯。
- 设计目标：
  1) 以 url 去重；
  2) 对无 URL 的来源（如部分列表内嵌快讯）以 (hash, site, source, published_at) 去重；
  3) 便于未来迁移到 PostgreSQL（接口尽量简洁）。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional


@dataclass
class Article:
    """文章/快讯的结构化数据。"""
    site: str
    source: str  # 例如 'detail', 'listing_flash', 'listing_data'
    title: str
    content: str
    published_at: Optional[str]  # ISO 格式日期或日期时间字符串，允许为空
    url: Optional[str] = None
    raw_html: Optional[str] = None
    extra_json: Optional[Dict[str, Any]] = None


def get_conn(db_path: str) -> sqlite3.Connection:
    """建立 SQLite 连接，启用 WAL 与较安全的 pragma。"""
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('PRAGMA foreign_keys=ON')
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """创建所需表与索引（幂等）。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site TEXT NOT NULL,
            source TEXT NOT NULL,
            url TEXT,
            title TEXT,
            content TEXT,
            published_at TEXT,
            raw_html TEXT,
            extra_json TEXT,
            hash TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    # URL 唯一（当 URL 存在时）
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_articles_url
        ON articles(url) WHERE url IS NOT NULL
        """
    )
    # 对无 URL 的记录，按 (hash, site, source, published_at) 唯一
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_articles_hash_site_source_pub
        ON articles(hash, site, source, published_at)
        WHERE url IS NULL
        """
    )
    conn.commit()


def _calc_hash(title: str, content: str) -> str:
    """基于标题与正文计算稳定哈希，便于无 URL 去重。"""
    base = (title or '').strip() + '\n' + (content or '').strip()
    return hashlib.sha1(base.encode('utf-8')).hexdigest()


def upsert_article(conn: sqlite3.Connection, art: Article) -> None:
    """插入或更新一条记录（兼容不支持部分唯一索引 UPSERT 的 SQLite 版本）。
    - 若 url 存在：按 url 先查后更/插。
    - 若 url 不存在：按 (hash, site, source, published_at) 先查后更/插。
    """
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    h = _calc_hash(art.title or '', art.content or '')
    extra_str = (
        json.dumps(art.extra_json, ensure_ascii=False) if art.extra_json else None
    )

    if art.url:
        row = conn.execute(
            'SELECT id FROM articles WHERE url = ?', (art.url,)
        ).fetchone()
        if row:
            conn.execute(
                '''
                UPDATE articles SET
                    site=?, source=?, title=?, content=?, published_at=?,
                    raw_html=?, extra_json=?, hash=?, updated_at=?
                WHERE id=?
                ''',
                (
                    art.site, art.source, art.title, art.content,
                    art.published_at, art.raw_html, extra_str, h, now,
                    row[0],
                ),
            )
        else:
            conn.execute(
                '''
                INSERT INTO articles (
                    site, source, url, title, content, published_at,
                    raw_html, extra_json, hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    art.site, art.source, art.url, art.title, art.content,
                    art.published_at, art.raw_html, extra_str, h, now, now,
                ),
            )
    else:
        # 以 (hash, site, source, published_at) 作为去重键；允许 published_at 为空
        row = conn.execute(
            '''
            SELECT id FROM articles
            WHERE url IS NULL
              AND hash = ? AND site = ? AND source = ?
              AND COALESCE(published_at, '') = COALESCE(?, '')
            ''',
            (h, art.site, art.source, art.published_at),
        ).fetchone()
        if row:
            conn.execute(
                '''
                UPDATE articles SET
                    title=?, content=?, raw_html=?, extra_json=?,
                    updated_at=?
                WHERE id=?
                ''',
                (
                    art.title, art.content, art.raw_html, extra_str, now, row[0]
                ),
            )
        else:
            conn.execute(
                '''
                INSERT INTO articles (
                    site, source, url, title, content, published_at,
                    raw_html, extra_json, hash, created_at, updated_at
                ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    art.site, art.source, art.title, art.content,
                    art.published_at, art.raw_html, extra_str, h, now, now,
                ),
            )
    conn.commit()


def upsert_many(conn: sqlite3.Connection, rows: Iterable[Article]) -> None:
    """批量 upsert，多条记录时更高效。"""
    for art in rows:
        upsert_article(conn, art)
