# -*- coding: utf-8 -*-
"""
通用 URL 抽取器：从给定的 URL 列表抓取新闻标题/正文与时间，导出为 CSV。
用法示例：
    python scripts/crawlers/fetch_from_urls.py \
        --urls data/raw/urls.txt \
        --out data/raw/news_sample.csv \
        --hint NVDA \
        --delay 1.0
输出列：Date, Title, Content, ticker_hint
注意：
- 本脚本不做站点适配，仅做通用解析；不同站点的标签结构可能不同，解析成功率取决于页面结构。
- 如果需要更高质量的适配，可在此基础上为特定站点编写 Provider。
"""
import argparse
import os
import time
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


def _read_urls(path: str) -> List[str]:
    # 读取 URL 列表（每行一个 URL），忽略空行与注释
    with open(path, 'r', encoding='utf-8') as f:
        lines = [x.strip() for x in f.readlines()]
    return [x for x in lines if x and not x.startswith('#')]


def _fetch_html(url: str, timeout: int = 15) -> Optional[str]:
    # 简单的请求封装，设置 UA 与语言头；失败返回 None
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/122.0 Safari/537.36'
        ),
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            return None
        # 有些站点会返回非 UTF-8，这里交由 requests 自动检测
        resp.encoding = resp.apparent_encoding or resp.encoding
        return resp.text
    except Exception:
        return None


def _extract_title(soup: BeautifulSoup) -> str:
    # 优先元标签，其次 <title>
    for key, attr in [
        ('og:title', 'property'),
        ('twitter:title', 'name'),
    ]:
        tag = soup.find('meta', attrs={attr: key})
        if tag and tag.get('content'):
            return tag['content'].strip()
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return ''


def _extract_date(soup: BeautifulSoup) -> Optional[pd.Timestamp]:
    # 常见发布时间字段
    meta_keys = [
        ('article:published_time', 'property'),
        ('og:updated_time', 'property'),
        ('datePublished', 'itemprop'),
        ('pubdate', 'name'),
    ]
    for key, attr in meta_keys:
        tag = soup.find('meta', attrs={attr: key})
        if tag and tag.get('content'):
            try:
                dt = pd.to_datetime(tag['content'], errors='coerce')
                if pd.notna(dt):
                    return dt.normalize()
            except Exception:
                pass
    # time 标签
    t = soup.find('time')
    if t is not None:
        dt = pd.to_datetime(
            t.get('datetime') or t.get_text(strip=True),
            errors='coerce',
        )
        if pd.notna(dt):
            return dt.normalize()
    return None


def _extract_content(soup: BeautifulSoup) -> str:
    # 先尝试 <article>，否则回退聚合所有 <p>
    art = soup.find('article')
    if art is not None:
        txt = art.get_text(separator='\n', strip=True)
        if txt:
            return txt
    ps = soup.find_all('p')
    parts = [p.get_text(' ', strip=True) for p in ps]
    return '\n'.join([x for x in parts if x])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--urls',
        required=True,
        help='包含 URL 列表的文本文件（每行一个 URL）',
    )
    ap.add_argument(
        '--out',
        default='data/raw/news_sample.csv',
        help='输出 CSV 路径',
    )
    ap.add_argument(
        '--hint',
        default='',
        help='统一的 ticker_hint（可选，例如 NVDA 或 600036.SH）',
    )
    ap.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='请求间隔（秒）',
    )
    ap.add_argument(
        '--timeout',
        type=int,
        default=15,
        help='请求超时（秒）',
    )
    args = ap.parse_args()

    urls = _read_urls(args.urls)
    rows = []
    for i, u in enumerate(urls, 1):
        html = _fetch_html(u, timeout=args.timeout)
        if html is None:
            continue
        soup = BeautifulSoup(html, 'html.parser')
        title = _extract_title(soup)
        content = _extract_content(soup)
        dt = _extract_date(soup) or pd.Timestamp.today()
        rows.append({
            'Date': dt.strftime('%Y-%m-%d'),
            'Title': title,
            'Content': content,
            'ticker_hint': args.hint,
        })
        time.sleep(max(0.0, args.delay))

    if not rows:
        print(
            'no rows parsed; please check your URLs file or site structures.'
        )
        return

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False, encoding='utf-8')
    print(f'saved: {args.out} (rows={len(rows)})')


if __name__ == '__main__':
    main()
