# -*- coding: utf-8 -*-
"""
Yahoo Finance 列表页解析 Provider。
输入：列表页 HTML（可能是片段）。
输出：字典列表，每项包含 url/title/date_text（可能为相对时间）。
"""
from __future__ import annotations

from typing import Dict, List
from bs4 import BeautifulSoup


def parse_listing(html: str) -> List[Dict[str, str]]:
    """从 Yahoo Finance 列表页 HTML 中提取文章链接与标题。
    - 尽量选择包含 'titles' 类名的标题链接；找不到则回退任意含 href 的 <a>。
    - 日期文本从 '.publishing' 容器提取，可能为相对时间
      （例如 39\u5206\u949f\u524d）。
    """
    soup = BeautifulSoup(html, 'html.parser')
    items: List[Dict[str, str]] = []

    for li in soup.select('li.stream-item.story-item'):
        # 优先标题链接（包含 class 'titles'）
        a = li.select_one('a.titles') or li.select_one('a[href]')
        if not a:
            continue
        href = a.get('href') or ''
        if not href:
            continue
        # Yahoo 列表通常给出绝对链接
        title = a.get('title') or a.get_text(strip=True) or ''
        pub = li.select_one('div.publishing')
        date_text = pub.get_text(strip=True) if pub else ''
        items.append(
            {
                'url': href,
                'title': title,
                'date_text': date_text,
            }
        )

    # 去重：按 url 去重，保持顺序
    seen = set()
    uniq: List[Dict[str, str]] = []
    for r in items:
        if r['url'] in seen:
            continue
        seen.add(r['url'])
        uniq.append(r)
    return uniq
