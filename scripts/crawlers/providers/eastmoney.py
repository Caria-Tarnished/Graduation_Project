# -*- coding: utf-8 -*-
"""
东方财富 列表页解析 Provider。
输入：列表页 HTML（可能是片段）。
输出：字典列表，每项包含 url/title/date_text（中文时间格式）。
"""
from __future__ import annotations

from typing import Dict, List
from bs4 import BeautifulSoup


def parse_listing(html: str) -> List[Dict[str, str]]:
    """从东方财富列表页 HTML 中提取文章链接与标题。
    - 选择 id 形如 newsTr* 的 <li>，在其中查找 <p class="title"> 下的链接。
    - 日期文本来自 <p class="time">。
    """
    soup = BeautifulSoup(html, 'html.parser')
    items: List[Dict[str, str]] = []

    for li in soup.select('li[id^="newsTr"]'):
        a = li.select_one('p.title a[href]')
        if not a:
            continue
        href = a.get('href') or ''
        if not href:
            continue
        title = a.get_text(strip=True) or a.get('title') or ''
        t = li.select_one('p.time')
        date_text = t.get_text(strip=True) if t else ''
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
