# -*- coding: utf-8 -*-
"""
从列表页 HTML 提取文章 URL 列表的实用脚本。

示例（推荐以模块方式运行，确保包内相对导入可用）：
    # 从 html.txt（包含 Yahoo Finance 列表片段）提取 URL 到 urls.txt
    python -m scripts.crawlers.parse_listing \
        --provider yahoo \
        --in-file html.txt \
        --out-urls data/raw/urls.txt \
        --out-csv data/raw/listing_items.csv

支持 provider：
- yahoo: Yahoo Finance 列表页
- eastmoney: 东方财富列表页
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

from .providers import yahoo_finance as prov_yahoo
from .providers import eastmoney as prov_eastmoney


PROVIDERS: Dict[str, object] = {
    'yahoo': prov_yahoo,
    'eastmoney': prov_eastmoney,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--provider',
        required=True,
        choices=PROVIDERS.keys(),
    )
    ap.add_argument(
        '--in-file',
        required=True,
        help='包含列表页 HTML 的文件路径',
    )
    ap.add_argument(
        '--out-urls',
        required=True,
        help='导出的 URL 列表路径（txt）',
    )
    ap.add_argument(
        '--out-csv',
        default='',
        help='可选：导出带标题/时间的 CSV',
    )
    args = ap.parse_args()

    # 兼容本地 HTML 片段的多种编码（例如 ANSI/GBK/UTF-8-SIG）
    # 读取为二进制后按常见编码尝试解码，最后回退为忽略错误的 UTF-8。
    def _read_text_any(path: str) -> str:
        with open(path, 'rb') as bf:
            data = bf.read()
        for enc in (
            'utf-8',
            'utf-8-sig',
            'gbk',
            'cp936',
            'gb18030',
            'latin-1',
        ):
            try:
                return data.decode(enc)
            except Exception:
                pass
        return data.decode('utf-8', errors='ignore')

    html = _read_text_any(args.in_file)

    mod = PROVIDERS[args.provider]
    items: List[Dict[str, str]] = mod.parse_listing(html)

    # 写出 URL 列表
    os.makedirs(os.path.dirname(args.out_urls) or '.', exist_ok=True)
    with open(args.out_urls, 'w', encoding='utf-8') as f:
        for r in items:
            f.write((r.get('url') or '').strip() + '\n')

    # 可选写出 CSV（包含标题与原始日期文本）
    if args.out_csv:
        import csv  # 延迟导入

        os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
        with open(args.out_csv, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['url', 'title', 'date_text'])
            for r in items:
                w.writerow([
                    (r.get('url') or '').strip(),
                    (r.get('title') or '').strip(),
                    (r.get('date_text') or '').strip(),
                ])

    print(f'extracted: {len(items)} items -> {args.out_urls}')
    if args.out_csv:
        print(f'also wrote metadata csv: {args.out_csv}')


if __name__ == '__main__':
    main()
