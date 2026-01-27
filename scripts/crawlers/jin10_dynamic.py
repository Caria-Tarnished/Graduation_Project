# -*- coding: utf-8 -*-
"""
使用 Playwright 抓取金十数据的动态页面：
- 模式 flash：抓取首页快讯，支持按日期区间回溯。
  通过反复点击/滚动“加载更多”，并基于每条快讯 id 中的数字推导时间。
- 模式 calendar：预留（后续接入按日访问方式或接口）。

注意：
- 需要先安装 Playwright 及浏览器内核：
  pip install playwright
  python -m playwright install chromium
- 建议配合 --db finance.db 直接入库。
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import time
from typing import Any, Dict, List, Optional

from playwright.sync_api import Page, Frame, ElementHandle, sync_playwright

# 可选写库（SQLite）
try:
    from .storage import Article, ensure_schema, get_conn, upsert_many
except Exception:  # noqa: BLE001
    Article = None  # type: ignore
    ensure_schema = None  # type: ignore
    get_conn = None  # type: ignore
    upsert_many = None  # type: ignore

DEFAULT_UA = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/122.0 Safari/537.36'
)


def _frame_query_all(page: Page, selector: str) -> List[ElementHandle]:
    """在所有 frame 中查询 selector，返回元素列表。
    兼容站点将内容放在 iframe 的情况。
    """
    elements: List[ElementHandle] = []
    frames: List[Frame] = [page.main_frame] + [f for f in page.frames]
    for fr in frames:
        try:
            elements.extend(fr.query_selector_all(selector))
        except Exception:
            continue
    return elements


def _safe_get_attr(el: ElementHandle, name: str) -> str:
    """安全读取元素属性，遇到刷新/销毁时返回空串以避免异常中断。"""
    try:
        return (el.get_attribute(name) or '').strip()
    except Exception:
        return ''


def _digits_to_iso(ts_digits: str) -> Optional[str]:
    """将快讯项 id 中的时间戳数字转换为 ISO 格式。
    例如：flash20260121192731970800 → 取前 14 位 20260121192731
    """
    if len(ts_digits) < 14:
        return None
    s = ts_digits[:14]
    try:
        dtm = dt.datetime.strptime(s, '%Y%m%d%H%M%S')
        return dtm.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return None


def _parse_flash_items(page: Page) -> List[Dict[str, str]]:
    """从当前页面 DOM 提取快讯项。
    解析规则：
    - id: flashYYYYMMDDhhmmss......，据此生成 published_at
    - 文本：优先 right-common-title，再追加 flash-text；
      若不存在，回退其它常见类名（更鲁棒）。
    - url: 若能取到详情链接则用之；否则基于 id 构造 detail URL
    """
    items: List[Dict[str, str]] = []
    # 更宽松的选择器：尝试多种容器匹配
    boxes = _frame_query_all(
        page,
        '[id^="flash"], [data-id], .jin-flash-item-container, '
        'li[class*="flash"], div[class*="flash"]',
    )
    for bx in boxes:
        id_attr = (
            _safe_get_attr(bx, 'id') + ' ' +
            _safe_get_attr(bx, 'data-id') + ' ' +
            _safe_get_attr(bx, 'data-news-id')
        ).strip()
        # 提取 14+ 位数字作为时间戳源（含毫秒则截取前 14 位）
        m = re.search(r'(\d{14,})', id_attr)
        ts = None
        if m:
            ts = _digits_to_iso(m.group(1))
        # 进一步：部分节点可能有 data-time 或 datetime 属性
        if not ts:
            for attr in ('data-time', 'datetime', 'data-published'):
                v = _safe_get_attr(bx, attr)
                if re.fullmatch(r'\d{14,}', v):
                    ts = _digits_to_iso(v)
                    break
        # 详情链接
        a_detail = None
        for sel in (
            'a[href*="flash.jin10.com/detail"]',
            'a[href*="/detail/"]',
            'a[href^="/detail/"]',
        ):
            try:
                a_detail = bx.query_selector(sel)
                if a_detail:
                    break
            except Exception:
                continue
        try:
            url = (a_detail.get_attribute('href') if a_detail else '') or ''
        except Exception:
            url = ''
        if (not url) and m:
            # 回退：基于 id 构造
            url = f'https://flash.jin10.com/detail/{m.group(1)}'
        # 标题与正文（多套选择器以兼容 DOM 差异）
        title_el = None
        for sel in (
            '.right-common-title',
            '.jin-flash-item-title',
            '.flash-title',
            'h3',
        ):
            try:
                title_el = bx.query_selector(sel)
                if title_el:
                    break
            except Exception:
                continue
        title = title_el.inner_text().strip() if title_el else ''
        text_el = None
        for sel in (
            '.flash-text',
            '.right-common-content',
            '.right-content',
            '.jin-flash-item-content',
            'article',
            'p',
        ):
            try:
                text_el = bx.query_selector(sel)
                if text_el:
                    break
            except Exception:
                continue
        text = text_el.inner_text().strip() if text_el else ''
        content = text
        if title and text and title not in text:
            content = f'{title} | {text}'
        items.append(
            {
                'url': url,
                'title': title or (text[:40] if text else ''),
                'date_text': '',  # 直接用 id 反推时间
                'content_text': content,
                'published_at': ts or '',
            }
        )
    return items


def _click_load_more(page: Page) -> bool:
    """尝试点击“加载更多”按钮；如无则滚动以触发懒加载。
    返回是否执行了有效的加载动作。
    """
    selectors = [
        'text=点击加载',
        'text=点击加载更多',
        'text=加载更多',
        'a:has-text("加载")',
        'button:has-text("加载")',
    ]
    for sel in selectors:
        # 在所有 frame 尝试点击
        frames: List[Frame] = [page.main_frame] + [f for f in page.frames]
        for fr in frames:
            try:
                el = fr.query_selector(sel)
                if el:
                    el.click(timeout=3000)
                    page.wait_for_load_state('networkidle', timeout=10000)
                    return True
            except Exception:
                continue
    # 回退：滚动
    try:
        # 多次滚动到底部以触发懒加载
        page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
        time.sleep(0.5)
        page.mouse.wheel(0, 2000)
        time.sleep(0.8)
        return True
    except Exception:
        return False


def _install_request_blocker(ctx) -> None:
    try:
        def _route_handler(route):
            try:
                r = route.request
                rtype = (r.resource_type or '').lower()
                if rtype in {'image', 'media', 'font', 'beacon', 'manifest'}:
                    return route.abort()
                return route.continue_()
            except Exception:
                try:
                    return route.continue_()
                except Exception:
                    return route.abort()

        ctx.route('**/*', _route_handler)
    except Exception:
        pass


def _ensure_data_tab(page: Page) -> None:
    try:
        if page.locator(
            'div.jin-table-body, .jin-list .jin-list-item'
        ).count() > 0:
            return
        tab = page.locator('text=经济数据').first
        if tab and tab.count():
            tab.click()
            page.wait_for_timeout(300)
    except Exception:
        pass


def _ensure_important_only(page: Page) -> None:
    try:
        try:
            sel_switch = "div.jin-switch:has-text('只看重要')"
            page.wait_for_selector(sel_switch, timeout=1500)
        except Exception:
            pass
        x_sel = (
            "xpath=//*[contains(text(),'只看重要')]/ancestor::*["
            "contains(@class,'liquid-switch') or "
            "contains(@class,'jin-switch') or "
            "contains(@class,'switch')][1]"
        )
        containers = [
            "div.jin-switch:has-text('只看重要'):visible",
            x_sel,
        ]
        for sel in containers:
            cand = page.locator(sel)
            cnt = cand.count()
            for i in range(cnt):
                container = cand.nth(i)
                try:
                    cb = container.locator(
                        "input.liquid-switch__input, input[type='checkbox']"
                    ).first
                    if cb and cb.count():
                        try:
                            if cb.is_checked():
                                return
                        except Exception:
                            pass
                        try:
                            cb.check(force=True, timeout=1200)
                            page.wait_for_timeout(150)
                        except Exception:
                            try:
                                cb.click()
                                page.wait_for_timeout(150)
                            except Exception:
                                pass
                    if container.locator(
                        "input.liquid-switch__input:checked, "
                        "input[type='checkbox']:checked"
                    ).count() > 0:
                        return
                except Exception:
                    continue
    except Exception:
        pass


def _filter_by_date(
    items: List[Dict[str, str]],
    since: dt.date,
    until: dt.date,
    allow_undated: bool = False,
) -> List[Dict[str, str]]:
    kept: List[Dict[str, str]] = []
    for r in items:
        pub = (r.get('published_at') or '').strip()
        if not pub:
            if allow_undated:
                kept.append(r)
            continue
        try:
            d = dt.datetime.strptime(pub[:10], '%Y-%m-%d').date()
        except Exception:
            continue
        if d < since or d > until:
            continue
        kept.append(r)
    return kept


def crawl_flash(
    start_date: str,
    end_date: str,
    out_csv: str,
    db_path: str,
    source: str,
    delay: float,
    headless: bool = True,
    max_loads: int = 300,
    allow_undated: bool = False,
    debug_dir: str = '',
    storage: str = '',
    user_data_dir: str = '',
    login_wait: float = 0.0,
    keep_open: bool = False,
    verbose: bool = False,
    drop_noise: bool = False,
) -> None:
    since = dt.datetime.strptime(start_date, '%Y-%m-%d').date()
    until = dt.datetime.strptime(end_date, '%Y-%m-%d').date()

    all_items: List[Dict[str, str]] = []
    # 注：此模式当前不统计边爬边入库条数，移除未使用的占位变量避免 F841

    with sync_playwright() as pw:
        # 注：此模式未使用 storage_state，移除未使用的占位变量避免 F841
        # 支持持久化上下文（用户数据目录）或基于 storage state 的无头登录
        use_persist = bool(user_data_dir)
        if use_persist:
            ctx = pw.chromium.launch_persistent_context(
                user_data_dir,
                headless=headless,
                user_agent=DEFAULT_UA,
                locale='zh-CN',
                extra_http_headers={'Accept-Language': 'zh-CN,zh;q=0.9'},
            )
            br = None
        else:
            br = pw.chromium.launch(headless=headless)
            ctx = br.new_context(
                user_agent=DEFAULT_UA,
                locale='zh-CN',
                extra_http_headers={'Accept-Language': 'zh-CN,zh;q=0.9'},
                storage_state=(
                    storage if (storage and os.path.exists(storage)) else None
                ),
            )
        page = ctx.new_page()
        # 直接打开快讯专页，更稳定
        page.goto('https://flash.jin10.com/', timeout=60000)
        page.wait_for_load_state(
            'domcontentloaded', timeout=60000
        )
        page.wait_for_load_state('networkidle', timeout=60000)
        # 等待任一快讯项出现（更稳健），最长 60s
        try:
            page.wait_for_selector(
                '[id^="flash"], [data-id], .jin-flash-item-container',
                timeout=60000,
            )
        except Exception:
            pass
        time.sleep(1.0)
        # 调试：保存首屏截图与 HTML
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            try:
                page.screenshot(path=os.path.join(debug_dir, 'init.png'))
                with open(
                    os.path.join(debug_dir, 'init.html'),
                    'w', encoding='utf-8'
                ) as f:
                    f.write(page.content())
            except Exception:
                pass
        if verbose and debug_dir:
            try:
                _dbg = os.path.abspath(debug_dir)
                print(
                    '[flash] init snapshot saved to',
                    _dbg,
                )
            except Exception:
                pass
        # 首次运行可人工登录并保存状态
        if login_wait and login_wait > 0:
            try:
                page.wait_for_timeout(int(login_wait * 1000))
            except Exception:
                pass
            if storage and not use_persist:
                try:
                    ctx.storage_state(path=storage)
                except Exception:
                    pass

        loads = 0
        min_date: Optional[dt.date] = None
        no_increase_rounds = 0
        last_count = 0
        if verbose:
            print(
                '[flash] start crawling',
                since,
                '→',
                until,
                'max_loads=',
                max_loads,
            )
        while loads < max_loads:
            # 先快速统计当前可见快讯节点数量，便于在解析前给出“心跳”反馈
            if verbose and loads == 0:
                try:
                    approx = len(_frame_query_all(
                        page,
                        (
                            '[id^="flash"], [data-id], '
                            '.jin-flash-item-container, '
                            'li[class*="flash"], '
                            'div[class*="flash"]'
                        ),
                    ))
                    print('[flash] approx dom items', approx)
                except Exception:
                    pass
            try:
                cur = _parse_flash_items(page)
                if verbose:
                    print('[flash] load', loads + 1, 'parsed', len(cur))
            except Exception:
                # 页面被关闭/刷新导致解析异常，终止循环以便输出已获取部分
                break
            if cur:
                all_items.extend(cur)
                # 更新最早日期（用于停止条件）
                try:
                    ds = [
                        dt.datetime.strptime(
                            i['published_at'][:10], '%Y-%m-%d'
                        ).date()
                        for i in cur
                        if i.get('published_at')
                    ]
                    if ds:
                        dmin = min(ds)
                        if (min_date is None) or (dmin < min_date):
                            min_date = dmin
                except Exception:
                    pass
            # 若已经向前加载到比 since 更早，则停止
            if min_date is not None and min_date < since:
                break
            # 判断元素数量是否增加，避免死循环（跨 frame 统计）
            cur_count = len(_frame_query_all(
                page,
                (
                    '[id^="flash"], [data-id], '
                    '.jin-flash-item-container, '
                    'li[class*="flash"], '
                    'div[class*="flash"]'
                ),
            ))
            if cur_count <= last_count:
                no_increase_rounds += 1
            else:
                no_increase_rounds = 0
            last_count = cur_count
            if verbose:
                print(
                    '[flash] dom_count',
                    cur_count,
                    'no_increase',
                    no_increase_rounds,
                )

            loads += 1
            _clicked = _click_load_more(page)
            if verbose:
                print('[flash] click_load_more', bool(_clicked))
            if not _clicked:
                break
            time.sleep(max(0.2, delay))
            if no_increase_rounds >= 3:
                break
            if verbose and (loads % 20 == 0):
                print('[flash] progress loads=', loads,
                      'total_items=', len(all_items),
                      'earliest=', min_date)
            # 调试：保存增量页 HTML 片段
            if debug_dir and (loads % 5 == 0):
                try:
                    with open(
                        os.path.join(debug_dir, f'round_{loads}.html'),
                        'w', encoding='utf-8'
                    ) as f:
                        f.write(page.content())
                except Exception:
                    pass

        if keep_open and (not headless):
            try:
                input('按回车关闭浏览器...')
            except Exception:
                pass
        if use_persist:
            ctx.close()
        else:
            br.close()

    # 去重（按 url 或 (title,content,published_at)）
    seen = set()
    uniq: List[Dict[str, str]] = []
    for r in all_items:
        u = (r.get('url') or '').strip()
        if u:
            if u in seen:
                continue
            seen.add(u)
            uniq.append(r)
        else:
            k = (
                (r.get('title') or '').strip(),
                (r.get('content_text') or '').strip(),
                (r.get('published_at') or '').strip(),
            )
            if k in seen:
                continue
            seen.add(k)
            uniq.append(r)

    # 过滤日期（可允许无时间项用于调试/人工补齐）
    uniq = _filter_by_date(uniq, since, until, allow_undated=allow_undated)
    # 排序：已标注时间的按时间降序，无时间的放在最后
    dated = [r for r in uniq if (r.get('published_at') or '').strip()]
    undated = [r for r in uniq if not (r.get('published_at') or '').strip()]
    dated.sort(key=lambda r: (r.get('published_at') or ''), reverse=True)
    uniq = dated + undated

    # 可选：过滤噪声（如 VIP/扫码分享/直播等非资讯内容）
    def _is_noise_item(r: Dict[str, str]) -> bool:
        t = (r.get('title') or '').strip()
        c = (r.get('content_text') or '').strip()
        bad = ['VIP', '扫码分享', '直播', '重要事件', '金十期货']
        if t in bad:
            return True
        if any(x in c for x in bad):
            return True
        return False

    if drop_noise:
        uniq = [r for r in uniq if not _is_noise_item(r)]

    # 输出 CSV
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        import csv

        with open(out_csv, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'url',
                'title',
                'date_text',
                'content_text',
                'published_at',
            ])
            for r in uniq:
                w.writerow([
                    (r.get('url') or '').strip(),
                    (r.get('title') or '').strip(),
                    (r.get('date_text') or '').strip(),
                    (r.get('content_text') or '').strip(),
                    (r.get('published_at') or '').strip(),
                ])

    # 入库
    if db_path and Article is not None:
        conn = get_conn(db_path)
        ensure_schema(conn)
        rows = []
        for r in uniq:
            rows.append(
                Article(
                    site='www.jin10.com',
                    source=source or 'listing_flash',
                    title=(r.get('title') or '').strip(),
                    content=(r.get('content_text') or '').strip(),
                    published_at=(
                        (r.get('published_at') or '').strip() or None
                    ),
                    url=(r.get('url') or '').strip() or None,
                    raw_html=None,
                    extra_json={
                        'date_text': (r.get('date_text') or '').strip()
                    },
                )
            )
        if rows:
            upsert_many(conn, rows)
        conn.close()

    print('flash done: items=', len(uniq))


def watch_flash(
    db_path: str,
    source: str,
    delay: float,
    headless: bool = True,
    debug_dir: str = '',
    storage: str = '',
    user_data_dir: str = '',
    login_wait: float = 0.0,
    watch_interval: float = 30.0,
) -> None:
    """实时轮询快讯页，发现新快讯则入库（基于页面解析）。
    - 启动时进行基线解析并建立内存去重集合（不立即入库）。
    - 每次循环可选择刷新页面，随后解析并 upsert 新增项。
    """
    seen = set()

    with sync_playwright() as pw:
        use_persist = bool(user_data_dir)
        if use_persist:
            ctx = pw.chromium.launch_persistent_context(
                user_data_dir,
                headless=headless,
                user_agent=DEFAULT_UA,
                locale='zh-CN',
                extra_http_headers={'Accept-Language': 'zh-CN,zh;q=0.9'},
            )
            br = None
        else:
            br = pw.chromium.launch(headless=headless)
            ctx = br.new_context(
                user_agent=DEFAULT_UA,
                locale='zh-CN',
                extra_http_headers={'Accept-Language': 'zh-CN,zh;q=0.9'},
                storage_state=(
                    storage if (storage and os.path.exists(storage)) else None
                ),
            )
        page = ctx.new_page()
        page.goto('https://flash.jin10.com/', timeout=60000)
        page.wait_for_load_state('domcontentloaded', timeout=60000)
        try:
            page.wait_for_selector(
                '[id^="flash"], [data-id], .jin-flash-item-container',
                timeout=60000,
            )
        except Exception:
            pass
        time.sleep(1.0)

        # 调试首屏
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            try:
                page.screenshot(
                    path=os.path.join(debug_dir, 'watch_init.png')
                )
                with open(
                    os.path.join(debug_dir, 'watch_init.html'),
                    'w', encoding='utf-8'
                ) as f:
                    f.write(page.content())
            except Exception:
                pass

        # 基线解析，仅建立 seen
        try:
            base_items = _parse_flash_items(page)
        except Exception:
            base_items = []
        for r in base_items:
            u = (r.get('url') or '').strip()
            if u:
                seen.add(u)
            else:
                k = (
                    (r.get('title') or '').strip(),
                    (r.get('content_text') or '').strip(),
                    (r.get('published_at') or '').strip(),
                )
                seen.add(k)
        print('watch bootstrap items:', len(seen))

        # DB 连接（可选）
        conn = None
        if db_path and Article is not None:
            conn = get_conn(db_path)
            ensure_schema(conn)

        try:
            rounds = 0
            while True:
                rounds += 1
                try:
                    # 简单刷新以促使新内容加载
                    try:
                        page.reload(timeout=60000)
                        page.wait_for_load_state(
                            'domcontentloaded', timeout=60000
                        )
                    except Exception:
                        pass
                    cur = _parse_flash_items(page)
                except Exception:
                    cur = []
                new_items = []
                for r in cur:
                    u = (r.get('url') or '').strip()
                    if u and (u not in seen):
                        seen.add(u)
                        new_items.append(r)
                    elif not u:
                        k = (
                            (r.get('title') or '').strip(),
                            (r.get('content_text') or '').strip(),
                            (r.get('published_at') or '').strip(),
                        )
                        if k not in seen:
                            seen.add(k)
                            new_items.append(r)

                if new_items:
                    print('watch new items:', len(new_items))
                    if conn is not None and Article is not None:
                        rows = []
                        for r in new_items:
                            pub = (
                                (r.get('published_at') or '').strip() or None
                            )
                            url_val = (
                                (r.get('url') or '').strip() or None
                            )
                            date_txt = (r.get('date_text') or '').strip()
                            rows.append(
                                Article(
                                    site='www.jin10.com',
                                    source=source or 'listing_flash',
                                    title=(r.get('title') or '').strip(),
                                    content=(
                                        (r.get('content_text') or '').strip()
                                    ),
                                    published_at=pub,
                                    url=url_val,
                                    raw_html=None,
                                    extra_json={
                                        'date_text': date_txt
                                    },
                                )
                            )
                        if rows:
                            upsert_many(conn, rows)
                else:
                    print('watch no new items')

                try:
                    time.sleep(max(1.0, watch_interval))
                except KeyboardInterrupt:
                    break
                except Exception:
                    pass
        finally:
            if conn is not None:
                conn.close()
            if use_persist:
                ctx.close()
            else:
                br.close()

    print('watch stopped')


def _parse_calendar_items(page: Page, assume_date: str) -> List[Dict[str, str]]:
    """解析日历页当日的经济数据/事件列表。
    由于站点结构频繁变动，采用多套选择器并做保守回退：
    - 时间：'.time', 'time', 'td.time', 'span.time'
    - 文本：'.data-name a', '.event-title', 'a', '.name', '.title'
    - 容器：'li[data-id]', 'li', 'tr', 'div[class*="item"]'
    返回 title/content_text/published_at（由 assume_date + 时间 构造）。
    """
    out: List[Dict[str, str]] = []
    try:
        table_rows = _frame_query_all(
            page,
            'div.jin-table-body div.jin-table-row',
        )
    except Exception:
        table_rows = []
    if table_rows:
        for r in table_rows:
            cols = []
            try:
                cols = r.query_selector_all('div.jin-table-column')
            except Exception:
                cols = []
            if not cols or len(cols) < 2:
                continue
            try:
                ttxt = (cols[0].inner_text() or '').strip()
            except Exception:
                ttxt = ''
            m = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?)', ttxt)
            if not m:
                continue
            name_el = None
            try:
                name_el = (
                    cols[1].query_selector('.data-name-text')
                    or cols[1].query_selector('a')
                    or cols[1]
                )
            except Exception:
                name_el = None
            try:
                title = (
                    (name_el.inner_text() or '').strip()
                    if name_el else ''
                )
            except Exception:
                title = ''
            if not title:
                continue
            hhmmss = m.group(1)
            if hhmmss.count(':') == 1:
                hhmmss = hhmmss + ':00'
            pub = f'{assume_date} {hhmmss}'
            out.append(
                {
                    'url': '',
                    'title': title,
                    'date_text': ttxt,
                    'content_text': title,
                    'published_at': pub,
                }
            )
        if out:
            return out
    try:
        list_items = _frame_query_all(
            page,
            'div.jin-list .jin-list-item',
        )
    except Exception:
        list_items = []
    if list_items:
        for it in list_items:
            t_el = None
            try:
                t_el = it.query_selector(
                    '.jin-list-item__header-left .time'
                )
            except Exception:
                t_el = None
            try:
                ttxt = (t_el.inner_text() or '').strip() if t_el else ''
            except Exception:
                ttxt = ''
            m = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?)', ttxt)
            if not m:
                continue
            name_el = None
            for sel in (
                '.data-name a',
                '.data-name',
                '.event-title',
                'a',
            ):
                try:
                    name_el = it.query_selector(sel)
                    if name_el:
                        break
                except Exception:
                    continue
            try:
                title = (
                    (name_el.inner_text() or '').strip()
                    if name_el else ''
                )
            except Exception:
                title = ''
            if not title:
                continue
            hhmmss = m.group(1)
            if hhmmss.count(':') == 1:
                hhmmss = hhmmss + ':00'
            pub = f'{assume_date} {hhmmss}'
            out.append(
                {
                    'url': '',
                    'title': title,
                    'date_text': ttxt,
                    'content_text': title,
                    'published_at': pub,
                }
            )
        if out:
            return out
    boxes = _frame_query_all(
        page,
        'li[data-id], li, tr, div[class*="item"]',
    )
    for bx in boxes:
        # 时间
        t_el = None
        for sel in ('.time', 'time', 'td.time', 'span.time', 'td'):
            try:
                t_el = bx.query_selector(sel)
                if t_el:
                    break
            except Exception:
                continue
        ttxt = t_el.inner_text().strip() if t_el else ''
        # 要求包含明确的时间（避免侧边栏等噪声）
        m = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?)', ttxt)
        if not m:
            continue
        # 标题/名称
        a_el = None
        for sel in (
            '.data-name a', '.event-title', 'a', '.name', '.title', 'td a'
        ):
            try:
                a_el = bx.query_selector(sel)
                if a_el:
                    break
            except Exception:
                continue
        title = a_el.inner_text().strip() if a_el else ''
        if not title:
            # 回退：取整行文本
            try:
                title = (bx.inner_text() or '').strip()
            except Exception:
                title = ''
        if not title:
            continue
        # published_at：使用匹配到的 HH:MM(:SS)
        hhmmss = m.group(1)
        if hhmmss.count(':') == 1:
            hhmmss = hhmmss + ':00'
        pub = f'{assume_date} {hhmmss}'
        out.append(
            {
                'url': '',
                'title': title,
                'date_text': ttxt,
                'content_text': title,
                'published_at': pub,
            }
        )
    return out


def _calendar_extract_rows_fast(page: Page, target_date: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        data = page.evaluate(
            """
            () => {
              const res = [];
              const text = (el) => (
                (el && el.textContent ? el.textContent.trim() : null) || null
              );
              const shown = (el) => {
                if (!el) return false;
                const cs = window.getComputedStyle(el);
                if (
                  cs &&
                  (cs.display === 'none' || cs.visibility === 'hidden')
                ) return false;
                const rect = el.getBoundingClientRect();
                if (
                  rect && (rect.width <= 0 || rect.height <= 0)
                ) return false;
                // offsetParent 在部分布局下可能为 null，这里放宽为尺寸检测
                return true;
              };
              const getStar = (container) => {
                if (!container) return null;
                const lit = container
                  .querySelectorAll(".jin-star i[style*='var(--rise)']")
                  .length;
                if (lit > 0) return lit;
                const total = container
                  .querySelectorAll(".jin-star i")
                  .length;
                const gray = container
.querySelectorAll(
                    ".jin-star i[style*='on-rise-" +
                    "light-" +
                    "lowest']"
                  )
.length;
                if (total && gray && gray > 0) {
                  return Math.max(0, total - gray);
                }
                return null;
              };
              const tableRows = Array.from(
                document.querySelectorAll(
                  'div.jin-table-body ' + 'div.jin-table-row'
                )
              );
              if (tableRows.length > 0) {
                for (const r of tableRows) {
                  if (!shown(r)) continue;
                  const cols = r.querySelectorAll('div.jin-table-column');
                  if (!cols || cols.length < 2) continue;
                  const time_text = text(cols[0]);
                  const name_cell = cols[1];
                  const name_text = (
                    text(name_cell.querySelector('.data-name-text')) ||
                    text(name_cell)
                  );
                  const star = getStar(cols[2] || null);
                  res.push({time: time_text, name: name_text, star});
                }
                return res;
              }
              const items = Array.from(
                document.querySelectorAll(
                  'div.jin-list ' + '.jin-list-item'
                )
              );
              for (const item of items) {
                if (!shown(item)) continue;
                const dataSlot = item.querySelector(
                  '.jin-list-item__slot .data'
                );
                if (!dataSlot) continue;
                const time_text = text(
                  item.querySelector('.jin-list-item__header-left .time')
                );
                const a = dataSlot.querySelector('.data-name a');
                let name_text = null;
                if (a) {
                  name_text = text(a);
                } else {
                  name_text = text(dataSlot.querySelector('.data-name'));
                }
                const star = getStar(
                  item.querySelector('.jin-list-item__header-right')
                );
                res.push({time: time_text, name: name_text, star});
              }
              return res;
            }
            """
        )
    except Exception:
        data = []
    for it in (data or []):
        rows.append({
            'time': (it or {}).get('time'),
            'name': (it or {}).get('name'),
            'star': (it or {}).get('star'),
        })
    return rows


def crawl_calendar(
    start_date: str,
    end_date: str,
    out_csv: str,
    db_path: str,
    source: str,
    delay: float,
    headless: bool = True,
    debug_dir: str = '',
    important_only: bool = False,
    user_data_dir: str = '',
    setup_seconds: int = 0,
) -> None:
    since = dt.datetime.strptime(start_date, '%Y-%m-%d').date()
    until = dt.datetime.strptime(end_date, '%Y-%m-%d').date()
    # 允许用户传入反向日期区间，自动交换保证自早至晚
    if since > until:
        since, until = until, since

    all_items: List[Dict[str, str]] = []
    total_streamed = 0  # 边爬边入库的累计条数（仅 calendar 用）

    with sync_playwright() as pw:
        storage_state_path = ''
        # 可选：运行前开启带界面浏览器供人工确认（例如登录态/筛选），随后自动关闭
        if setup_seconds and setup_seconds > 0:
            tmp_br = None
            tmp_ctx = None
            try:
                if user_data_dir:
                    tmp_ctx = pw.chromium.launch_persistent_context(
                        user_data_dir=user_data_dir,
                        headless=False,
                        args=[
                            "--disable-blink-features=AutomationControlled"
                        ],
                        user_agent=DEFAULT_UA,
                        locale='zh-CN',
                        extra_http_headers={
                            'Accept-Language': 'zh-CN,zh;q=0.9'
                        },
                    )
                else:
                    tmp_br = pw.chromium.launch(
                        headless=False,
                        args=[
                            "--disable-blink-features=AutomationControlled"
                        ],
                    )
                    tmp_ctx = tmp_br.new_context(
                        user_agent=DEFAULT_UA,
                        locale='zh-CN',
                        extra_http_headers={
                            'Accept-Language': 'zh-CN,zh;q=0.9'
                        },
                    )
                tmp_page = tmp_ctx.new_page()
                ds = since.strftime('%Y-%m-%d')
                first_url = f"https://rili.jin10.com/day/{ds}"
                try:
                    tmp_page.goto(first_url, timeout=60000)
                    tmp_page.wait_for_load_state(
                        'domcontentloaded', timeout=60000
                    )
                except Exception:
                    pass
                try:
                    print(f"[DEBUG] 预览阶段，等待 {setup_seconds} 秒后开始正式抓取…")
                except Exception:
                    pass
                try:
                    tmp_page.wait_for_timeout(setup_seconds * 1000)
                except Exception:
                    pass
                # 将预览阶段的登录态导出为 storage_state，供正式阶段复用
                try:
                    base_dir = (debug_dir or '.')
                    os.makedirs(base_dir, exist_ok=True)
                    storage_state_path = os.path.join(
                        base_dir, 'jin10_calendar_state.json'
                    )
                    tmp_ctx.storage_state(path=storage_state_path)
                    try:
                        print('[DEBUG] 已导出登录态 storage_state')
                    except Exception:
                        pass
                except Exception:
                    storage_state_path = ''
            finally:
                try:
                    tmp_ctx.close()
                except Exception:
                    pass
                if tmp_br:
                    try:
                        tmp_br.close()
                    except Exception:
                        pass

        # 正式抓取：根据是否使用预览阶段决定是否有界面
        try:
            print('[DEBUG] 预览阶段结束，启动正式抓取…')
        except Exception:
            pass
        try:
            time.sleep(0.8)
        except Exception:
            pass
        run_headless = (
            True if (setup_seconds and setup_seconds > 0) else headless
        )
        br = None
        prefer_state = (
            bool(user_data_dir)
            and bool(setup_seconds and setup_seconds > 0)
            and bool(
                storage_state_path and os.path.exists(storage_state_path)
            )
        )
        if user_data_dir and not prefer_state:
            try:
                ctx = pw.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir,
                    headless=run_headless,
                    args=["--disable-blink-features=AutomationControlled"],
                    user_agent=DEFAULT_UA,
                    locale='zh-CN',
                    extra_http_headers={'Accept-Language': 'zh-CN,zh;q=0.9'},
                )
            except Exception:
                try:
                    print('[WARN] 持久化上下文启动失败，改用非持久化重试')
                except Exception:
                    pass
                br = pw.chromium.launch(
                    headless=run_headless,
                    args=["--disable-blink-features=AutomationControlled"],
                )
                ctx = br.new_context(
                    user_agent=DEFAULT_UA,
                    locale='zh-CN',
                    extra_http_headers={'Accept-Language': 'zh-CN,zh;q=0.9'},
                )
        else:
            br = pw.chromium.launch(
                headless=run_headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            if prefer_state:
                ctx = br.new_context(
                    user_agent=DEFAULT_UA,
                    locale='zh-CN',
                    extra_http_headers={'Accept-Language': 'zh-CN,zh;q=0.9'},
                    storage_state=storage_state_path,
                )
            else:
                ctx = br.new_context(
                    user_agent=DEFAULT_UA,
                    locale='zh-CN',
                    extra_http_headers={'Accept-Language': 'zh-CN,zh;q=0.9'},
                )

        try:
            print(
                '[DEBUG] 正式阶段: headless=', run_headless,
                ' persist=', bool(user_data_dir)
            )
        except Exception:
            pass
        _install_request_blocker(ctx)
        page = ctx.new_page()

        cur = until
        try:
            print('[DEBUG] 抓取区间: ', since, ' -> ', until)
        except Exception:
            pass
        # 可选：打开数据库连接用于边爬边入库
        conn = None
        if db_path and Article is not None:
            try:
                conn = get_conn(db_path)
                ensure_schema(conn)
            except Exception:
                conn = None
        while cur >= since:
            date_s = cur.strftime('%Y-%m-%d')
            # 直达每日页面（更稳定）
            url = f'https://rili.jin10.com/day/{date_s}'
            try:
                print('[DEBUG] 加载日期页: ', date_s)
            except Exception:
                pass
            page.goto(url, timeout=60000)
            page.wait_for_load_state('domcontentloaded', timeout=60000)
            try:
                page.wait_for_load_state('networkidle', timeout=15000)
            except Exception:
                pass
            try:
                tab = page.locator('text=经济数据').first
                if tab and tab.count():
                    tab.click()
                    page.wait_for_timeout(200)
            except Exception:
                pass
            try:
                page.wait_for_selector(
                    'div.jin-table-body div.jin-table-row, '
                    '.jin-list .jin-list-item',
                    timeout=10000,
                )
            except Exception:
                pass
            try:
                page.wait_for_timeout(800)
            except Exception:
                pass
            # 可选：尝试开启“只看重要”开关（若存在）
            if important_only:
                try:
                    _ensure_important_only(page)
                except Exception:
                    pass
            # 调试：保存每日页面
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                try:
                    with open(
                        os.path.join(debug_dir, f'calendar_{date_s}.html'),
                        'w', encoding='utf-8'
                    ) as f:
                        f.write(page.content())
                except Exception:
                    pass
            fast_rows = _calendar_extract_rows_fast(page, target_date=date_s)
            if important_only:
                try:
                    fast_rows = [
                        r for r in (fast_rows or [])
                        if (
                            isinstance(r.get('star'), int) and
                            r.get('star') >= 3
                        )
                    ]
                except Exception:
                    pass
            cur_items = []
            if fast_rows:
                for r in fast_rows:
                    ttxt = (r.get('time') or '')
                    if not ttxt:
                        continue
                    m = re.search(
                        r'(\d{1,2}:[0-9]{2}(?::[0-9]{2})?)',
                        ttxt or ''
                    )
                    if not m:
                        continue
                    hhmmss = m.group(1)
                    if hhmmss.count(':') == 1:
                        hhmmss = hhmmss + ':00'
                    pub = f'{date_s} {hhmmss}'
                    title = (r.get('name') or '').strip()
                    if not title:
                        continue
                    cur_items.append({
                        'url': '',
                        'title': title,
                        'date_text': ttxt,
                        'content_text': title,
                        'published_at': pub,
                    })
            else:
                cur_items = _parse_calendar_items(page, assume_date=date_s)
            if cur_items:
                all_items.extend(cur_items)
                # 边爬边入库（若提供了 db_path）
                if conn is not None:
                    try:
                        rows = []
                        for r in cur_items:
                            rows.append(
                                Article(
                                    site='rili.jin10.com',
                                    source=source or 'listing_data',
                                    title=(r.get('title') or '').strip(),
                                    content=(
                                        (r.get('content_text') or '').strip()
                                    ),
                                    published_at=(
                                        (
                                            r.get('published_at') or ''
                                        ).strip()
                                        or None
                                    ),
                                    url=(
                                        (r.get('url') or '').strip() or None
                                    ),
                                    raw_html=None,
                                    extra_json={
                                        'date_text': (
                                            (r.get('date_text') or '').strip()
                                        )
                                    },
                                )
                            )
                        if rows:
                            upsert_many(conn, rows)
                            total_streamed += len(rows)
                    except Exception:
                        pass
                try:
                    print(
                        f"{date_s} 提取 {len(cur_items)} 条（累计入库 "
                        f"{total_streamed}）"
                    )
                except Exception:
                    pass
            else:
                try:
                    print(f"{date_s} 提取 0 条")
                except Exception:
                    pass
            cur -= dt.timedelta(days=1)
            time.sleep(max(0.1, delay))

        # 关闭上下文
        if br:
            br.close()
        else:
            ctx.close()

    # 去重与输出
    seen = set()
    uniq: List[Dict[str, str]] = []
    for r in all_items:
        k = (
            (r.get('title') or '').strip(),
            (r.get('content_text') or '').strip(),
            (r.get('published_at') or '').strip(),
        )
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)

    # 输出 CSV
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        import csv

        with open(out_csv, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'url', 'title', 'date_text', 'content_text', 'published_at'
            ])
            for r in uniq:
                w.writerow([
                    (r.get('url') or '').strip(),
                    (r.get('title') or '').strip(),
                    (r.get('date_text') or '').strip(),
                    (r.get('content_text') or '').strip(),
                    (r.get('published_at') or '').strip(),
                ])

    # 入库（若前面已边爬边保存，这里可以跳过；但保留兜底逻辑，避免关闭连接后无数据写入）
    if db_path and Article is not None:
        try:
            conn = get_conn(db_path)
            ensure_schema(conn)
            rows = []
            for r in uniq:
                rows.append(
                    Article(
                        site='rili.jin10.com',
                        source=source or 'listing_data',
                        title=(r.get('title') or '').strip(),
                        content=(r.get('content_text') or '').strip(),
                        published_at=(
                            (r.get('published_at') or '').strip() or None
                        ),
                        url=(r.get('url') or '').strip() or None,
                        raw_html=None,
                        extra_json={
                            'date_text': (r.get('date_text') or '').strip()
                        },
                    )
                )
            if rows:
                upsert_many(conn, rows)
            conn.close()
        except Exception:
            pass

    print('calendar done: items=', len(uniq))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--mode',
        required=False,
        default='calendar',
        choices=['calendar'],
    )
    ap.add_argument('--start-date', required=True)
    ap.add_argument('--end-date', required=True)
    ap.add_argument('--out-csv', default='')
    ap.add_argument('--db', default='')
    ap.add_argument('--source', default='listing_data')
    ap.add_argument('--delay', type=float, default=1.0)
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--debug-dir', default='')
    ap.add_argument('--user-data-dir', default='')
    ap.add_argument('--setup-seconds', type=int, default=0)
    ap.add_argument('--important-only', action='store_true')
    args = ap.parse_args()

    crawl_calendar(
        start_date=args.start_date,
        end_date=args.end_date,
        out_csv=args.out_csv,
        db_path=args.db,
        source=args.source,
        delay=args.delay,
        headless=args.headless,
        debug_dir=args.debug_dir,
        important_only=args.important_only,
        user_data_dir=args.user_data_dir,
        setup_seconds=args.setup_seconds,
    )


if __name__ == '__main__':
    main()
