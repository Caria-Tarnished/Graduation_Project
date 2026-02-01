# -*- coding: utf-8 -*-
"""
金十数据 日历页 爬虫（Playwright）

思路：
- 直接按日期访问 https://rili.jin10.com/day/YYYY-MM-DD 页面，避免复杂滑块交互不确定性。
- 对于最近 N 个月的每一天，逐日抓取当日经济数据表格。
- 解析字段：date(日期)、time(时间)、country(国家，尝试从国旗图片URL推断)、name(指标名)、star(重要性)、previous(前值)、consensus(预测值)、actual(公布值)。
- 输出 CSV。

使用：
python jin10_calendar.py --months 3 --output calendar_last_3m.csv
python jin10_calendar.py --start 2025-10-01 --end 2026-01-15 --output calendar_q4.csv

首次运行需安装浏览器：
python -m playwright install
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional

from playwright.sync_api import sync_playwright, Page
import pandas as pd
import csv
from urllib.parse import urljoin
import os
import re

# 统一导出编码
CSV_ENCODING = "utf-8"

# 可选写库（SQLite）：若 storage 不存在，则优雅降级为仅 CSV 输出
try:
    from .storage import Article, ensure_schema, get_conn, upsert_many  # type: ignore
except Exception:  # noqa: BLE001
    Article = None  # type: ignore
    ensure_schema = None  # type: ignore
    get_conn = None  # type: ignore
    upsert_many = None  # type: ignore

# 统一的浏览器 UA 字符串（避免在调用处出现超长行）
DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def _text_or_none(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s or s == "--":
        return None
    return s


def _to_float_or_str(v: Optional[str]) -> Optional[str]:
    """尝试解析数值，不做强制转换；若无法确定，返回原文本（保留百分号等单位）。"""
    if v is None:
        return None
    return v.strip()


def _parse_country_from_flag_src(src: str) -> Optional[str]:
    """从国旗图片 URL 中尽力解析国家名（中文文件名例如 .../flag/美国.png/flags）。失败返回 None。"""
    if not src:
        return None
    # 去掉查询参数
    s = src.split("?")[0]
    parts = s.split("/")
    for p in reversed(parts):
        if ".png" in p or ".jpg" in p:
            name = p.split(".")[0]
            return name or None
    return None


def _count_stars_in_row(row) -> Optional[int]:
    try:
        cols = row.locator("div.jin-table-column")
        if cols.count() < 3:
            return None
        star_cell = cols.nth(2)
        # 优先统计点亮的星（颜色为 var(--rise)）
        lit = 0
        try:
            lit = star_cell.locator(".jin-star i[style*='var(--rise)']").count()
        except Exception:
            lit = 0
        if lit and lit > 0:
            return lit
        # 兜底：仅当能检测到灰星时，才用总星减去灰星；否则不做武断推断
        try:
            total = star_cell.locator(".jin-star i").count()
            gray = star_cell.locator(".jin-star i[style*='on-rise-light-lowest']").count()
            if total and gray and gray > 0:
                return max(0, total - gray)
        except Exception:
            pass
        return None
    except Exception:
        return None


def _ensure_data_tab(page: Page, debug: bool = False) -> None:
    """确保处于“经济数据”页签，否则切换到该页签。"""
    try:
        # 若数据表/列表已可见，视为在经济数据页签
        if page.locator("div.jin-table-body, .jin-list .jin-list-item").count() > 0:
            return
        # 点击“经济数据”文字所在的页签
        tab = page.locator("text=经济数据").first
        if tab and tab.count():
            tab.click()
            page.wait_for_timeout(300)
    except Exception:
        pass


def _ensure_important_only(page: Page, debug: bool = False) -> None:
    """确保页面上的“只看重要”已开启。"""
    try:
        try:
            page.wait_for_selector(
                "div.jin-switch:has-text('只看重要')",
                timeout=1200,
            )
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
                    cb = container.locator("input.liquid-switch__input, input[type='checkbox']").first
                    if cb and cb.count():
                        try:
                            if cb.is_checked():
                                if debug:
                                    try:
                                        print("[DEBUG] 已开启‘只看重要’开关")
                                    except Exception:
                                        pass
                                return
                        except Exception:
                            pass
                        try:
                            cb.check(force=True, timeout=1500)
                            page.wait_for_timeout(200)
                        except Exception:
                            try:
                                cb.click()
                                page.wait_for_timeout(200)
                            except Exception:
                                pass
                    if container.locator(
                        "input.liquid-switch__input:checked, "
                        "input[type='checkbox']:checked"
                    ).count() > 0:
                        if debug:
                            try:
                                print("[DEBUG] 已开启‘只看重要’开关")
                            except Exception:
                                pass
                        return
                    lbl = container.locator("label.liquid-switch__label, label").first
                    if lbl and lbl.count():
                        try:
                            lbl.click()
                            page.wait_for_timeout(200)
                        except Exception:
                            pass
                    if container.locator(
                        "input.liquid-switch__input:checked, "
                        "input[type='checkbox']:checked"
                    ).count() > 0:
                        if debug:
                            try:
                                print("[DEBUG] 已开启‘只看重要’开关")
                            except Exception:
                                pass
                        return
                except Exception:
                    continue
    except Exception:
        pass


def _install_request_blocker(context, debug: bool = False) -> None:
    """在上下文层面拦截大资源类型以提速（保留脚本与XHR）。"""
    try:
        def _handler(route):
            try:
                rt = route.request.resource_type
                # 允许 stylesheet 与 websocket 通过，避免影响DOM呈现与数据加载
                if rt in ("image", "media", "font", "beacon", "manifest"):
                    return route.abort()
            except Exception:
                pass
            try:
                route.continue_()
            except Exception:
                try:
                    route.abort()
                except Exception:
                    pass
        context.route("**/*", _handler)
        if debug:
            try:
                print("[DEBUG] 已启用请求拦截：image/media/font/beacon 将被跳过")
            except Exception:
                pass
    except Exception:
        pass


def _extract_rows_from_page(page: Page, target_date: str) -> List[Dict[str, Any]]:
    """解析当日表格数据，返回结构化列表。"""
    rows: List[Dict[str, Any]] = []
    seen_keys = set()

    # 某些天数据较多，表格容器需要滚动以加载全部
    try:
        body_wrap = page.locator("div.jin-table-body__wrapper").first
        if body_wrap and body_wrap.count():
            prev_cnt = -1
            stable = 0
            for _ in range(8):
                page.evaluate("el => el.scrollTop = el.scrollHeight", body_wrap)
                page.wait_for_timeout(120)
                cur_cnt = page.locator("div.jin-table-body div.jin-table-row").count()
                if cur_cnt == prev_cnt:
                    stable += 1
                    if stable >= 2:
                        break
                else:
                    stable = 0
                prev_cnt = cur_cnt
    except Exception:
        pass

    # 数据行定位
    row_locs = page.locator("div.jin-table-body div.jin-table-row")
    count = row_locs.count()
    if count > 0:
        for i in range(count):
            r = row_locs.nth(i)
            cols = r.locator("div.jin-table-column")
            n = cols.count()
            if n < 6:
                continue
            time_text = _text_or_none(cols.nth(0).text_content())
            name_cell = cols.nth(1)
            name_text = _text_or_none(
                name_cell.locator(".data-name-text").text_content()
                if name_cell.locator(".data-name-text").count()
                else name_cell.text_content()
            )
            flag_src = ""
            try:
                flag_el = name_cell.locator("img.jin-flag").first
                if flag_el.count():
                    flag_src = flag_el.get_attribute("src") or ""
            except Exception:
                flag_src = ""
            country = _parse_country_from_flag_src(flag_src)
            star = _count_stars_in_row(r)
            previous = _text_or_none(cols.nth(3).text_content())
            consensus = _text_or_none(cols.nth(4).text_content())
            actual = _text_or_none(cols.nth(5).text_content())
            detail_url = None
            try:
                a = name_cell.locator("a[href^='/detail/']").first
                if a and a.count():
                    href = a.get_attribute("href") or ""
                    if href:
                        detail_url = urljoin("https://rili.jin10.com/", href)
            except Exception:
                pass
            affect = None
            try:
                aff_el = r.locator(".data-affect").first
                if aff_el and aff_el.count():
                    affect = _text_or_none(aff_el.text_content())
            except Exception:
                pass
            row = {
                "date": target_date,
                "time": time_text,
                "country": country,
                "name": name_text,
                "star": star,
                "previous": _to_float_or_str(previous),
                "consensus": _to_float_or_str(consensus),
                "actual": _to_float_or_str(actual),
                "affect": affect,
                "detail_url": detail_url,
            }
            key = (row.get("time") or "", row.get("name") or "", row.get("previous") or "", row.get("actual") or "")
            if key not in seen_keys:
                seen_keys.add(key)
                rows.append(row)
        return rows

    # 解析列表结构（jin-list）中的“数据”卡片
    try:
        li_items = page.locator("div.jin-list .jin-list-item")
        m = li_items.count()
        for i in range(m):
            item = li_items.nth(i)
            # 仅解析数据型（排除事件）
            if item.locator(".jin-list-item__slot .data").count() <= 0:
                continue
            time_text = _text_or_none(
                item.locator(".jin-list-item__header-left .time").text_content()
                if item.locator(".jin-list-item__header-left .time").count()
                else None
            )
            # 国家
            c_src = ""
            try:
                fimg = item.locator(".jin-list-item__header-left img.jin-flag").first
                if fimg and fimg.count():
                    c_src = fimg.get_attribute("src") or ""
            except Exception:
                pass
            country = _parse_country_from_flag_src(c_src)
            # 星级（优先统计点亮）
            try:
                lit = item.locator(".jin-list-item__header-right .jin-star i[style*='var(--rise)']").count()
                if lit and lit > 0:
                    star = lit
                else:
                    total = item.locator(".jin-list-item__header-right .jin-star i").count()
                    gray = item.locator(
                        ".jin-list-item__header-right .jin-star "
                        "i[style*='on-rise-light-lowest']"
                    ).count()
                    star = (max(0, total - gray) if total and gray and gray > 0 else None)
            except Exception:
                star = None
            # 名称与详情链接
            slot = item.locator(".jin-list-item__slot .data").first
            name_text = None
            detail_url = None
            try:
                a = slot.locator(".data-name a").first
                if a and a.count():
                    name_text = _text_or_none(a.text_content())
                    href = a.get_attribute("href") or ""
                    if href:
                        detail_url = urljoin("https://rili.jin10.com/", href)
                else:
                    name_text = _text_or_none(slot.locator(".data-name").text_content())
            except Exception:
                name_text = _text_or_none(
                    slot.locator(".data-name").text_content()
                    if slot.locator(".data-name").count()
                    else None
                )
            # 利多/利空标签
            affect = None
            try:
                aff_el = slot.locator(".data-affect").first
                if aff_el and aff_el.count():
                    affect = _text_or_none(aff_el.text_content())
            except Exception:
                pass
            # 前值/预期/公布
            previous = consensus = actual = None
            try:
                vs = slot.locator(".data-footer .data-value")
                if vs.count() >= 1:
                    previous = _text_or_none(vs.nth(0).locator(".data-value__num").text_content())
                if vs.count() >= 2:
                    consensus = _text_or_none(vs.nth(1).locator(".data-value__num").text_content())
                if vs.count() >= 3:
                    actual = _text_or_none(vs.nth(2).locator(".data-value__num").text_content())
            except Exception:
                pass

            row = {
                "date": target_date,
                "time": time_text,
                "country": country,
                "name": name_text,
                "star": star,
                "previous": _to_float_or_str(previous),
                "consensus": _to_float_or_str(consensus),
                "actual": _to_float_or_str(actual),
                "affect": affect,
                "detail_url": detail_url,
            }
            key = (row.get("time") or "", row.get("name") or "", row.get("previous") or "", row.get("actual") or "")
            if key not in seen_keys:
                seen_keys.add(key)
                rows.append(row)
    except Exception:
        pass

    return rows


def _extract_rows_fast(page: Page, target_date: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        data = page.evaluate(
            """
            () => {
              const res = [];
              const text = (el) => (el && el.textContent ? el.textContent.trim() : null) || null;
              const getStar = (container) => {
                if (!container) return null;
                const lit = container.querySelectorAll(".jin-star i[style*='var(--rise)']").length;
                if (lit > 0) return lit;
                const total = container.querySelectorAll(".jin-star i").length;
                const gray = container.querySelectorAll(".jin-star i[style*='on-rise-light-lowest']").length;
                if (total && gray && gray > 0) return Math.max(0, total - gray);
                return null;
              };
              const tableRows = Array.from(document.querySelectorAll('div.jin-table-body div.jin-table-row'));
              if (tableRows.length > 0) {
                for (const r of tableRows) {
                  const cols = r.querySelectorAll('div.jin-table-column');
                  if (!cols || cols.length < 6) continue;
                  const time_text = text(cols[0]);
                  const name_cell = cols[1];
                  const name_text = text(name_cell.querySelector('.data-name-text')) || text(name_cell);
                  const flag_img = name_cell.querySelector('img.jin-flag');
                  const flag_src = flag_img ? (flag_img.getAttribute('src') || '') : '';
                  const star = getStar(cols[2]);
                  const previous = text(cols[3]);
                  const consensus = text(cols[4]);
                  const actual = text(cols[5]);
                  const a = name_cell.querySelector("a[href^='/detail/']");
                  const detail_href = a ? (a.getAttribute('href') || '') : '';
                  const affect = text(r.querySelector('.data-affect'));
                  res.push({
                    time: time_text,
                    name: name_text,
                    flag_src,
                    star,
                    previous,
                    consensus,
                    actual,
                    affect,
                    detail_href,
                  });
                }
                return res;
              }
              const items = Array.from(document.querySelectorAll('div.jin-list .jin-list-item'));
              for (const item of items) {
                const dataSlot = item.querySelector('.jin-list-item__slot .data');
                if (!dataSlot) continue;
                const time_text = text(item.querySelector('.jin-list-item__header-left .time'));
                const fimg = item.querySelector('.jin-list-item__header-left img.jin-flag');
                const flag_src = fimg ? (fimg.getAttribute('src') || '') : '';
                const star = getStar(item.querySelector('.jin-list-item__header-right'));
                const a = dataSlot.querySelector('.data-name a');
                let name_text = null, detail_href = '';
                if (a) {
                  name_text = text(a);
                  detail_href = a.getAttribute('href') || '';
                } else {
                  name_text = text(dataSlot.querySelector('.data-name'));
                }
                const affect = text(dataSlot.querySelector('.data-affect'));
                let previous = null, consensus = null, actual = null;
                const vs = dataSlot.querySelectorAll('.data-footer .data-value .data-value__num');
                if (vs && vs.length >= 1) previous = text(vs[0]);
                if (vs && vs.length >= 2) consensus = text(vs[1]);
                if (vs && vs.length >= 3) actual = text(vs[2]);
                res.push({
                  time: time_text,
                  name: name_text,
                  flag_src,
                  star,
                  previous,
                  consensus,
                  actual,
                  affect,
                  detail_href,
                });
              }
              return res;
            }
            """
        )
    except Exception:
        try:
            return _extract_rows_from_page(page, target_date)
        except Exception:
            return []

    seen = set()
    for it in (data or []):
        time_text = _text_or_none((it or {}).get("time"))
        name_text = _text_or_none((it or {}).get("name"))
        previous = _to_float_or_str(_text_or_none((it or {}).get("previous")))
        consensus = _to_float_or_str(_text_or_none((it or {}).get("consensus")))
        actual = _to_float_or_str(_text_or_none((it or {}).get("actual")))
        affect = _text_or_none((it or {}).get("affect"))
        flag_src = (it or {}).get("flag_src") or ""
        detail_href = (it or {}).get("detail_href") or ""
        country = _parse_country_from_flag_src(flag_src)
        detail_url = urljoin("https://rili.jin10.com/", detail_href) if detail_href else None
        row = {
            "date": target_date,
            "time": time_text,
            "country": country,
            "name": name_text,
            "star": (it or {}).get("star"),
            "previous": previous,
            "consensus": consensus,
            "actual": actual,
            "affect": affect,
            "detail_url": detail_url,
        }
        key = (row.get("time") or "", row.get("name") or "", row.get("previous") or "", row.get("actual") or "")
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return rows


def _current_date_from_url(page: Page) -> Optional[date]:
    try:
        m = re.search(r"/day/(\d{4}-\d{2}-\d{2})", page.url)
        if not m:
            return None
        return datetime.strptime(m.group(1), "%Y-%m-%d").date()
    except Exception:
        return None


def _switch_date_via_slider(page: Page, cur_date: date, target_date: date, debug: bool = False) -> bool:
    try:
        _ensure_data_tab(page, debug=debug)
        try:
            page.wait_for_selector("div.date-slider", timeout=5000)
        except Exception:
            pass
        slider = page.locator("div.date-slider").first
        if not slider or slider.count() == 0:
            return False
        # 仅处理7天窗口内的切换，避免过度滚动
        ad = _current_date_from_url(page) or cur_date
        delta_days = (ad - target_date).days
        if abs(delta_days) > 7:
            return False
        # 直接点击具体的日号；若不在窗口则尝试移动周窗口
        target_day = str(target_date.day)
        try:
            max_shift = 6
            for _ in range(max_shift + 1):
                items = page.locator("ul.date-slider__day li.date-slider__day-item")
                n = items.count()
                idx = -1
                for i in range(n):
                    txt = (items.nth(i).locator("span.date-text").text_content() or "").strip()
                    if txt == target_day:
                        idx = i
                        break
                if idx != -1:
                    items.nth(idx).click()
                    page.wait_for_timeout(150)
                    break
                # 未找到则移动窗口一格并等待窗口变化
                # 记录窗口首日文本用于检测变化
                if n > 0:
                    first_item = items.nth(0)
                    first_txt_el = first_item.locator("span.date-text")
                    raw_before = first_txt_el.text_content() or ""
                    first_text_before = raw_before.strip()
                else:
                    first_text_before = ""
                if target_date < ad:
                    page.locator("div.date-slider__prev").first.click()
                else:
                    page.locator("div.date-slider__next").first.click()
                # 等待首日文本发生变化，避免连续无效点击
                for _wait in range(10):
                    page.wait_for_timeout(80)
                    items2 = page.locator(
                        "ul.date-slider__day li.date-slider__day-item"
                    )
                    if items2.count() > 0:
                        first_item2 = items2.nth(0)
                        first_txt_el2 = first_item2.locator("span.date-text")
                        raw_after = first_txt_el2.text_content() or ""
                        first_text_after = raw_after.strip()
                        if first_text_after != first_text_before:
                            break
            else:
                return False
        except Exception:
            return False
        # 最终确认
        ad_final = _current_date_from_url(page)
        if ad_final != target_date:
            return False
        try:
            page.wait_for_selector("div.jin-table-body, .jin-list .jin-list-item", timeout=15000)
        except Exception:
            pass
        return True
    except Exception:
        return False


def crawl_calendar(
    months: int = 3,
    start: Optional[str] = None,
    end: Optional[str] = None,
    headless: bool = True,
    debug: bool = False,
    important_only: bool = False,
    user_data_dir: Optional[str] = None,
    out_path: Optional[str] = None,
    db_path: Optional[str] = None,
    source: str = 'listing_data',
    recheck_important_every: int = 30,
    use_slider: bool = False,
    setup_seconds: int = 0,
    no_important_only: bool = False,
) -> int:
    """抓取最近 N 个月或指定日期区间的日历数据。"""
    # 计算日期区间
    if start and end:
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()
    else:
        end_date = date.today()
        start_date = end_date - timedelta(days=months * 30)

    total_written = 0

    with sync_playwright() as p:
        # 计算起始 URL（从区间终止日开始）
        first_day_str = end_date.strftime("%Y-%m-%d")
        first_url = f"https://rili.jin10.com/day/{first_day_str}"

        # 手动调试阶段：开启带界面，等待后自动关闭
        if setup_seconds and setup_seconds > 0:
            tmp_browser = None
            tmp_context = None
            try:
                if user_data_dir:
                    tmp_context = p.chromium.launch_persistent_context(
                        user_data_dir=user_data_dir,
                        headless=False,
                        args=[
                            "--disable-blink-features=AutomationControlled"
                        ],
                        locale="zh-CN",
                        user_agent=DEFAULT_UA,
                    )
                else:
                    tmp_browser = p.chromium.launch(
                        headless=False,
                        args=["--disable-blink-features=AutomationControlled"],
                    )
                    tmp_context = tmp_browser.new_context(
                        locale="zh-CN",
                        user_agent=DEFAULT_UA,
                    )
                tmp_page = tmp_context.new_page()
                try:
                    tmp_page.goto(first_url, wait_until="domcontentloaded", timeout=30000)
                    try:
                        tmp_page.wait_for_load_state("networkidle", timeout=30000)
                    except Exception:
                        if debug:
                            try:
                                print("[DEBUG] networkidle 等待超时，继续")
                            except Exception:
                                pass
                except Exception:
                    try:
                        tmp_page.goto(first_url, wait_until="load", timeout=45000)
                    except Exception:
                        pass
                if debug:
                    try:
                        print(
                            f"[DEBUG] 手动设置阶段，等待 {setup_seconds} 秒后切换到无界面运行"
                        )
                    except Exception:
                        pass
                tmp_page.wait_for_timeout(setup_seconds * 1000)
            finally:
                try:
                    tmp_context.close()
                except Exception:
                    pass
                if tmp_browser:
                    try:
                        tmp_browser.close()
                    except Exception:
                        pass

        # 正式运行阶段：若存在手动阶段，则强制无界面运行
        browser = None
        run_headless = True if (setup_seconds and setup_seconds > 0) else headless
        if user_data_dir:
            context = p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                headless=run_headless,
                args=[
                    "--disable-blink-features=AutomationControlled"
                ],
                locale="zh-CN",
                user_agent=DEFAULT_UA,
            )
        else:
            browser = p.chromium.launch(
                headless=run_headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = browser.new_context(
                locale="zh-CN",
                user_agent=DEFAULT_UA,
            )
        # 正式阶段启用请求拦截以提升速度
        _install_request_blocker(context, debug=debug)
        page = context.new_page()
        # 可选：打开数据库连接
        conn = None
        if db_path and Article is not None:
            try:
                conn = get_conn(db_path)
                ensure_schema(conn)
            except Exception:
                conn = None

        if debug:
            try:
                print("[DEBUG] 开始逐日抓取并增量写入CSV")
            except Exception:
                pass
        else:
            try:
                print("开始逐日抓取...")
            except Exception:
                pass

        cur = end_date
        since_last_check = 999999
        while cur >= start_date:
            day_str = cur.strftime("%Y-%m-%d")
            if debug:
                try:
                    print(f"[DEBUG] 目标日期: {day_str}")
                except Exception:
                    pass

            url = f"https://rili.jin10.com/day/{day_str}"
            if debug:
                try:
                    print(f"[DEBUG] 直达页面: {url}")
                except Exception:
                    pass
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
            except Exception:
                try:
                    page.goto(url, wait_until="load", timeout=20000)
                except Exception:
                    if debug:
                        try:
                            print("[DEBUG] 页面打开失败，跳过当日")
                        except Exception:
                            pass
                    cur -= timedelta(days=1)
                    continue
            # 仅按周期确保一次“只看重要”，减少每日固定成本
            if not no_important_only and since_last_check >= recheck_important_every:
                _ensure_data_tab(page, debug=debug)
                _ensure_important_only(page, debug=debug)
                since_last_check = 0

            try:
                page.wait_for_load_state("networkidle", timeout=12000)
            except Exception:
                pass
            try:
                page.wait_for_selector(
                    "div.jin-table-body, "
                    ".jin-list .jin-list-item",
                    timeout=20000,
                )
            except Exception:
                try:
                    page.wait_for_timeout(2000)
                except Exception:
                    pass

            # 使用URL中的日期作为“当日”标识（直达可靠）；若缺失则退回目标日
            used_day = _current_date_from_url(page) or cur
            used_day_str = used_day.strftime("%Y-%m-%d")

            # 快速解析（失败时自动退回慢速解析）
            day_rows = _extract_rows_fast(page, used_day_str)
            # 客户端兜底过滤：若要求“只看重要”，按星级>=3保留
            if not no_important_only:
                try:
                    day_rows = [r for r in (day_rows or []) if isinstance(r.get("star"), int) and r.get("star") >= 3]
                except Exception:
                    pass
            if debug:
                try:
                    print(
                        f"[DEBUG] {used_day_str} 提取 {len(day_rows)} 条"
                    )
                except Exception:
                    pass
            else:
                try:
                    print(
                        f"{used_day_str} 提取 {len(day_rows)} 条"
                    )
                except Exception:
                    pass

            # 边爬边入库（若提供了 db_path 且 storage 可用）
            if day_rows and (Article is not None) and (conn is not None):
                try:
                    rows_db = []
                    for r in day_rows:
                        ttxt = (r.get('time') or '').strip() if isinstance(r.get('time'), str) else ''
                        pub = None
                        if re.search(r'\d{1,2}:[0-9]{2}', ttxt or ''):
                            hhmmss = ttxt if ttxt.count(':') >= 2 else f"{ttxt}:00"
                            pub = f"{r.get('date')} {hhmmss}"
                        rows_db.append(
                            Article(
                                site='rili.jin10.com',
                                source=source or 'listing_data',
                                title=(r.get('name') or '').strip(),
                                content=(r.get('name') or '').strip(),
                                published_at=pub,
                                url=((r.get('detail_url') or '').strip() or None),
                                raw_html=None,
                                extra_json={
                                    'date': r.get('date'),
                                    'time': r.get('time'),
                                    'country': r.get('country'),
                                    'star': r.get('star'),
                                    'previous': r.get('previous'),
                                    'consensus': r.get('consensus'),
                                    'actual': r.get('actual'),
                                    'affect': r.get('affect'),
                                },
                            )
                        )
                    if rows_db:
                        upsert_many(conn, rows_db)
                except Exception:
                    pass

            if out_path and day_rows:
                prefer = [
                    "date",
                    "time",
                    "country",
                    "name",
                    "star",
                    "previous",
                    "consensus",
                    "actual",
                    "affect",
                    "detail_url",
                ]
                exists = os.path.exists(out_path)
                mode = "a" if exists else "w"
                try:
                    with open(
                        out_path,
                        mode,
                        encoding=CSV_ENCODING,
                        newline="",
                    ) as f:
                        writer = csv.DictWriter(f, fieldnames=prefer)
                        if not exists:
                            writer.writeheader()
                        # 补全缺失列为 None
                        for r in day_rows:
                            for c in prefer:
                                if c not in r:
                                    r[c] = None
                            writer.writerow(r)
                    total_written += len(day_rows)
                    if debug:
                        try:
                            print(
                                f"[DEBUG] 已写入 {len(day_rows)} 条到 {out_path}"
                            )
                        except Exception:
                            pass
                    else:
                        try:
                            print(
                                f"已写入 {len(day_rows)} 条到 {out_path}"
                            )
                        except Exception:
                            pass
                except Exception:
                    pass
            elif out_path and not day_rows:
                if debug:
                    try:
                        print(
                            f"[DEBUG] {day_str} 无数据，跳过写入"
                        )
                    except Exception:
                        pass
                else:
                    try:
                        print(
                            f"{day_str} 无数据，跳过写入"
                        )
                    except Exception:
                        pass

            cur -= timedelta(days=1)
            since_last_check += 1

        context.close()
        if browser:
            browser.close()
        # 关闭数据库连接
        if db_path and (conn is not None):
            try:
                conn.close()
            except Exception:
                pass

    return total_written


def to_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    df = pd.DataFrame(rows or [])
    # 固定列顺序
    prefer = [
        "date",
        "time",
        "country",
        "name",
        "star",
        "previous",
        "consensus",
        "actual",
        "affect",
        "detail_url",
    ]
    for c in prefer:
        if c not in df.columns:
            df[c] = None
    df = df[prefer]
    df.to_csv(out_path, index=False, encoding=CSV_ENCODING)


def main():
    parser = argparse.ArgumentParser(description="金十 日历数据抓取")
    parser.add_argument("--months", type=int, default=3, help="最近月份数，与 start/end 互斥")
    parser.add_argument("--start", type=str, default="", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--output", type=str, default="calendar_last_3m.csv", help="输出CSV路径")
    parser.add_argument("--db", type=str, default="", help="SQLite 数据库路径（留空则不入库）")
    parser.add_argument("--source", type=str, default="listing_data", help="Article.source 字段值")
    parser.add_argument("--headed", action="store_true", help="以带界面模式运行浏览器，便于调试")
    parser.add_argument("--debug", action="store_true", help="打印调试信息与抓取进度")
    parser.add_argument("--important-only", action="store_true", help="页面内开启‘只看重要’过滤")
    parser.add_argument("--user-data-dir", type=str, default="", help="持久化浏览器用户数据目录，用于复用登录状态")
    parser.add_argument(
        "--recheck-important-every",
        type=int,
        default=30,
        help=(
            "每隔N天重新校验一次‘只看重要’开关（0表示每次都校验，"
            "1表示仅切换后校验一次）"
        ),
    )
    parser.add_argument("--use-slider", action="store_true", help="启用7天内的滑块切换（默认关闭，使用直达URL）")
    parser.add_argument("--setup-seconds", type=int, default=0, help="手动调试阶段的秒数，等待后自动关闭界面并转入无界面运行")
    args = parser.parse_args()

    total = crawl_calendar(
        months=args.months,
        start=args.start or None,
        end=args.end or None,
        headless=not args.headed,
        debug=args.debug,
        important_only=getattr(args, "important_only", False),
        user_data_dir=(args.user_data_dir or None),
        out_path=args.output,
        db_path=(args.db or None),
        source=args.source,
        recheck_important_every=getattr(args, "recheck_important_every", 30),
        use_slider=getattr(args, "use_slider", False),
        setup_seconds=getattr(args, "setup_seconds", 0),
    )
    print(f"已输出 {total} 条记录到: {args.output}")


if __name__ == "__main__":
    main()
