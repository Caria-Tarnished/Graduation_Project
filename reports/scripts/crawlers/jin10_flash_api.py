# -*- coding: utf-8 -*-
"""
金十数据 快讯 API 抓取脚本（集成参考方案，支持接口发现与流式写入）

- 首选“API 模式”：直接轮询快讯接口，按 max_time/last_id 翻页，
  效率高、稳定。
- 可选“接口发现”：首次使用可通过 Playwright 打开快讯页，
  拦截到真实接口与初始参数/请求头。
- 本地兜底过滤：即使后端未按筛选返回，仍会在客户端按
  important/hot 等条件过滤后再落盘。

运行示例（PowerShell）：
python -m scripts.crawlers.jin10_flash_api \
  --months 12 \
  --output data/raw/flash_last_12m.csv \
  --api-base https://xxxxxx.jin10.com/flash \
  --stream --important-only \
  --hot-levels "2,3,4" --sleep 1.8 --debug

若未知接口地址，可先用“接口发现”模式：
python -m scripts.crawlers.jin10_flash_api \
  --months 12 --output data/raw/flash_last_12m.csv \
  --page-url https://www.jin10.com/flash \
  --headed --user-data-dir .pw_jin10 --setup-seconds 12 \
  --stream --important-only \
  --hot-levels "爆,沸,热" --sleep 1.8 --debug
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlsplit

import requests

# 可选写库（SQLite）
try:
    from .storage import (
        Article,
        ensure_schema,
        get_conn,
        upsert_many,
    )
except Exception:  # noqa: BLE001
    Article = None  # type: ignore
    ensure_schema = None  # type: ignore
    get_conn = None  # type: ignore
    upsert_many = None  # type: ignore

# 统一 CSV 列顺序（与参考实现一致）
CSV_FIELDS = [
    "id",
    "time",
    "type",
    "content",
    "important",
    "hot",
    "indicator_name",
    "previous",
    "consensus",
    "actual",
    "star",
    "country",
    "unit",
]

# HTML 标签去除（快讯 type:0 的 content 可能包含 <b> 等标签）
TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    """去除字符串中的 HTML 标签。"""
    return TAG_RE.sub("", text or "").strip()


def _normalize_time(val: Any) -> str:
    """将各种时间字段规范化为 'YYYY-MM-DD HH:MM:SS'。
    兼容：
    - 13/10 位时间戳（毫秒/秒）
    - ISO 字符串（含 T/时区）
    - 常见格式 'YYYY-MM-DD HH:MM(:SS)'、'YYYY/MM/DD HH:MM(:SS)'
    解析失败返回空串。
    """
    try:
        if val is None:
            return ""
        # 数字或数字串：epoch
        if isinstance(val, (int, float)):
            x = int(val)
            if x > 10_000_000_000:  # 13 位毫秒
                dtm = datetime.fromtimestamp(x / 1000)
            else:  # 10 位秒
                dtm = datetime.fromtimestamp(x)
            return dtm.strftime("%Y-%m-%d %H:%M:%S")
        s = str(val).strip()
        if not s:
            return ""
        if s.isdigit():
            x = int(s[:13]) if len(s) >= 13 else int(s)
            if len(s) >= 13:
                dtm = datetime.fromtimestamp(x / 1000)
            else:
                dtm = datetime.fromtimestamp(x)
            return dtm.strftime("%Y-%m-%d %H:%M:%S")
        # ISO 与常见格式
        ss = s.replace("T", " ").replace("Z", "+00:00")
        try:
            # 带时区或微秒
            dtm = datetime.fromisoformat(ss)
            # 去掉时区以统一输出
            if hasattr(dtm, "tzinfo") and dtm.tzinfo is not None:
                dtm = dtm.replace(tzinfo=None)
            return dtm.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
        ):
            try:
                dtm = datetime.strptime(ss, fmt)
                return dtm.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
        return ""
    except Exception:
        return ""


def parse_flash_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """将快讯接口中的单条记录标准化为统一结构。"""
    typ = item.get("type") or item.get("category") or item.get("ty")
    rid = (
        item.get("id")
        or item.get("news_id")
        or item.get("_id")
        or item.get("i")
        or item.get("uuid")
        or item.get("docid")
        or item.get("rid")
        or item.get("nid")
        or ""
    )
    tstr = (
        item.get("time")
        or item.get("create_time")
        or item.get("publish_time")
        or item.get("public_time")
        or item.get("created_at")
        or item.get("ctime")
        or item.get("show_time")
        or item.get("display_time")
        or item.get("time_at")
        or item.get("timestamp")
        or item.get("ts")
        or item.get("t")
        or ""
    )
    hot = item.get("hot") or item.get("heat") or item.get("h") or ""
    important = item.get("important") or item.get("imp") or item.get("impt")
    data = item.get("data") or item.get("detail") or item.get("d") or {}
    # 若顶层没有时间，尝试从 data 中兜底
    if not tstr:
        for k in (
            "time", "time_at", "show_time", "display_time",
            "created_at", "ctime", "publish_time",
            "public_time", "timestamp", "ts",
        ):
            v = data.get(k)
            if v:
                tstr = v
                break
    # 统一规范化时间
    tnorm = _normalize_time(tstr)

    parsed: Dict[str, Any] = {
        "id": rid,
        "time": tnorm,
        "type": typ,
        "hot": hot,
        "important": (
            int(bool(important))
            if isinstance(important, (bool, int))
            else 0
        ),
        "content": "",
        "indicator_name": None,
        "previous": None,
        "consensus": None,
        "actual": None,
        "star": None,
        "country": None,
        "unit": None,
    }

    if (
        typ == 0
        or (
            typ is None
            and (
                data.get("content")
                or item.get("content")
                or item.get("text")
                or item.get("title")
            )
        )
    ):
        # 纯文本快讯
        content = (
            data.get("content")
            or item.get("content")
            or item.get("text")
            or item.get("title")
            or ""
        )
        parsed["content"] = strip_html(content)
    elif typ == 1:
        # 结构化经济指标
        parsed["indicator_name"] = data.get("name")
        parsed["previous"] = data.get("previous")
        parsed["consensus"] = data.get("consensus")
        parsed["actual"] = data.get("actual")
        parsed["star"] = data.get("star")
        parsed["country"] = data.get("country")
        parsed["unit"] = data.get("unit")
        parts = [
            (
                f"{data.get('country') or ''}"
                f"{data.get('time_period') or ''}"
                f"{data.get('name') or ''}"
            ),
            f"前值:{data.get('previous')}",
            f"预期:{data.get('consensus')}",
            f"公布:{data.get('actual')}",
        ]
        parsed["content"] = " ".join(
            [p for p in parts if p and p != "None"]
        ).strip()
    else:
        content = (data.get("content") or data.get("title") or "").strip()
        parsed["content"] = strip_html(content)

    return parsed


def parse_flash_response(
    payload: Dict[str, Any],
) -> Tuple[
    List[Dict[str, Any]],
    Optional[str],
    Optional[str],
]:
    """解析一次响应，返回标准化后的记录列表以及下一次翻页需要的 max_time/last_id。"""
    items = payload.get("data")
    if items is None:
        for k in ("data", "list", "items", "rows", "result"):
            v = payload.get(k)
            if isinstance(v, list):
                items = v
                break
            if isinstance(v, dict):
                for kk in ("list", "items", "data", "rows", "result"):
                    vv = v.get(kk)
                    if isinstance(vv, list):
                        items = vv
                        break
                if isinstance(items, list):
                    break
    elif isinstance(items, dict):
        inner = None
        for kk in ("list", "items", "data", "rows", "result"):
            vv = items.get(kk)
            if isinstance(vv, list):
                inner = vv
                break
        if inner is None:
            for vv in items.values():
                if isinstance(vv, list):
                    inner = vv
                    break
        items = inner if isinstance(inner, list) else []
    elif not isinstance(items, list):
        items = []

    if not isinstance(items, list):
        for vv in payload.values():
            if isinstance(vv, list):
                items = vv
                break
            if isinstance(vv, dict):
                for vvv in vv.values():
                    if isinstance(vvv, list):
                        items = vvv
                        break
                if isinstance(items, list):
                    break
    if not isinstance(items, list):
        items = []

    result: List[Dict[str, Any]] = []
    for it in items:
        try:
            result.append(parse_flash_item(it))
        except Exception:
            continue

    next_max_time, next_last_id = None, None
    if items:
        last = items[-1]
        next_max_time = last.get("time") or last.get("create_time")
        next_last_id = last.get("id")
    if not next_max_time:
        for container in (
            payload,
            (
                payload.get("data")
                if isinstance(payload.get("data"), dict)
                else {}
            ),
        ):
            try:
                if isinstance(container, dict):
                    mt = (
                        container.get("max_time")
                        or container.get("min_time")
                    )
                    if mt:
                        next_max_time = mt
                        break
            except Exception:
                pass
    if not next_last_id:
        for container in (
            payload,
            (
                payload.get("data")
                if isinstance(payload.get("data"), dict)
                else {}
            ),
        ):
            try:
                if isinstance(container, dict):
                    lid = (
                        container.get("last_id")
                        or container.get("min_id")
                        or container.get("bottom_id")
                    )
                    if lid:
                        next_last_id = lid
                        break
            except Exception:
                pass
    return result, next_max_time, next_last_id


def _ensure_csv_header(out_path: str) -> None:
    try:
        need_header = (not out_path) or (not out_path.strip())
        if need_header:
            return
        need_header = (
            (not (os.path.exists(out_path)))
            or (os.path.getsize(out_path) == 0)
        )
    except Exception:
        need_header = True
    if need_header:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


def append_rows_to_csv(rows: List[Dict[str, Any]], out_path: str) -> int:
    if not rows:
        return 0
    _ensure_csv_header(out_path)
    wrote = 0
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        for r in rows:
            rec = {c: r.get(c) for c in CSV_FIELDS}
            writer.writerow(rec)
            wrote += 1
    return wrote


def discover_flash_endpoint(
    page_urls: List[str],
    headed: bool = False,
    wait_seconds: int = 20,
    user_data_dir: Optional[str] = None,
    setup_seconds: int = 0,
) -> Tuple[str, Dict[str, Any], Dict[str, str]]:
    """使用 Playwright 打开页面并拦截 XHR/Fetch，请求中自动发现 flash 接口。"""
    from playwright.sync_api import sync_playwright  # 按需导入，避免硬依赖

    target_req = None
    with sync_playwright() as p:
        browser = None
        if user_data_dir:
            context = p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                headless=not headed,
                args=["--disable-blink-features=AutomationControlled"],
                locale="zh-CN",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
            )
        else:
            browser = p.chromium.launch(
                headless=not headed,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = browser.new_context(
                locale="zh-CN",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
            )
        page = context.new_page()

        capture_enabled = (
            False if (setup_seconds and setup_seconds > 0) else True
        )

        def _match_flash_url(url: str) -> bool:
            try:
                sp = urlsplit(url)
                host = (sp.netloc or "").lower()
                basehost = host.split(":")[0]
                path = (sp.path or "").lower()
                q = (sp.query or "").lower()
                if "jin10.com" not in host:
                    return False
                # 排除主站与日历等非 API 域
                if basehost in ("www.jin10.com", "rili.jin10.com"):
                    return False
                # 明确排除非列表型接口（如 /flash/hot 等）
                disallow = (
                    "/flash/hot",
                    "/flash/recommend",
                    "/flash/subscribe",
                    "/flash/search",
                )
                for d in disallow:
                    if d in path:
                        return False
                # 优先匹配 get_flash_list
                if "get_flash_list" in path:
                    return True
                # 允许带 params 的 /flash
                if (path.rstrip("/") == "/flash") and ("params=" in q):
                    return True
            except Exception:
                return False
            return False

        def on_request(req):
            url = req.url or ""
            rtype = getattr(req, "resource_type", None) or req.resource_type
            if rtype in ("xhr", "fetch") and req.method in ("GET", "POST"):
                if not capture_enabled:
                    return
                # 仅在 URL 上包含 params= 时考虑
                if (
                    "params=" in (urlsplit(url).query or "")
                ) and _match_flash_url(url):
                    nonlocal target_req
                    target_req = req

        def on_response(resp):
            try:
                req = resp.request
                if req and _match_flash_url(req.url or ""):
                    # 仅在返回 JSON 时确认捕获
                    ctype = (
                        (
                            resp.headers.get("content-type")
                            or resp.headers.get("Content-Type")
                            or ""
                        ).lower()
                    )
                    if "json" not in ctype:
                        return
                    # 同时确保 URL 上包含 params=（绝大多数列表接口特征）
                    if "params=" not in (
                        urlsplit(req.url or "").query or ""
                    ):
                        return
                    nonlocal target_req
                    if target_req is None:
                        target_req = req
            except Exception:
                pass

        page.on("request", on_request)
        page.on("response", on_response)

        for u in page_urls:
            try:
                page.goto(u, wait_until="domcontentloaded", timeout=30000)
                try:
                    page.wait_for_load_state("networkidle", timeout=30000)
                except Exception:
                    pass
                if setup_seconds and setup_seconds > 0:
                    try:
                        print(
                            "请在 "
                            f"{setup_seconds}"
                            " 秒内完成页面内筛选，随后将自动拦截接口…"
                        )
                    except Exception:
                        pass
                    try:
                        page.wait_for_timeout(setup_seconds * 1000)
                    except Exception:
                        pass
                    capture_enabled = True
                if wait_seconds <= 0:
                    while target_req is None:
                        page.wait_for_timeout(1000)
                else:
                    # 细粒度轮询等待，最短时间内发现即返回
                    rounds = max(1, int(wait_seconds * 4))
                    for _ in range(rounds):
                        if target_req is not None:
                            break
                        page.wait_for_timeout(250)
            except Exception:
                pass
            if target_req:
                break

        context.close()
        if browser is not None:
            browser.close()

    if not target_req:
        raise RuntimeError(
            "未能自动发现 flash 接口，请尝试指定 --page-url 或使用 "
            "--api-base 直连模式，或增加 --wait/--setup-seconds"
        )

    full_url = target_req.url
    sp = urlsplit(full_url)
    base_url = f"{sp.scheme}://{sp.netloc}{sp.path}"
    base_params = dict(parse_qsl(sp.query))
    if "params" not in base_params:
        # 再保险：如果未带 params=，视为发现无效
        raise RuntimeError("发现到的快讯接口缺少 params 参数，请在页面停留于\"快讯/最新\"并重试")

    # 保留浏览器请求头的完整集合，避免丢失服务端校验所需的自定义头
    headers = dict(getattr(target_req, "headers", {}) or {})
    return base_url, base_params, headers


def crawl_flash_via_api(
    base_url: str,
    months: int = 3,
    sleep_secs: float = 1.8,
    headers: Optional[Dict[str, str]] = None,
    base_params: Optional[Dict[str, Any]] = None,
    stream_out_path: Optional[str] = None,
    important_only: bool = False,
    hot_levels: Optional[str] = None,
    debug: bool = False,
    db_path: Optional[str] = None,
    source: str = "flash_api",
) -> List[Dict[str, Any]]:
    """通过逆向 API 方式抓取快讯。返回抓取到的全部行（若 stream 模式则为空）。"""
    assert base_url, "必须提供可用的快讯接口 base_url"
    cutoff = datetime.now() - timedelta(days=30 * months)

    all_rows: List[Dict[str, Any]] = []
    total_count: int = 0  # stream 模式下累计写入条数
    params: Dict[str, Any] = dict(base_params or {})

    # 解析热度等级字符串为数字列表（1=火,2=热,3=沸,4=爆）
    def _parse_hot_levels(s: Optional[str]) -> Optional[List[int]]:
        if not s:
            return None
        try:
            parts = re.split(r"[\s,|]+", s.strip())
            lv: List[int] = []
            for p in parts:
                if not p:
                    continue
                p = p.strip()
                if p.isdigit():
                    lv.append(int(p))
                    continue
                if p in ("火", "?", "huo", "fire"):
                    lv.append(1)
                elif p in ("热", "re", "hot"):
                    lv.append(2)
                elif p in ("沸", "fei", "boil"):
                    lv.append(3)
                elif p in ("爆", "bao", "boom"):
                    lv.append(4)
            lv = sorted(set([x for x in lv if 1 <= x <= 4]))
            return lv or None
        except Exception:
            return None

    hot_levels_list = _parse_hot_levels(hot_levels)
    # 可选：打开 SQLite 连接
    conn = None
    if db_path and Article is not None:
        try:
            conn = get_conn(db_path)
            ensure_schema(conn)
        except Exception:
            conn = None

    def _level_to_hot_text(level: int) -> Optional[str]:
        mapping = {1: "火", 2: "热", 3: "沸", 4: "爆"}
        return mapping.get(level)

    def _hot_text_to_level(hot_val: Any) -> Optional[int]:
        try:
            if hot_val is None:
                return None
            s = str(hot_val).strip()
            if not s:
                return None
            if s.isdigit():
                v = int(s)
                return v if 1 <= v <= 4 else None
            rev = {"火": 1, "热": 2, "沸": 3, "爆": 4}
            return rev.get(s)
        except Exception:
            return None

    def _apply_filters_to_dict(
        d: Dict[str, Any], *, as_inner_json: bool,
        prefer_hot_key: Optional[str] = None,
        prefer_hot_style: Optional[str] = None,  # "str_list" | "int_list"
        prefer_imp_key: Optional[str] = None,
    ) -> None:
        try:
            if important_only:
                imp_key = prefer_imp_key or "only_important"
                d[imp_key] = 1
            if hot_levels_list:
                if as_inner_json:
                    if prefer_hot_key:
                        if prefer_hot_style == "str_list":
                            d[prefer_hot_key] = [
                                _level_to_hot_text(x)
                                for x in hot_levels_list
                                if _level_to_hot_text(x)
                            ]
                        else:
                            d[prefer_hot_key] = hot_levels_list
                    else:
                        d["hot"] = [
                            _level_to_hot_text(x)
                            for x in hot_levels_list
                            if _level_to_hot_text(x)
                        ]
                else:
                    d["hot_levels"] = ",".join([
                        str(x) for x in hot_levels_list
                    ])
        except Exception:
            pass

    if "params" in params:
        inner = params.get("params")
        try:
            inner_obj = (
                json.loads(inner)
                if isinstance(inner, str)
                else dict(inner or {})
            )
        except Exception:
            inner_obj = {}

        prefer_hot_key: Optional[str] = None
        prefer_hot_style: Optional[str] = None
        prefer_imp_key: Optional[str] = None
        try:
            for hk in ("hot", "hots", "heat", "hot_levels", "heat_levels"):
                if hk in inner_obj:
                    prefer_hot_key = hk
                    hv = inner_obj.get(hk)
                    if isinstance(hv, list) and hv:
                        if all(isinstance(x, int) for x in hv):
                            prefer_hot_style = "int_list"
                        else:
                            prefer_hot_style = "str_list"
                    break
        except Exception:
            pass
        try:
            for ik in ("only_important", "important", "imp", "impt"):
                if ik in inner_obj:
                    prefer_imp_key = ik
                    break
        except Exception:
            pass
        for bad in (
            "hot", "hots", "heat", "heat_levels",
            "hot_levels", "important", "imp", "impt",
        ):
            try:
                inner_obj.pop(bad, None)
            except Exception:
                pass
        _apply_filters_to_dict(
            inner_obj,
            as_inner_json=True,
            prefer_hot_key=prefer_hot_key,
            prefer_hot_style=prefer_hot_style,
            prefer_imp_key=prefer_imp_key,
        )
        params["params"] = json.dumps(inner_obj, ensure_ascii=False)
    else:
        try:
            for bad in (
                "hot", "hots", "heat", "heat_levels",
                "important", "imp", "impt",
            ):
                params.pop(bad, None)
        except Exception:
            pass
        _apply_filters_to_dict(params, as_inner_json=False)

    seen_ids: set = set()
    reached_cutoff = False  # 是否已到达时间截止线（用于提前结束主循环）
    while True:
        # 组装请求参数（保持 params JSON 形式）
        # 若已到达截止时间，则跳出主循环
        if reached_cutoff:
            break

        if "params" in params:
            inner = params.get("params")
            try:
                inner_obj = (
                    json.loads(inner)
                    if isinstance(inner, str)
                    else dict(inner or {})
                )
            except Exception:
                inner_obj = {}
            req_params: Dict[str, Any] = {
                "params": json.dumps(inner_obj, ensure_ascii=False)
            }
        else:
            req_params = dict(params)

        resp = requests.get(
            base_url,
            params=req_params,
            headers=headers or {},
            timeout=20,
        )
        resp.raise_for_status()
        try:
            payload = resp.json()
        except Exception:
            preview = (resp.text or "")[:200]
            print("警告: 接口返回的内容不是JSON，可能未登录或被风控。预览:", preview)
            break
        page_rows, next_max_time, next_last_id = parse_flash_response(payload)
        if not page_rows:
            break

        new_batch: List[Dict[str, Any]] = []
        for r in page_rows:
            rid = r.get("id")
            if not rid:
                try:
                    key = json.dumps(r, ensure_ascii=False, sort_keys=True)
                    rid = "auto_" + hashlib.md5(
                        key.encode("utf-8")
                    ).hexdigest()
                    r["id"] = rid
                except Exception:
                    rid = None
            if rid and rid not in seen_ids:
                ok = True
                t = r.get("time")
                try:
                    if t:
                        dt = datetime.strptime(
                            str(t), "%Y-%m-%d %H:%M:%S"
                        )
                        if dt < cutoff:
                            ok = False
                except Exception:
                    pass
                # 客户端兜底过滤
                if ok and important_only:
                    try:
                        if not bool(int(r.get("important") or 0)):
                            ok = False
                    except Exception:
                        if not r.get("important"):
                            ok = False
                if ok and hot_levels_list:
                    lv = _hot_text_to_level(r.get("hot"))
                    if (lv is None) or (lv not in set(hot_levels_list)):
                        ok = False
                seen_ids.add(rid)
                if ok:
                    new_batch.append(r)
                    if not stream_out_path:
                        all_rows.append(r)

        if stream_out_path and new_batch:
            wrote = append_rows_to_csv(new_batch, stream_out_path)
            total_count += (wrote or 0)
        # 可选：入库
        if conn is not None and new_batch:
            try:
                db_rows = []
                for r in new_batch:
                    content = (r.get("content") or "").strip()
                    title = content[:40] if content else ""
                    pub = (r.get("time") or "").strip() or None
                    db_rows.append(
                        Article(
                            site="www.jin10.com",
                            source=source or "flash_api",
                            title=title,
                            content=content,
                            published_at=pub,
                            url=None,
                            raw_html=None,
                            extra_json={
                                "id": r.get("id"),
                                "type": r.get("type"),
                                "important": r.get("important"),
                                "hot": r.get("hot"),
                                "indicator_name": r.get("indicator_name"),
                                "previous": r.get("previous"),
                                "consensus": r.get("consensus"),
                                "actual": r.get("actual"),
                                "star": r.get("star"),
                                "country": r.get("country"),
                                "unit": r.get("unit"),
                            },
                        )
                    )
                if db_rows:
                    upsert_many(conn, db_rows)
            except Exception:
                pass

        # 从本页返回的原始记录中，倒序寻找第一个带时间的项用于调试与截止判断
        last_time_str_dbg = None
        reached_cutoff = False
        try:
            for rr in reversed(page_rows):
                tval = rr.get("time")
                if tval:
                    last_time_str_dbg = tval
                    try:
                        last_dt = datetime.strptime(
                            str(tval),
                            "%Y-%m-%d %H:%M:%S",
                        )
                        if last_dt < cutoff:
                            reached_cutoff = True
                    except Exception:
                        pass
                    break
        except Exception:
            pass

        if debug:
            try:
                added = 0
                if stream_out_path:
                    added = (wrote if ('wrote' in locals()) else 0) or 0
                    total = total_count
                else:
                    added = len(new_batch)
                    total = len(all_rows)
                tail = last_time_str_dbg or "未知"
                print(
                    f"[DEBUG] 本轮新增 {added} 条，总计 {total} 条。"
                    f"最后一条时间: {tail}"
                )
            except Exception:
                pass

        if "params" in params:
            inner = params.get("params")
            try:
                inner_obj = (
                    json.loads(inner)
                    if isinstance(inner, str)
                    else dict(inner or {})
                )
            except Exception:
                inner_obj = {}
            if next_max_time:
                inner_obj["max_time"] = next_max_time
            if next_last_id:
                inner_obj["last_id"] = next_last_id
            params["params"] = json.dumps(inner_obj, ensure_ascii=False)
        else:
            if next_max_time:
                params["max_time"] = next_max_time
            if next_last_id:
                params["last_id"] = next_last_id

        time.sleep(sleep_secs)

    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    return all_rows


def main():
    parser = argparse.ArgumentParser(description="金十快讯 API 抓取")
    parser.add_argument("--months", type=int, default=3, help="回溯月份窗口，默认3")
    parser.add_argument("--output", required=True, help="输出 CSV 文件路径")
    parser.add_argument("--api-base", help="已知快讯接口基础地址（可跳过接口发现）")
    parser.add_argument(
        "--page-url",
        default="https://www.jin10.com/flash",
        help="快讯页 URL（用于接口发现）",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="是否打开有界面浏览器用于发现接口",
    )
    parser.add_argument("--user-data-dir", help="持久化目录，用于复用登录态")
    parser.add_argument(
        "--setup-seconds", type=int, default=0,
        help="打开页面后等待手动设置筛选的秒数",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="启用流式写入，边抓边写",
    )
    parser.add_argument("--important-only", action="store_true", help="只看重要")
    parser.add_argument("--hot-levels", help="热度等级（如: 2,3,4 或 爆,沸,热）")
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.8,
        help="请求间隔秒数，建议≥1.5",
    )
    parser.add_argument("--debug", action="store_true", help="打印调试信息")
    parser.add_argument("--db", default="", help="SQLite 数据库路径（可选）")
    parser.add_argument("--source", default="flash_api", help="入库来源标识")
    args = parser.parse_args()

    if args.api_base:
        base_url = args.api_base
        base_params: Dict[str, Any] = {}
        headers: Dict[str, str] = {}
        if args.debug:
            print(f"[DEBUG] 使用直连 API: {base_url}")
    else:
        base_url, base_params, headers = discover_flash_endpoint(
            [args.page_url],
            headed=args.headed,
            wait_seconds=20,
            user_data_dir=args.user_data_dir,
            setup_seconds=args.setup_seconds,
        )
        if args.debug:
            print(f"[DEBUG] 已捕获接口: {base_url}")
            print(f"[DEBUG] 初始参数: {base_params}")

    rows = crawl_flash_via_api(
        base_url=base_url,
        months=args.months,
        sleep_secs=args.sleep,
        headers=headers,
        base_params=base_params,
        stream_out_path=args.output if args.stream else None,
        important_only=args.important_only,
        hot_levels=args.hot_levels,
        debug=args.debug,
        db_path=(args.db or None),
        source=args.source,
    )

    # 非流式：统一落盘
    if rows and (not args.stream):
        _ensure_csv_header(args.output)
        append_rows_to_csv(rows, args.output)
        print(f"已输出 {len(rows)} 条记录到: {args.output}")


if __name__ == "__main__":
    main()
