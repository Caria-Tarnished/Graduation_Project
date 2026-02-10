# -*- coding: utf-8 -*-
"""
核心工具模块
"""
from app.core.utils.cache import (
    LRUCache,
    cached,
    get_cache,
    clear_all_caches,
    get_all_cache_stats
)

__all__ = [
    "LRUCache",
    "cached",
    "get_cache",
    "clear_all_caches",
    "get_all_cache_stats"
]
