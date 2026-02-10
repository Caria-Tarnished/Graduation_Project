# -*- coding: utf-8 -*-
"""
缓存工具类

提供 LRU 缓存和 TTL 缓存功能，用于优化系统性能。

使用示例：
    from app.core.utils.cache import LRUCache, cached
    
    # 方式 1：直接使用缓存类
    cache = LRUCache(maxsize=100, ttl=300)
    cache.set("key", "value")
    value = cache.get("key")
    
    # 方式 2：使用装饰器
    @cached(maxsize=100, ttl=300)
    def expensive_function(arg):
        return result
"""
import time
import hashlib
import json
from typing import Any, Optional, Callable
from functools import wraps
from collections import OrderedDict
from threading import Lock


class LRUCache:
    """
    LRU 缓存（带 TTL 支持）
    
    特性：
    - 最近最少使用（LRU）淘汰策略
    - 支持过期时间（TTL）
    - 线程安全
    """
    
    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None):
        """
        初始化缓存
        
        Args:
            maxsize: 最大缓存条目数
            ttl: 过期时间（秒），None 表示永不过期
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = OrderedDict()
        self.lock = Lock()
        
        # 统计信息
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, key: Any) -> str:
        """
        生成缓存键
        
        Args:
            key: 原始键（可以是任何可序列化对象）
        
        Returns:
            字符串键
        """
        if isinstance(key, str):
            return key
        
        # 对于复杂对象，使用 JSON + MD5
        try:
            key_str = json.dumps(key, sort_keys=True, ensure_ascii=False)
            return hashlib.md5(key_str.encode()).hexdigest()
        except (TypeError, ValueError):
            # 如果无法序列化，使用 repr
            return hashlib.md5(repr(key).encode()).hexdigest()
    
    def get(self, key: Any, default: Any = None) -> Any:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值（键不存在时返回）
        
        Returns:
            缓存值或默认值
        """
        cache_key = self._make_key(key)
        
        with self.lock:
            if cache_key not in self.cache:
                self.misses += 1
                return default
            
            # 检查是否过期
            value, timestamp = self.cache[cache_key]
            if self.ttl is not None and time.time() - timestamp > self.ttl:
                # 过期，删除并返回默认值
                del self.cache[cache_key]
                self.misses += 1
                return default
            
            # 移动到末尾（最近使用）
            self.cache.move_to_end(cache_key)
            self.hits += 1
            return value
    
    def set(self, key: Any, value: Any) -> None:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        cache_key = self._make_key(key)
        
        with self.lock:
            # 如果键已存在，先删除
            if cache_key in self.cache:
                del self.cache[cache_key]
            
            # 添加新值
            self.cache[cache_key] = (value, time.time())
            
            # 如果超过最大大小，删除最旧的条目
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> dict:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "ttl": self.ttl
            }


def cached(maxsize: int = 128, ttl: Optional[float] = None):
    """
    缓存装饰器
    
    Args:
        maxsize: 最大缓存条目数
        ttl: 过期时间（秒）
    
    Returns:
        装饰器函数
    
    使用示例：
        @cached(maxsize=100, ttl=300)
        def expensive_function(arg1, arg2):
            return result
    """
    cache = LRUCache(maxsize=maxsize, ttl=ttl)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))
            
            # 尝试从缓存获取
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # 缓存未命中，执行函数
            result = func(*args, **kwargs)
            
            # 存入缓存
            cache.set(cache_key, result)
            
            return result
        
        # 添加缓存管理方法
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.get_stats
        
        return wrapper
    
    return decorator


# 全局缓存实例（用于跨模块共享）
_global_caches = {}


def get_cache(name: str, maxsize: int = 128, ttl: Optional[float] = None) -> LRUCache:
    """
    获取全局缓存实例
    
    Args:
        name: 缓存名称
        maxsize: 最大缓存条目数
        ttl: 过期时间（秒）
    
    Returns:
        缓存实例
    """
    if name not in _global_caches:
        _global_caches[name] = LRUCache(maxsize=maxsize, ttl=ttl)
    return _global_caches[name]


def clear_all_caches() -> None:
    """清空所有全局缓存"""
    for cache in _global_caches.values():
        cache.clear()


def get_all_cache_stats() -> dict:
    """
    获取所有缓存的统计信息
    
    Returns:
        缓存统计信息字典
    """
    return {
        name: cache.get_stats()
        for name, cache in _global_caches.items()
    }


# 测试代码
if __name__ == "__main__":
    print("=" * 80)
    print("缓存工具测试")
    print("=" * 80)
    
    # 测试 1: 基础功能
    print("\n测试 1: 基础功能")
    cache = LRUCache(maxsize=3, ttl=None)
    
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")
    
    print(f"获取 key1: {cache.get('key1')}")
    print(f"获取 key2: {cache.get('key2')}")
    print(f"获取不存在的键: {cache.get('key4', 'default')}")
    
    # 测试 2: LRU 淘汰
    print("\n测试 2: LRU 淘汰")
    cache.set("key4", "value4")  # 应该淘汰 key3
    print(f"获取 key3（应该被淘汰）: {cache.get('key3', 'not found')}")
    print(f"获取 key1: {cache.get('key1')}")
    
    # 测试 3: TTL 过期
    print("\n测试 3: TTL 过期")
    cache_ttl = LRUCache(maxsize=10, ttl=1)
    cache_ttl.set("temp_key", "temp_value")
    print(f"立即获取: {cache_ttl.get('temp_key')}")
    
    import time
    time.sleep(1.5)
    print(f"1.5秒后获取（应该过期）: {cache_ttl.get('temp_key', 'expired')}")
    
    # 测试 4: 统计信息
    print("\n测试 4: 统计信息")
    stats = cache.get_stats()
    print(f"缓存统计: {stats}")
    
    # 测试 5: 装饰器
    print("\n测试 5: 装饰器")
    
    @cached(maxsize=10, ttl=None)
    def expensive_function(x, y):
        print(f"  执行函数: {x} + {y}")
        return x + y
    
    print(f"第一次调用: {expensive_function(1, 2)}")
    print(f"第二次调用（应该命中缓存）: {expensive_function(1, 2)}")
    print(f"第三次调用（不同参数）: {expensive_function(2, 3)}")
    
    print(f"\n缓存统计: {expensive_function.cache_stats()}")
    
    print("\n" + "=" * 80)
    print("缓存工具测试完成")
    print("=" * 80)
