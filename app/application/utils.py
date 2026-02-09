# -*- coding: utf-8 -*-
"""
应用层工具函数

提供超时控制、缓存、降级策略等功能。
"""
import time
import functools
from typing import Callable, Any, Optional
from datetime import datetime, timedelta


class TimeoutError(Exception):
    """超时异常"""
    pass


def with_timeout(timeout_seconds: float):
    """
    超时装饰器（简化版，仅记录时间）
    
    注意：Python 的超时控制比较复杂，这里仅做时间记录和警告
    真正的超时需要使用 threading 或 multiprocessing
    
    Args:
        timeout_seconds: 超时时间（秒）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            
            if elapsed > timeout_seconds:
                print(f"警告: {func.__name__} 执行时间 {elapsed:.2f}秒，超过限制 {timeout_seconds}秒")
            
            return result
        return wrapper
    return decorator


class SimpleCache:
    """简单的内存缓存"""
    
    def __init__(self, ttl_seconds: int = 300):
        """
        初始化缓存
        
        Args:
            ttl_seconds: 缓存过期时间（秒），默认 5 分钟
        """
        self.cache = {}
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
        
        Returns:
            缓存值，如果不存在或已过期则返回 None
        """
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        
        # 检查是否过期
        if datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds):
            del self.cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: Any):
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        self.cache[key] = (value, datetime.now())
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)


def with_cache(cache: SimpleCache, key_func: Optional[Callable] = None):
    """
    缓存装饰器
    
    Args:
        cache: 缓存对象
        key_func: 生成缓存键的函数，默认使用函数名和参数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # 尝试从缓存获取
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 存入缓存
            cache.set(cache_key, result)
            
            return result
        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, delay_seconds: float = 1.0):
    """
    失败重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay_seconds: 重试间隔（秒）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        print(f"警告: {func.__name__} 失败（第 {attempt + 1} 次），{delay_seconds}秒后重试...")
                        time.sleep(delay_seconds)
            
            # 所有重试都失败
            print(f"错误: {func.__name__} 失败，已重试 {max_retries} 次")
            raise last_exception
        return wrapper
    return decorator


# 测试代码
if __name__ == "__main__":
    print("=" * 80)
    print("应用层工具函数测试")
    print("=" * 80)
    
    # 测试 1: 超时装饰器
    print("\n测试 1: 超时装饰器")
    print("-" * 80)
    
    @with_timeout(1.0)
    def slow_function():
        time.sleep(0.5)
        return "完成"
    
    @with_timeout(0.1)
    def very_slow_function():
        time.sleep(0.5)
        return "完成"
    
    result1 = slow_function()
    print(f"快速函数: {result1}")
    
    result2 = very_slow_function()
    print(f"慢速函数: {result2}")
    
    # 测试 2: 缓存
    print("\n测试 2: 缓存")
    print("-" * 80)
    
    cache = SimpleCache(ttl_seconds=2)
    
    @with_cache(cache)
    def expensive_function(x: int) -> int:
        print(f"  执行计算: {x} * 2")
        time.sleep(0.1)
        return x * 2
    
    print("第一次调用:")
    result = expensive_function(5)
    print(f"结果: {result}")
    
    print("\n第二次调用（应该使用缓存）:")
    result = expensive_function(5)
    print(f"结果: {result}")
    
    print(f"\n缓存大小: {cache.size()}")
    
    # 测试 3: 重试
    print("\n测试 3: 重试")
    print("-" * 80)
    
    attempt_count = 0
    
    @retry_on_failure(max_retries=2, delay_seconds=0.5)
    def flaky_function():
        global attempt_count
        attempt_count += 1
        print(f"  尝试 {attempt_count}")
        if attempt_count < 3:
            raise ValueError("模拟失败")
        return "成功"
    
    try:
        result = flaky_function()
        print(f"结果: {result}")
    except Exception as e:
        print(f"最终失败: {e}")
    
    print("\n" + "=" * 80)
    print("工具函数测试完成")
    print("=" * 80)
