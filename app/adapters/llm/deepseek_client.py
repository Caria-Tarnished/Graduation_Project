# -*- coding: utf-8 -*-
"""
Deepseek LLM 客户端

功能：
1. 调用 Deepseek API 生成文本
2. 处理超时和错误
3. 支持重试机制

使用示例：
    client = DeepseekClient()
    response = client.complete("你好，请介绍一下自己", timeout_seconds=10.0)
"""
import os
import time
import requests
from typing import Optional, Dict, Any


class DeepseekClient:
    """Deepseek API 客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/v1/chat/completions",
        model: str = "deepseek-chat"
    ):
        """
        初始化 Deepseek 客户端
        
        Args:
            api_key: API 密钥（如果为 None，从环境变量 DEEPSEEK_API_KEY 读取）
            base_url: API 基础 URL
            model: 模型名称
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not found. "
                "Please set it in .env file or pass it as parameter."
            )
        
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def complete(
        self,
        prompt: str,
        timeout_seconds: float = 10.0,
        temperature: float = 0.7,
        max_tokens: int = 500,
        retry_times: int = 2
    ) -> str:
        """
        调用 Deepseek API 生成文本
        
        Args:
            prompt: 输入提示词
            timeout_seconds: 超时时间（秒）
            temperature: 温度参数（0-1，越高越随机）
            max_tokens: 最大生成 token 数
            retry_times: 重试次数
        
        Returns:
            生成的文本
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        last_error = None
        
        for attempt in range(retry_times + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=timeout_seconds
                )
                response.raise_for_status()
                
                result = response.json()
                return result['choices'][0]['message']['content']
            
            except requests.exceptions.Timeout:
                last_error = f"[超时] Deepseek API 响应超时（{timeout_seconds}秒）"
                if attempt < retry_times:
                    time.sleep(1)  # 等待 1 秒后重试
                    continue
            
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                if status_code == 401:
                    return "[错误] API Key 无效，请检查 DEEPSEEK_API_KEY"
                elif status_code == 429:
                    last_error = "[错误] API 调用频率超限，请稍后重试"
                    if attempt < retry_times:
                        time.sleep(2)  # 等待 2 秒后重试
                        continue
                else:
                    last_error = f"[错误] HTTP {status_code}: {e.response.text}"
            
            except requests.exceptions.RequestException as e:
                last_error = f"[错误] 网络请求失败: {str(e)}"
            
            except (KeyError, IndexError) as e:
                last_error = f"[错误] API 响应格式异常: {str(e)}"
            
            except Exception as e:
                last_error = f"[错误] 未知错误: {str(e)}"
        
        # 所有重试都失败，返回错误信息
        return last_error or "[错误] API 调用失败"
    
    def complete_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_seconds: float = 10.0,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        使用系统提示词调用 API
        
        Args:
            system_prompt: 系统提示词（定义 AI 角色）
            user_prompt: 用户提示词
            timeout_seconds: 超时时间
            temperature: 温度参数
            max_tokens: 最大生成 token 数
        
        Returns:
            生成的文本
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=timeout_seconds
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
        
        except Exception as e:
            return f"[错误] API 调用失败: {str(e)}"


# 测试代码
if __name__ == "__main__":
    print("=" * 80)
    print("Deepseek LLM 客户端测试")
    print("=" * 80)
    
    # 检查环境变量
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("\n⚠ 警告: 未设置 DEEPSEEK_API_KEY 环境变量")
        print("请在 .env 文件中添加: DEEPSEEK_API_KEY=your_key_here")
        exit(1)
    
    try:
        # 初始化客户端
        client = DeepseekClient()
        print("\n✓ Deepseek 客户端初始化成功")
        
        # 测试 1: 简单问答
        print("\n" + "=" * 80)
        print("测试 1: 简单问答")
        print("=" * 80)
        
        prompt = "请用一句话介绍什么是财经分析。"
        print(f"\n提示词: {prompt}")
        
        response = client.complete(prompt, timeout_seconds=10.0)
        print(f"\n回复: {response}")
        
        # 测试 2: 带系统提示词
        print("\n" + "=" * 80)
        print("测试 2: 带系统提示词")
        print("=" * 80)
        
        system_prompt = "你是一个专业的财经分析师，擅长解读市场动态。"
        user_prompt = "美联储加息对黄金价格有什么影响？请用2-3句话简要说明。"
        
        print(f"\n系统提示词: {system_prompt}")
        print(f"用户提示词: {user_prompt}")
        
        response = client.complete_with_system(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_seconds=10.0
        )
        print(f"\n回复: {response}")
        
        print("\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80)
    
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
