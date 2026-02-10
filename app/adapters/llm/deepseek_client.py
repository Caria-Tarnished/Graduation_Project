# -*- coding: utf-8 -*-
"""Deepseek LLM Client"""
import os
import time
import requests
from typing import Optional


class DeepseekClient:
    """Deepseek API Client"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/v1/chat/completions",
        model: str = "deepseek-chat"
    ):
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found")
        
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
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
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
                last_error = f"Timeout after {timeout_seconds}s"
                if attempt < retry_times:
                    time.sleep(1)
                    continue
            
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                if status_code == 401:
                    return "Error: Invalid API Key"
                elif status_code == 429:
                    last_error = "Error: Rate limit exceeded"
                    if attempt < retry_times:
                        time.sleep(2)
                        continue
                else:
                    last_error = f"HTTP {status_code}: {e.response.text}"
            
            except Exception as e:
                last_error = f"Error: {str(e)}"
        
        return last_error or "API call failed"
    
    def complete_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_seconds: float = 10.0,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
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
            return f"Error: {str(e)}"
