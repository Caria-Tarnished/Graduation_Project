# -*- coding: utf-8 -*-
"""
LLM Adapters

大语言模型适配器，支持多种 LLM 服务：
- Deepseek API
"""

from .deepseek_client import DeepseekClient

__all__ = ['DeepseekClient']
