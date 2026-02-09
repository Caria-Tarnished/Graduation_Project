# -*- coding: utf-8 -*-
"""
手动下载 bge-m3 模型

使用 HuggingFace 镜像站加速下载
"""
import os
from huggingface_hub import snapshot_download

# 设置镜像站（可选，如果官方站点慢）
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("开始下载 bge-m3 模型...")
print("模型大小约 2.27GB，请耐心等待...")
print()

try:
    model_path = snapshot_download(
        repo_id="BAAI/bge-m3",
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
        resume_download=True,  # 支持断点续传
        local_files_only=False
    )
    print()
    print(f"✓ 下载完成！模型路径: {model_path}")
    print()
    print("现在可以运行向量化脚本：")
    print("python scripts/rag/build_vector_index.py --input data/reports/chunks.json --output data/reports/chroma_db")
except Exception as e:
    print(f"✗ 下载失败: {e}")
    print()
    print("如果下载速度太慢，可以尝试：")
    print("1. 取消注释第 9 行，使用国内镜像站")
    print("2. 或者使用方案 B（浏览器手动下载）")
