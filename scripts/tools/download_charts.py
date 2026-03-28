# -*- coding: utf-8 -*-
"""
一键从本地挂载的 Google Drive 将最新生成的三张 RAG 消融实验图表
原封不动同步覆盖到本地仓库里 (thesis_assets/charts)
"""
import os
import shutil
from pathlib import Path

def find_google_drive_charts_dir():
    home = os.path.expanduser("~")
    candidates = [
        r"G:\我的云端硬盘\Graduation_Project\thesis_assets\charts",
        r"G:\My Drive\Graduation_Project\thesis_assets\charts",
        os.path.join(home, "Google Drive", "我的云端硬盘", "Graduation_Project", "thesis_assets", "charts"),
        os.path.join(home, "Google Drive", "My Drive", "Graduation_Project", "thesis_assets", "charts"),
    ]
    
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def main():
    print("开始自动扫描本地计算机挂载的 Google Drive 缓存...")
    src_dir = find_google_drive_charts_dir()
    
    if not src_dir:
        print("❌ 错误：未能在常见路径 (G: 等) 下找到 Google Drive 的图表。")
        print("请确保本地 Windows 系统已登录并开启了 Google Drive 同步，且 Colab 那边的脚本已经把图存进去了！")
        return
        
    print(f"✅ 定位到云端图表目录：{src_dir}")
    
    # 获取此脚本所在项目的绝对路径作为目标
    # 当前文件位于 e:\Projects\Graduation_Project\scripts\tools\download_charts.py
    repo_root = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
    dst_dir = os.path.join(repo_root, "thesis_assets", "charts")
    os.makedirs(dst_dir, exist_ok=True)
    
    print(f"📦 目标拉取目录：{dst_dir}")
    
    # 查找所有图片并复制
    success_count = 0
    for file_name in os.listdir(src_dir):
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.svg')):
            src_file = os.path.join(src_dir, file_name)
            dst_file = os.path.join(dst_dir, file_name)
            
            try:
                shutil.copy2(src_file, dst_file)
                print(f"  --> 已覆盖同步图表: {file_name}")
                success_count += 1
            except Exception as e:
                print(f"  --> 同步失败 {file_name}: {e}")
                
    if success_count > 0:
        print(f"\n🎉 搞定！你可以在目前 VS Code 或本地文件夹的 `thesis_assets/charts/` 下查看最新的 {success_count} 张图表了。")
    else:
        print("\n⚠️ 目标文件夹里没找到图片，是不是 Colab 那边图还没画完或者同步盘还没刷新？")

if __name__ == "__main__":
    main()
