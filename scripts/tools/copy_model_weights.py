# -*- coding: utf-8 -*-
"""
将训练好的模型权重从 reports/ 复制到 models/ 目录

用途：
- 将 Colab 训练完成并同步到本地的模型权重复制到标准模型目录
- 便于后续推理和部署使用

使用方法：
    # 复制 3 类模型
    python scripts/tools/copy_model_weights.py --src reports/bert_3cls_enhanced_v1/best --dst models/bert_3cls/best
    
    # 查看帮助
    python scripts/tools/copy_model_weights.py --help
"""
import argparse
import os
import shutil
from pathlib import Path


def copy_model_weights(src: str, dst: str, dry_run: bool = False, verbose: bool = False):
    """
    复制模型权重文件
    
    Args:
        src: 源目录（reports/ 下的模型目录）
        dst: 目标目录（models/ 下的模型目录）
        dry_run: 是否仅预览，不实际复制
        verbose: 是否输出详细日志
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    # 检查源目录
    if not src_path.exists():
        raise FileNotFoundError(f"源目录不存在: {src}")
    
    if not src_path.is_dir():
        raise NotADirectoryError(f"源路径不是目录: {src}")
    
    # 检查必要文件
    required_files = ["config.json"]
    missing_files = [f for f in required_files if not (src_path / f).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"源目录缺少必要文件: {', '.join(missing_files)}\n"
            f"请确保已完成训练并同步到本地"
        )
    
    # 统计文件
    all_files = list(src_path.rglob("*"))
    files_to_copy = [f for f in all_files if f.is_file()]
    
    if verbose:
        print(f"源目录: {src_path.absolute()}")
        print(f"目标目录: {dst_path.absolute()}")
        print(f"找到 {len(files_to_copy)} 个文件")
    
    # 创建目标目录
    if not dry_run:
        dst_path.mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    copied = 0
    for src_file in files_to_copy:
        rel_path = src_file.relative_to(src_path)
        dst_file = dst_path / rel_path
        
        if dry_run or verbose:
            print(f"{'[DRY-RUN] ' if dry_run else ''}复制: {rel_path}")
        
        if not dry_run:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
        
        copied += 1
    
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}完成：{'将复制' if dry_run else '已复制'} {copied} 个文件")
    
    if not dry_run:
        print(f"\n模型已复制到: {dst_path.absolute()}")
        print("\n现在可以使用以下命令测试模型：")
        print(f"  python scripts/test_sentiment_analyzer.py")


def main():
    parser = argparse.ArgumentParser(
        description="将训练好的模型权重从 reports/ 复制到 models/ 目录"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="reports/bert_3cls_enhanced_v1/best",
        help="源目录（默认: reports/bert_3cls_enhanced_v1/best）"
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="models/bert_3cls/best",
        help="目标目录（默认: models/bert_3cls/best）"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅预览，不实际复制"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细日志"
    )
    
    args = parser.parse_args()
    
    try:
        copy_model_weights(
            src=args.src,
            dst=args.dst,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        return 0
    except Exception as e:
        print(f"错误: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
