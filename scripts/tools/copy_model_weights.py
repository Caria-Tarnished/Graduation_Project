#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Copy trained model weights from reports/ to models/ directory."""
import argparse
import shutil
from pathlib import Path


def copy_model_weights(src: str, dst: str, dry_run: bool = False, verbose: bool = False):
    """Copy model weight files from source to destination."""
    src_path = Path(src)
    dst_path = Path(dst)
    
    # Check source directory
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")
    
    if not src_path.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {src}")
    
    # Check required files
    required_files = ["config.json"]
    missing_files = [f for f in required_files if not (src_path / f).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Source directory missing required files: {', '.join(missing_files)}\n"
            f"Please ensure training is complete and synced locally"
        )
    
    # Count files
    all_files = list(src_path.rglob("*"))
    files_to_copy = [f for f in all_files if f.is_file()]
    
    if verbose:
        print(f"Source: {src_path.absolute()}")
        print(f"Destination: {dst_path.absolute()}")
        print(f"Found {len(files_to_copy)} files")
    
    # Create destination directory
    if not dry_run:
        dst_path.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    copied = 0
    for src_file in files_to_copy:
        rel_path = src_file.relative_to(src_path)
        dst_file = dst_path / rel_path
        
        if dry_run or verbose:
            print(f"{'[DRY-RUN] ' if dry_run else ''}Copy: {rel_path}")
        
        if not dry_run:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
        
        copied += 1
    
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Complete: {'Will copy' if dry_run else 'Copied'} {copied} files")
    
    if not dry_run:
        print(f"\nModel copied to: {dst_path.absolute()}")
        print("\nYou can now test the model with:")
        print(f"  python scripts/test_sentiment_analyzer.py")


def main():
    parser = argparse.ArgumentParser(
        description="Copy trained model weights from reports/ to models/"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="reports/bert_3cls_enhanced_v1/best",
        help="Source directory (default: reports/bert_3cls_enhanced_v1/best)"
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="models/bert_3cls/best",
        help="Destination directory (default: models/bert_3cls/best)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview only, do not actually copy"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logs"
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
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
