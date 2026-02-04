# -*- coding: utf-8 -*-
"""
将 Google Drive 上的训练结果（experiments 下的运行目录）
同步到本地代码仓库的 reports/ 目录，便于分析与调参。

特点：
- 仅同步关键小文件（指标、报告、配置、预测明细），默认排除大权重文件。
- 按相对路径保持目录结构：experiments/.../file -> reports/.../file。
- 若目标文件已存在且大小一致，则跳过复制（避免重复 IO）。
- 支持 dry-run 预览。

示例（Windows）：
    python scripts/tools/sync_results.py \
        --src_root "G:\\我的云端硬盘\\Graduation_Project\\experiments" \
        --dst_root "E:\\Projects\\Graduation_Project\\reports" \
        --dry_run

Colab 侧建议：
- 将 output_dir 指向 /content/drive/MyDrive/Graduation_Project/experiments/<run_name>
- 训练脚本已自动输出 eval_results.json/metrics_*.json/report_test.txt/pred_test.csv/best/config.json

注意：本脚本不会删除任何文件，仅执行有条件的复制。
"""
import argparse
import fnmatch
import os
import shutil
from typing import Iterable, List, Tuple


def _repo_root_default() -> str:
    """推断仓库根目录（以本脚本相对路径回退）。"""
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.normpath(os.path.join(here, "..", ".."))


def _default_src_root() -> str:
    """根据操作系统给出一个合理的 Drive 默认路径。"""
    if os.name == "nt":
        return r"G:\\我的云端硬盘\\Graduation_Project\\experiments"
    # 非 Windows，优先 MyDrive 路径
    return "/content/drive/MyDrive/Graduation_Project/experiments"


def _default_dst_root() -> str:
    return os.path.join(_repo_root_default(), "reports")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _match_any(path_posix: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path_posix, pat) for pat in patterns)


def _rel_posix(path: str, start: str) -> str:
    rel = os.path.relpath(path, start)
    return rel.replace(os.sep, "/")


def _should_copy(src: str, dst: str) -> bool:
    if not os.path.exists(dst):
        return True
    try:
        s_stat = os.stat(src)
        d_stat = os.stat(dst)
        return s_stat.st_size != d_stat.st_size
    except OSError:
        return True


def _walk_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            yield os.path.join(dirpath, f)


def sync(
    src_root: str,
    dst_root: str,
    includes: List[str],
    excludes: List[str],
    dry_run: bool = False,
    verbose: bool = False,
) -> Tuple[int, int]:
    copied = 0
    skipped = 0
    src_root = os.path.abspath(src_root)
    dst_root = os.path.abspath(dst_root)

    for src in _walk_files(src_root):
        rel = _rel_posix(src, src_root)
        # 只在 include 匹配且不在 exclude 时复制
        if not _match_any(rel, includes):
            continue
        if excludes and _match_any(rel, excludes):
            continue

        dst = os.path.join(dst_root, rel)
        _ensure_dir(os.path.dirname(dst))

        if _should_copy(src, dst):
            if dry_run or verbose:
                print("COPY:", src, "->", dst)
            if not dry_run:
                shutil.copy2(src, dst)
            copied += 1
        else:
            if verbose:
                print("SKIP:", src)
            skipped += 1

    return copied, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="同步 Drive 训练结果到本地 reports 目录")
    parser.add_argument("--src_root", type=str, default=_default_src_root())
    parser.add_argument("--dst_root", type=str, default=_default_dst_root())
    parser.add_argument(
        "--include",
        action="append",
        help="包含的通配模式（可多次传入）",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        help="排除的通配模式（可多次传入）",
    )
    parser.add_argument("--dry_run", action="store_true", help="仅预览，将不实际复制")
    parser.add_argument("--verbose", action="store_true", help="输出详细日志")
    args = parser.parse_args()

    # 默认包含与排除模式
    default_includes = [
        "**/eval_results.json",
        "**/metrics*.json",
        "**/report*.txt",
        "**/pred*.csv",
        "**/best/config.json",
    ]
    default_excludes = [
        "**/*.bin",
        "**/*.safetensors",
        "**/*.pt",
        "**/pytorch_model*",
        "**/optimizer*",
        "**/checkpoint-*/**",
    ]

    includes = args.include if args.include else default_includes
    excludes = args.exclude if args.exclude else default_excludes

    if args.verbose:
        print("SRC=", args.src_root)
        print("DST=", args.dst_root)
        print("INCLUDES=", includes)
        print("EXCLUDES=", excludes)

    copied, skipped = sync(
        src_root=args.src_root,
        dst_root=args.dst_root,
        includes=includes,
        excludes=excludes,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    print(f"完成：复制 {copied} 个文件，跳过 {skipped} 个文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
