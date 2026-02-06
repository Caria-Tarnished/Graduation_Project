# -*- coding: utf-8 -*-
"""同步本地数据集目录到 Google Drive，并进行版本化管理。

目标目录结构（drive_root/dataset_name 下）：
- versions/<version>/...   # 版本快照（可回滚）
- latest/...               # 指向当前最新版本的“工作集”（Colab 永远读这里）
- latest_version.txt       # 记录 latest 对应的版本号

说明：
- 默认会镜像同步（mirror）到 versions/<version> 与 latest：
  - 会复制/覆盖变化的文件
  - 会删除目标目录中“源目录不存在”的旧文件（仅限该数据集目录内）
- 如果你不希望删除 latest 中的历史文件，可传入 --no_mirror_latest。

用法示例（Windows）：
    python scripts/tools/sync_dataset_to_drive.py \
        --src_dir "<repo_root>\\data\\processed_stale300" \
        --drive_root "<drive_root>\\Graduation_Project\\datasets" \
        --dataset_name processed_stale300
Colab 侧读取：
    /content/drive/MyDrive/Graduation_Project/datasets/processed_stale300/
    latest/train_enhanced.csv
"""
import argparse
import os
import shutil
import time
from dataclasses import dataclass
from typing import Iterable, Set


def _repo_root_default() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.normpath(os.path.join(here, "..", ".."))


def _default_drive_root() -> str:
    if os.name == "nt":
        candidates = [
            r"G:\\我的云端硬盘\\Graduation_Project\\datasets",
            r"G:\\My Drive\\Graduation_Project\\datasets",
        ]
        for p in candidates:
            if os.path.isdir(p):
                return p
        return candidates[0]
    return "/content/drive/MyDrive/Graduation_Project/datasets"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _walk_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            yield os.path.join(dirpath, f)


def _rel_posix(path: str, start: str) -> str:
    rel = os.path.relpath(path, start)
    return rel.replace(os.sep, "/")


def _should_copy(src: str, dst: str) -> bool:
    if not os.path.exists(dst):
        return True
    try:
        s_stat = os.stat(src)
        # 以大小和 mtime 做快速判断；Drive 文件系统的时间戳可能有轻微漂移，允许只要大小一致就跳过
        if s_stat.st_size != os.stat(dst).st_size:
            return True
        return False
    except OSError:
        return True


@dataclass
class SyncStats:
    copied: int = 0
    skipped: int = 0
    deleted: int = 0


def sync_tree(
    src_root: str,
    dst_root: str,
    mirror: bool,
    dry_run: bool,
    verbose: bool,
) -> SyncStats:
    src_root = os.path.abspath(src_root)
    dst_root = os.path.abspath(dst_root)

    stats = SyncStats()
    expected: Set[str] = set()

    for src in _walk_files(src_root):
        rel = _rel_posix(src, src_root)
        expected.add(rel)

        dst = os.path.join(dst_root, rel.replace("/", os.sep))
        _ensure_dir(os.path.dirname(dst))

        if _should_copy(src, dst):
            if dry_run or verbose:
                print("COPY:", src, "->", dst)
            if not dry_run:
                shutil.copy2(src, dst)
            stats.copied += 1
        else:
            if verbose:
                print("SKIP:", src)
            stats.skipped += 1

    if mirror and os.path.isdir(dst_root):
        for dst in _walk_files(dst_root):
            rel = _rel_posix(dst, dst_root)
            if rel in expected:
                continue
            if dry_run or verbose:
                print("DELETE:", dst)
            if not dry_run:
                try:
                    os.remove(dst)
                except OSError:
                    pass
            stats.deleted += 1

        # 清理空目录（自底向上）
        for dirpath, dirnames, filenames in os.walk(dst_root, topdown=False):
            if dirnames or filenames:
                continue
            if dry_run or verbose:
                print("RMDIR:", dirpath)
            if not dry_run:
                try:
                    os.rmdir(dirpath)
                except OSError:
                    pass

    return stats


def _write_text(path: str, text: str, dry_run: bool) -> None:
    if dry_run:
        print("WRITE:", path)
        return
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="同步本地数据集到 Drive（版本化 + latest）"
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default=os.path.join(_repo_root_default(), "data", "processed_stale300"),
        help="本地数据集目录（包含 train/val/test CSV 等）",
    )
    parser.add_argument(
        "--drive_root",
        type=str,
        default=_default_drive_root(),
        help=(
            "Drive 上 datasets 根目录（Windows 形如 G:\\我的云端硬盘\\...）"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="数据集名称（默认取 src_dir 的目录名）",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="版本号（默认自动生成 YYYYmmdd_HHMMSS）",
    )
    parser.add_argument("--dry_run", action="store_true", help="仅预览，不实际复制/删除")
    parser.add_argument("--verbose", action="store_true", help="输出详细日志")
    parser.add_argument(
        "--no_mirror_latest",
        action="store_true",
        help="不同步删除 latest 中多余文件（默认 mirror，会删除旧文件）",
    )
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src_dir)
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"src_dir not found: {src_dir}")

    dataset_name = args.dataset_name or os.path.basename(src_dir.rstrip("\\/"))
    version = args.version or time.strftime("%Y%m%d_%H%M%S")

    dataset_root = os.path.join(os.path.abspath(args.drive_root), dataset_name)
    versions_root = os.path.join(dataset_root, "versions")
    version_dir = os.path.join(versions_root, version)
    latest_dir = os.path.join(dataset_root, "latest")

    if args.verbose:
        print("SRC=", src_dir)
        print("DRIVE_ROOT=", os.path.abspath(args.drive_root))
        print("DATASET_ROOT=", dataset_root)
        print("VERSION=", version)
        print("VERSION_DIR=", version_dir)
        print("LATEST_DIR=", latest_dir)

    _ensure_dir(version_dir)
    _ensure_dir(latest_dir)

    # 1) 写入版本快照（可回滚）
    st1 = sync_tree(
        src_root=src_dir,
        dst_root=version_dir,
        mirror=True,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    # 2) 更新 latest（Colab 永远读 latest）
    st2 = sync_tree(
        src_root=src_dir,
        dst_root=latest_dir,
        mirror=not args.no_mirror_latest,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    # 3) 更新指针文件
    _write_text(
        os.path.join(dataset_root, "latest_version.txt"),
        version + "\n",
        args.dry_run,
    )

    print(
        "完成：\n"
        f"- versions/{version}: copied={st1.copied}, "
        f"skipped={st1.skipped}, deleted={st1.deleted}\n"
        f"- latest: copied={st2.copied}, skipped={st2.skipped}, "
        f"deleted={st2.deleted}\n"
        f"- latest_version.txt = {version}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
