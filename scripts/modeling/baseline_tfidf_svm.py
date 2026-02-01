# -*- coding: utf-8 -*-
"""
基线文本分类（TF-IDF + LinearSVC）。

- 读取 train/val/test 三个 CSV（UTF-8），要求至少包含列：text, label。
- 可选：使用源信息前缀特征（[SRC]/[STAR]/[IMP]/[CTRY]）拼接到文本前，提升简单模型的可分性。
- 中文推荐使用字符 n-gram（char 2-4），对短文本与噪声具有更强鲁棒性。
- 输出：
  - models_dir/tfidf.pkl, models_dir/model.pkl（可复用推理）
  - models_dir/metrics_val.json, metrics_test.json
  - models_dir/report_val.txt, report_test.txt
  - models_dir/pred_test.csv（含 event_id 若存在）

注意：本脚本仅作为快速基线，CPU 可运行，便于对比后续 BERT 微调效果。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import joblib  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics import classification_report, f1_score  # type: ignore
from sklearn.svm import LinearSVC  # type: ignore


def _ensure_dir(path: str) -> None:
    """确保目录存在（中文路径兼容）。"""
    d = os.path.abspath(path)
    os.makedirs(d, exist_ok=True)


def _read_csv(path: str) -> pd.DataFrame:
    """以 UTF-8 读取 CSV，统一列名为小写；缺列时补空字符串。"""
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in ["text", "label", "source", "star", "important", "country", "event_id"]:
        if col not in df.columns:
            df[col] = ""
    return df


def _build_text(df: pd.DataFrame, use_prefix: bool) -> np.ndarray:
    """构造文本：可选加入前缀特征，降低信息丢失。

    参数：
    - use_prefix: 是否在文本前拼接形如 [SRC=flash] 的特征 token。
    """
    base = df["text"].fillna("").astype(str)
    if not use_prefix:
        return base.values
    src = df["source"].fillna("").astype(str)
    star = df["star"].fillna("").astype(str)
    imp = df["important"].fillna("").astype(str)
    ctry = df["country"].fillna("").astype(str)
    prefix = (
        "[SRC=" + src + "] "
        + "[STAR=" + star + "] "
        + "[IMP=" + imp + "] "
        + "[CTRY=" + ctry + "] "
    )
    return (prefix + base).values


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[dict, str]:
    """计算 macro-F1 与分类报告。"""
    m = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    rpt = classification_report(y_true, y_pred, digits=4)
    return m, rpt


def _to_str_list(arr: np.ndarray) -> list[str]:
    """将任意 ndarray 转为字符串列表，兜底处理 None/NaN。

    说明：尽管上游做了 fillna 与 astype(str)，但在字符串拼接等步骤中，
    仍可能出现 NaN 传播到数组中。此处再做一次最终兜底，避免
    TfidfVectorizer 抛出 "np.nan is an invalid document"。
    """
    out: list[str] = []
    for x in list(arr):
        try:
            # pandas 的 isna 可同时识别 None/NaN
            import pandas as pd  # type: ignore

            if pd.isna(x):
                out.append("")
                continue
        except Exception:
            pass
        if isinstance(x, str):
            out.append(x)
        else:
            try:
                out.append(str(x))
            except Exception:
                out.append("")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF + LinearSVC 基线分类器")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models/baseline_tfidf_svm")
    parser.add_argument("--analyzer", type=str, default="char", choices=["char", "word"], help="分词粒度")
    parser.add_argument("--ngram_min", type=int, default=2)
    parser.add_argument("--ngram_max", type=int, default=4)
    parser.add_argument("--min_df", type=int, default=3)
    parser.add_argument("--max_features", type=int, default=200_000)
    parser.add_argument("--no_prefix", action="store_true", help="不使用前缀特征")
    parser.add_argument(
        "--class_weight",
        type=str,
        default="",
        choices=["", "balanced"],
        help="类别权重：空=不使用；balanced=按类频率自动加权",
    )
    parser.add_argument("--C", type=float, default=1.0, help="LinearSVC 的正则化强度")
    parser.add_argument(
        "--sublinear_tf",
        action="store_true",
        help="TF 子线性缩放（对长文本词频做 log 缩放），默认关闭以保持兼容",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="l2",
        choices=["l2", "l1", "none"],
        help="TF-IDF 向量归一化方式（l2/l1/none）",
    )
    parser.add_argument(
        "--dual",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="LinearSVC 的 dual 选项：auto=按样本/特征数自动选择",
    )

    args = parser.parse_args()
    use_prefix = not args.no_prefix

    _ensure_dir(args.output_dir)

    # 读取数据
    train = _read_csv(args.train_csv)
    val = _read_csv(args.val_csv)
    test = _read_csv(args.test_csv)

    # 文本与标签
    X_tr = _build_text(train, use_prefix)
    y_tr = train["label"].astype(int).values

    X_va = _build_text(val, use_prefix)
    y_va = val["label"].astype(int).values

    X_te = _build_text(test, use_prefix)
    y_te = test["label"].astype(int).values

    # 向量化器（字符 n-gram 更适合中文短文本）
    vec = TfidfVectorizer(
        analyzer=args.analyzer,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_features=args.max_features,
        sublinear_tf=args.sublinear_tf,
        norm=(None if args.norm == "none" else args.norm),
    )
    Xtr = vec.fit_transform(_to_str_list(X_tr))
    Xva = vec.transform(_to_str_list(X_va))
    Xte = vec.transform(_to_str_list(X_te))

    # 线性 SVM 作为强基线
    cw = None if (args.class_weight or "").strip() == "" else args.class_weight
    if args.dual == "auto":
        dual_flag = True if Xtr.shape[0] < Xtr.shape[1] else False
    else:
        dual_flag = True if args.dual == "true" else False
    clf = LinearSVC(C=args.C, class_weight=cw, dual=dual_flag)
    clf.fit(Xtr, y_tr)

    # 评估与保存
    for split_name, X, y in [("val", Xva, y_va), ("test", Xte, y_te)]:
        pred = clf.predict(X)
        m, rpt = _metrics(y, pred)
        with open(os.path.join(args.output_dir, f"metrics_{split_name}.json"), "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.output_dir, f"report_{split_name}.txt"), "w", encoding="utf-8") as f:
            f.write(rpt)
        if split_name == "test":
            # 输出预测明细，包含 event_id（若存在）
            out = {
                "event_id": test.get("event_id", pd.Series(range(len(pred)))).astype(str),
                "label": pd.Series(y),
                "pred": pd.Series(pred),
            }
            pd.DataFrame(out).to_csv(
                os.path.join(args.output_dir, "pred_test.csv"), index=False, encoding="utf-8"
            )

    # 持久化模型与向量器
    joblib.dump(vec, os.path.join(args.output_dir, "tfidf.pkl"))
    joblib.dump(clf, os.path.join(args.output_dir, "model.pkl"))

    print("完成：模型与评估结果已保存至", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
