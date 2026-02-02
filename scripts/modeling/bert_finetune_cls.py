# -*- coding: utf-8 -*-
"""
BERT 中文文本分类微调脚本（财经快讯/日历 30 分钟窗口标签）。

- 输入：train/val/test 三个 CSV（UTF-8），至少包含列：text, label。
- 标签映射：{-1,0,1} → {0,1,2}，保证交叉熵训练的类别索引从 0 开始。
- 可选前缀特征：[SRC]/[STAR]/[IMP]/[CTRY] 拼接至文本前，提升非语义结构信息的利用。
- 默认模型：hfl/chinese-roberta-wwm-ext。
- 输出：
  - 最优模型目录（args.output_dir/best）
  - 评估指标 JSON：val/test（accuracy、macro_f1）
  - 测试集预测明细 CSV：含 event_id（若存在）与预测标签（-1/0/1 范畴映射回原值）。

注意：
- 需先安装 transformers、datasets、evaluate、torch；CPU 亦可运行（较慢）。
- 本脚本仅作为基线/对照，便于与 TF-IDF + LinearSVC 对比。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from datasets import Dataset, DatasetDict  # type: ignore
from sklearn.metrics import classification_report, f1_score  # type: ignore
from transformers import (  # type: ignore
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import evaluate  # type: ignore


def _ensure_dir(path: str) -> None:
    """确保目录存在（中文路径兼容）。"""
    os.makedirs(path, exist_ok=True)


def _read_csv(path: str) -> pd.DataFrame:
    """以 UTF-8 读取 CSV，统一列名为小写；缺列时补空字符串。"""
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in ["text", "label", "source", "star", "important", "country", "event_id"]:
        if col not in df.columns:
            df[col] = ""
    return df


def _build_text(df: pd.DataFrame, use_prefix: bool) -> pd.Series:
    """构造文本：可选加入前缀特征，降低信息丢失。

    参数：
    - use_prefix: 是否在文本前拼接形如 [SRC=flash] 的特征 token。
    """
    base = df["text"].fillna("").astype(str)
    if not use_prefix:
        return base
    src = df["source"].fillna("").astype(str)
    star = df["star"].astype(str)
    imp = df["important"].astype(str)
    ctry = df["country"].astype(str)
    prefix = (
        "[SRC=" + src + "] "
        + "[STAR=" + star + "] "
        + "[IMP=" + imp + "] "
        + "[CTRY=" + ctry + "] "
    )
    return (prefix + base)


def _map_label_series(y: pd.Series) -> pd.Series:
    """将 {-1,0,1} → {0,1,2}，用于交叉熵；并保留原值供回映射。"""
    mp = {-1: 0, 0: 1, 1: 2}
    return y.astype(int).map(mp)


def _inv_map_label_array(y_hat: np.ndarray) -> np.ndarray:
    """将 {0,1,2} → {-1,0,1}。"""
    inv = {0: -1, 1: 0, 2: 1}
    return np.vectorize(lambda x: inv.get(int(x), 0))(y_hat)


def _compute_metrics_builder() -> callable:
    """构造 metrics 函数：accuracy 与 macro_f1。"""
    acc = evaluate.load("accuracy")

    def _fn(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        # 直接在 {0,1,2} 空间计算
        return {
            "accuracy": float(acc.compute(predictions=preds, references=labels)["accuracy"]),
            "macro_f1": float(f1_score(labels, preds, average="macro")),
        }

    return _fn


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT 中文文本分类微调脚本")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models/bert_xauusd_cls")
    parser.add_argument("--model_name", type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_bs", type=int, default=8)
    parser.add_argument("--eval_bs", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--no_prefix", action="store_true", help="不使用前缀特征")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_col", type=str, default="label")

    args = parser.parse_args()
    use_prefix = not args.no_prefix

    _ensure_dir(args.output_dir)

    # 读取数据并构造文本
    train = _read_csv(args.train_csv)
    val = _read_csv(args.val_csv)
    test = _read_csv(args.test_csv)

    # 文本
    for df in (train, val, test):
        df["text2"] = _build_text(df, use_prefix)

    # 标签列与映射
    lbl_col = args.label_col
    y_train = train[lbl_col].astype(int)
    y_val = val[lbl_col].astype(int)
    y_test = test[lbl_col].astype(int)
    classes_sorted = sorted(set(list(y_train.values) + list(y_val.values) + list(y_test.values)))
    if classes_sorted == [-1, 0, 1]:
        mp = {-1: 0, 0: 1, 1: 2}
    else:
        mp = {v: i for i, v in enumerate(classes_sorted)}
    inv = {i: v for v, i in mp.items()}
    for df in (train, val, test):
        df["label_mapped"] = df[lbl_col].astype(int).map(mp)
        df["labels"] = df["label_mapped"]

    # 转为 HF Datasets
    ds = DatasetDict(
        {
            "train": Dataset.from_pandas(train[["text2", "labels"]]),
            "val": Dataset.from_pandas(val[["text2", "labels"]]),
            "test": Dataset.from_pandas(test[["text2", "labels"]]),
        }
    )

    # 分词器与编码
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tok_fn(batch):
        texts = batch["text2"]
        if not isinstance(texts, list):
            texts = [texts]
        # 兜底转为字符串，处理 None/NaN/其他类型
        texts = ["" if (t is None) else (t if isinstance(t, str) else str(t)) for t in texts]
        return tok(
            texts,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text2"])

    # 模型与训练器（允许分类头尺寸不匹配，如从 3 类迁移到 5 类）
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(mp),
            ignore_mismatched_sizes=True,
        )
    except TypeError:
        # 兼容旧版 transformers 不支持 ignore_mismatched_sizes 的情况
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(mp),
        )

    # 兼容老版本 transformers：若不支持 evaluation_strategy 等参数，则退化为最小参数集
    try:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.train_bs,
            per_device_eval_batch_size=args.eval_bs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            save_total_limit=2,
            logging_steps=50,
            report_to=[],  # 关闭 wandb 等外部上报，避免无意联网
            seed=args.seed,
        )
    except TypeError:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.train_bs,
            per_device_eval_batch_size=args.eval_bs,
            seed=args.seed,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["val"],
        tokenizer=tok,
        compute_metrics=_compute_metrics_builder(),
    )

    trainer.train()

    # 评估：val 与 test
    metrics_val = trainer.evaluate(ds_tok["val"])  # 在 {0,1,2} 空间
    with open(os.path.join(args.output_dir, "metrics_val.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics_val.items()}, f, ensure_ascii=False, indent=2)

    metrics_test = trainer.evaluate(ds_tok["test"])  # 在 {0,1,2} 空间
    with open(os.path.join(args.output_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics_test.items()}, f, ensure_ascii=False, indent=2)

    # 生成测试集分类报告与预测明细（映射回 {-1,0,1}）
    preds = trainer.predict(ds_tok["test"]).predictions
    yhat_idx = np.argmax(preds, axis=-1)
    yhat = np.vectorize(lambda x: inv.get(int(x), 0))(yhat_idx)
    ytrue = test[lbl_col].astype(int).values

    rpt = classification_report(ytrue, yhat, digits=4)
    with open(os.path.join(args.output_dir, "report_test.txt"), "w", encoding="utf-8") as f:
        f.write(rpt)

    # 输出预测明细（含 event_id 若存在）
    out = {
        "event_id": test.get("event_id", pd.Series(range(len(yhat)))).astype(str),
        "label": pd.Series(ytrue),
        "pred": pd.Series(yhat.astype(int)),
    }
    pd.DataFrame(out).to_csv(
        os.path.join(args.output_dir, "pred_test.csv"), index=False, encoding="utf-8"
    )

    # 保存最优模型
    best_dir = os.path.join(args.output_dir, "best")
    _ensure_dir(best_dir)
    trainer.save_model(best_dir)
    print("完成：模型与评估结果已保存至", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
