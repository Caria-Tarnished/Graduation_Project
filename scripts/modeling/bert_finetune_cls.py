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
import datasets  # type: ignore
from datasets import Dataset, DatasetDict  # type: ignore
from sklearn.metrics import classification_report, f1_score  # type: ignore
from transformers import (  # type: ignore
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging  # type: ignore
import evaluate  # type: ignore
import torch  # type: ignore
try:
    from transformers import EarlyStoppingCallback  # type: ignore
except Exception:  # 兼容旧版 transformers 无 EarlyStoppingCallback 的情况
    class EarlyStoppingCallback:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            pass


def _ensure_dir(path: str) -> None:
    """确保目录存在（中文路径兼容）。"""
    os.makedirs(path, exist_ok=True)


def _read_csv(path: str) -> pd.DataFrame:
    """以 UTF-8 读取 CSV，统一列名为小写；缺列时补空字符串。"""
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in [
        "text",
        "text_enhanced",
        "label",
        "source",
        "star",
        "important",
        "country",
        "event_id",
    ]:
        if col not in df.columns:
            df[col] = ""
    return df


def _build_text(df: pd.DataFrame, use_prefix: bool, text_col: str = "text") -> pd.Series:
    """构造文本：可选加入前缀特征，降低信息丢失。

    参数：
    - use_prefix: 是否在文本前拼接形如 [SRC=flash] 的特征 token。
    """
    base = df[text_col].fillna("").astype(str)
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


def _to_hf_dataset(df: pd.DataFrame) -> Dataset:
    """将 Pandas DataFrame 安全转换为 HF Dataset，兼容旧版 datasets 无 preserve_index 的情况。"""
    try:
        return Dataset.from_pandas(df, preserve_index=False)
    except TypeError:
        return Dataset.from_pandas(df)


class WeightedTrainer(Trainer):
    """带类权重的 Trainer，实现对少数类的损失放大。

    - 若未提供 `class_weights`，则退化为标准 CrossEntropyLoss。
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        # 兼容旧版 transformers.Trainer 不支持 tokenizer/callbacks 等参数
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            # 渐进式丢弃不被支持的关键字参数
            for k in ("tokenizer", "callbacks"):
                if k in kwargs:
                    kwargs.pop(k, None)
                    try:
                        super().__init__(*args, **kwargs)
                        break
                    except TypeError:
                        continue
            else:
                # 仍失败则直接再次触发异常，暴露真实问题
                super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 使用自定义的带权重交叉熵计算损失（中文数据需确保 labels 为整型索引）
        # 兼容新版 transformers 的 num_items_in_batch 参数
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        if labels is not None:
            # 明确转为 long，避免 labels 类型异常导致的 GPU 端错误
            try:
                labels = labels.long()
            except Exception:
                pass
            if self.class_weights is not None:
                loss_fn = torch.nn.CrossEntropyLoss(
                    weight=self.class_weights.to(logits.device)
                )
            else:
                loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            loss = outputs["loss"]
        if return_outputs:
            return loss, outputs
        return loss


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
    parser.add_argument(
        "--text_col",
        type=str,
        default="text",
        help="用于训练的文本列名（例如 text_enhanced）",
    )
    # 训练增强参数（默认与旧版兼容）
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Warmup ratio (deprecated, use warmup_steps instead)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps (recommended over warmup_ratio)",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1
    )
    parser.add_argument(
        "--class_weight", type=str, default="none", choices=["none", "auto"]
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="禁用训练/数据集处理过程的 tqdm 进度条输出（适用于 Colab，减少输出与缓存压力）",
    )

    args = parser.parse_args()
    use_prefix = not args.no_prefix

    # 计算 warmup_steps（如果用户提供了 warmup_ratio）
    # 优先使用 warmup_steps，如果为 0 则从 warmup_ratio 计算
    if args.warmup_steps == 0 and args.warmup_ratio > 0:
        # 稍后在知道总步数后计算
        use_warmup_ratio = True
    else:
        use_warmup_ratio = False

    # 禁用 HuggingFace 的警告日志（减少输出噪音）
    hf_logging.set_verbosity_error()
    
    # 禁用 transformers 的自动转换线程（避免 403 错误）
    os.environ["TRANSFORMERS_OFFLINE"] = "0"  # 允许在线，但不强制转换
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"  # 禁用隐式 token
    text_col = str(args.text_col).strip().lower() if args.text_col else "text"

    if bool(getattr(args, "disable_tqdm", False)):
        try:
            hf_logging.disable_progress_bar()
        except Exception:
            pass
        try:
            datasets.disable_progress_bars()
        except Exception:
            pass

    _ensure_dir(args.output_dir)

    # 读取数据并构造文本
    train = _read_csv(args.train_csv)
    val = _read_csv(args.val_csv)
    test = _read_csv(args.test_csv)

    # 文本
    for df in (train, val, test):
        if text_col not in df.columns:
            df[text_col] = ""
        df["text2"] = _build_text(df, use_prefix, text_col=text_col)

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

    # 训练前做标签合法性校验：避免 GPU 端 CrossEntropy 因 label 越界触发非法内存访问
    for name, df in (("train", train), ("val", val), ("test", test)):
        if df["labels"].isna().any():
            bad = df.loc[df["labels"].isna(), lbl_col].head(20).tolist()
            raise ValueError(f"{name} contains unmapped labels. examples={bad}")
        df["labels"] = df["labels"].astype(int)
        mn = int(df["labels"].min()) if len(df) > 0 else 0
        mx = int(df["labels"].max()) if len(df) > 0 else 0
        if mn < 0 or mx >= len(mp):
            raise ValueError(
                f"{name} label out of range: min={mn}, max={mx}, num_labels={len(mp)}"
            )

    # 转为 HF Datasets
    ds = DatasetDict(
        {
            "train": _to_hf_dataset(train[["text2", "labels"]]),
            "val": _to_hf_dataset(val[["text2", "labels"]]),
            "test": _to_hf_dataset(test[["text2", "labels"]]),
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

    # 类权重计算：若启用 --class_weight auto，则按训练集频次的反比例计算。
    class_weights = None
    if args.class_weight != "none":
        vc = train["labels"].value_counts().reindex(range(len(mp)), fill_value=0).astype(float)
        total = float(vc.sum()) if float(vc.sum()) > 0 else 1.0
        num_c = len(vc)
        # 经典平衡权重：N / (K * n_i)。若某类在训练集计数为 0，则权重置 0（训练中不会出现该类标签）。
        w = np.where(vc.values > 0, total / (num_c * vc.values), 0.0)
        class_weights = torch.tensor(w, dtype=torch.float)

    # 模型与训练器（允许分类头尺寸不匹配，如从 3 类迁移到 5 类）
    # 禁用模型加载时的警告
    import warnings
    warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
    warnings.filterwarnings("ignore", message=".*MISSING.*")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(mp),
            ignore_mismatched_sizes=True,
            use_safetensors=False,
        )
    except TypeError:
        # 兼容旧版 transformers 不支持 ignore_mismatched_sizes 的情况
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(mp),
        )

    # 计算总训练步数（用于 warmup_steps）
    total_steps = (len(train_df) // args.train_bs) * args.epochs
    if use_warmup_ratio:
        calculated_warmup_steps = int(total_steps * args.warmup_ratio)
        print(f"计算 warmup_steps: {calculated_warmup_steps} (total_steps={total_steps}, warmup_ratio={args.warmup_ratio})")
    else:
        calculated_warmup_steps = args.warmup_steps

    # 兼容老版本 transformers：若不支持部分参数，则退化为最小参数集
    try:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.train_bs,
            per_device_eval_batch_size=args.eval_bs,
            eval_strategy="steps",  # 新版本使用 eval_strategy 而非 evaluation_strategy
            save_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=2,
            logging_steps=50,
            report_to=[],  # 关闭 wandb 等外部上报，避免无意联网
            seed=args.seed,
            warmup_steps=calculated_warmup_steps,  # 使用 warmup_steps 而非 warmup_ratio
            weight_decay=args.weight_decay,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            dataloader_pin_memory=False,  # CPU 模式下禁用 pin_memory
        )
    except TypeError:
        # 如果 eval_strategy 不支持，尝试旧版本的 evaluation_strategy
        try:
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                learning_rate=args.lr,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.train_bs,
                per_device_eval_batch_size=args.eval_bs,
                evaluation_strategy="steps",
                save_strategy="steps",
                eval_steps=args.eval_steps,
                save_steps=args.save_steps,
                load_best_model_at_end=True,
                metric_for_best_model="macro_f1",
                greater_is_better=True,
                save_total_limit=2,
                logging_steps=50,
                report_to=[],
                seed=args.seed,
                warmup_steps=calculated_warmup_steps,  # 使用 warmup_steps 而非 warmup_ratio
                weight_decay=args.weight_decay,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                dataloader_pin_memory=False,  # CPU 模式下禁用 pin_memory
            )
        except TypeError:
            # 最小参数集
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                learning_rate=args.lr,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.train_bs,
                per_device_eval_batch_size=args.eval_bs,
                seed=args.seed,
            )
            # 手动设置必要字段
            try:
                setattr(training_args, "metric_for_best_model", "macro_f1")
                setattr(training_args, "greater_is_better", True)
                setattr(training_args, "load_best_model_at_end", True)
                setattr(training_args, "eval_strategy", "steps")
                setattr(training_args, "evaluation_strategy", "steps")
                setattr(training_args, "save_strategy", "steps")
                setattr(training_args, "eval_steps", args.eval_steps)
                setattr(training_args, "save_steps", args.save_steps)
                setattr(training_args, "logging_steps", 50)
                setattr(training_args, "report_to", [])
            except Exception:
                pass

    # 提前停止回调（若可用）
    callbacks = []
    try:
        # 仅当具备 metric_for_best_model 且启用了 load_best_model_at_end 且有 evaluation_strategy 时再启用早停
        # 明确检查 patience > 0，避免 0 值被误判
        has_eval_strategy = (
            getattr(training_args, "evaluation_strategy", None) == "steps" or
            getattr(training_args, "eval_strategy", None) == "steps"
        )
        if (
            args.early_stopping_patience is not None
            and args.early_stopping_patience > 0
            and getattr(training_args, "metric_for_best_model", None)
            and getattr(training_args, "load_best_model_at_end", False)
            and has_eval_strategy
        ):
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
            )
    except Exception:
        pass

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["val"],
        tokenizer=tok,
        compute_metrics=_compute_metrics_builder(),
        class_weights=class_weights,
        callbacks=callbacks,
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
    # 兼容旧版 Trainer 未持有 tokenizer：显式保存分词器，便于下游加载
    try:
        tok.save_pretrained(best_dir)
    except Exception:
        pass
    # 汇总评估结果到固定文件，便于本地 Agent/脚本监控（Drive 自动同步）
    try:
        summary = {
            "val": {
                "macro_f1": float(metrics_val.get("eval_macro_f1", metrics_val.get("macro_f1", 0.0))),
                "accuracy": float(metrics_val.get("eval_accuracy", metrics_val.get("accuracy", 0.0))),
                "loss": float(metrics_val.get("eval_loss", metrics_val.get("loss", 0.0))),
            },
            "test": {
                "macro_f1": float(metrics_test.get("eval_macro_f1", metrics_test.get("macro_f1", 0.0))),
                "accuracy": float(metrics_test.get("eval_accuracy", metrics_test.get("accuracy", 0.0))),
                "loss": float(metrics_test.get("eval_loss", metrics_test.get("loss", 0.0))),
            },
            "output_dir": os.path.abspath(args.output_dir),
            "best_dir": os.path.abspath(best_dir),
        }
        with open(os.path.join(args.output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    print("完成：模型与评估结果已保存至", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
