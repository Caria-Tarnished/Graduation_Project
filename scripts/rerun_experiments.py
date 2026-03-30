"""
重跑实验脚本：在统一的测试集(3,823条)上评估所有模型
1. 基线模型 TF-IDF+SVM 重评估
2. BERT 无增强推理（输入增强消融）
3. 规则引擎触发率统计
"""

import sys
import os
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

# ── 路径配置 ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_ENHANCED = os.path.join(PROJECT_ROOT, "data", "processed", "test_enhanced_3cls.csv")
TEST_PLAIN = os.path.join(PROJECT_ROOT, "data", "processed", "test_3cls.csv")
BASELINE_TFIDF = os.path.join(PROJECT_ROOT, "models", "baseline_tfidf_svm", "tfidf.pkl")
BASELINE_MODEL = os.path.join(PROJECT_ROOT, "models", "baseline_tfidf_svm", "model.pkl")
BERT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bert_3cls", "best")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "rerun_experiments")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_test_data():
    """加载测试集"""
    df_enhanced = pd.read_csv(TEST_ENHANCED)
    df_plain = pd.read_csv(TEST_PLAIN)
    print(f"增强测试集: {len(df_enhanced)} 条")
    print(f"原始测试集: {len(df_plain)} 条")
    print(f"标签分布:\n{df_enhanced['label'].value_counts().sort_index()}")
    return df_enhanced, df_plain


# ═══════════════════════════════════════════
# 实验1: 基线模型在BERT测试集上重评估
# ═══════════════════════════════════════════
def experiment_1_baseline(df):
    """在BERT的3,823条测试集上评估TF-IDF+SVM基线"""
    print("\n" + "=" * 60)
    print("实验1: 基线模型 TF-IDF+SVM 在BERT测试集上重评估")
    print("=" * 60)

    vec = joblib.load(BASELINE_TFIDF)
    clf = joblib.load(BASELINE_MODEL)

    texts = df["text"].fillna("").tolist()
    labels = df["label"].tolist()

    X = vec.transform(texts)
    preds = clf.predict(X)

    macro_f1 = f1_score(labels, preds, average="macro")
    accuracy = accuracy_score(labels, preds)
    report = classification_report(labels, preds, digits=4)
    cm = confusion_matrix(labels, preds, labels=[-1, 0, 1])

    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\n分类报告:\n{report}")
    print(f"混淆矩阵:\n{cm}")

    results = {
        "experiment": "baseline_on_bert_testset",
        "test_size": len(df),
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
        "report": report,
        "confusion_matrix": cm.tolist()
    }

    with open(os.path.join(OUTPUT_DIR, "exp1_baseline_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


# ═══════════════════════════════════════════
# 实验2: BERT无增强推理（输入增强消融）
# ═══════════════════════════════════════════
def experiment_2_bert_no_augmentation(df_enhanced, df_plain):
    """用已训练的增强BERT模型，分别在增强和非增强文本上推理"""
    print("\n" + "=" * 60)
    print("实验2: BERT输入增强消融实验")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        print("ERROR: 需要安装 torch 和 transformers")
        return None

    print("加载BERT模型...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"设备: {device}")

    # 模型输出 0/1/2 -> 标签 -1/0/1 的映射
    id2label = {0: -1, 1: 0, 2: 1}

    def predict_batch(texts, batch_size=32):
        all_preds = []
        all_confs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=384, return_tensors="pt"
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_ids = torch.argmax(probs, dim=-1).cpu().tolist()
                confs = probs.max(dim=-1).values.cpu().tolist()

            all_preds.extend([id2label[p] for p in pred_ids])
            all_confs.extend(confs)

            if (i // batch_size) % 10 == 0:
                print(f"  进度: {min(i + batch_size, len(texts))}/{len(texts)}")
        return all_preds, all_confs

    labels = df_enhanced["label"].tolist()

    # 有增强（使用 text_enhanced 列）
    print("\n推理: 有增强文本 (text_enhanced)...")
    texts_enhanced = df_enhanced["text_enhanced"].fillna("").tolist()
    preds_enhanced, confs_enhanced = predict_batch(texts_enhanced)
    f1_enhanced = f1_score(labels, preds_enhanced, average="macro")
    acc_enhanced = accuracy_score(labels, preds_enhanced)
    report_enhanced = classification_report(labels, preds_enhanced, digits=4)

    print(f"有增强 Macro F1: {f1_enhanced:.4f}, Accuracy: {acc_enhanced:.4f}")
    print(f"分类报告:\n{report_enhanced}")

    # 无增强（使用 text 列）
    print("\n推理: 无增强文本 (text)...")
    texts_plain = df_plain["text"].fillna("").tolist()
    preds_plain, confs_plain = predict_batch(texts_plain)
    f1_plain = f1_score(labels, preds_plain, average="macro")
    acc_plain = accuracy_score(labels, preds_plain)
    report_plain = classification_report(labels, preds_plain, digits=4)

    print(f"无增强 Macro F1: {f1_plain:.4f}, Accuracy: {acc_plain:.4f}")
    print(f"分类报告:\n{report_plain}")

    improvement = (f1_enhanced - f1_plain) / f1_plain * 100
    print(f"\n输入增强提升: {improvement:+.1f}%")

    results = {
        "experiment": "bert_augmentation_ablation",
        "test_size": len(labels),
        "with_augmentation": {
            "macro_f1": round(f1_enhanced, 4),
            "accuracy": round(acc_enhanced, 4),
            "report": report_enhanced
        },
        "without_augmentation": {
            "macro_f1": round(f1_plain, 4),
            "accuracy": round(acc_plain, 4),
            "report": report_plain
        },
        "improvement_percent": round(improvement, 1)
    }

    with open(os.path.join(OUTPUT_DIR, "exp2_augmentation_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


# ═══════════════════════════════════════════
# 实验3: 规则引擎触发率统计
# ═══════════════════════════════════════════
def experiment_3_rule_engine(df, bert_preds=None, bert_confs=None):
    """在测试集上统计规则引擎触发率"""
    print("\n" + "=" * 60)
    print("实验3: 规则引擎触发率统计")
    print("=" * 60)

    # 规则引擎阈值（与 sentiment_analyzer.py 一致）
    PRICED_IN_THRESHOLD = 0.01       # 1%
    HIGH_VOLATILITY_THRESHOLD = 0.015  # 1.5%
    LOW_NET_CHANGE_THRESHOLD = 0.002   # 0.2%

    # 如果没有BERT预测结果，从已保存的报告中加载
    if bert_preds is None:
        pred_file = os.path.join(PROJECT_ROOT, "reports", "bert_3cls_enhanced_v1", "pred_test.csv")
        if os.path.exists(pred_file):
            pred_df = pd.read_csv(pred_file)
            bert_preds = pred_df["pred"].tolist()
            print(f"从 {pred_file} 加载了 {len(bert_preds)} 条BERT预测")
        else:
            print("ERROR: 无法找到BERT预测结果文件")
            return None

    labels = df["label"].tolist()
    pre_rets = df["pre_ret"].fillna(0).tolist()
    range_ratios = df["range_ratio"].fillna(0).tolist()

    # 确保长度一致
    n = min(len(bert_preds), len(labels), len(pre_rets))
    print(f"评估样本数: {n}")

    # 统计触发
    rule_results = []
    for i in range(n):
        pred = bert_preds[i]
        pre_ret = pre_rets[i]
        rr = range_ratios[i]
        true_label = labels[i]

        rule_triggered = None
        final_sentiment = pred

        # 规则1: 预期兑现
        if pred == 1 and pre_ret > PRICED_IN_THRESHOLD:
            rule_triggered = "bullish_priced_in"
        elif pred == -1 and pre_ret < -PRICED_IN_THRESHOLD:
            rule_triggered = "bearish_priced_in"

        # 规则2: 建议观望（优先级更高）
        if rr > HIGH_VOLATILITY_THRESHOLD and abs(pre_ret) < LOW_NET_CHANGE_THRESHOLD:
            rule_triggered = "watch"

        rule_results.append({
            "true_label": true_label,
            "bert_pred": pred,
            "pre_ret": pre_ret,
            "range_ratio": rr,
            "rule_triggered": rule_triggered
        })

    rule_df = pd.DataFrame(rule_results)

    # 统计触发率
    total = len(rule_df)
    triggered = rule_df[rule_df["rule_triggered"].notna()]
    n_triggered = len(triggered)

    print(f"\n总样本数: {total}")
    print(f"规则触发总数: {n_triggered} ({n_triggered / total * 100:.1f}%)")

    # 分规则统计
    rule_stats = {}
    for rule_name in ["bullish_priced_in", "bearish_priced_in", "watch"]:
        subset = rule_df[rule_df["rule_triggered"] == rule_name]
        count = len(subset)
        rate = count / total * 100
        print(f"  {rule_name}: {count}条 ({rate:.1f}%)")
        rule_stats[rule_name] = {"count": count, "rate": round(rate, 1)}

    results = {
        "experiment": "rule_engine_evaluation",
        "test_size": total,
        "total_triggered": n_triggered,
        "trigger_rate": round(n_triggered / total * 100, 1),
        "rules": rule_stats,
        "thresholds": {
            "priced_in": PRICED_IN_THRESHOLD,
            "high_volatility": HIGH_VOLATILITY_THRESHOLD,
            "low_net_change": LOW_NET_CHANGE_THRESHOLD
        }
    }

    with open(os.path.join(OUTPUT_DIR, "exp3_rule_engine.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


# ═══════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("毕设实验重跑脚本")
    print("=" * 60)

    df_enhanced, df_plain = load_test_data()

    # 实验1: 基线模型
    exp1 = experiment_1_baseline(df_enhanced)

    # 实验2: BERT输入增强消融
    exp2 = experiment_2_bert_no_augmentation(df_enhanced, df_plain)

    # 实验3: 规则引擎
    exp3 = experiment_3_rule_engine(df_enhanced)

    # 总结
    print("\n" + "=" * 60)
    print("实验结果总结")
    print("=" * 60)
    if exp1:
        print(f"基线 TF-IDF+SVM (3,823条): Macro F1 = {exp1['macro_f1']}")
    if exp2:
        print(f"BERT 有增强:   Macro F1 = {exp2['with_augmentation']['macro_f1']}")
        print(f"BERT 无增强:   Macro F1 = {exp2['without_augmentation']['macro_f1']}")
        print(f"输入增强提升:  {exp2['improvement_percent']:+.1f}%")
    if exp3:
        print(f"规则引擎触发率: {exp3['trigger_rate']}% ({exp3['total_triggered']}/{exp3['test_size']})")

    print(f"\n结果已保存到: {OUTPUT_DIR}/")
