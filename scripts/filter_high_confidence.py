# -*- coding: utf-8 -*-
"""
高置信度样本过滤脚本 (方案三)

策略：在原始代理标注的基础上，进一步筛选掉"边界模糊"的样本，
只保留每类中收益率信号最明确的子集，提高标签纯度。

过滤规则（均基于训练集的 ret_post 分布计算，避免数据泄漏）：
  - 利空 (-1)：保留 ret_post 低于训练集第15分位的样本（最负的 ~ 靠近极端下跌）
  - 利多 (+1)：保留 ret_post 高于训练集第85分位的样本（最正的 ~ 靠近极端上涨）
  - 中立 (0)：保留 |ret_post| 排名位于训练集绝对值第25分位以内（最接近0的）

输出：
  - data/processed/train_hc_3cls.csv
  - data/processed/val_hc_3cls.csv
  - data/processed/test_hc_3cls.csv
  对应的 Colab 训练时替换同名文件即可
"""
import pandas as pd
import numpy as np
import os

BASE_DIR = r"e:\Projects\Graduation_Project"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RET_COL = "ret_post"

# ─── 1. 加载三个分割 ───────────────────────────────────────────────
print("加载原始数据集...")
df_train = pd.read_csv(os.path.join(DATA_DIR, "train_3cls.csv"), encoding="utf-8", on_bad_lines="skip")
df_val   = pd.read_csv(os.path.join(DATA_DIR, "val_3cls.csv"),   encoding="utf-8", on_bad_lines="skip")
df_test  = pd.read_csv(os.path.join(DATA_DIR, "test_3cls.csv"),  encoding="utf-8", on_bad_lines="skip")

print(f"原始训练集: {len(df_train)} | 验证集: {len(df_val)} | 测试集: {len(df_test)}")

# ─── 2. 基于训练集计算分位阈值（避免标签泄漏）───────────────────────
train_ret = df_train[RET_COL].dropna()

# 利空/利多的边界阈值
q15_lo = train_ret.quantile(0.15)  # 收益率最低15% → 置信利空
q85_hi = train_ret.quantile(0.85)  # 收益率最高15% → 置信利多
# 中立的绝对值上界（仅保留靠近0的）
q25_abs = train_ret.abs().quantile(0.25)

print("\n=== 阈值（基于训练集 ret_post 计算）===")
print(f"  利空上界  (q15): ret < {q15_lo:.6f}")
print(f"  利多下界  (q85): ret > {q85_hi:.6f}")
print(f"  中立绝对值上界 (q25_abs): |ret| < {q25_abs:.6f}")

# ─── 3. 定义过滤函数 ────────────────────────────────────────────────
def filter_hc(df):
    """保留高置信度样本：各类别内ret_post分布最极端的子集"""
    mask_bearish = (df["label"] == -1) & (df[RET_COL] < q15_lo)
    mask_neutral = (df["label"] ==  0) & (df[RET_COL].abs() < q25_abs)
    mask_bullish = (df["label"] ==  1) & (df[RET_COL] > q85_hi)
    return df[mask_bearish | mask_neutral | mask_bullish].reset_index(drop=True)

# ─── 4. 过滤三个分割 ────────────────────────────────────────────────
df_train_hc = filter_hc(df_train)
df_val_hc   = filter_hc(df_val)
df_test_hc  = filter_hc(df_test)

print("\n=== 过滤后规模 ===")
for name, orig, filt in [
    ("训练集", df_train, df_train_hc),
    ("验证集", df_val,   df_val_hc),
    ("测试集", df_test,  df_test_hc),
]:
    pct = len(filt) / len(orig) * 100
    print(f"  {name}: {len(orig)} → {len(filt)} ({pct:.1f}%)")
    dist_orig = orig["label"].value_counts().sort_index().to_dict()
    dist_filt = filt["label"].value_counts().sort_index().to_dict()
    print(f"    原始分布: {dist_orig}")
    print(f"    过滤后:   {dist_filt}")

# ─── 5. 验证标签纯度改善情况 ────────────────────────────────────────
print("\n=== 标签纯度验证（训练集 ret_post 均值对比）===")
print("原始:")
for l in [-1, 0, 1]:
    s = df_train[df_train["label"] == l][RET_COL]
    print(f"  label={l:+d}: mean={s.mean():+.6f}  std={s.std():.6f}  noise_ratio(|ret|<q85-q15)={((s.abs() < (q85_hi - q15_lo)).mean()):.3f}")

print("过滤后:")
for l in [-1, 0, 1]:
    s = df_train_hc[df_train_hc["label"] == l][RET_COL]
    print(f"  label={l:+d}: mean={s.mean():+.6f}  std={s.std():.6f}  n={len(s)}")

# ─── 6. 保存结果 ────────────────────────────────────────────────────
df_train_hc.to_csv(os.path.join(DATA_DIR, "train_hc_3cls.csv"), index=False, encoding="utf-8")
df_val_hc.to_csv(  os.path.join(DATA_DIR, "val_hc_3cls.csv"),   index=False, encoding="utf-8")
df_test_hc.to_csv( os.path.join(DATA_DIR, "test_hc_3cls.csv"),  index=False, encoding="utf-8")

print("\n=== 完成！文件已保存 ===")
print(f"  {os.path.join(DATA_DIR, 'train_hc_3cls.csv')}")
print(f"  {os.path.join(DATA_DIR, 'val_hc_3cls.csv')}")
print(f"  {os.path.join(DATA_DIR, 'test_hc_3cls.csv')}")
print("\n接下来：将 train_hc_3cls.csv / val_hc_3cls.csv 上传 Colab 替换原文件后，重新训练 BERT 即可。")
