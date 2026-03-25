# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False 

# 路径配置
BASE_DIR = r"e:\Projects\Graduation_Project"
TRAIN_CSV = os.path.join(BASE_DIR, "data", "processed", "train_3cls.csv")
PRED_CSV = os.path.join(BASE_DIR, "reports", "bert_3cls_enhanced_v1", "pred_test.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "thesis_assets", "charts")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_charts():
    # 1. 数据集分布图表
    print(f"读取数据: {TRAIN_CSV}...")
    df_train = pd.read_csv(TRAIN_CSV, encoding='utf-8', on_bad_lines='skip')
    
    # 1.1 数据来源分布饼图
    plt.figure(figsize=(8, 8))
    # 填充缺失值为'Unknown'
    sources = df_train['source'].fillna('Unknown')
    source_counts = sources.value_counts()
    
    # 如果类别太多，只展示前几个，其他的归为"Other"
    if len(source_counts) > 5:
        top_sources = source_counts[:4]
        other_sum = source_counts[4:].sum()
        top_sources['Other'] = other_sum
        source_counts = top_sources
        
    plt.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title('训练集数据来源分布')
    plt.savefig(os.path.join(OUTPUT_DIR, 'source_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("已生成来源分布饼图")

    # 1.2 情感类别分布柱状图
    plt.figure(figsize=(8, 6))
    label_counts = df_train['label'].value_counts().sort_index()
    label_map = {-1: '利空 (-1)', 0: '中立 (0)', 1: '利多 (1)'}
    mapped_labels = [label_map.get(lbl, str(lbl)) for lbl in label_counts.index]
    
    # 使用较新的seaborn语法 (hue代替palette以免被弃用警告)
    sns.barplot(x=mapped_labels, y=label_counts.values, hue=mapped_labels, legend=False, palette="muted")
    plt.title('训练集情感类别分布')
    plt.xlabel('情感类别')
    plt.ylabel('样本数量')
    
    for i, v in enumerate(label_counts.values):
        plt.text(i, v + max(label_counts.values)*0.01, str(v), ha='center', va='bottom')
        
    plt.savefig(os.path.join(OUTPUT_DIR, 'label_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("已生成情感类别分布柱状图")

    # 2. 预测混淆矩阵图表
    print(f"读取预测数据: {PRED_CSV}...")
    df_pred = pd.read_csv(PRED_CSV, encoding='utf-8')
    
    # 使用 pandas crosstab 计算混淆矩阵，避免依赖 sklearn
    cm_df = pd.crosstab(df_pred['label'], df_pred['pred'], dropna=False)
    # 确保有 -1, 0, 1 这些列和行
    labels_order = [-1, 0, 1]
    cm_df = cm_df.reindex(index=labels_order, columns=labels_order, fill_value=0)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['利空(-1)', '中立(0)', '利多(1)'], 
                yticklabels=['利空(-1)', '中立(0)', '利多(1)'])
    plt.title('BERT模型测试集预测混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("已生成混淆矩阵热力图")
    
    print(f"所有图表均已成功保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_charts()
