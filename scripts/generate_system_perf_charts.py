# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False 

BASE_DIR = "e:/Projects/Graduation_Project"
OUTPUT_DIR = os.path.join(BASE_DIR, "thesis_assets", "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_performance_charts():
    # 1. 响应时间对比图 (Latency Comparison)
    labels = ['重复查询\n(缓存命中)', '混合场景\n(整体平均)', '数据库检索\n(百万级记录)']
    before = [850, 800, 155.83]  # 优化前：0.85s, 0.8s, 155.83ms
    after = [1.0, 276, 0.20]     # 优化后：<1ms, 276ms, 0.20ms
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, before, width, label='优化前 (Before)', color='lightcoral')
    rects2 = ax.bar(x + width/2, after, width, label='优化后 (After)', color='mediumseagreen')
    
    # 采用对数坐标轴，因为差异太大（从 850ms 到 0.2ms）
    ax.set_yscale('log')
    ax.set_ylabel('响应延时 / ms (Log Scale)')
    ax.set_title('系统多层级性能优化对比 (延迟显著下降)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # 在柱子上添加文字
    def autolabel(rects, vals):
        for rect, val in zip(rects, vals):
            height = rect.get_height()
            ax.annotate('{}ms'.format(val),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel(rects1, before)
    autolabel(rects2, after)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'system_latency_optimization.png'), dpi=300)
    plt.close()

    # 2. 缓存命中率饼图 (Cache Hit Ratios)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 查询结果缓存
    axes[0].pie([90, 10], labels=['命中 (Hit)', '未命中 (Miss)'], autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#E0E0E0'])
    axes[0].set_title('顶层：查询结果缓存命中率')
    
    # RAG 检索缓存
    axes[1].pie([60, 40], labels=['命中 (Hit)', '未命中 (Miss)'], autopct='%1.1f%%', startangle=90, colors=['#2196F3', '#E0E0E0'])
    axes[1].set_title('中间层：RAG 检索缓存命中率')
    
    # 市场上下文缓存
    axes[2].pie([37.5, 62.5], labels=['命中 (Hit)', '未命中 (Miss)'], autopct='%1.1f%%', startangle=90, colors=['#FF9800', '#E0E0E0'])
    axes[2].set_title('底层：市场上下文缓存命中率')
    
    plt.suptitle('系统三层缓存架构命中率分析', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'system_cache_hit_ratios.png'), dpi=300)
    plt.close()
    
    print(f"System performance charts generated in {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_performance_charts()
