# -*- coding: utf-8 -*-
"""
PDF 文件分析脚本

功能：
1. 自动识别 PDF 类型（行情周报 vs 研报）
2. 检测语言（中文 vs 英文）
3. 分析页面布局（单栏 vs 双栏）
4. 提取元数据（标题、日期、作者等）
5. 生成分析报告

使用方法：
    python scripts/rag/analyze_pdfs.py --input data/raw/reports/research_reports
"""
import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json

try:
    import pdfplumber
except ImportError:
    print("错误：需要安装 pdfplumber")
    print("请运行：pip install pdfplumber")
    exit(1)


def detect_language(text: str) -> str:
    """
    检测文本语言
    
    Args:
        text: 文本内容
    
    Returns:
        "zh": 中文
        "en": 英文
        "mixed": 混合
    """
    # 统计中文字符数量
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 统计英文字符数量
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    total = chinese_chars + english_chars
    if total == 0:
        return "unknown"
    
    chinese_ratio = chinese_chars / total
    
    if chinese_ratio > 0.5:
        return "zh"
    elif chinese_ratio < 0.1:
        return "en"
    else:
        return "mixed"


def detect_layout(page) -> str:
    """
    检测页面布局（单栏 vs 双栏）
    
    Args:
        page: pdfplumber page 对象
    
    Returns:
        "single": 单栏
        "double": 双栏
    """
    # 简单启发式：检查文本块的 x 坐标分布
    words = page.extract_words()
    if not words:
        return "unknown"
    
    # 统计文本块的 x0 坐标
    x_coords = [w['x0'] for w in words]
    
    # 如果有明显的两个聚类，则是双栏
    page_width = page.width
    left_count = sum(1 for x in x_coords if x < page_width * 0.4)
    right_count = sum(1 for x in x_coords if x > page_width * 0.6)
    
    if left_count > 10 and right_count > 10:
        return "double"
    else:
        return "single"


def has_tables(page) -> bool:
    """
    检测页面是否包含表格
    
    Args:
        page: pdfplumber page 对象
    
    Returns:
        是否包含表格
    """
    tables = page.extract_tables()
    return len(tables) > 0


def extract_title(pdf) -> str:
    """
    提取 PDF 标题
    
    Args:
        pdf: pdfplumber PDF 对象
    
    Returns:
        标题文本
    """
    # 尝试从元数据提取
    if pdf.metadata and pdf.metadata.get('Title'):
        return pdf.metadata['Title']
    
    # 尝试从首页提取（通常标题字体最大）
    if len(pdf.pages) > 0:
        first_page = pdf.pages[0]
        text = first_page.extract_text()
        if text:
            lines = text.split('\n')
            # 返回第一个非空行
            for line in lines:
                line = line.strip()
                if len(line) > 5:  # 标题通常不会太短
                    return line
    
    return "Unknown"


def extract_date_from_filename(filename: str) -> str:
    """
    从文件名提取日期
    
    Args:
        filename: 文件名
    
    Returns:
        日期字符串（YYYY-MM-DD 格式）或 "Unknown"
    """
    # 模式1：YYMMDD（如 260201）
    match = re.search(r'(\d{2})(\d{2})(\d{2})', filename)
    if match:
        yy, mm, dd = match.groups()
        # 假设 20xx 年
        return f"20{yy}-{mm}-{dd}"
    
    # 模式2：YYYYMMDD（如 20250406）
    match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if match:
        yyyy, mm, dd = match.groups()
        return f"{yyyy}-{mm}-{dd}"
    
    # 模式3：H3_AP 格式（如 H3_AP202601291818533670_1.pdf）
    match = re.search(r'H3_AP(\d{4})(\d{2})(\d{2})', filename)
    if match:
        yyyy, mm, dd = match.groups()
        return f"{yyyy}-{mm}-{dd}"
    
    return "Unknown"


def classify_pdf_type(filename: str, first_page_text: str, has_many_tables: bool) -> str:
    """
    分类 PDF 类型
    
    Args:
        filename: 文件名
        first_page_text: 首页文本
        has_many_tables: 是否包含大量表格
    
    Returns:
        "weekly_report": 行情周报（表格为主）
        "research_report": 研报（文本为主）
    """
    # 规则1：文件名包含 H3_AP 的是行情周报
    if filename.startswith("H3_AP"):
        return "weekly_report"
    
    # 规则2：首页包含"行情周报"关键词
    if "行情周报" in first_page_text or "市场周报" in first_page_text:
        return "weekly_report"
    
    # 规则3：表格占比很高（超过 50% 的页面有表格）
    if has_many_tables:
        return "weekly_report"
    
    # 默认：研报
    return "research_report"


def analyze_pdf(pdf_path: str) -> Dict:
    """
    分析单个 PDF 文件
    
    Args:
        pdf_path: PDF 文件路径
    
    Returns:
        分析结果字典
    """
    filename = os.path.basename(pdf_path)
    
    result = {
        "filename": filename,
        "path": pdf_path,
        "type": "unknown",
        "language": "unknown",
        "layout": "unknown",
        "title": "Unknown",
        "date": "Unknown",
        "pages": 0,
        "has_tables": False,
        "table_pages": 0,
        "error": None
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            result["pages"] = len(pdf.pages)
            
            # 提取标题
            result["title"] = extract_title(pdf)
            
            # 从文件名提取日期
            result["date"] = extract_date_from_filename(filename)
            
            # 分析前 3 页（或全部页面，如果少于 3 页）
            sample_pages = min(3, len(pdf.pages))
            
            all_text = ""
            table_count = 0
            layout_samples = []
            
            for i in range(sample_pages):
                page = pdf.pages[i]
                
                # 提取文本
                page_text = page.extract_text() or ""
                all_text += page_text + "\n"
                
                # 检测表格
                if has_tables(page):
                    table_count += 1
                
                # 检测布局
                layout = detect_layout(page)
                layout_samples.append(layout)
            
            # 检测语言
            result["language"] = detect_language(all_text)
            
            # 检测布局（取众数）
            if layout_samples:
                result["layout"] = max(set(layout_samples), key=layout_samples.count)
            
            # 检测表格
            result["has_tables"] = table_count > 0
            result["table_pages"] = table_count
            
            # 分类 PDF 类型
            has_many_tables = (table_count / sample_pages) > 0.5
            result["type"] = classify_pdf_type(filename, all_text, has_many_tables)
    
    except Exception as e:
        result["error"] = str(e)
    
    return result


def analyze_directory(input_dir: str) -> List[Dict]:
    """
    分析目录下的所有 PDF 文件
    
    Args:
        input_dir: 输入目录
    
    Returns:
        分析结果列表
    """
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    
    print(f"找到 {len(pdf_files)} 个 PDF 文件")
    print("=" * 80)
    
    results = []
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] 分析: {pdf_path.name}")
        
        result = analyze_pdf(str(pdf_path))
        results.append(result)
        
        # 打印简要信息
        if result["error"]:
            print(f"  ❌ 错误: {result['error']}")
        else:
            print(f"  类型: {result['type']}")
            print(f"  语言: {result['language']}")
            print(f"  布局: {result['layout']}")
            print(f"  页数: {result['pages']}")
            print(f"  表格: {result['table_pages']}/{result['pages']} 页")
            print(f"  日期: {result['date']}")
        
        print()
    
    return results


def generate_report(results: List[Dict], output_path: str):
    """
    生成分析报告
    
    Args:
        results: 分析结果列表
        output_path: 输出路径
    """
    # 统计信息
    total = len(results)
    weekly_reports = sum(1 for r in results if r["type"] == "weekly_report")
    research_reports = sum(1 for r in results if r["type"] == "research_report")
    
    zh_count = sum(1 for r in results if r["language"] == "zh")
    en_count = sum(1 for r in results if r["language"] == "en")
    mixed_count = sum(1 for r in results if r["language"] == "mixed")
    
    double_layout = sum(1 for r in results if r["layout"] == "double")
    single_layout = sum(1 for r in results if r["layout"] == "single")
    
    errors = sum(1 for r in results if r["error"])
    
    # 生成报告
    report = {
        "summary": {
            "total_files": total,
            "weekly_reports": weekly_reports,
            "research_reports": research_reports,
            "languages": {
                "chinese": zh_count,
                "english": en_count,
                "mixed": mixed_count
            },
            "layouts": {
                "double_column": double_layout,
                "single_column": single_layout
            },
            "errors": errors
        },
        "files": results
    }
    
    # 保存 JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("=" * 80)
    print("分析报告")
    print("=" * 80)
    print(f"总文件数: {total}")
    print(f"  - 行情周报: {weekly_reports}")
    print(f"  - 研报: {research_reports}")
    print()
    print(f"语言分布:")
    print(f"  - 中文: {zh_count}")
    print(f"  - 英文: {en_count}")
    print(f"  - 混合: {mixed_count}")
    print()
    print(f"布局分布:")
    print(f"  - 双栏: {double_layout}")
    print(f"  - 单栏: {single_layout}")
    print()
    if errors > 0:
        print(f"⚠️  错误: {errors} 个文件解析失败")
    print()
    print(f"详细报告已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="分析 PDF 文件")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/reports/research_reports",
        help="输入目录（默认：data/raw/reports/research_reports）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/reports/pdf_analysis.json",
        help="输出报告路径（默认：data/raw/reports/pdf_analysis.json）"
    )
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"错误：输入目录不存在: {args.input}")
        return
    
    # 分析 PDF
    results = analyze_directory(args.input)
    
    # 生成报告
    generate_report(results, args.output)


if __name__ == "__main__":
    main()
