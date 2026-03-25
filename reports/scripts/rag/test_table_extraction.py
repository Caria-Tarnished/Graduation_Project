# -*- coding: utf-8 -*-
"""
表格提取测试脚本

测试 pdfplumber 提取表格并转换为 Markdown 的效果
"""
import os
import pdfplumber
from pathlib import Path


def table_to_markdown(table: list) -> str:
    """
    将表格转换为 Markdown 格式
    
    Args:
        table: pdfplumber 提取的表格（二维列表）
    
    Returns:
        Markdown 格式的表格字符串
    """
    if not table or len(table) < 2:
        return ""
    
    # 清洗表格数据（去除 None 和空字符串）
    cleaned_table = []
    for row in table:
        cleaned_row = [str(cell).strip() if cell else "" for cell in row]
        # 跳过全空行
        if any(cleaned_row):
            cleaned_table.append(cleaned_row)
    
    if len(cleaned_table) < 2:
        return ""
    
    # 构建 Markdown 表格
    lines = []
    
    # 表头
    header = cleaned_table[0]
    lines.append("| " + " | ".join(header) + " |")
    
    # 分隔线
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    
    # 数据行
    for row in cleaned_table[1:]:
        # 补齐列数（如果某行列数不足）
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")
    
    return "\n".join(lines)


def test_pdf_extraction(pdf_path: str, max_pages: int = 3):
    """
    测试 PDF 提取效果
    
    Args:
        pdf_path: PDF 文件路径
        max_pages: 最多测试几页
    """
    print("=" * 80)
    print(f"测试文件: {os.path.basename(pdf_path)}")
    print("=" * 80)
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        test_pages = min(max_pages, total_pages)
        
        print(f"总页数: {total_pages}")
        print(f"测试页数: {test_pages}")
        print()
        
        for i in range(test_pages):
            page = pdf.pages[i]
            print(f"\n{'='*80}")
            print(f"第 {i+1} 页")
            print(f"{'='*80}")
            
            # 提取文本
            text = page.extract_text()
            if text:
                text_preview = text[:300].replace('\n', ' ')
                print(f"\n文本预览（前300字符）:")
                print(f"{text_preview}...")
            
            # 提取表格
            tables = page.extract_tables()
            if tables:
                print(f"\n发现 {len(tables)} 个表格")
                
                for j, table in enumerate(tables, 1):
                    print(f"\n--- 表格 {j} ---")
                    print(f"行数: {len(table)}")
                    print(f"列数: {len(table[0]) if table else 0}")
                    
                    # 转换为 Markdown
                    markdown = table_to_markdown(table)
                    if markdown:
                        print("\nMarkdown 格式:")
                        print(markdown)
                    else:
                        print("⚠️  表格为空或格式异常")
            else:
                print("\n未发现表格")


def main():
    # 测试文件列表
    test_files = [
        # 中文研报（有表格）
        "data/raw/reports/research_reports/H3_AP202410311640648532_1.pdf",
        # 英文研报（摩根士丹利）
        "data/raw/reports/research_reports/Morgan Stanley-Gold：Bull Case in Play-260123.pdf",
        # 中文研报（国新证券）
        "data/raw/reports/research_reports/H3_AP202507241714875603_1.pdf",
    ]
    
    for pdf_path in test_files:
        if os.path.exists(pdf_path):
            test_pdf_extraction(pdf_path, max_pages=2)
            print("\n\n")
        else:
            print(f"⚠️  文件不存在: {pdf_path}")


if __name__ == "__main__":
    main()
