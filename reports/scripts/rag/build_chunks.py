# -*- coding: utf-8 -*-
"""
PDF 解析与切片脚本

功能：
1. 解析 PDF 文件（支持双栏排版）
2. ROI 裁剪去除页眉页脚
3. 智能表格处理（简单表格→Markdown，复杂表格→描述）
4. 正则清洗（免责声明、联系方式等）
5. 提取元数据（日期、语言、来源）
6. 文本切片（RecursiveCharacterTextSplitter）
7. 输出 JSON 格式的切片数据

使用方法：
    python scripts/rag/build_chunks.py --input data/raw/reports/research_reports --output data/reports/chunks.json
"""
import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

try:
    import pdfplumber
except ImportError:
    print("错误：需要安装 pdfplumber")
    print("请运行：pip install pdfplumber")
    exit(1)

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        print("错误：需要安装 langchain-text-splitters")
        print("请运行：pip install langchain-text-splitters")
        exit(1)


# ============================================================================
# 表格处理
# ============================================================================

def is_simple_table(table: list) -> bool:
    """
    判断是否为简单表格
    
    简单表格定义：
    - 行数 <= 10
    - 列数 <= 5
    - 单元格内容不太长（平均长度 < 50 字符）
    
    Args:
        table: pdfplumber 提取的表格
    
    Returns:
        是否为简单表格
    """
    if not table or len(table) < 2:
        return False
    
    rows = len(table)
    cols = len(table[0]) if table else 0
    
    # 规则1：行列数限制
    if rows > 10 or cols > 5:
        return False
    
    # 规则2：单元格内容长度
    total_length = 0
    cell_count = 0
    
    for row in table:
        for cell in row:
            if cell:
                total_length += len(str(cell))
                cell_count += 1
    
    if cell_count > 0:
        avg_length = total_length / cell_count
        if avg_length > 50:
            return False
    
    return True


def table_to_markdown(table: list) -> str:
    """
    将表格转换为 Markdown 格式
    
    Args:
        table: pdfplumber 提取的表格
    
    Returns:
        Markdown 格式的表格字符串
    """
    if not table or len(table) < 2:
        return ""
    
    # 清洗表格数据
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
        # 补齐列数
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")
    
    return "\n".join(lines)


def describe_table(table: list) -> str:
    """
    为复杂表格生成描述性文本
    
    Args:
        table: pdfplumber 提取的表格
    
    Returns:
        描述性文本
    """
    if not table:
        return "空表格"
    
    rows = len(table)
    cols = len(table[0]) if table else 0
    
    # 尝试提取表头
    header = []
    if rows > 0:
        header = [str(cell).strip() for cell in table[0] if cell]
    
    if header:
        header_text = "、".join(header[:5])  # 最多显示前5列
        if len(header) > 5:
            header_text += "等"
        return f"{rows}行×{cols}列表格，包含{header_text}"
    else:
        return f"{rows}行×{cols}列表格"


# ============================================================================
# 文本清洗
# ============================================================================

def clean_text(text: str) -> str:
    """
    清洗文本
    
    清洗内容：
    1. 免责声明
    2. 分析师联系方式
    3. 特殊字符
    4. 多余空白
    
    Args:
        text: 原始文本
    
    Returns:
        清洗后的文本
    """
    if not text:
        return ""
    
    # 规则1：去除免责声明（通常在文末）
    disclaimer_patterns = [
        r'免责声明[\s\S]*',
        r'法律声明[\s\S]*',
        r'重要声明[\s\S]*',
        r'本报告[\s\S]*仅供[\s\S]*参考',
        r'请务必阅读[\s\S]*免责条款',
    ]
    
    for pattern in disclaimer_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 规则2：去除分析师联系方式
    contact_patterns = [
        r'分析师[：:]\s*[\u4e00-\u9fff]+\s*电话[：:]\s*[\d\-\+\(\)]+',
        r'联系人[：:]\s*[\u4e00-\u9fff]+\s*邮箱[：:]\s*[\w\.\-]+@[\w\.\-]+',
        r'证书编号[：:]\s*[A-Z0-9]+',
    ]
    
    for pattern in contact_patterns:
        text = re.sub(pattern, '', text)
    
    # 规则3：去除特殊字符（保留中英文、数字、常用标点）
    text = re.sub(r'[^\u4e00-\u9fffa-zA-Z0-9\s\.,;:!?()（）、，。；：！？""''《》【】\-\+\*\/\%\$\€\£\¥]', '', text)
    
    # 规则4：压缩多余空白
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


# ============================================================================
# 元数据提取
# ============================================================================

def extract_date_from_filename(filename: str) -> str:
    """
    从文件名提取日期
    
    Args:
        filename: 文件名
    
    Returns:
        日期字符串（YYYY-MM-DD）或 "Unknown"
    """
    # 模式1：YYMMDD（如 260201）
    match = re.search(r'(\d{2})(\d{2})(\d{2})', filename)
    if match:
        yy, mm, dd = match.groups()
        return f"20{yy}-{mm}-{dd}"
    
    # 模式2：YYYYMMDD（如 20250406）
    match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if match:
        yyyy, mm, dd = match.groups()
        return f"{yyyy}-{mm}-{dd}"
    
    # 模式3：H3_AP 格式
    match = re.search(r'H3_AP(\d{4})(\d{2})(\d{2})', filename)
    if match:
        yyyy, mm, dd = match.groups()
        return f"{yyyy}-{mm}-{dd}"
    
    return "Unknown"


def detect_language(text: str) -> str:
    """
    检测文本语言
    
    Args:
        text: 文本内容
    
    Returns:
        "zh": 中文, "en": 英文, "mixed": 混合
    """
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
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


def extract_source(filename: str) -> str:
    """
    从文件名提取来源机构
    
    Args:
        filename: 文件名
    
    Returns:
        来源机构名称
    """
    # 常见机构名称
    sources = [
        "Goldman Sachs", "J.P. Morgan", "Morgan Stanley", "UBS",
        "国新证券", "联储证券", "华源证券", "开源证券",
        "世界黄金协会", "上海黄金交易所"
    ]
    
    for source in sources:
        if source in filename:
            return source
    
    # 如果是 H3_AP 开头，返回"国内券商"
    if filename.startswith("H3_AP"):
        return "国内券商"
    
    return "Unknown"


# ============================================================================
# PDF 解析
# ============================================================================

def process_page(page, roi_margin: float = 0.1) -> str:
    """
    处理单个 PDF 页面
    
    Args:
        page: pdfplumber page 对象
        roi_margin: ROI 裁剪边距（0.1 表示去除上下各 10%）
    
    Returns:
        处理后的文本
    """
    # 1. ROI 裁剪（去除页眉页脚）
    bbox = (
        0,
        page.height * roi_margin,
        page.width,
        page.height * (1 - roi_margin)
    )
    cropped = page.within_bbox(bbox)
    
    # 2. 提取文本
    text = cropped.extract_text() or ""
    
    # 3. 提取表格
    tables = page.extract_tables()
    table_texts = []
    
    for table in tables:
        if is_simple_table(table):
            # 简单表格 → Markdown
            markdown = table_to_markdown(table)
            if markdown:
                table_texts.append(f"\n\n{markdown}\n\n")
        else:
            # 复杂表格 → 描述
            desc = describe_table(table)
            table_texts.append(f"\n\n[表格: {desc}]\n\n")
    
    # 4. 合并文本和表格
    combined = text + "".join(table_texts)
    
    # 5. 清洗文本
    cleaned = clean_text(combined)
    
    return cleaned


def parse_pdf(pdf_path: str) -> Dict:
    """
    解析 PDF 文件
    
    Args:
        pdf_path: PDF 文件路径
    
    Returns:
        解析结果字典
    """
    filename = os.path.basename(pdf_path)
    
    result = {
        "filename": filename,
        "path": pdf_path,
        "date": extract_date_from_filename(filename),
        "source": extract_source(filename),
        "language": "unknown",
        "pages": 0,
        "full_text": "",
        "error": None
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            result["pages"] = len(pdf.pages)
            
            # 处理所有页面
            page_texts = []
            for page in pdf.pages:
                page_text = process_page(page)
                if page_text:
                    page_texts.append(page_text)
            
            # 合并所有页面
            full_text = "\n\n".join(page_texts)
            result["full_text"] = full_text
            
            # 检测语言
            result["language"] = detect_language(full_text)
    
    except Exception as e:
        result["error"] = str(e)
    
    return result


# ============================================================================
# 文本切片
# ============================================================================

def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    切片文本
    
    Args:
        text: 文本内容
        chunk_size: 切片大小
        chunk_overlap: 切片重叠
    
    Returns:
        切片列表
    """
    if not text:
        return []
    
    # 使用 LangChain 的 RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    return chunks


def build_chunks_from_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    """
    从 PDF 构建切片
    
    Args:
        pdf_path: PDF 文件路径
        chunk_size: 切片大小
        chunk_overlap: 切片重叠
    
    Returns:
        切片列表
    """
    # 1. 解析 PDF
    parsed = parse_pdf(pdf_path)
    
    if parsed["error"]:
        print(f"  ❌ 解析失败: {parsed['error']}")
        return []
    
    # 2. 切片
    chunks = split_text(parsed["full_text"], chunk_size, chunk_overlap)
    
    # 3. 构建切片数据
    chunk_data = []
    for i, chunk_text in enumerate(chunks):
        chunk_data.append({
            "chunk_id": f"{parsed['filename']}_{i}",
            "text": chunk_text,
            "metadata": {
                "source_file": parsed["filename"],
                "source_path": parsed["path"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "date": parsed["date"],
                "source": parsed["source"],
                "language": parsed["language"],
                "pages": parsed["pages"]
            }
        })
    
    return chunk_data


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PDF 解析与切片")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/reports/research_reports",
        help="输入目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/reports/chunks.json",
        help="输出文件"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="切片大小（默认：500）"
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=50,
        help="切片重叠（默认：50）"
    )
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"错误：输入目录不存在: {args.input}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 查找所有 PDF 文件
    pdf_files = list(Path(args.input).glob("*.pdf"))
    
    print(f"找到 {len(pdf_files)} 个 PDF 文件")
    print(f"切片参数: chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}")
    print("=" * 80)
    
    # 处理所有 PDF
    all_chunks = []
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] 处理: {pdf_path.name}")
        
        chunks = build_chunks_from_pdf(
            str(pdf_path),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        if chunks:
            all_chunks.extend(chunks)
            print(f"  ✓ 生成 {len(chunks)} 个切片")
        else:
            print(f"  ⚠️  未生成切片")
    
    # 保存结果
    print()
    print("=" * 80)
    print(f"总切片数: {len(all_chunks)}")
    print(f"保存到: {args.output}")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print("✓ 完成")


if __name__ == "__main__":
    main()
