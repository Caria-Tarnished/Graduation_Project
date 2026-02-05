# -*- coding: utf-8 -*-
"""
Data augmentation script: Add market context prefixes to financial event texts (Input Augmentation)

Theory: Time Series Textualization (Financial LLM Survey Lecture 7 & 8)
Core idea: Convert pre-release market state (trend, volatility) into natural language prefixes for BERT input.

Input: train/val/test_multi_labeled.csv (with pre_ret, range_ratio features)
Output: train/val/test_enhanced.csv (with new text_enhanced column)

Usage:
    python scripts/modeling/build_enhanced_dataset.py \
        --input_dir data/processed \
        --output_dir data/processed \
        --trend_threshold_high 0.005 \
        --trend_threshold_low 0.002 \
        --volatility_threshold 0.008
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd  # type: ignore


def _ensure_dir(path: str) -> None:
    """Ensure directory exists"""
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _safe_float(x: object) -> Optional[float]:
    """Safely convert to float, return None on failure"""
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def generate_trend_prefix(
    pre_ret: Optional[float],
    range_ratio: Optional[float],
    trend_high: float,
    trend_low: float,
    vol_threshold: float,
) -> str:
    """
    Generate trend prefix based on pre-return and volatility
    
    Args:
        pre_ret: Pre-return (120 minutes)
        range_ratio: Volatility (Range/Price)
        trend_high: Strong trend threshold (e.g. 0.5%)
        trend_low: Weak trend threshold (e.g. 0.2%)
        vol_threshold: High volatility threshold (e.g. 0.8%)
    
    Returns:
        Trend prefix string
    """
    # Handle missing values
    if pre_ret is None:
        return "[Market Data Missing]"
    
    vol = range_ratio if range_ratio is not None else 0.0
    
    # Decision logic (by priority)
    if pre_ret > trend_high:
        return "[Strong Rally]"
    elif pre_ret < -trend_high:
        return "[Sharp Decline]"
    elif abs(pre_ret) < trend_low and vol > vol_threshold:
        return "[High Volatility]"
    elif -trend_high < pre_ret < -trend_low:
        return "[Weak Decline]"
    elif trend_low < pre_ret < trend_high:
        return "[Mild Rally]"
    else:
        return "[Sideways]"


def generate_data_prefix(
    actual: Optional[float],
    consensus: Optional[float],
    previous: Optional[float],
    indicator_name: Optional[str],
) -> str:
    """
    Generate special prefix for macro data (actual vs consensus)
    
    Args:
        actual: Actual value
        consensus: Market consensus
        previous: Previous value
        indicator_name: Indicator name
    
    Returns:
        Data prefix string (empty if not macro data)
    """
    # Only generate prefix when both actual and consensus exist
    if actual is None or consensus is None:
        return ""
    
    # Calculate surprise
    surprise = actual - consensus
    
    # Build prefix
    prefix_parts = []
    
    # Add indicator type (if available)
    if indicator_name and str(indicator_name).strip():
        prefix_parts.append(f"[{indicator_name}]")
    else:
        prefix_parts.append("[Economic Data]")
    
    # Add numeric info
    prefix_parts.append(f"[Actual{actual:.2f} Exp{consensus:.2f}]")
    
    # Add surprise direction
    if abs(surprise) > 0.01:  # Significant difference
        if surprise > 0:
            prefix_parts.append("[Beat]")
        else:
            prefix_parts.append("[Miss]")
    
    return " ".join(prefix_parts)


def generate_enhanced_text(
    row: pd.Series,
    trend_high: float,
    trend_low: float,
    vol_threshold: float,
) -> str:
    """
    Generate enhanced text for a single record
    
    Args:
        row: A DataFrame row
        trend_high: Strong trend threshold
        trend_low: Weak trend threshold
        vol_threshold: High volatility threshold
    
    Returns:
        Enhanced text
    """
    # 1. Get trend prefix
    pre_ret = _safe_float(row.get('pre_ret'))
    range_ratio = _safe_float(row.get('range_ratio'))
    trend_prefix = generate_trend_prefix(
        pre_ret, range_ratio, trend_high, trend_low, vol_threshold
    )
    
    # 2. Get macro data prefix (if applicable)
    actual = _safe_float(row.get('actual'))
    consensus = _safe_float(row.get('consensus'))
    previous = _safe_float(row.get('previous'))
    indicator_name = row.get('indicator_name')
    data_prefix = generate_data_prefix(actual, consensus, previous, indicator_name)
    
    # 3. Get original text
    original_text = str(row.get('text', '')).strip()
    if not original_text:
        original_text = str(row.get('content', '')).strip()
    if not original_text:
        original_text = str(row.get('name', '')).strip()
    
    # 4. Concatenate (trend prefix + data prefix + original text)
    parts = [trend_prefix]
    if data_prefix:
        parts.append(data_prefix)
    parts.append(original_text)
    
    return " ".join(parts)


def process_dataset(
    input_path: str,
    output_path: str,
    trend_high: float,
    trend_low: float,
    vol_threshold: float,
) -> None:
    """
    Process a single dataset file
    
    Args:
        input_path: Input CSV path
        output_path: Output CSV path
        trend_high: Strong trend threshold
        trend_low: Weak trend threshold
        vol_threshold: High volatility threshold
    """
    print(f"\nProcessing file: {input_path}")
    
    # Read data
    df = pd.read_csv(input_path, encoding='utf-8')
    print(f"  Original samples: {len(df)}")
    
    # Generate enhanced text
    df['text_enhanced'] = df.apply(
        lambda row: generate_enhanced_text(row, trend_high, trend_low, vol_threshold),
        axis=1
    )
    
    # Count prefix distribution (for verification)
    prefix_counts = df['text_enhanced'].str.extract(r'^\[([^\]]+)\]')[0].value_counts()
    print(f"  Prefix distribution:")
    for prefix, count in prefix_counts.head(10).items():
        print(f"    [{prefix}]: {count}")
    
    # Save (keep all original columns + new text_enhanced)
    _ensure_dir(output_path)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"  Saved to: {output_path}")
    print(f"  Output samples: {len(df)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add market context prefixes to financial event texts (Input Augmentation)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/processed",
        help="Input directory (containing train/val/test_multi_labeled.csv)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory (will generate train/val/test_enhanced.csv)"
    )
    parser.add_argument(
        "--trend_threshold_high",
        type=float,
        default=0.005,
        help="Strong trend threshold (default 0.5%%)"
    )
    parser.add_argument(
        "--trend_threshold_low",
        type=float,
        default=0.002,
        help="Weak trend threshold (default 0.2%%)"
    )
    parser.add_argument(
        "--volatility_threshold",
        type=float,
        default=0.008,
        help="High volatility threshold (default 0.8%%)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Data Augmentation: Add Market Context Prefixes")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Threshold settings:")
    print(f"  Strong trend: +/-{args.trend_threshold_high*100:.2f}%")
    print(f"  Weak trend: +/-{args.trend_threshold_low*100:.2f}%")
    print(f"  High volatility: {args.volatility_threshold*100:.2f}%")
    
    # Process three datasets
    for split in ['train', 'val', 'test']:
        input_file = os.path.join(args.input_dir, f"{split}_multi_labeled.csv")
        output_file = os.path.join(args.output_dir, f"{split}_enhanced.csv")
        
        if not os.path.exists(input_file):
            print(f"\nWarning: File not found, skipping {input_file}")
            continue
        
        process_dataset(
            input_file,
            output_file,
            args.trend_threshold_high,
            args.trend_threshold_low,
            args.volatility_threshold,
        )
    
    print("\n" + "=" * 60)
    print("Data augmentation completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check generated files: data/processed/*_enhanced.csv")
    print("2. Verify prefix distribution is reasonable")
    print("3. Train model with enhanced data:")
    print("   python scripts/modeling/bert_finetune_cls.py \\")
    print("     --train_csv data/processed/train_enhanced.csv \\")
    print("     --val_csv data/processed/val_enhanced.csv \\")
    print("     --test_csv data/processed/test_enhanced.csv \\")
    print("     --output_dir models/bert_enhanced_v1 \\")
    print("     --label_col label_multi_cls \\")
    print("     --class_weight auto \\")
    print("     --epochs 5 --lr 1e-5 --max_length 384")


if __name__ == "__main__":
    main()
