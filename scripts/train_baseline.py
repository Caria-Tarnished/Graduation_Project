# -*- coding: utf-8 -*-
import argparse
import json
import os
import sqlite3
from typing import List, Optional, Tuple

import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import LinearSVC


# 加载项目配置
def _load_cfg() -> dict:
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# 读取 SQLite 数据库路径，默认 finance.db
def _db_path(cfg: dict) -> str:
    return cfg.get('storage', {}).get('sqlite_path', 'finance.db')


# 将“天”为单位的窗口转为分钟（与 labels 表 window_min 对齐）
def _to_window_min(days: int) -> int:
    return int(days) * 1440


# 从 DB 读取 labels 与 news 联结的数据，构造 (ts, text, label)
def _load_db(
    conn: sqlite3.Connection,
    window_min: int,
    tickers: Optional[List[str]],
    text_source: str,
) -> pd.DataFrame:
    parts = [
        "SELECT l.news_id, l.ticker, l.window_min, l.label, ",
        "n.title, n.content, n.ts AS ts ",
        "FROM labels l JOIN news n ON l.news_id = n.id ",
        "WHERE l.window_min = ?",
    ]
    sql = "".join(parts)
    params: List = [int(window_min)]
    if tickers:
        placeholders = ",".join(["?"] * len(tickers))
        sql += f" AND l.ticker IN ({placeholders})"
        params.extend(tickers)

    try:
        df = pd.read_sql_query(sql, conn, params=params)
    except Exception as e:
        raise RuntimeError(f"read_sql_query failed: {e}") from e

    if df.empty:
        return df

    df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
    if text_source == 'title':
        df['text'] = df['title'].fillna('')
    elif text_source == 'content':
        df['text'] = df['content'].fillna('')
    else:
        # both：标题 + 内容
        df['text'] = (df['title'].fillna('') + '。' + df['content'].fillna(''))

    df = df[['ts', 'text', 'label']].dropna()
    return df


# 从 CSV 读取 (ts, text, label)；假设 CSV 已带标签
def _load_csv(
    csv_path: str,
    text_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # 解析时间列：优先 ts，其次 date；都没有则 NaT
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
    elif 'date' in df.columns:
        df['ts'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['ts'] = pd.NaT

    if text_col not in df.columns:
        raise ValueError(f'text column {text_col} not found in CSV')
    if 'label' not in df.columns:
        raise ValueError('CSV must contain label column')

    out = df[['ts', text_col, 'label']].rename(columns={text_col: 'text'})
    out = out[out['text'].notna()].copy()
    return out


# 按时间进行切分；可指定 split_date 或按比例切分
def _time_split(
    df: pd.DataFrame,
    split_date: Optional[str],
    train_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values('ts')
    if split_date:
        dt = pd.to_datetime(split_date, errors='coerce')
        if pd.isna(dt):
            raise ValueError('invalid --split-date')
        tr = df[df['ts'] <= dt]
        te = df[df['ts'] > dt]
        return tr, te

    n = len(df)
    cut = max(1, int(round(n * train_ratio)))
    tr = df.iloc[:cut]
    te = df.iloc[cut:]
    return tr, te


# TF-IDF 向量化
def _vectorize(
    tr_text: List[str],
    te_text: List[str],
    analyzer: str,
    ngram_min: int,
    ngram_max: int,
    max_features: Optional[int],
):
    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=(ngram_min, ngram_max),
        max_features=max_features,
        lowercase=False,
    )
    Xtr = vec.fit_transform(tr_text)
    Xte = vec.transform(te_text)
    return vec, Xtr, Xte


# 训练 LinearSVC 并评估，返回指标与混淆矩阵
def _train_and_eval(Xtr, ytr, Xte, yte) -> dict:
    clf = LinearSVC(class_weight='balanced')
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    report = classification_report(yte, yhat, output_dict=True)
    f1m = f1_score(yte, yhat, average='macro')
    cm = confusion_matrix(yte, yhat).tolist()
    return {
        'f1_macro': float(f1m),
        'report': report,
        'cm': cm,
        'y_pred': yhat.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--from-db', action='store_true', help='train from DB join')
    p.add_argument('--labels-csv', help='train from labels CSV (with text)')
    p.add_argument(
        '--text-source',
        default='title',
        choices=['title', 'content', 'both', 'csv_col'],
        help='DB: title/content/both; CSV: use csv_col',
    )
    p.add_argument(
        '--csv-text-col',
        default='title',
        help='text column name in labels CSV',
    )
    p.add_argument(
        '--window-days',
        type=int,
        default=1,
        help='label window in days for DB mode',
    )
    p.add_argument('--tickers', nargs='*', help='filter tickers in DB mode')
    p.add_argument(
        '--analyzer',
        default='char_wb',
        choices=['char', 'char_wb', 'word'],
    )
    p.add_argument('--ngram-min', type=int, default=2)
    p.add_argument('--ngram-max', type=int, default=4)
    p.add_argument('--max-features', type=int, default=50000)
    p.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='time split ratio if no --split-date',
    )
    p.add_argument('--split-date', help='YYYY-MM-DD for time split')
    p.add_argument('--save-report', default='reports/baseline_report.json')
    p.add_argument('--save-preds', default='reports/baseline_preds.csv')
    args = p.parse_args()

    if not args.from_db and not args.labels_csv:
        raise SystemExit('specify --from-db or --labels-csv')

    cfg = _load_cfg()

    if args.from_db:
        conn = sqlite3.connect(_db_path(cfg))
        try:
            df = _load_db(
                conn=conn,
                window_min=_to_window_min(args.window_days),
                tickers=args.tickers,
                text_source=(
                    args.text_source if args.text_source != 'csv_col'
                    else 'title'
                ),
            )
        finally:
            conn.close()
    else:
        df = _load_csv(args.labels_csv, args.csv_text_col)

    if df.empty:
        raise SystemExit('no training data available')

    tr, te = _time_split(df, args.split_date, args.train_ratio)
    if tr.empty or te.empty:
        raise SystemExit(
            'empty train/test after split; adjust --split-date/ratio'
        )

    _, Xtr, Xte = _vectorize(
        tr['text'].astype(str).tolist(),
        te['text'].astype(str).tolist(),
        analyzer=args.analyzer,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        max_features=args.max_features,
    )

    res = _train_and_eval(Xtr, tr['label'].tolist(), Xte, te['label'].tolist())
    print(
        json.dumps(
            {'f1_macro': res['f1_macro']},
            ensure_ascii=False,
            indent=2,
        )
    )

    # 保存结果文件（若目录存在）
    rep_dir = os.path.dirname(args.save_report)
    if rep_dir:
        os.makedirs(rep_dir, exist_ok=True)
    with open(args.save_report, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'f1_macro': res['f1_macro'],
                'report': res['report'],
                'cm': res['cm'],
                'params': {
                    'analyzer': args.analyzer,
                    'ngram_range': [args.ngram_min, args.ngram_max],
                    'window_days': args.window_days,
                    'from_db': args.from_db,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    preds_dir = os.path.dirname(args.save_preds)
    if preds_dir:
        os.makedirs(preds_dir, exist_ok=True)
    pd.DataFrame(
        {
            'ts': te['ts'].astype(str).tolist(),
            'text': te['text'].tolist(),
            'label': te['label'].tolist(),
            'pred': res['y_pred'],
        }
    ).to_csv(args.save_preds, index=False)
    print(f"saved report -> {args.save_report}")
    print(f"saved preds  -> {args.save_preds}")


if __name__ == '__main__':
    main()
