# -*- coding: utf-8 -*-
import sqlite3, pandas as pd

conn = sqlite3.connect(r'e:\Projects\Graduation_Project\finance_analysis.db')

# 先看看表结构
cur = conn.cursor()
cur.execute("PRAGMA table_info(event_impacts)")
cols = cur.fetchall()
print('=== event_impacts 列名 ===', [c[1] for c in cols])

df2 = pd.read_sql("SELECT * FROM event_impacts LIMIT 3", conn)
print(df2)

conn.close()
