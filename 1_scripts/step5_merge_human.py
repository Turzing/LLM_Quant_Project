# step5_merge_human.py
import pandas as pd
from pathlib import Path

def read_excel_flexible(filepath):
    """读取Excel文件（支持 .xlsx 和 .xls）"""
    return pd.read_excel(filepath, engine='openpyxl')

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "0_data"

# 加载常规样本（可能是CSV或Excel）
samples_path = DATA_DIR / "regular_samples.csv"
if not samples_path.exists():
    samples_path = DATA_DIR / "regular_samples.xlsx"
df_samples = pd.read_csv(samples_path) if samples_path.suffix == '.csv' else pd.read_excel(samples_path)
df_samples['id'] = df_samples.index

# 加载三位评分员的结果（Excel格式）
human_A = read_excel_flexible(DATA_DIR / "human_A.xlsx")
human_B = read_excel_flexible(DATA_DIR / "human_B.xlsx")
human_C = read_excel_flexible(DATA_DIR / "human_C.xlsx")

# 添加评分员标识
human_A['rater'] = 'A'
human_B['rater'] = 'B'
human_C['rater'] = 'C'

# 合并
human_all = pd.concat([human_A, human_B, human_C], ignore_index=True)
human_all.to_csv(DATA_DIR / "human_baseline_full.csv", index=False, encoding='utf-8-sig')
print(f"人类基线已合并，共 {len(human_all)} 条（3人 × {len(df_samples)}条）")
print(f"保存至 {DATA_DIR / 'human_baseline_full.csv'}")