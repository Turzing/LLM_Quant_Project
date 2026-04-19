# step6_prepare_scoring_full.py
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "0_data"
OUTPUT_DIR = PROJECT_ROOT / "2_outputs"

# 加载LLM回复
df_llm = pd.read_csv(OUTPUT_DIR / "llm_responses_all.csv")
df_llm = df_llm[['id', 'type', 'reply']]
df_llm['source'] = 'LLM'

# 加载人类基线
df_human = pd.read_csv(DATA_DIR / "human_baseline_full.csv")
print("人类基线列名：", df_human.columns.tolist())  # 调试：查看有哪些列

# 如果没有 'id' 列，则用行号作为id
if 'id' not in df_human.columns:
    df_human['id'] = df_human.index

# 构造回复文本
df_human['reply'] = "建议：" + df_human['human_suggestion'] + "\n解释：" + df_human['human_explanation']
df_human = df_human[['id', 'reply']].copy()
df_human['source'] = 'Human'
df_human['type'] = 'human_baseline'
# 重新生成唯一id（因为可能重复）
df_human['id'] = "human_" + df_human.index.astype(str)

# 合并
df_all = pd.concat([df_llm, df_human], ignore_index=True)

# 随机排序
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

# 添加评分列
for col in ['逻辑性', '一致性', '完整性', '清晰度']:
    df_all[col] = ''
if 'adversarial' in df_all['type'].values:
    df_all['矛盾识别'] = ''
    df_all['权衡合理性'] = ''

# 保存
output_path = OUTPUT_DIR / "scoring_table_full.xlsx"
df_all.to_excel(output_path, index=False)
print(f"全量评分表已生成：{output_path}")
print(f"共 {len(df_all)} 条待评分（LLM: {len(df_llm)}, Human: {len(df_human)}）")