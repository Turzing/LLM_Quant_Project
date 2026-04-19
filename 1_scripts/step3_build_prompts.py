# step3_build_prompts.py
# 构造三种 Prompt：原始常规、原始对抗、CoT修复版

import pandas as pd
import json
import random
from pathlib import Path

# 路径设置
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "0_data"

# 加载常规样本
df_regular = pd.read_csv(DATA_DIR / "regular_samples.csv")
print(f"常规样本总数: {len(df_regular)}")

# 设置随机种子
random.seed(42)

# ------------------------------
# 1. 构造原始版 Prompt（常规样本）
# ------------------------------
def build_original_prompt(row):
    prompt = f"""<s>[INST] <<SYS>>
你是一位专业的量化交易分析师。根据以下技术指标，给出明确的交易建议（买入/卖出/持有），并用一两句话解释理由。请引用具体数值。最后，请用0-100分评估你对这条建议的信心程度（0=完全不确定，100=绝对确定）。
<</SYS>>

- 收盘价：{row['close']}
- 20日均线：{row['ma20']}
- RSI(14)：{row['rsi']}
- MACD：{row['macd']}

输出格式：
建议：买入/卖出/持有
解释：因为...
置信度：0-100 [/INST]"""
    return prompt

# 为所有常规样本生成原始 prompt
original_prompts = []
for idx, row in df_regular.iterrows():
    original_prompts.append({
        'id': idx,
        'type': 'original',
        'stock': row['stock'],
        'date': row['date'],
        'prompt': build_original_prompt(row)
    })

# ------------------------------
# 2. 构造对抗性样本（基于常规样本，随机选30条修改指标）
# ------------------------------
# 随机抽取30条作为对抗性样本的基底
adversarial_base = df_regular.sample(n=30, random_state=42).copy()

def create_adversarial_row(row):
    """修改指标制造矛盾：RSI超买(>70) + MACD为正(金叉) + 价格远高于均线"""
    new_row = row.copy()
    new_row['rsi'] = 82.0               # 超买
    new_row['macd'] = 0.05              # 金叉
    new_row['close'] = new_row['ma20'] * 1.2   # 价格高于均线20%
    return new_row

adversarial_samples = adversarial_base.apply(create_adversarial_row, axis=1)

# 生成对抗性 prompt（使用相同的原始模板，只是指标变了）
adversarial_prompts = []
for idx, row in adversarial_samples.iterrows():
    adversarial_prompts.append({
        'id': f"adv_{idx}",
        'type': 'adversarial',
        'stock': row['stock'],
        'date': row['date'],
        'prompt': build_original_prompt(row)   # 复用原始模板
    })

# ------------------------------
# 3. 构造 CoT 修复版 Prompt（从常规样本中再随机选60条）
# ------------------------------
cot_base = df_regular.sample(n=60, random_state=123).copy()

def build_cot_prompt(row):
    prompt = f"""<s>[INST] <<SYS>>
你是一位专业的量化交易分析师。请严格按照以下步骤思考，最后给出交易建议和解释。
<</SYS>>

【步骤1】逐一分析每个指标：
- 收盘价与20日均线的关系（高于/低于/持平）
- RSI(14)的数值含义（<30超卖，30-70中性，>70超买）
- MACD值的方向（正/负/金叉/死叉）

【步骤2】综合所有指标，判断当前市场状态（强势/弱势/震荡/矛盾）。

【步骤3】基于以上分析，给出交易建议（买入/卖出/持有）和简要解释。

【步骤4】用0-100分评估你对这条建议的信心。

指标如下：
- 收盘价：{row['close']}
- 20日均线：{row['ma20']}
- RSI(14)：{row['rsi']}
- MACD：{row['macd']}

输出格式：
建议：买入/卖出/持有
解释：因为...
置信度：0-100 [/INST]"""
    return prompt

cot_prompts = []
for idx, row in cot_base.iterrows():
    cot_prompts.append({
        'id': f"cot_{idx}",
        'type': 'cot',
        'stock': row['stock'],
        'date': row['date'],
        'prompt': build_cot_prompt(row)
    })

# ------------------------------
# 4. 保存所有 Prompt 到 JSON 文件
# ------------------------------
all_prompts = original_prompts + adversarial_prompts + cot_prompts
print(f"总计生成 Prompt: {len(all_prompts)} 条")
print(f"  - 原始常规: {len(original_prompts)}")
print(f"  - 原始对抗: {len(adversarial_prompts)}")
print(f"  - CoT修复版: {len(cot_prompts)}")

output_path = DATA_DIR / "all_prompts.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_prompts, f, ensure_ascii=False, indent=2)

print(f"已保存到: {output_path}")

# 同时保存对抗样本和CoT样本的CSV，便于后续参考
adversarial_samples.to_csv(DATA_DIR / "adversarial_samples.csv", index=False)
cot_base.to_csv(DATA_DIR / "cot_samples.csv", index=False)
print("对抗样本和CoT样本的原始数据已保存为CSV。")