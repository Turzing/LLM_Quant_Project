# step7_score_cli.py
# 命令行评分工具：逐条显示回复，输入分数，自动保存进度

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "2_outputs"

# 加载评分表（如果没有完整Excel，直接从原始数据生成）
def load_or_create_scoring_data():
    # 尝试加载已有的评分进度文件
    progress_file = OUTPUT_DIR / "scoring_progress.csv"
    if progress_file.exists():
        df = pd.read_csv(progress_file)
        print(f"加载已有进度，已评分 {df['逻辑性'].notna().sum()} 条")
        return df

    # 否则重新生成需要评分的条目
    df_llm = pd.read_csv(OUTPUT_DIR / "llm_responses_all.csv")
    df_llm = df_llm[['id', 'type', 'reply']]
    df_llm['source'] = 'LLM'

    df_human = pd.read_csv(PROJECT_ROOT / "0_data" / "human_baseline_full.csv")
    if 'id' not in df_human.columns:
        df_human['id'] = df_human.index
    df_human['reply'] = "建议：" + df_human['human_suggestion'] + "\n解释：" + df_human['human_explanation']
    df_human = df_human[['id', 'reply']]
    df_human['source'] = 'Human'
    df_human['type'] = 'human_baseline'
    df_human['id'] = "human_" + df_human.index.astype(str)

    df_all = pd.concat([df_llm, df_human], ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    # 添加评分列
    for col in ['逻辑性', '一致性', '完整性', '清晰度']:
        df_all[col] = pd.NA
    if 'adversarial' in df_all['type'].values:
        df_all['矛盾识别'] = pd.NA
        df_all['权衡合理性'] = pd.NA

    return df_all

def save_progress(df):
    progress_file = OUTPUT_DIR / "scoring_progress.csv"
    df.to_csv(progress_file, index=False, encoding='utf-8-sig')
    print(f"进度已保存至 {progress_file}")

def main():
    df = load_or_create_scoring_data()
    total = len(df)
    print(f"共 {total} 条待评分。输入数字分数，按Enter确认。输入 q 退出。")

    for idx, row in df.iterrows():
        # 跳过已评分的
        if pd.notna(row['逻辑性']):
            continue

        print("\n" + "="*60)
        print(f"[{idx+1}/{total}] ID: {row['id']} | 类型: {row['type']} | 来源: {row['source']}")
        print("-"*60)
        print("回复内容:")
        print(row['reply'])
        print("-"*60)

        # 获取评分
        def get_score(prompt):
            while True:
                val = input(prompt).strip()
                if val.lower() == 'q':
                    save_progress(df)
                    sys.exit(0)
                if val.isdigit() and 1 <= int(val) <= 5:
                    return int(val)
                print("请输入1-5之间的整数")

        df.at[idx, '逻辑性'] = get_score("逻辑性 (1-5): ")
        df.at[idx, '一致性'] = get_score("一致性 (1-5): ")
        df.at[idx, '完整性'] = get_score("完整性 (1-5): ")
        df.at[idx, '清晰度'] = get_score("清晰度 (1-5): ")

        if row['type'] == 'adversarial' and row['source'] == 'LLM':
            df.at[idx, '矛盾识别'] = get_score("矛盾识别 (1-5): ")
            df.at[idx, '权衡合理性'] = get_score("权衡合理性 (1-5): ")

        # 每10条自动保存一次
        if (idx+1) % 10 == 0:
            save_progress(df)

    save_progress(df)
    print("\n🎉 评分完成！")
    # 生成最终评分表
    df.to_excel(OUTPUT_DIR / "scoring_table_full.xlsx", index=False)
    print(f"已导出完整评分表至 {OUTPUT_DIR / 'scoring_table_full.xlsx'}")

if __name__ == "__main__":
    main()