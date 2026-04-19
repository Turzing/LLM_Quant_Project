# step8_analysis_full.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（避免图表中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "2_outputs"
FIGURE_DIR = PROJECT_ROOT / "3_figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 加载评分数据
def load_scores():
    # 优先尝试加载完整评分表
    excel_path = OUTPUT_DIR / "scoring_table_full.xlsx"
    csv_path = OUTPUT_DIR / "scoring_progress.csv"
    if excel_path.exists():
        df = pd.read_excel(excel_path)
        print(f"加载评分表: {excel_path}")
    elif csv_path.exists():
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"加载评分进度: {csv_path}")
    else:
        raise FileNotFoundError("未找到评分数据文件，请先完成第七步评分")
    
    # 确保分数列为数值
    score_cols = ['逻辑性', '一致性', '完整性', '清晰度']
    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if '矛盾识别' in df.columns:
        df['矛盾识别'] = pd.to_numeric(df['矛盾识别'], errors='coerce')
    if '权衡合理性' in df.columns:
        df['权衡合理性'] = pd.to_numeric(df['权衡合理性'], errors='coerce')
    
    # 计算EDCS（四个维度平均）
    df['edcs'] = df[score_cols].mean(axis=1)
    return df

df = load_scores()
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# 1. 整体统计
print("\n=== 1. 整体EDCS统计 ===")
print(df['edcs'].describe())

# 2. 按来源（LLM vs Human）统计
print("\n=== 2. 按来源统计 ===")
src_stats = df.groupby('source')['edcs'].agg(['mean', 'std', 'count'])
print(src_stats)

# t检验
llm_scores = df[df['source']=='LLM']['edcs'].dropna()
human_scores = df[df['source']=='Human']['edcs'].dropna()
t_stat, p_val = stats.ttest_ind(llm_scores, human_scores)
print(f"\nLLM vs Human t检验: t={t_stat:.3f}, p={p_val:.4f}")

# 3. 按类型统计（original, adversarial, cot, human_baseline）
print("\n=== 3. 按类型统计 ===")
type_stats = df.groupby('type')['edcs'].agg(['mean', 'std', 'count'])
print(type_stats)

# 4. 对抗性样本分析
adv_df = df[df['type'] == 'adversarial']
if not adv_df.empty and '矛盾识别' in adv_df.columns:
    print("\n=== 4. 对抗性样本矛盾识别与权衡合理性 ===")
    print(f"矛盾识别平均分: {adv_df['矛盾识别'].mean():.2f}")
    print(f"权衡合理性平均分: {adv_df['权衡合理性'].mean():.2f}")
    # 失败率（矛盾识别<3）
    fail_rate = (adv_df['矛盾识别'] < 3).mean() * 100
    print(f"矛盾识别失败率（<3分）: {fail_rate:.1f}%")

# 5. 置信度分析（如果LLM回复中有confidence列）
# 需要从原始llm_responses_all.csv中提取置信度
if 'id' in df.columns:
    llm_responses = pd.read_csv(OUTPUT_DIR / "llm_responses_all.csv")
    if 'confidence' in llm_responses.columns:
        # 合并置信度
        df_conf = df[df['source']=='LLM'].merge(
            llm_responses[['id', 'confidence']], on='id', how='left'
        )
        df_conf = df_conf.dropna(subset=['confidence', 'edcs'])
        if len(df_conf) > 1:
            corr, p_corr = stats.pearsonr(df_conf['confidence'], df_conf['edcs'])
            print(f"\n=== 5. 置信度与EDCS相关性 ===")
            print(f"皮尔逊相关系数: r={corr:.3f}, p={p_corr:.4f}")
            # 校准曲线
            df_conf['conf_bin'] = pd.cut(df_conf['confidence'], bins=10, labels=False)
            calib = df_conf.groupby('conf_bin').agg(
                mean_conf=('confidence', 'mean'),
                mean_edcs=('edcs', 'mean')
            )
            # 绘图
            plt.figure(figsize=(6,6))
            plt.plot([0,100], [0,5], 'k--', alpha=0.5, label='Ideal')
            plt.scatter(calib['mean_conf'], calib['mean_edcs'], c='red', label='LLaMA-2')
            plt.xlabel('Mean Confidence')
            plt.ylabel('Mean EDCS')
            plt.title('Calibration Curve: Confidence vs EDCS')
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / 'calibration_curve.png')
            print(f"校准曲线保存至 {FIGURE_DIR / 'calibration_curve.png'}")

# 6. 绘图
# 图1：箱线图（按类型）
plt.figure(figsize=(10,6))
order = ['original', 'adversarial', 'cot', 'human_baseline']
order = [o for o in order if o in df['type'].unique()]
sns.boxplot(data=df, x='type', y='edcs', order=order)
plt.title('EDCS Distribution by Prompt Type')
plt.ylabel('EDCS (1-5)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'edcs_by_type.png')
print(f"箱线图保存至 {FIGURE_DIR / 'edcs_by_type.png'}")

# 图2：柱状图（LLM vs Human）
plt.figure(figsize=(6,5))
means = src_stats['mean']
errors = src_stats['std'] / np.sqrt(src_stats['count'])
plt.bar(means.index, means.values, yerr=errors.values, capsize=5, color=['blue', 'orange'])
plt.ylabel('Mean EDCS')
plt.title('LLM vs Human Performance')
plt.ylim(0, 5)
for i, (idx, val) in enumerate(means.items()):
    plt.text(i, val + 0.1, f'{val:.2f}', ha='center')
plt.savefig(FIGURE_DIR / 'llm_vs_human.png')
print(f"柱状图保存至 {FIGURE_DIR / 'llm_vs_human.png'}")

# 图3：各维度得分对比（雷达图或分组柱状图）
dimensions = ['逻辑性', '一致性', '完整性', '清晰度']
llm_dims = df[df['source']=='LLM'][dimensions].mean()
human_dims = df[df['source']=='Human'][dimensions].mean()
x = np.arange(len(dimensions))
width = 0.35
plt.figure(figsize=(8,5))
plt.bar(x - width/2, llm_dims, width, label='LLM')
plt.bar(x + width/2, human_dims, width, label='Human')
plt.xticks(x, dimensions)
plt.ylabel('Score')
plt.ylim(0,5)
plt.legend()
plt.title('Dimension-wise Comparison')
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'dimension_comparison.png')
print(f"维度对比图保存至 {FIGURE_DIR / 'dimension_comparison.png'}")

print("\n✅ 分析完成！图表已保存至 3_figures 文件夹。")