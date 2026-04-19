# step2_sample.py
# 从 TickFlow 原始数据计算指标，并随机抽样

import pandas as pd
import pickle
import random
import ta
from pathlib import Path

# 路径设置
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "0_data"   # 注意是数字0，不是字母o

# 加载原始数据
with open(DATA_DIR / "stock_data.pkl", 'rb') as f:
    raw_data = pickle.load(f)

def add_indicators(df):
    """计算技术指标，输入 df 索引是日期，列包含 open, high, low, close, volume"""
    df = df.copy()
    # 将日期索引变成普通列
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    
    # 确保列名小写
    df.columns = [col.lower() for col in df.columns]
    
    # 计算指标
    df['ma20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # 删除前20行的NaN
    df = df.dropna().reset_index(drop=True)
    return df

# 为每只股票计算指标
processed_data = {}
for stock, df in raw_data.items():
    print(f"Processing {stock}...")
    try:
        processed_data[stock] = add_indicators(df)
        print(f"  -> {len(processed_data[stock])} valid days")
    except Exception as e:
        print(f"  -> Error: {e}")

# 随机抽样（每只股票抽6个交易日）
random.seed(42)
samples = []

for stock, df in processed_data.items():
    valid_indices = df.index.tolist()
    n = min(6, len(valid_indices))
    if n == 0:
        continue
    sampled_indices = random.sample(valid_indices, n)
    for idx in sampled_indices:
        row = df.loc[idx]
        # 日期已经是 datetime 对象
        date_val = row['date']
        if hasattr(date_val, 'strftime'):
            date_str = date_val.strftime('%Y-%m-%d')
        else:
            date_str = str(date_val)
        samples.append({
            'stock': stock,
            'date': date_str,
            'close': round(row['close'], 2),
            'ma20': round(row['ma20'], 2),
            'rsi': round(row['rsi'], 1),
            'macd': round(row['macd'], 4)
        })

df_samples = pd.DataFrame(samples)
output_path = DATA_DIR / "regular_samples.csv"
df_samples.to_csv(output_path, index=False)
print(f"总样本数: {len(df_samples)}")
print(f"已保存: {output_path}")