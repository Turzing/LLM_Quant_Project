# step1_download_data.py
from tickflow import TickFlow
import pandas as pd
import pickle
from pathlib import Path

# 路径设置
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "0_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 20只股票代码（TickFlow 格式：代码.市场后缀）
stocks = [
    '000001.SZ', '000002.SZ', '000063.SZ', '000858.SZ', '300015.SZ',
    '300750.SZ', '601318.SH', '600036.SH', '600519.SH', '600276.SH',
    '600887.SH', '002415.SZ', '000725.SZ', '002594.SZ', '300059.SZ',
    '688981.SH', '601888.SH', '603259.SH', '000333.SZ', '600900.SH'
]

tf = TickFlow.free()  # 免费模式，无需注册

def download_data(stock_list):
    """下载所有股票的日线数据"""
    data = {}
    for stock in stock_list:
        print(f"Downloading {stock}...")
        try:
            df = tf.klines.get(stock, period="1d", count=250, as_dataframe=True)
            if df.empty:
                print(f"Warning: {stock} no data")
                continue
            data[stock] = df
        except Exception as e:
            print(f"Error downloading {stock}: {e}")
    return data

print("开始下载20只股票1年日线数据...")
raw_data = download_data(stocks)

# 保存原始数据
with open(DATA_DIR / "stock_data.pkl", 'wb') as f:
    pickle.dump(raw_data, f)

print(f"数据下载完成，保存至: {DATA_DIR / 'stock_data.pkl'}")
print(f"成功下载 {len(raw_data)} 只股票")