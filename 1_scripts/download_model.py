# download_model.py
import os
import sys

# 设置镜像站（国内加速）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

model_id = "NousResearch/Llama-2-7b-chat-hf"
local_dir = "D:/LLM_Quant_Project/models/llama-2-7b-chat-hf"

print(f"开始下载模型 {model_id}")
print(f"目标文件夹: {local_dir}")
print("由于模型较大（约13GB），请耐心等待...")

# 开始下载
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    ignore_patterns=["*.h5", "*.ot", "*.msgpack"]  # 忽略不必要的大文件
)

print(f"\n✅ 下载完成！模型保存在: {local_dir}")