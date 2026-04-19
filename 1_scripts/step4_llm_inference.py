# step4_llm_inference.py
# 大模型推理脚本：支持4-bit量化，自动回退CPU模式，带详细进度

import json
import torch
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm
import sys
import subprocess

# ==================== 路径设置 ====================
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "0_data"
OUTPUT_DIR = PROJECT_ROOT / "2_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 模型路径（请确认本地模型已下载）====================
# 你可以改成你自己的本地模型路径
MODEL_PATH = "D:/LLM_Quant_Project/models/Llama-2-7b-chat-hf"
# 如果上面路径不存在，尝试另一个常见命名
if not Path(MODEL_PATH).exists():
    MODEL_PATH = "D:/LLM_Quant_Project/models/llama-2-7b-chat-hf"

print(f"模型路径: {MODEL_PATH}")
if not Path(MODEL_PATH).exists():
    print("错误：找不到本地模型文件夹！")
    print("请确保已经下载模型到上述路径，或修改 MODEL_PATH 变量。")
    sys.exit(1)

# ==================== 依赖检查 ====================
def check_and_install(package):
    try:
        __import__(package)
        print(f"✓ {package} 已安装")
    except ImportError:
        print(f"✗ {package} 未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 必须的库
required_packages = ['transformers', 'torch', 'sentencepiece', 'accelerate', 'bitsandbytes']
for pkg in required_packages:
    check_and_install(pkg)

# ==================== 加载 prompts ====================
with open(DATA_DIR / "all_prompts.json", 'r', encoding='utf-8') as f:
    all_prompts = json.load(f)

print(f"共加载 {len(all_prompts)} 条prompt")
print("类型分布:")
type_counts = {}
for p in all_prompts:
    t = p['type']
    type_counts[t] = type_counts.get(t, 0) + 1
for t, cnt in type_counts.items():
    print(f"  {t}: {cnt}")

# ==================== 加载模型（尝试4-bit，失败则CPU）====================
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 尝试使用GPU量化
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"可用设备: {device}")

model = None
if device == "cuda":
    try:
        print("尝试使用 4-bit 量化加载模型...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("✓ 4-bit 量化加载成功")
    except Exception as e:
        print(f"⚠️ 4-bit 量化失败: {e}")
        print("将回退到 CPU 模式（可能较慢）")
        device = "cpu"

if model is None:
    # CPU 模式
    print("加载模型到 CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    print("✓ CPU 加载完成")

model.eval()

# ==================== 推理函数 ====================
def generate_response(prompt, max_new_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    if device == "cuda":
        inputs = inputs.to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取模型生成的部分（去掉输入prompt）
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    return response

# ==================== 批量推理 ====================
results = []
start_total = time.time()

with tqdm(total=len(all_prompts), desc="推理进度", unit="条") as pbar:
    for i, item in enumerate(all_prompts):
        start_time = time.time()
        prompt = item['prompt']
        pbar.set_postfix_str(f"当前: {item['type']}_{item['id']}")
        try:
            reply = generate_response(prompt)
            elapsed = time.time() - start_time
            results.append({
                'id': item['id'],
                'type': item['type'],
                'stock': item['stock'],
                'date': item['date'],
                'prompt': prompt,
                'reply': reply,
                'inference_time': round(elapsed, 2)
            })
            # 每10条打印示例
            if (i+1) % 10 == 0:
                sample = reply[:100].replace('\n', ' ')
                tqdm.write(f"[示例] ID={item['id']} 回复: {sample}...")
        except Exception as e:
            elapsed = time.time() - start_time
            tqdm.write(f"[错误] ID={item['id']}: {str(e)}")
            results.append({
                'id': item['id'],
                'type': item['type'],
                'stock': item['stock'],
                'date': item['date'],
                'prompt': prompt,
                'reply': f"ERROR: {str(e)}",
                'inference_time': round(elapsed, 2)
            })
        pbar.update(1)
        # 每10条保存中间结果
        if (i+1) % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(OUTPUT_DIR / "llm_responses_temp.csv", index=False)
            tqdm.write(f"[保存] 已保存 {len(results)} 条中间结果")

# 最终保存
df_results = pd.DataFrame(results)
output_file = OUTPUT_DIR / "llm_responses_all.csv"
df_results.to_csv(output_file, index=False)

total_time = time.time() - start_total
avg_time = df_results['inference_time'].mean()
print(f"\n✅ 推理完成！共 {len(results)} 条回复")
print(f"   保存至: {output_file}")
print(f"   平均每条耗时: {avg_time:.2f} 秒")
print(f"   总耗时: {total_time/60:.1f} 分钟")