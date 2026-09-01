#!/usr/bin/env python3
"""合并 SFT LoRA adapter 到 Qwen3.5-4B 基座模型

Output: ~/autodl-tmp/qwen3.5-4b-sft-merged/  (完整 fp16 模型, ~8GB)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/root/models/Qwen3.5-4B"
ADAPTER = "/root/autodl-tmp/sft_checkpoint"
OUT = "/root/autodl-tmp/qwen3.5-4b-sft-merged"

print("[1/3] Loading base model in fp16 ...")
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
)

print("[2/3] Merging LoRA adapter ...")
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()

print(f"[3/3] Saving to {OUT} ...")
model.save_pretrained(OUT)

tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tokenizer.save_pretrained(OUT)

total_gb = sum(p.numel() * 2 for p in model.parameters()) / 1e9
print(f"Done! {OUT}/  ({total_gb:.1f} GB fp16)")
