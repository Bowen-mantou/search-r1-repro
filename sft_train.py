#!/usr/bin/env python3
"""
SFT 冷启动微调脚本 (4-bit QLoRA on Qwen3.5-4B)

适配 AutoDL / 本地 GPU，对蒸馏轨迹进行 LoRA SFT 微调。

Usage:
    python sft_train.py [--data distilled_trajectories_fixed.jsonl] [--output ./sft_checkpoint]
                         [--model Qwen/Qwen3.5-4B] [--epochs 3]

AutoDL 环境准备:
    pip install transformers peft trl accelerate bitsandbytes datasets

Data:   distilled_trajectories_fixed.jsonl  (397 trajectories)
Output: ./sft_checkpoint/  (LoRA adapter weights only)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer


# ============================================================================
# Configuration (override via command line)
# ============================================================================

AUTODL_TMP = Path.home() / "autodl-tmp"
BASE_MODEL = os.environ.get("SFT_MODEL", str(Path.home() / "models" / "Qwen3.5-4B"))
DATA_PATH = Path(os.environ.get("SFT_DATA", str(AUTODL_TMP / "distilled_trajectories_fixed.jsonl")))
OUTPUT_DIR = Path(os.environ.get("SFT_OUTPUT", str(AUTODL_TMP / "sft_checkpoint")))

# ---------- QLoRA ----------
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# ---------- Training ----------
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.1
MAX_LENGTH = 2048

# ============================================================================
# Data processing
# ============================================================================


def load_and_format_data(data_path: Path) -> Dataset:
    """Load JSONL records and wrap trajectory_text in Qwen3.5 chat format.

    Each record contains:
      - messages:   [{"role": "system/user/assistant/tool", "content": "..."}]
      - trajectory_text:  pure text of the full multi-turn search trajectory

    We extract system prompt & user question from `messages`, then combine
    with `trajectory_text` (as assistant response) using the Qwen3.5
    ``<|im_start|>role\\ncontent<|im_end|>`` chat markers.

    The resulting text is a single sequence on which standard CLM loss is
    computed (next-token prediction over the whole chat string).
    """
    texts = []

    with open(data_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            system_prompt = ""
            question = ""
            for msg in record.get("messages", []):
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                elif msg["role"] == "user":
                    question = msg["content"]

            trajectory = record.get("trajectory_text", "")
            if not question or not trajectory:
                print(f"  Skipping line {lineno}: missing question or trajectory_text")
                continue

            # Build Qwen3.5 chat format
            parts = []
            if system_prompt:
                parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
            parts.append(f"<|im_start|>user\n{question}<|im_end|>")
            parts.append(f"<|im_start|>assistant\n{trajectory}<|im_end|>")

            texts.append({"text": "\n".join(parts)})

    print(f"  Loaded {len(texts)} formatted examples from {data_path.name}")
    return Dataset.from_list(texts)


# ============================================================================
# Callback  --  print loss at each logging step
# ============================================================================


class LossLogCallback(TrainerCallback):
    """Print training loss and learning rate at every logging step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            lr = logs.get("learning_rate", 0)
            print(
                f"  Step {state.global_step:5d} | "
                f"loss = {logs['loss']:.6f} | "
                f"lr   = {lr:.2e}"
            )


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT Cold-Start Fine-Tuning")
    parser.add_argument("--model", default=BASE_MODEL, help="Base model name or path")
    parser.add_argument("--data", default=str(DATA_PATH), help="JSONL data file path")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Training epochs")
    parser.add_argument("--hf-mirror", action="store_true", help="Use HF mirror (hf-mirror.com) for China")
    args_cli = parser.parse_args()

    # --- HF mirror ---
    if args_cli.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # --- System info ---
    print("=" * 62)
    print("  SFT Cold-Start Fine-Tuning  |  Qwen3.5-4B  |  4-bit QLoRA")
    print("=" * 62)
    print(f"  PyTorch   : {torch.__version__}")
    if not torch.cuda.is_available():
        print("  GPU       : NOT AVAILABLE -- aborting")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"  GPU       : {gpu_name} ({vram_gb:.1f} GB)")
    print(f"  Base model: {args_cli.model}")
    print(f"  Data      : {args_cli.data}")
    print(f"  Output    : {args_cli.output}")
    print(f"  Epochs    : {args_cli.epochs}")
    print()

    # ====================================================================
    # 1. Data
    # ====================================================================
    print("[1/5] Loading & formatting data ...")
    t0 = time.perf_counter()
    dataset = load_and_format_data(Path(args_cli.data))
    print(f"       Done in {time.perf_counter() - t0:.1f}s\n")

    # ====================================================================
    # 2. Tokenizer
    # ====================================================================
    print("[2/5] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args_cli.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"       Vocab size: {len(tokenizer)}\n")

    # ====================================================================
    # 3. Model  --  4-bit QLoRA
    # ====================================================================
    print("[3/5] Loading model with 4-bit quantization ...")
    t0 = time.perf_counter()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args_cli.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    # Prepare for k-bit quantized training and enable gradient checkpointing
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    torch.cuda.synchronize()
    mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"       Model loaded in {time.perf_counter() - t0:.1f}s")
    print(f"       GPU memory: {mem_alloc:.2f} GB allocated / {mem_reserved:.2f} GB reserved\n")

    # ====================================================================
    # 4. LoRA
    # ====================================================================
    print("[4/5] Configuring LoRA ...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print()

    # ====================================================================
    # 5. Training
    # ====================================================================
    effective_bs = BATCH_SIZE * GRAD_ACCUM
    print("[5/5] Starting SFT training ...")
    print(f"       Config: epochs={args_cli.epochs}  bs={BATCH_SIZE}  grad_accum={GRAD_ACCUM}  "
          f"effective_bs={effective_bs}")
    print(f"       lr={LEARNING_RATE}  warmup={WARMUP_RATIO}  max_length={MAX_LENGTH}")
    print(f"       LoRA: r={LORA_R}  alpha={LORA_ALPHA}  dropout={LORA_DROPOUT}")
    print()

    sft_config = SFTConfig(
        output_dir=args_cli.output,
        num_train_epochs=args_cli.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        max_length=MAX_LENGTH,
        dataset_text_field="text",
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        save_only_model=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to=[],
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[LossLogCallback()],
    )

    train_start = time.perf_counter()
    trainer.train()
    train_minutes = (time.perf_counter() - train_start) / 60.0

    # --- Save final adapter ---
    out_dir = Path(args_cli.output)
    print(f"\n  Saving LoRA adapter to {out_dir} ...")
    trainer.save_model()
    tokenizer.save_pretrained(out_dir)

    print()
    print("=" * 62)
    print(f"  Training complete!  Time: {train_minutes:.1f} min")
    print(f"  Adapter saved to: {out_dir.absolute()}")
    print()
    print("  Load for inference:")
    print(f"    from peft import PeftModel")
    print(f"    from transformers import AutoModelForCausalLM")
    print(f"    model = AutoModelForCausalLM.from_pretrained(")
    print(f"        '{args_cli.model}',")
    print(f"        load_in_4bit=True,")
    print(f"        device_map='auto',")
    print(f"        trust_remote_code=True,")
    print(f"    )")
    print(f"    model = PeftModel.from_pretrained(model, '{out_dir}')")
    print("=" * 62)


if __name__ == "__main__":
    main()
