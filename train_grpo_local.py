#!/usr/bin/env python3
"""
SFT+GRPO 本地训练 (4-bit QLoRA, AutoDL 3060 12GB)

从 SFT 合并模型出发做 GRPO 精调，不依赖 PyTRIO。
支持 LLDS 正则化（已接入训练循环）。

Usage:
    # 纯 GRPO（无 LLDS）
    python train_grpo_local.py --max-steps 50 --questions-per-batch 8 --group-size 8

    # GRPO + LLDS
    python train_grpo_local.py --max-steps 50 --questions-per-batch 8 --group-size 8 \
        --llds-lambda 0.05 --llds-variant A
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from data import SearchExample, shuffled_examples, take_batch
from protocol import (
    build_prompt,
    initial_messages,
    parse_assistant,
    stop_sequences,
)
from reward import score_answer
from reward_lite import PRMLiteScorer, apply_lata
from search import (
    SEARCH_BACKENDS,
    SearchClient,
    SearchResult,
    create_search_client,
    format_item,
    resolve_search_concurrency,
    resolve_search_timeout,
)

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

BASE_MODEL = "wang072266/qwen3.5-4b-search-r1-sft"
OUTPUT_DIR = Path("grpo_checkpoint")

MAX_SEARCH_CALLS = 4
MAX_ASSISTANT_TURNS = 6
MAX_TRAJECTORY_TOKENS = 8192
MAX_ASSISTANT_TOKENS = 1024
MAX_TOOL_RESPONSE_TOKENS = 1024

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

LEARNING_RATE = 1e-5
GRAD_ACCUM = 4

MICRO_BATCH_MAX_TOKENS = 4000
MAX_TRAIN_SEQ_LEN = 4096  # hard cap per training sequence to prevent OOM


# ═══════════════════════════════════════════════════════════════════════════
# Stopping criteria
# ═══════════════════════════════════════════════════════════════════════════

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids: list[int]):
        self.stop_ids = set(stop_ids)

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        for sid in self.stop_ids:
            if input_ids[0, -1].item() == sid:
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Turn:
    prompt_tokens: list[int]
    completion_tokens: list[int]
    completion_text: str
    is_tool_call: bool
    tool_result: SearchResult | None = None
    tool_content: str = ""
    logprobs: list[float] = field(default_factory=list)


@dataclass
class Trajectory:
    example: SearchExample
    turns: list[Turn]
    final_text: str = ""
    reward: float = 0.0
    advantage: float = 0.0
    search_calls: int = 0
    valid_format: bool = False
    anomalous: bool = False
    messages: list = field(default_factory=list)
    question_index: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Rollout
# ═══════════════════════════════════════════════════════════════════════════

def _run_search(search_client: SearchClient, query: str, call_id: str, timeout: float) -> SearchResult:
    """Execute a search query synchronously."""
    return search_client.search(query)


def _batch_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[list[int]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_ids: list[int],
) -> list[list[int]]:
    """Batch-generate completions for multiple prompts."""
    n = len(prompts)
    if n == 0:
        return []
    max_len = max(len(p) for p in prompts)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long, device=model.device)
    for i, p in enumerate(prompts):
        input_ids[i, -len(p):] = torch.tensor(p, device=model.device)

    attention_mask = (input_ids != pad_id).long()
    stop = StoppingCriteriaList([StopOnTokens(stop_ids)]) if stop_ids else None

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            stopping_criteria=stop,
            pad_token_id=pad_id,
        )

    results = []
    for i in range(n):
        new_ids = outputs[i, max_len:].tolist()
        results.append(new_ids)
    return results


def _compute_completion_logprobs(
    model: AutoModelForCausalLM,
    prompts: list[list[int]],
    completions: list[list[int]],
    pad_token_id: int,
) -> list[list[float]]:
    """Batch forward pass to get per-token logprobs for completion tokens.

    For each prompt+completion pair, does one forward pass and extracts
    log_softmax at the positions corresponding to completion tokens.
    These serve as the reference logprobs for LLDS regularization.
    """
    n = len(prompts)
    if n == 0:
        return []

    # Concatenate prompt + completion, right-aligned with padding
    full_seqs = [prompts[i] + completions[i] for i in range(n)]
    max_len = max(len(s) for s in full_seqs)

    input_ids = torch.full((n, max_len), pad_token_id, dtype=torch.long, device=model.device)
    for i, s in enumerate(full_seqs):
        input_ids[i, :len(s)] = torch.tensor(s, device=model.device)
    attention_mask = (input_ids != pad_token_id).long()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        log_probs = F.log_softmax(outputs.logits, dim=-1)  # [n, max_len, vocab]

    results = []
    for i in range(n):
        prompt_len = len(prompts[i])
        comp_lps = []
        for j, token_id in enumerate(completions[i]):
            # logits at position (prompt_len + j - 1) predict token at (prompt_len + j)
            pos = prompt_len + j - 1
            if 0 <= pos < log_probs.shape[1] - 1:
                comp_lps.append(log_probs[i, pos, token_id].item())
            else:
                comp_lps.append(0.0)
        results.append(comp_lps)
    return results


def rollout_question(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    search_client: SearchClient,
    example: SearchExample,
    group_size: int,
    temperature: float,
    top_p: float,
    base_seed: int,
    search_timeout: float,
    question_index: int = 0,
) -> list[Trajectory]:
    """Generate group_size trajectories for one question using batched inference."""
    stop_strs = stop_sequences(tokenizer)
    stop_ids = []
    for s in stop_strs:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            stop_ids.extend(ids)

    # Each trajectory maintains its own conversation state
    all_messages = [initial_messages(example.question) for _ in range(group_size)]
    all_turns: list[list[Turn]] = [[] for _ in range(group_size)]
    all_prompt_full: list[list[int]] = [[] for _ in range(group_size)]
    finished = [False] * group_size
    final_texts = [""] * group_size
    valid_formats = [False] * group_size
    search_counts = [0] * group_size

    for turn_idx in range(MAX_ASSISTANT_TURNS):
        # Build prompts for unfinished trajectories
        active_indices = [i for i in range(group_size) if not finished[i]]
        if not active_indices:
            break

        prompts = [build_prompt(tokenizer, all_messages[i]) for i in active_indices]
        if not prompts or not any(prompts):
            break

        # Batch generate
        completions = _batch_generate(
            model, tokenizer, prompts,
            MAX_ASSISTANT_TOKENS, temperature, top_p, stop_ids,
        )

        # Compute reference logprobs for LLDS (one extra forward pass per turn)
        logprobs_batch = _compute_completion_logprobs(
            model, prompts, completions,
            tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        for batch_idx, gi in enumerate(active_indices):
            completion_ids = completions[batch_idx]
            completion_logprobs = logprobs_batch[batch_idx]
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            for ss in stop_strs:
                completion_text = completion_text.replace(ss, "")

            prompt_tokens = prompts[batch_idx]
            turn = Turn(
                prompt_tokens=list(prompt_tokens),
                completion_tokens=completion_ids,
                completion_text=completion_text,
                is_tool_call=False,
                logprobs=completion_logprobs,
            )
            all_turns[gi].append(turn)
            all_prompt_full[gi].extend(prompt_tokens)
            all_prompt_full[gi].extend(completion_ids)

            parsed = parse_assistant(completion_text)
            if parsed.kind == "tool" and parsed.query:
                if search_counts[gi] >= MAX_SEARCH_CALLS:
                    finished[gi] = True
                    continue
                search_counts[gi] += 1
                turn.is_tool_call = True
                result = _run_search(search_client, parsed.query, f"q{search_counts[gi]}", search_timeout)
                turn.tool_result = result

                if not result.ok:
                    tc = f"Search error: {result.error or 'unknown'}"
                elif not result.items:
                    tc = "Search returned no results."
                else:
                    tc = "\n\n".join(format_item(item, i) for i, item in enumerate(result.items, 1))
                turn.tool_content = tc
                tool_msg = {"role": "user", "content": f"<tool_response>{tc}</tool_response>"}
                all_messages[gi].append({"role": "assistant", "content": completion_text})
                all_messages[gi].append(tool_msg)
            elif parsed.kind == "answer":
                final_texts[gi] = parsed.content
                valid_formats[gi] = True
                all_messages[gi].append({"role": "assistant", "content": completion_text})
                finished[gi] = True
            else:
                all_messages[gi].append({"role": "assistant", "content": completion_text})
                finished[gi] = True

    trajectories = []
    for gi in range(group_size):
        result = score_answer(final_texts[gi], example.answers)
        trajectories.append(Trajectory(
            example=example,
            turns=all_turns[gi],
            final_text=final_texts[gi],
            reward=result.reward,
            search_calls=search_counts[gi],
            valid_format=result.valid_format,
            messages=all_messages[gi],
            question_index=question_index,
        ))
    return trajectories


def rollout_batch_local(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    search_client: SearchClient,
    examples: list[SearchExample],
    group_size: int,
    temperature: float,
    top_p: float,
    base_seed: int,
    search_timeout: float,
    progress_callback=None,
) -> list[Trajectory]:
    """Generate trajectories for a batch of questions with batched per-question inference."""
    all_trajs: list[Trajectory] = []
    for qi, ex in enumerate(examples):
        t0 = time.time()
        trajs = rollout_question(model, tokenizer, search_client, ex, group_size, temperature, top_p, base_seed + qi, search_timeout, question_index=qi)
        all_trajs.extend(trajs)
        rewards = [t.reward for t in trajs]
        t_sec = time.time() - t0
        print(f"    q={qi+1}/{len(examples)} {ex.question[:50]} | reward={np.mean(rewards):.2f} searches={np.mean([t.search_calls for t in trajs]):.1f} time={t_sec:.0f}s", flush=True)
        if progress_callback:
            progress_callback(len(trajs))
    return all_trajs


# ═══════════════════════════════════════════════════════════════════════════
# Anomaly detection
# ═══════════════════════════════════════════════════════════════════════════

def is_anomalous(traj: Trajectory) -> bool:
    if not traj.final_text or not traj.final_text.strip():
        return True
    if traj.search_calls >= MAX_SEARCH_CALLS and not traj.valid_format:
        return True
    if traj.search_calls == 0 and not traj.valid_format:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# GRPO training utilities
# ═══════════════════════════════════════════════════════════════════════════

def compute_group_advantages(trajectories: list[Trajectory], group_size: int) -> None:
    """Assign group-relative advantage to each trajectory."""
    n_questions = len(trajectories) // group_size
    for qi in range(n_questions):
        group = trajectories[qi * group_size:(qi + 1) * group_size]
        rewards = [t.reward for t in group if not t.anomalous]
        if not rewards:
            for t in group:
                t.advantage = 0.0
            continue
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-8
        for t in group:
            if t.anomalous:
                t.advantage = 0.0
            else:
                t.advantage = (t.reward - mean_r) / std_r


def build_training_sequences(
    trajectories: list[Trajectory],
    tokenizer: AutoTokenizer,
) -> list[dict]:
    """Build training sequences from trajectories with reference logprobs.

    Returns list of dicts with keys:
        input_ids, loss_mask, advantage, ref_logprobs, trajectory

    ref_logprobs are aligned with input_ids: 0.0 for prompt tokens,
    rollout logprob for completion tokens. Used by grpo_loss for LLDS.
    """
    sequences = []
    for traj in trajectories:
        if traj.anomalous or traj.advantage == 0.0:
            continue

        all_ids: list[int] = []
        loss_mask: list[float] = []
        ref_logprobs: list[float] = []
        prev_len = 0

        for turn in traj.turns:
            # Prompt tokens (no loss, no LLDS reference)
            all_ids.extend(turn.prompt_tokens)
            cur_len = len(all_ids)
            n_prompt = cur_len - prev_len
            loss_mask.extend([0.0] * n_prompt)
            ref_logprobs.extend([0.0] * n_prompt)
            prev_len = cur_len

            # Completion tokens (loss, LLDS reference)
            all_ids.extend(turn.completion_tokens)
            cur_len = len(all_ids)
            n_comp = cur_len - prev_len
            loss_mask.extend([1.0] * n_comp)
            # Align logprobs with completion tokens; pad if lengths mismatch
            turn_lps = turn.logprobs if turn.logprobs else [0.0] * n_comp
            if len(turn_lps) != n_comp:
                turn_lps = list(turn_lps) + [0.0] * (n_comp - len(turn_lps))
            ref_logprobs.extend(turn_lps[:n_comp])
            prev_len = cur_len

        # Truncate overlong sequences to prevent OOM in forward pass
        if len(all_ids) > MAX_TRAIN_SEQ_LEN:
            all_ids = all_ids[:MAX_TRAIN_SEQ_LEN]
            loss_mask = loss_mask[:MAX_TRAIN_SEQ_LEN]
            ref_logprobs = ref_logprobs[:MAX_TRAIN_SEQ_LEN]

        sequences.append({
            "input_ids": all_ids,
            "loss_mask": loss_mask,
            "advantage": traj.advantage,
            "ref_logprobs": ref_logprobs,
            "trajectory": traj,
        })
    return sequences


def pack_micro_batches(sequences: list[dict], max_padded_tokens: int) -> list[dict]:
    """Split sequences into micro-batches that fit in GPU memory."""
    if not sequences:
        return []

    # Sort by length to minimize padding
    sequences = sorted(sequences, key=lambda s: len(s["input_ids"]))

    batches = []
    current_batch = []
    current_max_len = 0

    for seq in sequences:
        new_max = max(current_max_len, len(seq["input_ids"]))
        new_padded = new_max * (len(current_batch) + 1)
        if new_padded > max_padded_tokens and current_batch:
            batches.append(current_batch)
            current_batch = [seq]
            current_max_len = len(seq["input_ids"])
        else:
            current_batch.append(seq)
            current_max_len = new_max

    if current_batch:
        batches.append(current_batch)
    return batches


def _chunked_log_softmax_gather(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Compute per-token log-probabilities in chunks to limit peak memory.

    logits:  [n, seq_len, vocab]   (can be hundreds of GB in full log_softmax)
    labels:  [n, seq_len]           target token indices
    returns: [n, seq_len]           log P(label | context) per position
    """
    n, seq_len, vocab = logits.shape
    result = torch.empty((n, seq_len), dtype=logits.dtype, device=logits.device)
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = logits[:, start:end, :]  # [n, chunk, vocab]
        lp_chunk = F.log_softmax(chunk, dim=-1)  # [n, chunk, vocab]
        result[:, start:end] = lp_chunk.gather(2, labels[:, start:end].unsqueeze(-1)).squeeze(-1)
    return result


def grpo_loss(
    model: AutoModelForCausalLM,
    batch: list[dict],
    pad_token_id: int,
    llds_lambda: float = 0.0,
    llds_variant: str = "A",
) -> tuple[torch.Tensor, dict]:
    """Compute GRPO loss with optional LLDS penalty for one micro-batch.

    LLDS (Lazy Likelihood-Displacement Stabilization):
        Penalizes the new policy when its log-likelihood on rollout tokens
        drops below the reference (old policy) log-likelihood.

        Variants:
            "A"  - Action-level: penalize all completion tokens (default)
            "R"  - Response-level: only penalize if total likelihood decreased

        penalty = λ * Σ max(0, log π_ref - log π_θ) over completion tokens
    """
    max_len = max(len(s["input_ids"]) for s in batch)
    n = len(batch)

    input_ids = torch.full((n, max_len), pad_token_id, dtype=torch.long, device=model.device)
    loss_mask = torch.zeros((n, max_len), device=model.device)
    advantages = torch.zeros(n, device=model.device)
    ref_logprobs = torch.zeros((n, max_len), device=model.device)

    for i, seq in enumerate(batch):
        ids = seq["input_ids"]
        input_ids[i, :len(ids)] = torch.tensor(ids, device=model.device)
        mask = seq["loss_mask"]
        loss_mask[i, :len(mask)] = torch.tensor(mask, device=model.device)
        advantages[i] = seq["advantage"]
        ref_lps = seq.get("ref_logprobs", [])
        if ref_lps:
            ref_logprobs[i, :len(ref_lps)] = torch.tensor(ref_lps, device=model.device)

    # Forward pass
    outputs = model(input_ids)
    logits = outputs.logits  # [n, seq_len, vocab]

    # Shift: predict token t+1 from token t
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous()

    # Log probabilities (chunked to avoid OOM on large vocab)
    token_log_probs = _chunked_log_softmax_gather(shift_logits, shift_labels)  # [n, seq_len-1]

    # GRPO loss: -advantage * mean(log_prob over loss tokens)
    masked_lp = token_log_probs * shift_mask
    n_loss_tokens = shift_mask.sum(dim=1).clamp(min=1)
    mean_lp = masked_lp.sum(dim=1) / n_loss_tokens  # [n]

    loss = -(advantages * mean_lp).mean()

    # ---- LLDS penalty ----
    llds_penalty_val = 0.0
    if llds_lambda > 0:
        shift_ref_lp = ref_logprobs[:, 1:]  # align with token_log_probs (right-shift)

        # Per-token penalty: max(0, ref_lp - cur_lp) — only when current is worse
        token_penalty = torch.clamp(shift_ref_lp - token_log_probs, min=0)

        if llds_variant == "R":
            # Response-level gate: only penalize if total likelihood decreased
            cur_total = (token_log_probs * shift_mask).sum(dim=1)
            ref_total = (shift_ref_lp * shift_mask).sum(dim=1)
            gate = (cur_total < ref_total).float().unsqueeze(1)
            token_penalty = token_penalty * gate

        # Sum over masked tokens, normalize by number of active sequences
        llds_penalty = (token_penalty * shift_mask).sum()
        n_active = max(1, (shift_mask.sum(dim=1) > 0).sum().item())
        llds_penalty_val = llds_penalty.item() / n_active

        loss = loss + llds_lambda * (llds_penalty / n_active)

    # Metrics
    with torch.no_grad():
        total_loss_tokens = shift_mask.sum().item()

    return loss, {"loss_tokens": total_loss_tokens, "llds_penalty": llds_penalty_val}


# ═══════════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SFT+GRPO local training")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--data", default=str(Path(__file__).parent / "datasets" / "train.jsonl"))
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--questions-per-batch", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--search-backend", default="deepseek")
    parser.add_argument("--search-model", default="deepseek-v4-flash")
    parser.add_argument("--search-timeout", type=float, default=60.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="grpo_checkpoint")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--llds-lambda", type=float, default=0.0, help="LLDS regularization (0=off)")
    parser.add_argument("--llds-variant", default="A", choices=["A", "R"], help="LLDS variant: A=action-level, R=response-level")
    parser.add_argument("--prm-lite", action="store_true", help="Enable PRM-Lite process reward (12 penalty + 10 bonus rules)")
    parser.add_argument("--lata", action="store_true", help="Enable LATA length-adaptive advantage normalization")
    parser.add_argument("--hf-mirror", action="store_true")
    parser.add_argument("--resume", default="", help="LoRA adapter path to resume from")
    args = parser.parse_args()

    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    base_dir = Path(__file__).resolve().parent

    # ── System info ──
    print("=" * 62)
    print("  SFT+GRPO Local Training  |  4-bit QLoRA  |  LLDS integrated")
    print("=" * 62)
    assert torch.cuda.is_available(), "CUDA not available!"
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {gpu} ({vram:.1f} GB)")
    print(f"  Model: {args.base_model}")
    print(f"  Steps: {args.max_steps}  Questions/batch: {args.questions_per_batch}  Group: {args.group_size}")
    print(f"  LLDS: λ={args.llds_lambda} variant={args.llds_variant}")
    print()

    # ── Data ──
    print("[1/6] Loading data ...")
    examples = shuffled_examples(Path(args.data), args.seed)
    print(f"  {len(examples)} training questions")
    print()

    # ── Tokenizer ──
    print("[2/6] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  vocab={len(tokenizer)}")
    print()

    # ── Model ──
    print("[3/6] Loading model (4-bit QLoRA) ...")
    t0 = time.perf_counter()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Resume
    if args.resume:
        print(f"  Loading adapter from {args.resume} ...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)

    torch.cuda.synchronize()
    print(f"  Model ready in {time.perf_counter() - t0:.0f}s")
    print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.memory_reserved()/1e9:.1f}GB")
    print()

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable/1e6:.1f}M  LR: {LEARNING_RATE}  Grad accum: {GRAD_ACCUM}")
    print()

    # ── Search ──
    print("[4/6] Setting up search ...")
    search_client = create_search_client(
        args.search_backend,
        base_dir / ".env",
        model=args.search_model,
        timeout=args.search_timeout,
    )
    print(f"  Backend: {args.search_backend}  Model: {args.search_model}")
    print()

    # ── Feature flags ──
    flags = []
    if args.llds_lambda > 0:
        flags.append(f"LLDS(λ={args.llds_lambda}, {args.llds_variant})")
    else:
        flags.append("LLDS=off")
    if args.prm_lite:
        flags.append("PRM-Lite")
    if args.lata:
        flags.append("LATA")
    print(f"  Features: {', '.join(flags)}")
    if args.llds_lambda > 0:
        print(f"  (LLDS penalty = λ * Σ max(0, log π_ref - log π_θ) over completion tokens)")
    if args.prm_lite:
        print(f"  (PRM-Lite = 12 penalty + 10 bonus heuristic rules, capped at [-0.2, +0.2])")
    if args.lata:
        print(f"  (LATA = advantage / sqrt(completion_tokens), protects long reasoning chains)")
    print()

    # ── Training loop ──
    print("[5/6] Starting training ...")
    print()

    step_losses = []
    for step in range(args.max_steps):
        step_start = time.perf_counter()

        # Get batch
        batch = take_batch(examples, step * args.questions_per_batch, args.questions_per_batch)

        # Rollout
        model.eval()
        trajectories = rollout_batch_local(
            model, tokenizer, search_client, batch, args.group_size,
            args.temperature, args.top_p, args.seed + step,
            args.search_timeout,
        )

        # Detect anomalous
        for t in trajectories:
            t.anomalous = is_anomalous(t)
            if t.anomalous:
                t.advantage = 0.0

        # Compute group advantages
        compute_group_advantages(trajectories, args.group_size)

        # PRM-Lite: overlay process reward on outcome reward
        prm_total = 0.0
        if args.prm_lite:
            prm_scorer = PRMLiteScorer()
            for t in trajectories:
                if not t.anomalous:
                    result = prm_scorer.score(t)
                    t.reward += result.process_reward
                    prm_total += result.process_reward

        # LATA: length-adaptive advantage normalization
        if args.lata:
            apply_lata(trajectories)

        # Build training sequences (now includes ref_logprobs for LLDS)
        sequences = build_training_sequences(trajectories, tokenizer)
        micro_batches = pack_micro_batches(sequences, MICRO_BATCH_MAX_TOKENS)

        # Free rollout memory before training forward pass
        gc.collect()
        torch.cuda.empty_cache()

        # Train
        model.train()
        total_loss = 0.0
        total_tokens = 0
        total_llds_penalty = 0.0
        optimizer.zero_grad()

        for mb_idx, mb in enumerate(micro_batches):
            loss, metrics = grpo_loss(
                model, mb, tokenizer.pad_token_id or tokenizer.eos_token_id,
                llds_lambda=args.llds_lambda,
                llds_variant=args.llds_variant,
            )
            scaled_loss = loss / (len(micro_batches) * GRAD_ACCUM)
            scaled_loss.backward()
            total_loss += loss.item()
            total_tokens += metrics["loss_tokens"]
            total_llds_penalty += metrics.get("llds_penalty", 0.0)

        # Optimizer step
        if (step + 1) % GRAD_ACCUM == 0 or step == args.max_steps - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.empty_cache()

        # Metrics
        correct = sum(1 for t in trajectories if t.reward > 0.5)
        anomalous_count = sum(1 for t in trajectories if t.anomalous)
        mean_reward = np.mean([t.reward for t in trajectories])
        mean_search = np.mean([t.search_calls for t in trajectories])
        avg_loss = total_loss / len(micro_batches) if micro_batches else 0
        step_time = time.perf_counter() - step_start

        step_losses.append(avg_loss)
        extra = []
        if args.llds_lambda > 0:
            extra.append(f"llds={total_llds_penalty:.4f}")
        if args.prm_lite:
            extra.append(f"prm={prm_total/len(trajectories):.3f}")
        if args.lata:
            extra.append("lata")
        extra_str = " " + " ".join(extra) if extra else ""
        print(f"  step={step+1:>3d}/{args.max_steps}  "
              f"loss={avg_loss:.4f}  reward={mean_reward:.3f}  "
              f"correct={correct}/{len(trajectories)}({correct/len(trajectories):.1%})  "
              f"search={mean_search:.1f}  anomalous={anomalous_count}{extra_str}  "
              f"tokens={total_tokens}  time={step_time:.0f}s")

        # Save checkpoint
        if (step + 1) % args.save_every == 0 or step == args.max_steps - 1:
            ckpt_dir = Path(args.output) / f"step-{step+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  → checkpoint: {ckpt_dir}")

    # ── Final ──
    print()
    print(f"[6/6] Done! Avg loss: {np.mean(step_losses):.4f}")
    print(f"  Checkpoints: {Path(args.output).absolute()}")

    # Save final
    final_dir = Path(args.output) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"  Final adapter: {final_dir}")


if __name__ == "__main__":
    main()
