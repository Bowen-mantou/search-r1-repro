#!/usr/bin/env python3
"""
本地评估脚本：加载 train_grpo_local.py 保存的 LoRA checkpoint，在 dev 集上评测。

不依赖 PyTRIO，可在本地或 AutoDL 运行。

Usage:
    # 评测单个 checkpoint
    python eval_local.py --checkpoint grpo_checkpoint/step-25 \
        --base-model ~/autodl-tmp/qwen3.5-4b-sft-merged

    # 评测并输出详细 JSONL
    python eval_local.py --checkpoint grpo_checkpoint/final \
        --base-model ~/autodl-tmp/qwen3.5-4b-sft-merged \
        --output eval_result/e2d_step25.jsonl

    # 跑全部 checkpoint + 汇总
    python eval_local.py --checkpoint-dir grpo_checkpoint \
        --base-model ~/autodl-tmp/qwen3.5-4b-sft-merged \
        --output-dir eval_result
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from data import SearchExample, load_examples
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
# Config (mirrors train_grpo_local.py)
# ═══════════════════════════════════════════════════════════════════════════

MAX_SEARCH_CALLS = 4
MAX_ASSISTANT_TURNS = 6
MAX_ASSISTANT_TOKENS = 1024

DEV_DATA = Path(__file__).resolve().parent / "datasets" / "dev.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "eval_result"


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory (minimal, mirrors train_grpo_local.py)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EvalTurn:
    prompt_tokens: list[int]
    completion_tokens: list[int]
    completion_text: str
    is_tool_call: bool
    tool_result: SearchResult | None = None
    tool_content: str = ""
    logprobs: list[float] = field(default_factory=list)


@dataclass
class EvalTrajectory:
    example: SearchExample
    turns: list[EvalTurn]
    final_text: str = ""
    reward: float = 0.0
    exact_match: bool = False
    valid_format: bool = False
    search_calls: int = 0
    anomalous: bool = False
    messages: list = field(default_factory=list)
    question_index: int = 0

    @property
    def completion_text(self):
        """dummy for reward_lite compatibility"""
        return self.final_text


# ═══════════════════════════════════════════════════════════════════════════
# Rollout (adapted from train_grpo_local.py)
# ═══════════════════════════════════════════════════════════════════════════

def _run_search(search_client: SearchClient, query: str) -> SearchResult:
    return search_client.search(query)


def evaluate_question(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    search_client: SearchClient,
    example: SearchExample,
    question_index: int = 0,
) -> EvalTrajectory:
    """Generate a single deterministic trajectory for one question."""
    stop_strs = stop_sequences(tokenizer)
    stop_ids = []
    for s in stop_strs:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            stop_ids.extend(ids)

    messages = initial_messages(example.question)
    turns: list[EvalTurn] = []
    finished = False
    final_text = ""
    valid_format = False
    search_calls = 0

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for turn_idx in range(MAX_ASSISTANT_TURNS):
        if finished:
            break

        prompt = build_prompt(tokenizer, messages)
        if not prompt:
            break

        input_ids = torch.tensor([prompt], device=model.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_ASSISTANT_TOKENS,
                do_sample=False,  # greedy for evaluation
                temperature=1.0,
                pad_token_id=pad_id,
                stopping_criteria=None,
            )

        # Extract completion (right-aligned, so slice at prompt length)
        completion_ids = outputs[0, len(prompt):].tolist()
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        for ss in stop_strs:
            completion_text = completion_text.replace(ss, "")

        turn = EvalTurn(
            prompt_tokens=list(prompt),
            completion_tokens=completion_ids,
            completion_text=completion_text,
            is_tool_call=False,
        )
        turns.append(turn)

        parsed = parse_assistant(completion_text)
        if parsed.kind == "tool" and parsed.query:
            if search_calls >= MAX_SEARCH_CALLS:
                finished = True
                continue
            search_calls += 1
            turn.is_tool_call = True
            result = _run_search(search_client, parsed.query)
            turn.tool_result = result

            if not result.ok:
                tc = f"Search error: {result.error or 'unknown'}"
            elif not result.items:
                tc = "Search returned no results."
            else:
                tc = "\n\n".join(format_item(item, i) for i, item in enumerate(result.items, 1))
            turn.tool_content = tc
            tool_msg = {"role": "user", "content": f"<tool_response>{tc}</tool_response>"}
            messages.append({"role": "assistant", "content": completion_text})
            messages.append(tool_msg)
        elif parsed.kind == "answer":
            final_text = parsed.content
            valid_format = True
            messages.append({"role": "assistant", "content": completion_text})
            finished = True
        else:
            messages.append({"role": "assistant", "content": completion_text})
            finished = True

    result = score_answer(final_text, example.answers)
    return EvalTrajectory(
        example=example,
        turns=turns,
        final_text=final_text,
        reward=result.reward,
        exact_match=result.exact_match,
        valid_format=result.valid_format,
        search_calls=search_calls,
        messages=messages,
        question_index=question_index,
    )


def evaluate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    search_client: SearchClient,
    examples: list[SearchExample],
) -> list[EvalTrajectory]:
    """Evaluate all examples, one at a time."""
    trajectories = []
    for qi, ex in enumerate(tqdm(examples, desc="Eval", unit="q")):
        t0 = time.time()
        traj = evaluate_question(model, tokenizer, search_client, ex, question_index=qi)
        t_sec = time.time() - t0
        print(f"  q={qi+1}/{len(examples)} {ex.question[:60]} | "
              f"em={traj.exact_match} format={traj.valid_format} "
              f"searches={traj.search_calls} time={t_sec:.0f}s", flush=True)
        trajectories.append(traj)
    return trajectories


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(trajectories: list[EvalTrajectory]) -> dict[str, float]:
    """Compute evaluation metrics."""
    n = len(trajectories)

    # EM and format
    em_count = sum(1 for t in trajectories if t.exact_match)
    format_count = sum(1 for t in trajectories if t.valid_format)

    # Per-source breakdown
    by_source: dict[str, list[EvalTrajectory]] = {}
    for t in trajectories:
        by_source.setdefault(t.example.data_source, []).append(t)

    source_em = {}
    for src, items in sorted(by_source.items()):
        source_em[f"em/{src}"] = sum(1 for t in items if t.exact_match) / max(len(items), 1)

    # Search behavior
    search_calls = [t.search_calls for t in trajectories]
    no_search_rate = sum(1 for s in search_calls if s == 0) / max(n, 1)
    turns = [len(t.turns) for t in trajectories]

    # PRM-Lite scoring
    prm_scorer = PRMLiteScorer()
    prm_rewards = []
    prm_details: dict[str, list[float]] = {}
    for t in trajectories:
        result = prm_scorer.score(t)
        prm_rewards.append(result.process_reward)
        for rule_id, val in result.all_rules.items():
            prm_details.setdefault(rule_id, []).append(val)

    # Aggregate PRM rule hit rates
    prm_rule_hits = {}
    for rule_id, vals in prm_details.items():
        prm_rule_hits[f"prm/{rule_id}"] = sum(1 for v in vals if v != 0) / max(len(vals), 1)

    metrics = {
        "em/macro": em_count / max(n, 1),
        "format/rate": format_count / max(n, 1),
        "search/mean": np.mean(search_calls) if search_calls else 0,
        "search/no_search_rate": no_search_rate,
        "turns/mean": np.mean(turns) if turns else 0,
        "prm/process_reward_mean": np.mean(prm_rewards) if prm_rewards else 0,
        "prm/total_penalty_mean": np.mean([v for v in prm_rewards if v < 0]) if any(v < 0 for v in prm_rewards) else 0,
        "prm/total_bonus_mean": np.mean([v for v in prm_rewards if v > 0]) if any(v > 0 for v in prm_rewards) else 0,
        "evaluated_examples": n,
    }
    metrics.update(source_em)
    metrics.update(prm_rule_hits)

    return metrics


def save_results(
    trajectories: list[EvalTrajectory],
    metrics: dict[str, float],
    output_path: Path,
    checkpoint_label: str = "",
) -> None:
    """Save per-question trajectories and summary to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for t in trajectories:
            rec = {
                "type": "trajectory",
                "id": t.example.id,
                "question": t.example.question,
                "answers": t.example.answers,
                "data_source": t.example.data_source,
                "final_text": t.final_text,
                "reward": t.reward,
                "exact_match": t.exact_match,
                "valid_format": t.valid_format,
                "search_calls": t.search_calls,
                "assistant_turns": len(t.turns),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        summary = {
            "type": "summary",
            "checkpoint": checkpoint_label,
            "metrics": metrics,
        }
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"  Saved: {output_path}")


def print_metrics(metrics: dict[str, float], label: str = "") -> None:
    """Pretty-print evaluation metrics."""
    prefix = f"[{label}] " if label else ""
    print(f"\n{'='*55}")
    print(f"  {prefix}Evaluation Results")
    print(f"{'='*55}")
    print(f"  Questions:       {int(metrics.get('evaluated_examples', 0))}")
    print(f"  Macro EM:        {metrics.get('em/macro', 0):.2%}")
    print(f"  Format Rate:     {metrics.get('format/rate', 0):.2%}")
    print(f"  Avg Search:      {metrics.get('search/mean', 0):.1f}")
    print(f"  No-Search Rate:  {metrics.get('search/no_search_rate', 0):.2%}")
    print(f"  Avg Turns:       {metrics.get('turns/mean', 0):.1f}")
    print(f"  PRM Reward Avg:  {metrics.get('prm/process_reward_mean', 0):+.4f}")
    print()

    # Per-source breakdown
    source_ems = {k: v for k, v in metrics.items() if k.startswith("em/") and k != "em/macro"}
    if source_ems:
        print("  Per-source EM:")
        for src, val in sorted(source_ems.items()):
            print(f"    {src}:  {val:.2%}")

    # Top PRM rules
    prm_rules = {k: v for k, v in metrics.items() if k.startswith("prm/") and v > 0.05}
    if prm_rules:
        print(f"\n  PRM Rule Hit Rates (>5%):")
        for rule, rate in sorted(prm_rules.items(), key=lambda x: -x[1]):
            print(f"    {rule}:  {rate:.1%}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(
    base_model: str,
    checkpoint: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model with 4-bit QLoRA, then optionally merge LoRA checkpoint."""
    print(f"Loading base model: {base_model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.is_dir():
            print(f"Loading LoRA adapter: {checkpoint}")
            model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
        else:
            print(f"Checkpoint not found: {checkpoint}, using base model only")

    model.eval()
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Path to base/sft-merged model")
    parser.add_argument("--checkpoint", help="Path to a single LoRA checkpoint directory")
    parser.add_argument("--checkpoint-dir", help="Directory containing step-N checkpoint dirs (evaluate all)")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Evaluate the base model directly (full-parameter checkpoint, no LoRA)")
    parser.add_argument("--data", default=str(DEV_DATA), help="Evaluation JSONL file")
    parser.add_argument("--limit", type=int, default=0, help="Max questions to evaluate (0=all)")
    parser.add_argument("--output", help="Output JSONL path (single checkpoint mode)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Output dir (multi checkpoint mode)")
    parser.add_argument("--search-backend", default="deepseek")
    parser.add_argument("--search-model", default="deepseek-v4-flash")
    parser.add_argument("--search-timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    # ── Data ──
    print(f"Loading data: {args.data}")
    examples = load_examples(Path(args.data))
    if args.limit > 0:
        examples = examples[:args.limit]
    print(f"  {len(examples)} evaluation questions")

    # ── Search ──
    print(f"Setting up search: {args.search_backend}")
    search_client = create_search_client(
        args.search_backend,
        base_dir / ".env",
        model=args.search_model,
        timeout=args.search_timeout,
    )

    # ── Determine checkpoints to evaluate ──
    checkpoints: list[tuple[str, str]] = []  # (label, path)
    if args.no_adapter:
        checkpoints.append(("full-param", None))
    elif args.checkpoint:
        label = Path(args.checkpoint).name
        checkpoints.append((label, args.checkpoint))
    elif args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        if ckpt_dir.is_dir():
            for p in sorted(ckpt_dir.iterdir()):
                if p.is_dir() and p.name.startswith("step-"):
                    checkpoints.append((p.name, str(p)))
            # Also check for "final"
            final = ckpt_dir / "final"
            if final.is_dir():
                checkpoints.append(("final", str(final)))
    else:
        print("ERROR: need --checkpoint or --checkpoint-dir")
        sys.exit(1)

    if not checkpoints:
        print("No checkpoints found!")
        sys.exit(1)
    print(f"Checkpoints to evaluate: {[c[0] for c in checkpoints]}")
    print()

    # ── Evaluate each checkpoint ──
    all_results: list[dict] = []
    for ckpt_label, ckpt_path in checkpoints:
        print(f"\n{'#'*55}")
        print(f"# Evaluating: {ckpt_label}")
        print(f"{'#'*55}")

        # Load model (reload per checkpoint to avoid memory accumulation)
        model, tokenizer = load_model_and_tokenizer(args.base_model, ckpt_path)

        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu} ({vram:.1f} GB)")

        # Evaluate
        t0 = time.perf_counter()
        trajectories = evaluate_batch(model, tokenizer, search_client, examples)
        elapsed = time.perf_counter() - t0

        # Metrics
        metrics = compute_metrics(trajectories)
        metrics["eval_time_seconds"] = elapsed
        print_metrics(metrics, ckpt_label)

        # Save
        if args.checkpoint:
            output_path = Path(args.output) if args.output else DEFAULT_OUTPUT / f"eval_{ckpt_label}.jsonl"
        else:
            output_path = Path(args.output_dir) / f"eval_{ckpt_label}.jsonl"
        save_results(trajectories, metrics, output_path, ckpt_label)

        all_results.append({"checkpoint": ckpt_label, "metrics": metrics})

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Cross-checkpoint summary ──
    if len(all_results) > 1:
        print(f"\n{'='*55}")
        print(f"  Cross-Checkpoint Comparison")
        print(f"{'='*55}")
        print(f"  {'Checkpoint':<12} {'Macro EM':>10} {'Format':>10} {'Avg Search':>10} {'PRM Avg':>10}")
        print(f"  {'-'*52}")
        for r in all_results:
            m = r["metrics"]
            print(f"  {r['checkpoint']:<12} {m.get('em/macro', 0):>10.2%} "
                  f"{m.get('format/rate', 0):>10.2%} {m.get('search/mean', 0):>10.1f} "
                  f"{m.get('prm/process_reward_mean', 0):>+10.4f}")

    print(f"\nDone. Results in: {args.output_dir if args.checkpoint_dir else output_path}")


if __name__ == "__main__":
    main()
