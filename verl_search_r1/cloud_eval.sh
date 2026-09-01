#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# 云端评估：直接在训练实例上评测 veRL checkpoint（不用下载回本地）
#
# 前置：训练完成后（或训练中显存允许时）在同一实例运行。
# 依赖：pip install peft bitsandbytes  （transformers/torch 装 veRL 时已带）
#
# Usage (from $PROJECT_DIR):
#   bash verl_search_r1/cloud_eval.sh                          # 评测全部 checkpoint
#   bash verl_search_r1/cloud_eval.sh global_step_25           # 指定单个
#   LIMIT=20 bash verl_search_r1/cloud_eval.sh                 # 先抽 20 题快速看
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/autodl-tmp/search-r1}"
CKPT_ROOT="$PROJECT_DIR/verl_checkpoints"
OUT_DIR="$PROJECT_DIR/eval_result"
LIMIT="${LIMIT:-0}"
ONLY="${1:-}"

cd "$PROJECT_DIR"

# ── Dependencies ────────────────────────────────────────────────────────────
python -c "import peft, bitsandbytes" 2>/dev/null || pip install -q peft bitsandbytes

# ── Discover checkpoints ────────────────────────────────────────────────────
if [ -n "$ONLY" ]; then
    CKPTS=("$CKPT_ROOT/$ONLY")
else
    # 找所有实验目录下的 global_step_* checkpoint
    CKPTS=()
    while IFS= read -r d; do
        CKPTS+=("$d")
    done < <(find "$CKPT_ROOT" -maxdepth 3 -type d -name "global_step_*" | sort)
fi

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "ERROR: no checkpoints found under $CKPT_ROOT"
    exit 1
fi

mkdir -p "$OUT_DIR"

# ── Evaluate each (full-parameter checkpoint → --no-adapter) ───────────────
for ckpt in "${CKPTS[@]}"; do
    label="$(basename "$(dirname "$ckpt")")-$(basename "$ckpt")"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "Evaluating: $ckpt  →  $OUT_DIR/${label}.jsonl"
    echo "════════════════════════════════════════════════════════════"
    PYTHONIOENCODING=utf-8 python -u eval_local.py \
        --base-model "$ckpt" \
        --no-adapter \
        --output "$OUT_DIR/${label}.jsonl" \
        --limit "$LIMIT"
done

echo ""
echo "Done. Results in: $OUT_DIR/"
