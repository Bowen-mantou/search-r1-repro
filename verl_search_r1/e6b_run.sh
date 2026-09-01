#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# E6-B: veRL 全参数 GRPO + PRM-Lite (2×4090 多卡训练, 1×4090 省钱降级)
#
# 为什么不用 run.sh：run.sh 的 --config/--nnodes/--nproc_per_node 参数是
# hydra 不接受的（该路线上次被放弃在 API 对齐阶段，从未真正跑通）。
# 本脚本改用 veRL 官方扩展模式：以 veRL 自带 ppo_trainer 配置为基础 +
# 完整 CLI override。
#
# 显存/费用策略（2026-08-30，¥2.18/卡/h）：
#   2×4090: ¥22-28/50步 —— 主方案（FSDP 多卡分片 = 简历上的多卡训练经历）。
#   1×4090: ¥14-20/50步 —— 省钱降级（N_GPUS=1），offload 后 ~2GB 余量。
#   4×4090: ¥35-44/50步 —— 超预算，不推荐。
#
# Usage (from $VERL_DIR):
#   bash $PROJECT_DIR/verl_search_r1/e6b_run.sh --probe      # 先跑体检
#   bash $PROJECT_DIR/verl_search_r1/e6b_run.sh --dry-run    # 打印命令
#   N_GPUS=1 bash $PROJECT_DIR/verl_search_r1/e6b_run.sh     # 降级 1 卡
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
export PROJECT_DIR="${PROJECT_DIR:-$HOME/autodl-tmp/search-r1}"
VERL_DIR="${VERL_DIR:-$HOME/autodl-tmp/verl}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Tunables ────────────────────────────────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-wang072266/qwen3.5-4b-search-r1-sft}"
N_GPUS="${N_GPUS:-2}"
STEPS="${STEPS:-50}"
SAVE_FREQ="${SAVE_FREQ:-25}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-e6b-grpo-prmlite}"
PRM="${PRM:-true}"
LATA="${LATA:-false}"

# ── GPU-count-dependent memory preset ───────────────────────────────────────
if [ "$N_GPUS" -eq 1 ]; then
    GPU_UTIL="0.55"              # 1×4090: 训练侧 offload 后余量更大，给 vLLM 更多 KV cache
    MAX_BATCHED_TOKENS="32768"   # 限在飞 token 数，防 KV 撑爆
else
    GPU_UTIL="0.75"              # 2×4090: rollout 阶段训练侧 offload 后几乎不占 GPU，给 vLLM 更多 KV cache
    MAX_BATCHED_TOKENS="65536"   # 允许更多请求批处理，提升并发吞吐
fi

TRAIN_PARQUET="$PROJECT_DIR/datasets/train.parquet"
TEST_PARQUET="$PROJECT_DIR/datasets/test.parquet"
CHECKPOINT_DIR="$PROJECT_DIR/verl_checkpoints/$EXPERIMENT_NAME"

PROBE=false
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --probe) PROBE=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ── Preconditions ──────────────────────────────────────────────────────────
if [ ! -f "$TRAIN_PARQUET" ]; then
    echo "ERROR: $TRAIN_PARQUET missing. Run:"
    echo "  python $PROJECT_DIR/verl_search_r1/prepare_data.py $PROJECT_DIR/datasets/train.jsonl $TRAIN_PARQUET"
    exit 1
fi
cd "$VERL_DIR" || { echo "ERROR: $VERL_DIR missing (run setup.sh first)"; exit 1; }

# ── Build override list (single source of truth, consumed by probe + launch) ─
# 键路径已对齐 veRL main 0.10（2026-08-30）：reward 挂 reward.* 下，
# 自定义 reward 的额外参数走 reward.custom_reward_function.reward_kwargs.*
# （平铺键会被忽略 → PRM 静默失效）；ref 的动态 bsz 键是 log_prob_use_dynamic_bsz。
OVERRIDES=(
    "data.train_files=$TRAIN_PARQUET"
    "data.val_files=$TEST_PARQUET"
    "data.max_prompt_length=2048"
    "data.max_response_length=4096"
    "data.train_batch_size=32"
    "data.return_raw_chat=True"
    "data.shuffle=True"
    "data.truncation=left"
    "reward.reward_model.enable=false"
    # 2026-08-30 修复：veRL 0.10 load_extern_object 无前缀时把 module 当文件路径；
    # 必须 pkg:// 前缀才走 importlib（sys.path/.pth 生效）。
    "reward.custom_reward_function.path=pkg://verl_search_r1.reward_fn"
    "reward.custom_reward_function.name=search_r1_reward"
    "+reward.custom_reward_function.reward_kwargs.enable_prm_lite=$PRM"
    "+reward.custom_reward_function.reward_kwargs.enable_lata=$LATA"
    "algorithm.adv_estimator=grpo"
    "algorithm.use_kl_in_reward=false"
    "actor_rollout_ref.model.path=$BASE_MODEL"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.model.enable_gradient_checkpointing=true"
    "actor_rollout_ref.model.trust_remote_code=true"
    "actor_rollout_ref.actor.optim.lr=1e-6"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
    # 2026-08-30 OOM 修复：KL 关闭 → ref 不创建。
    # 根因：veRL 0.10 对 ref 强制 CPUOffload(offload_params=True) 的路径在
    # torch 2.13 下未生效（日志: ref FSDP 后 allocated 8.46GB 常驻 GPU），
    # 14.34 = ref 8.46 + actor offload 后训练峰值 → OOM。
    # Search-R1 官方 GRPO 配方本来无 KL 项，关闭后显存宽裕且每步更快。
    "actor_rollout_ref.actor.use_kl_loss=false"
    "actor_rollout_ref.actor.ppo_mini_batch_size=32"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.actor.use_dynamic_bsz=true"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096"
    "actor_rollout_ref.actor.loss_agg_mode=token-mean"
    # ── 显存账（2026-08-30 推算，24GB/卡）───────────────────────────────
    # 无 offload 时每卡：权重 8.4/N + 梯度 8.4/N + Adam 25.2/N + 激活 ~1.5 GB：
    #   N=1 → 51.9GB，N=2 → 26.7GB —— 都必爆（vLLM 8.4GB 权重还没算）。
    # 开 optimizer+param offload（CPU）后训练侧：梯度 8.4/N + 激活 ~1.5 GB
    # + vLLM(0.45/0.55 池)：
    #   N=1 → 8.4+1.5+10.8 ≈ 20.7GB ✓（余 ~2GB，最划算方案）
    #   N=2 → 4.2+1.5+13.2 ≈ 18.9GB ✓（余 ~4GB）
    # ref 已关闭（use_kl_loss=false）：veRL 0.10 的 ref CPUOffload 在
    # torch 2.13 不生效（实测常驻 8.46GB），关闭后不再计入。
    # 代价：参数每步流式进出，步耗时 +30-60s。
    # ───────────────────────────────────────────────────────────────────────
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true"
    "actor_rollout_ref.actor.fsdp_config.param_offload=true"
    "actor_rollout_ref.ref.fsdp_config.param_offload=true"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.n=4"
    "actor_rollout_ref.rollout.temperature=1.0"
    "actor_rollout_ref.rollout.top_p=1.0"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UTIL"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_BATCHED_TOKENS"
    # 2026-08-30 修复：默认 2048MB bucket 恰好超过 torch 2.13 reduce_tensor 的
    # CUDA IPC 上限（INT_MAX=2GB-1）→ 静默改走 shm 文件路径 → veRL rebuild_ipc
    # list_args[6] 越界。512MB 走正常 IPC 路径（且 <512MB 无直发权重路径）
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512"
    # vLLM 吞吐优化：启用 CUDA graph（推理阶段无动态形状，安全）
    "actor_rollout_ref.rollout.enforce_eager=false"
    "actor_rollout_ref.rollout.free_cache_engine=true"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    # 多轮上下文：prompt ~2048 + 6 turns × ~512 ≈ 5120，6144 够用且 KV cache 减半提升并发
    "actor_rollout_ref.rollout.max_model_len=6144"
    "actor_rollout_ref.rollout.max_num_seqs=229"
    # 2026-08-30 OOM 修复 #2：权重同步时分层 unshard（summon_full_params），
    # 避免 all-gather 全量 8.4 + 分片 4.2 并存（update_weights 峰值 12.6→~5）
    "actor_rollout_ref.rollout.layered_summon=true"
    "actor_rollout_ref.rollout.agent.default_agent_loop=search_r1_agent"
    "actor_rollout_ref.rollout.agent.agent_loop_config_path=$PROJECT_DIR/verl_search_r1/agent_config.yaml"
    "actor_rollout_ref.rollout.multi_turn.enable=true"
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3"
    "trainer.n_gpus_per_node=$N_GPUS"
    "trainer.nnodes=1"
    "trainer.project_name=search-r1"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.logger=['console']"
    "trainer.total_epochs=1"
    "trainer.total_training_steps=$STEPS"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=-1"
    "trainer.default_local_dir=$CHECKPOINT_DIR"
)

printf '%s\n' "${OVERRIDES[@]}" > "$VERL_DIR/.e6b_overrides.txt"

# ── Copy entry point into veRL dir (hydra resolves config-path here) ──────
cp "$SRC_DIR/e6b_main.py" "$VERL_DIR/e6b_main.py"

# 不传 --config-path：hydra 会把 main_ppo.py 装饰器里的 config_path="config"
# 相对 verl/trainer/main_ppo.py 解析 → $VERL_DIR/verl/trainer/config/
CMD=(python -u e6b_main.py --config-name=ppo_trainer "${OVERRIDES[@]}")

# ── Banner ─────────────────────────────────────────────────────────────────
cat <<EOF
╔══════════════════════════════════════════════════════════════╗
║  E6-B veRL GRPO + PRM-Lite                                  ║
╠══════════════════════════════════════════════════════════════╣
║  GPUs:          $N_GPUS
║  Steps:         $STEPS (save every $SAVE_FREQ)
║  Base model:    $BASE_MODEL
║  PRM-Lite:      $PRM   |  LATA: $LATA
║  Checkpoints:   $CHECKPOINT_DIR
║  veRL dir:      $VERL_DIR
╚══════════════════════════════════════════════════════════════╝
EOF

# ── Probe ──────────────────────────────────────────────────────────────────
if [ "$PROBE" = true ]; then
    echo ""
    echo "[PROBE] Running pre-flight checks (no training)..."
    PROJECT_DIR="$PROJECT_DIR" python -u "$PROJECT_DIR/verl_search_r1/probe_verl_api.py" --live-search \
        || { echo "PROBE FAILED — fix issues above before launching."; exit 1; }
    echo ""
    echo "Probe passed. Launch training with:"
    echo "  bash $SRC_DIR/e6b_run.sh"
    exit 0
fi

# ── Launch ─────────────────────────────────────────────────────────────────
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "  [DRY RUN] command:"
    printf '  %s\n' "${CMD[@]}"
    exit 0
fi

echo ""
echo "  Launching (log: $EXPERIMENT_NAME.log)..."
# Ray workers inherit the driver env：verl_search_r1（agent loop + reward fn）必须可导入
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ATTENTION_BACKEND=FLASHINFER
export HYDRA_FULL_ERROR=1
# 2026-08-30 修复：AutoDL 网络对 huggingface.co 只解析出 IPv6（无路由）→
# from_pretrained 卡死数十分钟。模型已全部缓存，强制离线（绕过网络探测）。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 2026-08-30 OOM 修复 #2：去碎片 + GC 回收 reserved 未分配块
# （update_weights 峰值 21.9GB，缺口 ~400MB；GC 可释放 ~700MB reserved）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.05
# 2026-08-30 TQ 修复：controller 启动时 thundering herd → ACK 超时 → pending 卡死
# 加大超时 + base.py 重试补丁双保险
export TQ_DATA_UPDATE_RESPONSE_TIMEOUT=120
# 搜索后端凭证：由 .env 文件提供（DEEPSEEK_API_KEY / ZHIHU_SEARCH_KEYS /
# MIMO_API_KEY），search.py 会在运行时读取。切勿把真实 key 写进本脚本。
# 历史记录：曾切换过 deepseek → zhihu → mimo 后端（见 docs/TUTORIAL.md 踩坑篇）。
exec "${CMD[@]}" 2>&1 | tee "$EXPERIMENT_NAME.log"
