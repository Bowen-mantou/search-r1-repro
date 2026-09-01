# Search-R1：检索增强推理的 GRPO 训练系统（复现 + 改进）

> 基于 [Search-R1 论文（arXiv:2503.09516）](https://arxiv.org/abs/2503.09516) 与
> [KMnO4-zx/agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab) 03-search-r1
> 教程的**独立改进版**。用 GRPO 强化学习训练 Qwen3.5-4B 学会「何时搜索、搜什么、
> 如何用搜索结果回答问题」——检索不是外部管线写死的，而是模型自己决策的工具调用。

```
用户问题: "Which shopping mall is owned by GGP and located in Texas?"
    ↓ 模型自主决策
<tool_call><function=search><parameter=query>GGP shopping mall Texas</parameter></function></tool_call>
    ↓ 搜索后端返回证据
<tool_response>[1] Title: Baybrook Mall ... Evidence: GGP owns Baybrook Mall ...</tool_response>
    ↓ 模型基于证据作答
Answer: Baybrook Mall
```

---

## 项目构成

完整流水线：**教师蒸馏（397 条轨迹）→ QLoRA SFT 冷启动 → GRPO 强化学习**，
两套训练体系互相印证：

| 维度 | 本地 QLoRA 管线（`train_grpo_local.py`） | veRL 全参数管线（`verl_search_r1/`） |
|------|-----------------------------------------|-------------------------------------|
| 训练方式 | 4-bit QLoRA（rank=16） | 全参数 FSDP（2×4090，CPU offload） |
| 推理引擎 | HF `model.generate()` | vLLM + PagedAttention |
| 定位 | 快速验证算法想法（~16min/步） | 生产级多卡分布式框架 |
| 验证结果 | correct_rate 最高 **81.2%** | 训练启动与全链路调试完成；TransferQueue 调度瓶颈定量分析 |

## 核心改进点（相对上游教程）

1. **异常轨迹过滤**：搜索失败/格式异常/截断的轨迹 advantage 置 0，通过 loss mask
   排除噪声梯度（10 步验证 160 条轨迹检出 4 条异常，检测率 2.5%）
2. **LLDS 正则化**（`llds.py`，47 个单元测试）：ICML 2026 LLDS 的复现实现，
   消除 LLD 死亡螺旋（v1 固定偏移 λ=0.05 被 GRPO 优势淹没、step 63 崩到 1.6%；
   v2 比例缩放 `min(delta, 10.0)` 后低点能恢复，不再永久崩溃）
3. **PRM-Lite 过程奖励**（`reward_lite.py`，782 行）：22 条启发式规则
   （12 惩罚 + 10 奖励）提供稠密反馈，capped 在 [-0.2, +0.2] 防止喧宾夺主
4. **LATA 长度归一化**：`advantage = (r - mean) / sqrt(L)`，保护长推理链不被
   系统性低估
5. **5 种可插拔搜索后端**（`search.py`）：DeepSeek Search / 知乎 / Wikipedia /
   Bing / MiMo LLM，统一 `SearchClient` 接口 + 运行时缓存 + 429 凭证轮转 +
   并发限速，全部免改训练代码即可切换
6. **veRL 全参数迁移**（`verl_search_r1/`）：自定义 `SearchR1AgentLoop`
   （TITO 协议 + token 级 `response_mask`）、veRL 0.10 API 对齐、
   FSDP OOM 修复（optimizer/param CPU offload）、Mamba 混合架构适配
   （max_num_seqs=229）、pre-flight probe（47 项检查）、TransferQueue
   瓶颈定量分析

## 结果（诚实声明）

| 实验 | 方法 | 结果 | 状态 |
|------|------|------|------|
| SFT 冷启动 | QLoRA（r=32），397 条蒸馏轨迹，3 epochs | loss 1.6515 → 0.1143 | ✅ |
| v2 纯 RL（GRPO） | QLoRA，DeepSeek 搜索 | correct_rate **81.2%**（step 3 peak） | ✅ |
| E3-B v2（+LLDS） | 比例缩放 LLDS | 最高 81.2%，低点可恢复，不再永久崩溃 | ✅ |
| veRL 全参数 GRPO | 2×4090，FSDP offload | 训练启动 + 12 轮调试完成；TQ 瓶颈定量分析 | ✅ |

**veRL 管线的性能分析**：环境侧（OOM、API 漂移、Mamba 限制等 12 轮调试）全部
解决、probe 47/47 全绿之后，对每步耗时做了完整定位：每个 agent loop 产生约 400
个 TQ 子操作，128 个 loop/步 ≈ 51K 操作/步，而 TQ 的 SimpleStorageUnit PUT_DATA
吞吐只有 ~736/min，推算出纯调度时间 ≈ 70 分钟，与实测步时 50-70 分钟吻合
（搜索后端换成 0.68s 的知乎后步时几乎不变，交叉验证了瓶颈在框架调度而非搜索
API）。完整的定位过程与优化方案见
[docs/TUTORIAL.md](docs/TUTORIAL.md) 与 [docs/INTERVIEW_QA.md](docs/INTERVIEW_QA.md)。

## 目录结构

```
├── protocol.py                 # <tool_call> 搜索协议 + chat template 处理
├── search.py                   # 5 种搜索后端 + 运行时缓存 + 限速控制
├── rollout.py                  # PyTRIO 版多轮 rollout（思考→搜索→回答）
├── reward.py                   # EM + Format outcome reward
├── reward_lite.py              # PRM-Lite 22 条规则 + LATA
├── llds.py                     # LLDS 长度感知 KL 正则化（3 变体）
├── train_grpo_local.py         # 本地 GRPO 训练入口（集成以上全部）
├── eval_local.py               # 本地评估（多 checkpoint 扫描 + 分源 EM）
├── distill_trajectories.py     # DeepSeek 教师蒸馏数据生成
├── fix_distill_tags.py         # 蒸馏数据格式修复
├── sft_train.py                # QLoRA SFT 微调
├── merge_sft.py                # LoRA 合并
├── prepare_data.py             # ModelScope NQ/HotpotQA 下载清洗
├── data.py / analyse.py        # 数据工具 / 训练曲线分析
├── test_llds.py                # LLDS 单元测试（47 项）
├── test_reward_lite.py         # PRM-Lite 单元测试
├── distilled_trajectories_fixed.jsonl  # 397 条蒸馏轨迹（生成方式见 TUTORIAL）
├── datasets/dev.jsonl          # 70 题固定评估集（train/test 用 prepare_data.py 再生成）
├── verl_search_r1/             # veRL 全参数迁移
│   ├── search_agent_loop.py    # SearchR1AgentLoop（TITO + response_mask）
│   ├── reward_fn.py            # veRL 奖励函数（EM+Format+PRM-Lite）
│   ├── agent_config.yaml       # agent loop 配置
│   ├── grpo_config.yaml        # 训练配置（设计参考）
│   ├── e6b_run.sh              # 训练启动脚本（override 全集 + 显存预设）
│   ├── e6b_main.py             # 入口（先注册 agent loop 再进 veRL）
│   ├── probe_verl_api.py       # 启动前体检（不花 GPU 分钟）
│   ├── prepare_data.py         # JSONL → parquet（veRL 版）
│   ├── cloud_eval.sh           # 云端评估
│   └── E6B_DEPLOY.md           # 服务器部署清单（含止损规则）
└── docs/
    ├── TUTORIAL.md             # 从理论到代码到工程的完整教程
    └── INTERVIEW_QA.md         # 28 道面试问答（含测量方法）
```

## 快速开始

### 依赖

```bash
pip install torch transformers peft bitsandbytes trl datasets accelerate
pip install httpx python-dotenv pandas pyarrow
# 可选：deepseek 搜索后端（KMnO4-zx/deepseek-search）
git clone https://github.com/KMnO4-zx/deepseek-search && pip install -e deepseek-search
```

### 1. 准备数据（本地管线）

```bash
python prepare_data.py            # 从 ModelScope 下载 NQ/HotpotQA 并生成 train/test/dev.jsonl
```

### 2. SFT 冷启动（可选，也可直接用已发布的合并模型）

```bash
python distill_trajectories.py    # 需要 .env 里的 DEEPSEEK_API_KEY（教师蒸馏）
python sft_train.py --model Qwen/Qwen3.5-4B --data distilled_trajectories_fixed.jsonl
```

也可直接使用已发布的 SFT 合并模型
[`wang072266/qwen3.5-4b-search-r1-sft`](https://huggingface.co/wang072266/qwen3.5-4b-search-r1-sft)。

### 3. 本地 GRPO 训练 + 评估

```bash
# 在项目根目录放 .env（参考 .env.example）：DEEPSEEK_API_KEY=...
python train_grpo_local.py --max-steps 50 --questions-per-batch 8 --group-size 8 \
    --search-backend deepseek --llds-lambda 0.05 --prm-lite --lata

# 评估全部 checkpoint（greedy 解码，输出分源 EM + PRM 规则命中率）
python eval_local.py --checkpoint-dir grpo_checkpoint --base-model <你的模型>
```

### 4. veRL 全参数训练（多卡）

见 `verl_search_r1/E6B_DEPLOY.md`：安装 veRL main → `e6b_run.sh --probe`
（47 项检查 0 FAIL 才启动）→ `N_GPUS=2 bash e6b_run.sh`。

## 已知限制

1. **TransferQueue 调度开销**（veRL 管线）：每步 ~50-70 分钟，其中绝大部分是
   TQ 调度时间；优化方向（批量调度/异步搜索管线/绕开 TQ）见 TUTORIAL 踩坑篇
2. 本地管线依赖 PyTRIO 平台（`rollout.py` 路径）；独立运行请用
   `train_grpo_local.py`（不依赖 PyTRIO）
3. 蒸馏轨迹的搜索结果由教师模型**模拟生成**（非真实网页检索），SFT 学的是
   格式与推理模式而非事实性检索

## 致谢与来源

- 上游教程与基础代码：[KMnO4-zx/agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab)
  （知乎文章「一杯喜茶，搞定 Search-R1」）
- 论文：[Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516)，
  官方实现 [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)
- DeepSeek 搜索客户端：[KMnO4-zx/deepseek-search](https://github.com/KMnO4-zx/deepseek-search)
- LLDS 正则化参考：Deng et al., "On GRPO Collapse in Search-R1: The Lazy
  Likelihood-Displacement Death Spiral", arXiv 2512.04220, ICML 2026
- 数据集：NQ / HotpotQA（经
  [zhuangzhuang2023/nq_hotpotqa_train](https://modelscope.cn/datasets/zhuangzhuang2023/nq_hotpotqa_train) 获取）

## License

[MIT](LICENSE)（与上游一致）。蒸馏数据与代码可自由使用，引用时请标注上游来源。
