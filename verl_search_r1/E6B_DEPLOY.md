# E6-B 部署清单（veRL 全参数 GRPO + PRM-Lite，2×4090 多卡主方案）

> 2026-08-30。此清单与 `deploy_e6b/` 打包件配套使用。

## 0. 为什么换路线（一句话）

原 `repo/` 的 veRL fork 锁死 `transformers<4.48` + `vllm<=0.6.3`，而 Qwen3.5-4B
需要 transformers ≥ 4.57 —— **fork 根本加载不了这个模型**。
E6-B 改走 `verl_search_r1/`：upstream veRL main（无版本锁定）+ 自定义
`SearchR1AgentLoop`（`response_mask` 原生实现状态掩码，替代 fork 的
`<information>` state_masking）。

## 1. 费用与时间预估（¥2.18/卡/h，2026-08-30）

| 方案 | 每步耗时 | 50步+安装1.5h | 总费用 | 显存风险 |
|------|---------|--------------|--------|---------|
| **2×4090（主方案）** | 4-6 min | 5-6.5 h | **¥22-28** | 安全（offload 后 ~4GB 余量） |
| 1×4090（省钱降级） | 6-8 min | 6.5-8 h | ¥14-20 | 紧（~2GB 余量） |
| 4×4090 | 3-4 min | 4-5 h | ¥35-44 | 安全，但超预算 |

**选 2 卡做主方案的理由**：¥2.18/卡/h 下 2 卡总价 ¥22-28 在预算内，且
这是真正的多卡训练——FSDP 全参数分片、跨卡梯度 all-reduce、Ray 分布式
调度、每卡独立 vLLM rollout 引擎，简历上可写"多卡分布式训练"。
省钱降级用 `N_GPUS=1`。

**止损规则**：前 2 步盯实测；>12min/步 或 连续崩溃 → 立即止损；
格式 rate 连续 5 步 <20% → 停止（拿日志回本地分析）。

## 1.1 显存账（2026-08-30 推算，24GB/卡）

无 offload 时每卡：权重 8.4/N + 梯度 8.4/N + Adam 25.2/N + ref 8.4/N
+ 激活 ~1.5 = **50.4/N + 1.5 GB**：
- N=1 → 51.9GB ❌　N=2 → 26.7GB ❌（vLLM 权重 8.4GB 还没算）

因此 e6b_run.sh 预设 **optimizer_offload + param_offload（CPU）**，
训练侧仅剩 梯度 8.4/N + 激活 ~1.5：
- N=1：8.4 + 1.5 + vLLM(0.45→10.8GB) ≈ **20.7GB ✓**（余 ~2GB）
- N=2：4.2 + 1.5 + vLLM(0.55→13.2GB) ≈ **18.9GB ✓**（余 ~4GB）

代价：参数每步流式进出，步耗时 +30-60s。

## 2. 打包件结构（已含 deploy_e6b/）

```
deploy_e6b/                          → 上传后解压为 ~/autodl-tmp/search-r1/
├── verl_search_r1/                  （AgentLoop + reward + 配置 + 本清单）
│   ├── e6b_run.sh                   ★ 启动脚本（hydra 正确参数 + 2卡显存预设）
│   ├── e6b_main.py                  ★ 入口（先注册 agent loop 再进 verl）
│   ├── probe_verl_api.py            ★ 启动前体检（不花 GPU 分钟）
│   ├── prepare_data.py              ★ JSONL → parquet（prompt/reward_model 列）
│   ├── search_agent_loop.py / reward_fn.py / agent_config.yaml
│   └── grpo_config.yaml             （设计参考；实际以 e6b_run.sh overrides 为准）
├── datasets/train.jsonl (29MB) / test.jsonl (12MB) / dev.jsonl (70题评估集)
├── eval_local.py / cloud_eval.sh    （云端评估：--no-adapter 评测全参数 checkpoint）
├── protocol.py / search.py / reward.py / reward_lite.py / data.py
├── deepseek_search/
├── .env                             （DEEPSEEK_API_KEY）
└── E6B_DEPLOY.md
```

## 3. 服务器操作步骤

### 3.1 开机 + 上传（~10 min）

1. AutoDL 开机 **2×4090** 实例（¥4.36/h）
2. 上传 `deploy_e6b.tar.gz` 到 `~/autodl-tmp/`
3. ```bash
   cd ~/autodl-tmp && tar xzf deploy_e6b.tar.gz
   mv deploy_e6b search-r1        # 或解压时直接命名 search-r1
   ```
4. 验证：`ls ~/autodl-tmp/search-r1/verl_search_r1/e6b_run.sh`

### 3.2 安装 veRL（~30-60 min）

```bash
export HF_ENDPOINT=https://hf-mirror.com    # 模型下载走镜像
bash ~/autodl-tmp/search-r1/verl_search_r1/setup.sh
# 注意：setup.sh 检测到 <4 张卡会问 "Continue? (y/n)" → 回答 y
source ~/.bashrc
```

vllm 版本若与 verl main 不兼容，以 verl pyproject 的 pin 为准：
```bash
grep -i vllm ~/autodl-tmp/verl/pyproject.toml   # 按其版本 pip install
```

### 3.3 准备数据（1 min）

```bash
cd ~/autodl-tmp/search-r1
python verl_search_r1/prepare_data.py datasets/train.jsonl datasets/train.parquet
python verl_search_r1/prepare_data.py datasets/test.jsonl  datasets/test.parquet
```

### 3.4 启动前体检（2 min，关键步骤）

```bash
cd ~/autodl-tmp/verl
PROJECT_DIR=~/autodl-tmp/search-r1 bash ~/autodl-tmp/search-r1/verl_search_r1/e6b_run.sh --probe
```

Probe 检查 6 项：AgentLoop API 兼容性、hydra 配置合成（抓 key 漂移）、
库版本、模型可加载、DeepSeek 实弹查询、数据文件。
**必须 0 FAIL 才继续**；有 FAIL 把输出贴回给我。

### 3.5 启动训练

```bash
cd ~/autodl-tmp/verl
PROJECT_DIR=~/autodl-tmp/search-r1 bash ~/autodl-tmp/search-r1/verl_search_r1/e6b_run.sh
```

- 后台跑建议：`nohup ... > e6b_train.log 2>&1 &`（脚本自身也会 tee 一份日志）
- Ray + vLLM 初始化 ~2-3 min，第 1 步前会下载 8GB 模型（镜像，~10 min）
- **OOM 或步速不达标 → 关机换 2×4090**：同一条命令加 `N_GPUS=2` 即可
  （注意换实例要重新安装环境，见 §1 决策规则）

## 4. 监控要点

| 指标 | 健康区间（1卡 / 2卡） | 止损信号 |
|------|----------|----------|
| 每步耗时 | 6-9 / 4-6 min | >12 min/步（大概率卡死/显存抖动） |
| format rate（console 输出） | >60% | 连续 5 步 <20% → 格式已崩，停止 |
| 平均搜索次数 | 0.5-3 | 连续 10 步 ≈0 → 模型退化成不搜索 |
| reward/em | 有波动但逐步抬升 | 连续 10 步无提升且 <5% → 停 |
| GPU 显存 | <22GB/卡 | OOM → 把 gpu_memory_utilization 降到 0.25 重启 |

**止损总原则**：预算 ~¥43 没有重试余量。OOM 一次可调参重试；崩溃两次以上
或格式崩坏 → 立即停止，把日志拿回来本地分析，别在服务器上耗。

Checkpoint 在 `~/autodl-tmp/search-r1/verl_checkpoints/e6b-grpo-prmlite/`
（step 25 和 50，全参数 HF 格式）。

## 5. 训练后：云端评估（同一实例，~¥2-4，不用下载 16GB checkpoint）

评估直接放在训练环境里做（与 blog 做法一致）：

```bash
cd ~/autodl-tmp/search-r1
bash verl_search_r1/cloud_eval.sh                # 自动找全部 global_step_* 评估
# 或先抽 20 题快速验证：LIMIT=20 bash verl_search_r1/cloud_eval.sh
```

- `eval_local.py` 新增 `--no-adapter` 开关：全参数 checkpoint 直接当 base 加载
- 70 题在 4090 上 ~15-25 min（本地 4060 要 1h）
- 结果 JSONL 很小（~30KB），只把这个拉回本地即可；checkpoint 想留就留
- 与 SFT 基线 1.43% 和 E1-A/E3-B 81.2% 直接可比（同一脚本同一评估集）

## 5.5 可选：同实例接着跑 E6-A（纯 GRPO 无 PRM，隔离 PRM 净贡献）

E6-B 结束后**不要关机**，同一实例（环境已装好，无安装成本）直接：

```bash
cd ~/autodl-tmp/verl
PROJECT_DIR=~/autodl-tmp/search-r1 \
PRM=false EXPERIMENT_NAME=e6a-grpo-pure STEPS=25 \
  bash ~/autodl-tmp/search-r1/verl_search_r1/e6b_run.sh
```

- 25 步 ≈ 1.5-2.5h ≈ **¥7-11**；E6-B(¥22-28) + E6-A(¥7-11) ≈ **¥29-39 ≤ 预算 ¥43**
- E6-B vs E6-A 唯一变量是 PRM-Lite → 科学上干净的消融
- 跑完再 `cloud_eval.sh` 一次，全部结果拉回本地对比

## 5.6 收尾

1. 评估完**立刻关机**
2. 本地只拉 eval_result/*.jsonl + 训练日志（小文件）
3. 数据齐了后做 E5 汇总表：SFT 1.43% → QLoRA GRPO 81.2% → 全参数 ±PRM

## 6. 已知风险（诚实清单）

1. **veRL main 版本漂移**：search_agent_loop.py 对齐的是 8 月初的 API；
   probe 会抓出大部分漂移，但 Ray worker 内部的隐性不兼容只有真跑才知道。
2. **1×4090 显存紧**：见 §1.1 显存账——已预设 optimizer+param CPU offload
   （vLLM 0.45、micro_batch=1、eager 模式、在飞 token 上限 16384）。
   若 OOM：改 `N_GPUS=2` 重跑（vLLM 0.55），总费用 ≤ ¥28。
   若实测 >9min/步，砍步数到 25-30 保预算（§1 决策规则）。
3. **run.sh 与 grpo_config.yaml 已过时**：此路线 8 月 8 日被放弃时这两个
   文件未验证（`--config`/`--nnodes` 不是 hydra 参数）。以 `e6b_run.sh` 为准。
4. **首次跑通概率**：预计 60-70%。probe 能挡掉配置类问题，Ray/AgentLoop
   运行时问题可能要到第 1-2 步才暴露 —— 所以前 2 步盯紧日志。
