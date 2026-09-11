# veRL 部分导读：本项目关键参数与代码速查

> 面向：面试前快速回顾 / 重新上手 verl_search_r1 代码。
> 完整教学版见《系统工程教学指南》第 3 章（Ray + veRL 架构）；
> 完整逐键解释见 `docs/02-TUTORIAL.md` §3.4。
> 代码位置：`C:\new\intern\plan\projects\search-r1\verl_search_r1\`

---

## 1. 文件地图（按"面试被问到的概率"排序）

| 文件 | 重要性 | 一句话作用 | 什么时候读 |
|---|---|---|---|
| `e6b_run.sh` | ★★★★★ | **训练启动的唯一事实来源**：60+ 个 override + 显存账注释，本身就是文档 | 第一个读 |
| `search_agent_loop.py` | ★★★★★ | SearchR1AgentLoop：多轮循环 + TITO + response_mask | 第二个读 |
| `reward_fn.py` | ★★★★ | reward 入口：EM+Format+PRM-Lite，字段流转范例 | 第三个读 |
| `probe_verl_api.py` | ★★★★ | 47 项启动前检查（方法论价值 > 代码价值） | 当踩坑案例集读 |
| `E6B_DEPLOY.md` | ★★★ | 部署清单 + 止损规则 + 显存账 | 面试前扫一遍 |
| `agent_config.yaml` | ★★★ | agent loop 配置（YAML **列表**格式，veRL 0.10 约定） | 1 分钟 |
| `e6b_main.py` | ★★★ | 入口：先注册 agent loop 再进 hydra | 1 分钟 |
| `grpo_config.yaml` | ★★ | 设计参考（实际以 e6b_run.sh 为准，勿混用） | 略读 |
| `prepare_data.py` | ★★ | JSONL → parquet（prompt/reward_model 列） | 略读 |
| `cloud_eval.sh` | ★★ | 云端评估全参 checkpoint（--no-adapter） | 略读 |

## 2. 关键参数四层导读（按"决定什么"分层）

### 第一梯队：显存与 OOM（决定能不能跑起来）

| 参数 | 值 | 为什么（一句话） |
|---|---|---|
| `actor.use_kl_loss` | **false** | 关 KL → ref 模型不创建。veRL 0.10 的 ref CPUOffload 在 torch 2.13 失效（实测常驻 GPU 8.46GB），2×4090 放不下；Search-R1 配方本来无 KL |
| `fsdp_config.optimizer_offload` + `param_offload` | **true** | Adam 状态（25.2GB）与参数流式放 CPU；每卡显存从 26.7GB❌ → 18.9GB✓，代价步时 +30-60s |
| `rollout.gpu_memory_utilization` | 0.55 / 0.75 | vLLM 的 KV 池上限；1 卡 0.55、2 卡 0.75（训练侧 offload 后 GPU 空出来） |
| `checkpoint_engine.update_weights_bucket_megabytes` | **512** | 默认 2048MB 恰好超 torch 2.13 CUDA IPC 上限（INT_MAX=2GB−1）→ 静默走 shm 文件路径 → rebuild_ipc 越界 |
| `rollout.layered_summon` | true | 权重同步分层 unshard，峰值 12.6→~5GB |
| `PYTORCH_CUDA_ALLOC_CONF` | expandable_segments:True, garbage_collection_threshold:0.05 | 去碎片 + GC 回收 reserved（OOM 缺口 400MB 时能释放 ~700MB） |

### 第二梯队：架构适配（决定模型能不能加载/推理）

| 参数 | 值 | 为什么 |
|---|---|---|
| `rollout.max_num_seqs` | **229** | Mamba cache block 上限（报错原文：256 exceeds available 229）；混合架构的并发由最受限组件决定 |
| `rollout.max_model_len` | 6144 | prompt ~2048 + 3 轮上下文，6144 够用且 KV 减半提升并发 |
| `rollout.enforce_eager` | false | 开 CUDA graph 提速（推理无动态形状） |
| `VLLM_ATTENTION_BACKEND` | FLASHINFER | FA2 编译不了 Qwen3.5 混合架构的替代后端 |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | 1 | AutoDL 网络对 huggingface.co 只解析出 IPv6 无路由 → from_pretrained 卡死；模型已缓存强制离线 |
| `actor_rollout_ref.model.trust_remote_code` | true | Qwen3.5 自定义架构 |

### 第三梯队：算法行为（决定 RL 怎么学）

| 参数 | 值 | 为什么 |
|---|---|---|
| `algorithm.adv_estimator` | grpo | 无 critic，组内归一化 advantage |
| `rollout.n` | 4 | 每题 4 条轨迹（GRPO 组大小）；train_batch_size=32 = 8 题 × 4 |
| `rollout.temperature / top_p` | 1.0 / 1.0 | 训练探索 |
| `multi_turn.enable` + `max_assistant_turns` | true / 3 | 多轮搜索循环（与 agent_config.yaml 里的 3 是两层限制，loop 自己再限一次） |
| `agent.default_agent_loop` + `agent_loop_config_path` | search_r1_agent | 启用自定义 loop（YAML 列表格式：name + _target_ + kwargs） |
| `reward.custom_reward_function.path` | **pkg://verl_search_r1.reward_fn** | pkg:// 前缀必须：无前缀时 veRL 把 module 当文件路径 |
| `+reward.custom_reward_function.reward_kwargs.enable_prm_lite` | $PRM | **reward_kwargs 下的键才传进 reward 函数**；平铺键被静默忽略（PRM 静默失效的坑） |
| `actor.optim.lr` + `lr_warmup_steps` | 1e-6 / 10 | 全参数 LR 比 QLoRA（1e-5）低一个量级 |
| `data.return_raw_chat` | True | parquet 的 prompt 列（消息列表）原样给 agent loop 的 raw_prompt |
| `data.truncation` | left | 超长保留**尾部**（答案在尾部） |

### 第四梯队：工程保障

| 参数 | 值 | 为什么 |
|---|---|---|
| `TQ_DATA_UPDATE_RESPONSE_TIMEOUT` | 120 | TQ ACK 超时 30s→120s（controller 启动期 thundering herd） |
| `export PYTHONPATH=$PROJECT_DIR` | — | Ray worker 不继承 shell 环境（reward_fn 曾被 FileNotFoundError） |
| `trainer.save_freq` / `default_local_dir` | 25 / checkpoint 目录 | 多 checkpoint 事后扫描评估 |
| `reward.reward_model.enable` | false | 关内置 RM 用自定义函数 |

## 3. 代码阅读顺序（30 分钟版）

```
e6b_run.sh（15 分钟，注释即文档，重点看显存账注释块）
  → search_agent_loop.py 的 run()（10 分钟，只看主循环 + mask extend 段）
  → reward_fn.py 的 search_r1_reward（3 分钟，看字段流转）
  → e6b_main.py + agent_config.yaml（2 分钟）
  → probe_verl_api.py 的目录注释（5 分钟，看 7 节检查设计，不必逐行）
```

## 4. 三个"为什么这么设"的经典追问（面试高发）

1. **为什么关 KL（use_kl_loss=false）？** 三层：配方不需要（Search-R1 官方 GRPO 无 KL）→ 显存放不下（ref CPUOffload 在 torch 2.13 失效、实测常驻 8.46GB）→ 防崩职责由 LLDS 接替（单向惩罚更精准）。关掉还省了 ref 前向时间。
2. **为什么 max_num_seqs=229？** Qwen3.5 是 Transformer+Mamba 混合架构，Mamba 层的 recurrent state 是固定数量的 cache block、按序列预分配、不可分页 → 并发上限 229。256 启动即报错。229 对 GRPO 组采样（batch 32）绰绰有余。
3. **为什么 bucket 是 512MB？** veRL 默认 2048MB 的权重同步分桶恰好超过 torch 2.13 CUDA IPC 上限（INT_MAX=2GB−1）→ 静默降级走 shm 文件 → rebuild_ipc 索引越界。512MB 走正常 IPC 路径。

## 5. 与面试问答的映射

| 本导读内容 | 对应 07-INTERVIEW_QA |
|---|---|
| 显存账 + OOM 参数 | Q14（20.67GB 峰值、400MB 缺口） |
| use_kl_loss=false | Q6（KL 约束怎么做） |
| max_num_seqs=229 | Q17（Mamba cache block） |
| 512MB bucket / layered_summon | Q14 追问（IPC 细节） |
| TQ 相关环境变量 | Q16 / Q21（TQ 瓶颈定位与优化） |
| probe 方法论 | Q18（47 项检查） |
| reward_kwargs 注入 | 坑 6（API 漂移、PRM 静默失效） |

---
**口径纪律**（引用本文件数字时遵守）：TQ 数据（51K 操作/步、736/min、步时 50-70min）
作为性能分析引用；不写「27%」跨论文对比；不写「veRL 训练未完成 50 步」，
veRL 部分统一为「环境调通 + TQ 瓶颈定量分析」正面框架。
