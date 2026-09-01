# Search-R1 完整教程：从理论到代码到工程

> 目标读者：想把「检索增强推理 + GRPO 强化学习」从零学到能独立复现、改进、部署的你。
> 本文按「理论 → 代码 → 工程」三层展开，代码部分逐文件逐段讲，工程部分每个坑给
> 现象→根因→修复→教训。读完本文 + 跑一遍代码，你应该能向任何人讲清楚这个项目。
>
> 配套阅读：`../README.md`（项目概览）、`INTERVIEW_QA.md`（28 道面试问答）。
> 上游来源：[Search-R1 论文](https://arxiv.org/abs/2503.09516)、
> [KMnO4-zx/agentic-rl-lab](https://github.com/KMnO4-zx/agentic-rl-lab)、
> [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)。

---

## 第 0 章 导读：这个项目在做什么

一句话：**用 GRPO 强化学习，训练 Qwen3.5-4B 学会「自己决定何时搜索、搜什么、
如何用搜索结果回答问题」**。

```
                        ┌──────────────────────────────────────────────┐
                        │           一次完整训练 step 的数据流           │
                        └──────────────────────────────────────────────┘

 数据                   rollout（采样）                 训练
──────    ───────────────────────────────────    ────────────────
 NQ /      问题 → 模型生成 → 解析 <tool_call>        EM + Format 奖励
 HotpotQA         │            │  query              （outcome）
 问题         <tool_call>    搜索后端 5 选 1        + PRM-Lite 22 规则
                  │            │  证据              + LATA 长度归一
              Answer:       <tool_response>              │
                  │            │                    组内 advantage
              最终答案    ← 拼接回对话 ←                （GRPO 无 critic）
                  │                                   │
              完整轨迹（turns + logprobs + mask）      │
                  └──────────────┬───────────────────┘
                                 ▼
              GRPO loss = -A·logπ + λ·LLDS（防似然崩塌）
                                 ▼
                    梯度更新（LoRA 或 全参数 FSDP）
```

项目有一条完整流水线，每个环节都能独立讲清楚：

1. **数据**：NQ / HotpotQA（17 万训练样本）→ 清洗成统一 JSONL
2. **教师蒸馏**：DeepSeek Chat API 生成 397 条「搜索-推理」示范轨迹
3. **SFT 冷启动**：QLoRA 微调基座，让模型先学会搜索格式与推理模式
   （loss 1.65 → 0.11）
4. **GRPO 强化学习**：两套实现——本地 QLoRA 管线（验证算法，correct_rate
   最高 81.2%）+ veRL 全参数管线（生产级多卡框架）
5. **评估**：70 题固定 dev 集，分数据源 EM + 格式率 + PRM 规则命中率

学习路线建议：先读第 1 章理论（每节末尾有「联系本项目」小结），再跟着第 2 章
把本地管线跑起来，最后读第 3、4 章理解 veRL 与工程细节。

---

# 第 1 章 理论篇

## 1.1 检索增强推理：RAG 和 Agentic Search 的区别

### 1.1.1 什么是 RAG

经典 RAG（Retrieval-Augmented Generation）的流程是**外部管线写死**的：

```
用户问题 → [固定检索器：embedding 召回 top-k 文档] → [拼接进 prompt] → LLM 回答
```

问题在于：**模型对检索过程没有任何决策权**。检索器什么时候查、查什么、查几次，
都是工程师预先写死的。三个具体缺陷：

1. **不会判断是否需要检索**。「法国的首都是什么」这类问题检索是浪费，
   但 RAG 管线一律检索；
2. **不会改写 query**。多跳问题（"《小王子》作者的出生国是哪个"）需要
   先查"作者是谁"，拿到结果再查"他的出生国"。固定检索器做不到这种链式推理；
3. **不会选择性使用结果**。召回的文档可能有噪音、矛盾，模型只能被动接受。

### 1.1.2 什么是 Agentic Search（Search-R1 的做法）

Search-R1 的思路：**把搜索变成模型可调用的工具，用 RL 训练模型学会怎么用**。

```
问题 → 模型生成 <search>query</search> → 搜索引擎返回结果
     → 模型看结果决定：继续搜（换 query）还是 <answer>xxx</answer>
```

训练完成后模型获得了三种能力：

- **判断**：这题需要搜吗？（内部知识能答就不搜）
- **检索**：搜什么关键词？搜几轮？第一轮结果不好时怎么改写 query？
- **整合**：从多条证据里提取、比较、综合出答案，并只输出答案而非复述文档

### 1.1.3 为什么「训练模型学会搜索」比「写死检索流程」好

写死流程假设了检索的**最优策略已知且固定**；训练模型则让策略**从数据中涌现**：

| | 写死 RAG 管线 | RL 训练搜索 Agent |
|---|---|---|
| 何时检索 | 规则决定（总是/阈值） | 模型学：简单题不搜省时 |
| query 质量 | 原问题原样 | 模型学：压缩成关键词（B1 规则奖励这个行为）|
| 多轮 | 不支持或规则固定 | 模型学：多跳问题搜 2-4 轮（B7 规则）|
| 结果利用 | 拼接即可 | 模型学：综合多条证据（B2 规则）|
| 不可搜索问题 | 会硬查 | 模型学：直接凭知识回答 |

**联系本项目**：我们的 dev 集实验给出了最直观的证据——SFT 冷启动模型（没经过
RL）在 70 题上 EM 只有 1.43%、平均搜索 0 次（它根本没学会调用工具）；
GRPO 训练后（本地管线）correct_rate 到 81.2%、平均搜索 1-2.8 次。
**「搜索」这个行为本身是 RL 训练出来的，不是 SFT 教出来的。**

### 1.1.4 Search-R1 论文的核心贡献（2503.09516）

1. **协议设计**：`<search>query</search>` → `<information>结果</information>` →
   `<answer>答案</answer>`。简单标签让 RL 的解析器可靠、奖励可计算。
2. **检索在环训练（retrieval-in-the-loop）**：训练时真的调用搜索引擎，
   轨迹 = 模型生成 + 真实环境反馈的交替序列。这是「环境交互式 RL」，
   与离线偏好数据（DPO 用）有本质区别。
3. **奖励稀疏性处理**：只有最终答案对错一个信号（outcome reward），
   但多轮搜索行为能从这一路信号中学会——GRPO 的组内比较是关键。
4. **State masking（信息掩码）**：搜索结果 token 不参与 loss（详见 §1.6）。

> 注意：我们项目的最终格式与论文略有不同——用 Qwen 原生的
> `<tool_call><function=search>...` 协议 + `Answer:` 行（见 §2.3），
> 结果用 `<tool_response>` 包裹。设计动机见 §2.3 和踩坑篇「SFT 格式 100% 无效」。

## 1.2 RLHF 基础：从 PPO 到 GRPO

### 1.2.1 先建三个概念

**策略（policy）π_θ**：语言模型本身，输入对话历史输出下一个 token 的分布。
θ 是我们要训练的参数。

**轨迹（trajectory）**：一次完整交互的 token 序列。本项目中 =
`[system prompt][问题] → [模型生成 search] → [搜索结果] → [模型生成 answer]`。

**奖励（reward）r**：轨迹结束后环境给的标量。本项目 = 答案是否正确（EM）+ 格式
+ 过程奖励（§1.3）。

强化学习的目标：**最大化期望奖励** `E_τ~π[ r(τ) ]`。
策略梯度定理告诉我们：`∇_θ E[r] = E[ ∇_θ log π_θ(τ) · A(τ) ]`，
其中 `A(τ)` 是轨迹的 advantage（这条轨迹比平均好多少）。这就是所有 RLHF 的根。

### 1.2.2 PPO：为什么需要 4 个模型

PPO 同时维护：

| 模型 | 作用 | 参数量（4B 基座） | 显存（bf16） |
|---|---|---|---|
| Actor（策略 π_θ） | 生成文本 | 4B | ~8GB |
| **Critic（价值 V_φ）** | 估计每个状态的期望回报，用来算 advantage | 4B | ~8GB |
| Reference（π_ref） | 计算 KL，防止策略跑太远 | 4B | ~8GB |
| Reward | 轨迹打分 | 规则或模型 | — |

Critic 的显存开销 = Actor 的显存开销（还要算它的优化器状态）。
PPO 的 advantage 是 `A = r + γV(s') - V(s)`（TD 残差），需要训练一个
神经网络 V 来估计价值——**这是 PPO 最贵的部分**。

PPO 的 clipped 目标（本项目 GRPO loss 同款）：

```
ratio = exp(log π_θ(a|s) − log π_old(a|s))    # 新旧策略概率比
loss  = −mean( min( ratio·A,  clip(ratio, 1−ε, 1+ε)·A ) )
```

clip 的意义：当 ratio 偏离 1 太远（ε=0.2）时截断梯度，
让策略「小步快走」，防止一次更新把策略打坏。`log π_old` 在 rollout 时用旧模型
算好存下来，训练时只需算当前模型的前向。

### 1.2.3 GRPO：砍掉 Critic，用组内比较

GRPO（Group Relative Policy Optimization，DeepSeekMath 2024 提出）的核心观察：
**同一道题采样 N 条轨迹，它们的相对好坏本身就是 advantage**，
不需要 Critic 来估计绝对价值：

```
对问题 q，采样 N 条轨迹 r_1, ..., r_N
mean = (1/N) Σ r_i
std  = sqrt( (1/N) Σ (r_i − mean)² )          # 组内标准差
A_i  = (r_i − mean) / (std + ε)               # 组内归一化 advantage
```

直觉：假设同题 8 条轨迹，4 条答对（r=1）、4 条答错（r=0）：
mean=0.5，std≈0.535，答对的 A≈+0.93，答错的 A≈−0.93。
**模型学到的是"这道题的正确做法相对错误做法"**，而不是绝对价值。

组内比较天然带来三个好处：

1. **无 Critic**：省掉一个 4B 模型的权重+梯度+优化器状态。对 4B 模型，
   全参数训练下省 ~50% 训练显存（8GB 权重 + 24GB Adam 状态量级）。
2. **奖励尺度自动归一**：reward 是 0~1 还是 0~100 都不重要，组内减均值
   除标准差后总是尺度无关的。不同题难度不同也不怕——每题的组各自归一。
3. **方差小**：同一 prompt 的 N 条轨迹之间对比，比跨 prompt 对比方差小
   （共享同一问题的难度、搜索环境）。

显存账（本项目实测推算，24GB/卡、bf16、Adam）：

```
PPO 全参数（4B）：权重 8.4 + 梯度 8.4 + Adam 25.2 + critic(8.4+8.4+25.2) + ref 8.4 ≈ 92.4GB
GRPO 全参数（4B）：权重 8.4 + 梯度 8.4 + Adam 25.2 + ref 8.4 = 50.4GB（再按卡数分片）
```

GRPO loss（本项目 `train_grpo_local.py:grpo_loss` 的实现）：

```python
loss = -(advantages * mean_lp).mean()
# mean_lp = 每条轨迹 loss_mask 内 token 的平均 log π_θ
# advantages = 组内归一化后的 advantage（向量化）
```

### 1.2.4 GRPO 的两个重要细节

**KL 约束**：GRPO 论文在 reward 里减 KL 项防止策略跑飞：
`r_final = r − β·KL(π_θ || π_ref)`。本项目本地管线用 **LLDS**（§1.4）替代
KL 的防崩作用；veRL 管线则直接关掉 ref（`use_kl_loss=false`，理由见 §3.4）。

**importance sampling 的 log π_old**：loss 里的 ratio 需要 rollout 时旧策略的
token logprob。本地管线在 rollout 时额外做一次前向（`_compute_completion_logprobs`）
存到 `Turn.logprobs`；veRL 由框架自动算 old_log_prob。

**联系本项目**：本地管线 group_size=8（每题 8 条轨迹），veRL 配置 n=4。
GRPO 是训练骨架，下面的所有改进（LLDS/PRM-Lite/LATA）都是在这个骨架上
修奖励或 loss。

## 1.3 奖励设计：outcome reward 与过程奖励（PRM）

### 1.3.1 Outcome reward：只看结果

本项目 outcome reward 分三档（`reward.py`）：

```
1.0   答案格式正确且与参考答案精确匹配（normalize 后字符串相等）
0.0   格式正确但答案错误
-0.1  格式错误（没有恰好一行的 "Answer: ..."）
```

`normalize_answer` 做三件事：小写、去标点（Unicode 类别 P）、去冠词
（a/an/the）、合并空格。这样 "Baybrook Mall" 和 "baybrook mall" 算对。

**优点**：零成本（规则计算）、无偏（答案对错客观）、与最终目标一致。
**缺点**：**极度稀疏**。一条 500 token 的轨迹，只有一个标量信号，
模型不知道是哪个动作导致了对/错——搜索 query 写得好但答案错 vs 根本没搜，
reward 都是 0。稀疏奖励下 RL 收敛慢、方差大。

### 1.3.2 真正的 PRM（Process Reward Model）为什么难

PRM 给每个中间步骤打分（数学题每一步对错）。难点：

1. **过程标注贵**：需要人工或强模型标注每步正确性，而 outcome 标注只需要最终答案；
2. **奖励偷懒（reward hacking）**：优化过程分数 ≠ 优化最终正确率。经典的
   PRM800K 教训：过程分数涨了，最终答案正确率没涨（分数可以被"写得像对的"
   骗过去）；
3. **训练一个 PRM 本身就是项目**：4B 的 PRM 又是 4B 的显存和训练成本。

### 1.3.3 PRM-Lite：用 22 条规则折中

本项目选择**规则化的过程奖励**（`reward_lite.py`，782 行）：
不训练模型，而是用 22 条启发式规则评估轨迹的**搜索行为质量**。
12 条惩罚（P1-P12）+ 10 条奖励（B1-B10），完整清单：

| 规则 | 触发条件 | 分值 |
|---|---|---|
| P1 空查询 | query 为空 | −0.05 |
| P2 查询过长 | query > 200 字符 | −0.03 |
| P3 重复查询 | 连续两次相同 query | −0.05 |
| P4 无有效回答 | 没有 Answer: 行 | −0.10 |
| P5 轮次过多 | turns > 6 | −0.05 |
| P6 回答过短 | 答案 < 5 字符 | −0.03 |
| P7 问句式查询 | query 是完整问句而非关键词 | −0.02 |
| P8 忽略搜索结果 | 搜了但答案与结果无词重叠 | −0.05 |
| P9 只看第一个结果 | 多结果但只参考第一个 | −0.03 |
| P10 复制粘贴 | 答案含结果中 50+ 字符连续子串 | −0.03 |
| P11 不相关查询 | query 与问题几乎无词重叠 | −0.05 |
| P12 不必要搜索 | 对"1+1 等于几"类问题搜索 | −0.02 |
| B1 关键词查询 | query ≤5 词且非问句 | +0.03 |
| B2 多证据引用 | 答案参考 ≥2 个结果块 | +0.05 |
| B3 推理文本 | 搜索前后有 ≥20 字符推理 | +0.02 |
| B4 首搜命中 | 第一个 query 与问题词重叠 >30% | +0.05 |
| B5 简洁回答 | 答案长度 ∈ [20,100] | +0.03 |
| B6 格式正确 | 恰好一个非空 Answer: 行 | +0.03 |
| B7 合理搜索次数 | 简单题 1-2 次 / 多跳题 2-4 次 | +0.03 |
| B8 自己的话 | 答案不含结果 30+ 字符子串 | +0.03 |
| B9 多样化关键词 | 相邻 query 词重叠 <50% | +0.04 |
| B10 逐步聚焦 | query 特异性单调递增 | +0.03 |

三个关键设计决策（面试必问）：

1. **cap 在 [−0.2, +0.2]**：outcome reward 是 ±1.0 的主信号，过程奖励只做
   「润色」。如果过程分太大，模型会学会刷规则分数而不是答对题（reward hacking
   的前奏）。0.2 的幅度 = 约 4-7 条规则打满，足以区分行为好坏，不足以喧宾夺主。
2. **互斥规则**：P4（无 Answer）与 B6（格式正确）逻辑互斥——同时命中时按绝对值
   大的保留，避免同一事实双重计分；P10（复制粘贴）命中时强制移除 B8（自己的话），
   因为同一段文本不能既"抄"又"原创"。
3. **词重叠用 Jaccard-like 度量**：`_word_overlap_ratio` 去停用词后取交集/较
   短集合大小，对"答案是否参考了搜索结果"这类模糊判断给出连续值，再用阈值
   二值化。

**联系本项目**：PRM-Lite 加在最终 reward 上：`reward = score_answer + process_reward`。
它给 GRPO 提供了**稠密的中间反馈**——即使最终答案错了，搜索行为好的轨迹
（reward −0.1+0.2）也会比行为差的轨迹（−0.1−0.2）advantage 高，模型先学会
"怎么搜"，再学会"搜对"。

## 1.4 训练稳定性：LLD 死亡螺旋与 LLDS

### 1.4.1 LLD 死亡螺旋是什么

这是本项目遇到的最核心的训练问题（ICML 2026 论文 *On GRPO Collapse in
Search-R1* 独立发现了同一现象，我们撞上时叫它"死亡螺旋"）。实测数据
（E1-A 纯 RL，PyTRIO）：

```
step 3   correct_rate = 81.2%   ← 峰值
step 56  correct_rate = 3.1%    ← 崩了
```

崩溃的机制链条（LLD = Lazy Likelihood Displacement，懒惰似然位移）：

1. **GRPO 的 loss = −A·log π_θ**。正 advantage 的 token 被拉高似然，
   负 advantage 的 token 被压低似然。更新只有"推"没有"拉回"。
2. **当组内答案方差大**，advantage 幅度大，梯度对 log π 的"压低"非常用力。
3. 某些原本高似然的正确 token（SFT 学的、或早期 RL 学的）被反复压低 →
   模型的似然（likelihood）整体下降 → 输出开始漂移。
4. **似然下降 → rollout 质量变差 → reward 方差更大 → advantage 更大 →
   压低更狠** → 正反馈循环，直到输出熵崩、格式全乱。
5. 观察到的特征：**reward 指标还在上升/高位，但 log-likelihood 在下降**——
   这就是"死亡螺旋"：指标好≠模型好，模型在靠"赌"少数对局维持 reward。

v1 训练（API key 过期那次）的教训还多一层：搜索全失败时模型学到
"不搜直接猜"的捷径，reward 反而"不错"——**环境噪声会诱导策略走捷径**，
这正是 RL 需要防崩机制的另一个原因。

### 1.4.2 LLDS 的数学形式

LLDS（Lazy Likelihood-Displacement Stabilization）在 GRPO loss 上叠加一个
**单向似然保持惩罚**：

```
L_llds = λ · Σ_t max( 0,  log π_ref(t) − log π_θ(t) )
```

解读：对每个参与训练的 token t：
- `log π_ref(t)` = 旧策略（rollout 时）对该 token 的 logprob（我们采样时
  顺手存的）；
- 若当前策略的似然**高于**旧策略 → max(0, 负数) = 0，不惩罚（允许变好）；
- 若当前策略的似然**低于**旧策略 → 惩罚与"掉了多少"成正比。
- **单向性（max 操作）是整个设计的灵魂**：它只阻止"变坏"，不阻止"变好"。
  这是它区别于普通 KL 惩罚的地方（KL 是双向对称的，会拖慢收敛）。

三个变体（`llds.py:LLDSConfig`）：

| 变体 | 门控 | 适用场景 |
|---|---|---|
| R（Response-level） | 轨迹总似然下降才激活惩罚 | 粗粒度 |
| **A（Action-level，默认）** | 逐 token 惩罚，仅 advantage≠0 的 token | 推荐：精确 |
| MA（Mask Answer） | 只惩罚工具调用 token，保护答案 token | 保护"回答行为" |

### 1.4.3 v1 为什么失败、v2 为什么成功（本项目实测）

**v1（固定偏移，λ=0.05）**：最高 71.9%（step 58），但 step 63 崩到 1.6%。
分析：v1 的惩罚是固定幅度的（惩罚不随似然下降量变化），λ=0.05 相对
GRPO 的 advantage 梯度太弱——优势项一放大就把惩罚淹没，等于没防。

**v2（比例缩放）**：惩罚改为与似然下降量成正比（`Σ max(0, log π_ref −
log π_θ)`，实现中早期还带 `min(delta, 10.0)` 上限防单 token 爆炸）：
- 最高 correct_rate **81.2%**（step 3，追平纯 RL 峰值）；
- 关键区别：**低点能恢复**。掉到 3.1% 后能爬回 54.7%，不再永久崩溃；
- 代价：批次方差大（难题多的批次 anomalous 高达 50/64 条）——防崩≠消除方差。

**教训**：正则化的**强度必须与被防护的量同阶**。固定惩罚挡不住优势项的量级
变化；比例惩罚自动匹配梯度大小，所以有效。这跟 KL 系数要调参是同一类问题，
LLDS 的巧妙在于用 max(0,·) 单向化后，比例惩罚不会拖慢正常学习。

**联系本项目**：本地管线 `grpo_loss` 里 LLDS 与 GRPO loss 相加：
`loss = grpo_loss + λ·llds_penalty`，参考 logprob 在 rollout 时
由 `_compute_completion_logprobs` 捕获（§2.8）。

## 1.5 LATA：长度自适应优势归一化

### 1.5.1 动机：长回答被系统性低估

GRPO 的 advantage 只比较 reward。但**长轨迹天然吃亏**：

1. 长轨迹 = 多轮搜索 = 更多出错机会，即使最终答案对，中间也可能有瑕疵；
2. PRM-Lite 的惩罚规则（P5 轮次过多、P2 查询过长等）与长度正相关，
   长轨迹的过程分系统性偏低；
3. 结果：**好但长的轨迹 advantage 可能低于差但短的轨迹**，模型被诱导
   "尽早结束"，学会敷衍——多跳问题的深度推理被压制。

### 1.5.2 公式与直觉

```
A_lata = (r − mean_group) / sqrt(L)
L = 该轨迹全部 completion token 数（assistant 生成部分）
```

为什么是 **sqrt(L)** 而不是 L 或 log(L)：

- **除以 L**：过强。假设轨迹是"每步独立同分布地贡献噪声"，则总奖励的标准差
  ∝ sqrt(L)（独立随机变量和的方差 ∝ L，标准差 ∝ sqrt(L)）。除以 sqrt(L)
  恰好把不同长度轨迹的 advantage **拉回同一尺度**，这是中心极限定理给的
  启发式。除以 L 会把长轨迹压成 0，信号全无。
- **不除**：长轨迹方差大，advantage 排序被长度主导。
- sqrt(L) 是最温和且理论有据的折中：L=100 → 除 10，L=400 → 除 20，
  差 2 倍而不是 4 倍。

**联系本项目**：`reward_lite.py:LATAScaler.compute_advantages` 按
question_index 分组重算 advantage，L=0（空轨迹）时退回不除。
开关是 `--lata`。注意 LATA 与 LLDS 正交：LLDS 管 loss（防崩），
LATA 管 advantage（公平性）。

## 1.6 TITO 协议与 response_mask：哪些 token 参与训练

### 1.6.1 一条轨迹的 token 结构

多轮搜索轨迹 = 模型 token 与工具 token 交替：

```
[system prompt][问题]                          ← prompt，不训练
<tool_call>...query...</tool_call>             ← 模型生成，训练 ✓
<tool_response>[1] Title: ... Evidence: ...    ← 搜索结果，不训练 ✗
<tool_call>...query2...</tool_call>            ← 模型生成，训练 ✓
<tool_response>...</tool_response>             ← 搜索结果，不训练 ✗
Answer: Baybrook Mall                          ← 模型生成，训练 ✓
```

TITO（Token-In-Token-Out）协议：**输入输出全是 token**。agent loop 把
搜索结果 token 化后直接拼进序列，`response_mask` 标记每个 token 是否参与训练：

- 模型生成的 token → mask = 1（这些是策略 π_θ 的产物，GRPO 优化它们）
- 搜索结果的 token → mask = 0（这些是**环境**的产物，不是策略的产物）

### 1.6.2 为什么搜索结果 token 必须 mask

两个原因（面试高频题）：

1. **防止"抄答案"捷径**。如果搜索结果 token 参与 loss，模型会发现：
   "把结果里的 `Baybrook Mall` 原样复读一遍就能获得高 logprob"。
   但高 logprob ≠ 会推理——模型在训练时看到的是结果文本，测试时如果
   结果变了或没有结果，它就崩。**RL 优化的是"生成答案"这个行为，
   不是"复读输入"这个行为。**
2. **信号稀释**。搜索结果通常几百 token，若全部参与 loss，模型的大部分
   梯度来自"预测搜索结果里的下一个词"——这是语言模型任务，不是
   搜索推理任务。mask 掉之后，梯度全部集中在模型自己生成的关键 token 上。

### 1.6.3 两种实现路径

| | 本地管线 | veRL 管线 |
|---|---|---|
| 实现 | `build_training_sequences` 里逐 turn 构造 `loss_mask`（prompt/tool 部分填 0，completion 部分填 1）| `SearchR1AgentLoop.run` 里边生成边填 `all_response_mask` |
| 训练时 | loss 只对 mask=1 的 token 求均值 | veRL 框架按 response_mask 自动处理 |
| 结果 token 的 mask 值 | 不参与 loss | 0 |

另外还有一个细节：**prompt token 也不参与 loss**（mask=0）——语言模型的
训练目标永远是"预测下一个 token"，prompt 部分只有"给定"没有"预测"。

> 补充：原版 Search-R1 用 `<information>` 标签 + veRL state_masking 配置
> 实现同样的效果；我们迁移到 veRL main 后改为 response_mask 原生实现
> （`search_agent_loop.py`），更精确且不依赖框架特例（见 §3.2）。

---

# 第 2 章 本地管线代码详解（逐文件）

> 本地管线 = 用 HF transformers + PEFT 在单卡上跑的完整 GRPO 训练，
> 不依赖任何训练平台（rollout.py 除外，它是 PyTRIO 平台版）。
> 数据流顺序：
>
> ```
> prepare_data.py → data.py → protocol.py → search.py → train_grpo_local.py
> (rollout: build_prompt → generate → parse_assistant → search → tool_response)
> → reward.py / reward_lite.py / llds.py → grpo_loss → 保存 LoRA
> → eval_local.py 评估
> ```

## 2.1 data.py：数据容器（33 行）

三个职责：`SearchExample` 数据类（id/question/answers/data_source）、
`load_examples` 读 JSONL、`take_batch` 循环取批（训练数据用不完时循环）。

```python
def take_batch(examples, start, batch_size):
    return [examples[(start + offset) % len(examples)] for offset in range(batch_size)]
```

`% len` 实现循环复用：17 万条数据永远取不完，训练 N 步不重复。

## 2.2 prepare_data.py：数据获取与清洗

数据源是 ModelScope 上的 NQ+HotpotQA 打包集（`zhuangzhuang2023/
nq_hotpotqa_train`，固定 revision 保证可复现），train.parquet 169615 行 /
test.parquet 51713 行。

```python
def download_dataset(raw_dir):
    path = dataset_snapshot_download(DATASET_ID, revision=DATASET_REVISION,
        local_dir=str(raw_dir), allow_patterns=["train.parquet", "test.parquet"])
```

固定 revision 是数据工程的好习惯：数据集更新不会悄悄改变实验结果。

清洗逻辑 `normalize_row`：抽取 question + golden_answers（兜底 reward_model.
ground_truth.target）→ 统一成 `{id, question, answers, data_source}` JSONL。
`select_dev` 从 test 集**按来源等量抽样**（7 个 benchmark 各 10 条 = 70 题）
生成固定 dev 集——这是评估口径的基础，训练、评估都在这 70 题上对比。

## 2.3 protocol.py：协议层（本项目最精妙的文件之一）

### 2.3.1 为什么用 Qwen 原生 tool-call 格式

`SEARCH_TOOL` 是 OpenAI 风格的 function 定义，`SYSTEM_PROMPT` 规定：
每轮一个动作（tool call 或 Answer）、query 用简洁英文、禁止同一轮又搜又答。
模型走的是 **Qwen 原生工具调用协议**（`<tool_call><function=search>
<parameter=query>...</parameter></function></tool_call>`）而不是论文的
`<search>` 标签。原因：

1. Qwen3.5 的 chat template 原生支持 tools 定义（`apply_chat_template(tools=[...])`
   会注入工具调用格式说明），模型**天生会**这种格式，不需要从零教；
2. 与 SFT 阶段（用 DeepSeek 蒸馏轨迹，同款格式）无缝衔接——SFT 模型
   输出什么，RL 就解析什么（踩坑篇会讲为什么这很重要）。

### 2.3.2 `_render_chat`：兼容 tokenizer 的多种返回类型

```python
rendered = tokenizer.apply_chat_template(messages, tools=[SEARCH_TOOL],
    tokenize=True, add_generation_prompt=add_generation_prompt, enable_thinking=False)
if isinstance(rendered, Mapping): rendered = rendered["input_ids"]
if hasattr(rendered, "tolist"):   rendered = rendered.tolist()
if rendered and isinstance(rendered[0], list): rendered = rendered[0]
return [int(token) for token in rendered]
```

不同 transformers 版本返回 list / tensor / BatchEncoding / 嵌套 list，
这里统一成 `list[int]`。`enable_thinking=False` 关掉 Qwen3 系思考模式。

### 2.3.3 `build_next_prompt`：token 级精确拼接（重点）

多轮对话最大的坑：**用 chat template 重新编码历史消息，可能和你采样时
的 token 对不上**（trim、重新格式化、模板版本差异都会造成漂移）。
漂移的后果：你训练用的 token 序列和模型实际看到的对不上，KL/ref 全部失真。

`build_next_prompt` 的解法是「真采样 token + 模板补丁」：

```python
canonical_prompt = build_prompt(tokenizer, messages_before_assistant)  # 重编码
# 用空 assistant 单独提取模板闭合 token（</assistant> 部分）
empty_assistant_end = _render_chat(tokenizer,
    [*messages_before_assistant, {"role": "assistant", "content": ""}],
    add_generation_prompt=False)
assistant_closing_tokens = empty_assistant_end[len(canonical_prompt):]  # 纯模板增量
# 下一步 prompt = 真采样 token + 补上缺失的结束符 + 新 tool observation
overlap = _suffix_prefix_overlap(completion_tokens, assistant_closing_tokens)
return [*previous_prompt_tokens, *completion_tokens,
        *assistant_closing_tokens[overlap:], *observation_tokens]
```

逻辑链：模型采样时可能已经生成了部分结束符 → 计算 suffix/prefix 重叠 →
只补缺失部分 → 再拼 tool observation。这样**下一轮的输入 token 中，
模型生成部分 100% 是真实采样的 token**（logprob 对齐才有意义），
模板控制字符部分由 canonical 渲染补全。两处断言
（`empty_assistant_end[:len(canonical_prompt)] == canonical_prompt` 等）
保证模板行为符合预期，不符就报错而不是静默漂移。

### 2.3.4 `parse_assistant`：动作解析

```python
def parse_assistant(text) -> ParsedAssistant:
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    if not matches:
        kind = "invalid" if "<tool_call>" in text else "answer"
        return ParsedAssistant(kind=kind, content=text.strip())
    if len(matches) != 1 or text[matches[0].end():].strip():
        return ParsedAssistant(kind="invalid", content=text.strip())  # 多个/尾随文本
    query = matches[0].group(1).strip()
    if not query or "<" in query or ">" in query:
        return ParsedAssistant(kind="invalid", ...)
    return ParsedAssistant(kind="tool", content=text[:matches[0].start()].strip(), query=query)
```

三种输出：`tool`（恰好一个合法 tool call，query 非空且不含标签）、
`answer`（无 tool call 的文本，final answer 从里面提取 `Answer:` 行）、
`invalid`（有 `<tool_call>` 字样但格式坏了，或一个 turn 里有多个动作）。
**严格性是故意的**：宁可不给搜索机会，也不执行解析错误的 query
（解析错误的 query 会把垃圾灌进上下文）。

## 2.4 search.py：5 种可插拔搜索后端（1012 行）

### 2.4.1 架构：统一接口 + 各后端自治

```
SearchClient = DeepSeekSearchClient | WikipediaSearchClient | ZhihuSearchClient
             | BingSearchClient | MiMoSearchClient

create_search_client("deepseek", env_path) → 读 .env → DeepSeekSearchClient
```

所有后端实现同一个接口：`search(query) -> SearchResult(ok, items, latency,
status, error)`；`SearchItem(title, content, source, url)`。
训练代码只依赖 `SearchClient` 协议——**换后端不改一行训练代码**
（这是 8-31 那天 DeepSeek API 挂了能快速切知乎/MiMo 的前提）。

| 后端 | 延迟 | 特点 |
|---|---|---|
| DeepSeek Search | 1-3s | 真实网页搜索，Evidence 模式，需 key（deepseek-search 包）|
| 知乎全局搜索 | ~0.68s | 真实网页搜索，多 key 轮转 |
| Wikipedia | ~0.3s/请求 | 免费稳定，英文百科 |
| Bing HTML | ~1s | 免 key，HTML 解析（`b_algo` 块正则）|
| MiMo LLM | 6-14s | LLM 模拟搜索（生成事实），无真实检索 |

### 2.4.2 公共基础设施（每个后端都有）

**重试策略**：429 / 5xx / 超时重试 1-2 次，指数退避
`retry_delay * (2**attempt)`。知乎后端多一层：429 时**轮转到下一个 key**
并把当前 key 拉黑（`_disable_credential`），全部 key 都 429 才报错。

**限速器** `_wait_for_rate_slot`：共享锁 + `_next_request_time` 时间戳，
保证相邻请求间隔 ≥ min_request_interval（Wikipedia 0.31s ≈ 200 RPM，
Bing 0.35s ≈ 170 RPM）——主动限速比被动吃 429 便宜（429 重试是纯浪费）。

**统计** `SearchStats`：requests/successes/timeouts/rate_limits/errors/
latency_total + 后端特有指标（DeepSeek 的 token 用量、知乎的 failover 次数），
`metrics()` 输出成功率/超时率/429 率/平均延迟，直接进 SwanLab 监控。
`_error_result` 统一把异常转成 `SearchResult(ok=False, ..., error=...)`
——**错误是数据不是异常**，rollout 拿到 ok=False 会继续跑，绝不中断训练。

**运行时缓存**（MiMo 之外的通用能力在 agent loop 层）：veRL 版在
`SearchR1AgentLoop` 里做了类级 `_shared_cache`，同 step 内相同 query 直接命中。
（教训见 §4.4「预生成缓存」：缓存 key 必须和运行时查询 key 一致。）

### 2.4.3 DeepSeek 后端细节

`DeepSeekSearchClient.search` 调 `deepseek_search(query, mode="evidence")`，
返回带编号的证据文本，`parse_evidence` 用 `EVIDENCE_PATTERN` 正则拆成
`[N] Source: ... Evidence: ...` 的条目，失败时两级降级（`[N] Title/Content`
格式 → 整段兜底）。`__post_init__` 里就 `resolve_api_key`——
**提前失败**：key 无效在初始化时报错，而不是 rollout 到一半才发现全部 401
（v1 训练 15 步全废的教训，见 §4.4）。

### 2.4.4 `format_item`：证据的统一格式

```python
def format_item(item, index):
    if item.source is None and item.url is None:
        return f"[{index}] Source: {item.title}\n    Evidence: {item.content}"
    lines = [f"[{index}] Title: {item.title}", f"    Content: {item.content}"]
    if item.source: lines.append(f"    Source: {item.source}")
    if item.url:    lines.append(f"    URL: {item.url}")
    return "\n".join(lines)
```

有来源的用 Title/Content/Source/URL 四段式，没来源的退化成 Source/Evidence
两段式。编号 [1] [2] 让模型能"引用"具体证据（PRM 的 B2 规则就靠编号块判定
多证据引用）。

## 2.5 reward.py：outcome reward（49 行）

`score_answer(text, references)` → `RewardResult(reward, valid_format,
exact_match, answer)`。核心在 `normalize_answer`（§1.3.1 讲过）与
`extract_answer`：`ANSWER_PATTERN` 是 `^\s*Answer:\s*(.*?)\s*$`（MULTILINE），
**恰好一行**才有效——多行 Answer 判格式错误。这个严格性是有意的：
格式就是协议，协议不稳 RL 就学不到动作（见 §4.4「格式 100% 无效」）。

## 2.6 reward_lite.py：PRM-Lite + LATA（782 行）

### 2.6.1 文本分析基元

四条工具函数（`_QUESTION_STARTS`、`_STOPWORDS` 两个词表 + 函数）：

- `_is_question_sentence`：首词是疑问词（what/who/.../must）或含 "?" →
  判断 query 是"完整问句"还是"关键词"（P7/B1 用）；
- `_word_overlap_ratio`：去停用词的 Jaccard-like 重叠，按较短集合归一
  （P8/P9/P11/B2/B4/B9 用）；
- `_common_substring_length`：最长公共连续子串，从 min(len,100) 向下搜
  （P10/B8 的"复制粘贴"检测）；
- `_query_specificity`：词数 + 大写首词比例的加权分（B10 的"逐步聚焦"）。

### 2.6.2 特征提取（duck-typing 设计）

```python
queries   = _extract_queries_from_turns(turns)      # 从每轮 text 用 TOOL_CALL_PATTERN 抠 query
answer    = _extract_answer(final_text)             # 复用 reward.py 的解析
messages  = getattr(trajectory, "messages", [])
search_blocks = _extract_search_result_blocks(messages)  # tool 消息按 \n\n 切块
```

`score()` 对所有对象都用 `getattr(x, name, default)` 鸭子类型访问——
所以本地 `Trajectory`、评估的 `EvalTrajectory`、veRL 的
`_PRMTrajectoryAdapter` 都能直接喂给它，零适配成本。

### 2.6.3 主流程与互斥

`score()` 遍历 12+10 条规则（每条都是 `_check_xxx` 方法返回 0.0 或分值），
`all_rules` 记录全部（评估时算"规则命中率"用），`penalties/bonuses` 只记
命中的。互斥调整（P4↔B6、P10→删 B8）后 cap：

```python
total_penalty = max(sum(penalties.values()), cf.max_penalty)   # ≥ −0.2
total_bonus   = min(sum(bonuses.values()),  cf.max_bonus)      # ≤ +0.2
process_reward = total_penalty + total_bonus
```

### 2.6.4 `LATAScaler` 与便捷包装

`apply_lata(trajectories)` 按 question_index 分组重算 advantage
（§1.5）。`score_with_prm(trajectory)` 单轨迹便捷函数。
`PRMLiteConfig` 给每条规则一个开关（`p1_empty_query=True/False`），
消融实验时能单独关规则。

## 2.7 llds.py：LLDS 正则化（488 行）

三个组件：

1. **`LLDSConfig`**：`lambda_reg`（默认 0.05）、`variant`（R/A/MA）、
   `mask_answer`。`__post_init__` 校验变体名，MA 自动开 mask_answer。
2. **`LLDSTracker`**：`store_reference(trajectory, id)` 把 rollout 采样的
   参考 logprob 按训练序列对齐存起来（关键：**与 build_datum 相同的
   token 顺序**——prompt/observation 区域补 0.0、completion 区域填真值，
   再整体右移一位对齐 next-token 预测）；`compute_regularization(id,
   current_logprobs)` 逐 token 算 `max(0, ref−cur)` 求和，按变体做门控
   （R 变体先比轨迹总似然，A 变体跳过 advantage=0 的 token，mask_answer
   跳过非工具 token）。
3. **`extract_completion_logprobs` + `llds_loss_fn`**：PyTRIO 集成辅助。

对齐是这个文件最容易被忽略的重点：LLDS 比较"同一个 token 位置"的旧/新
logprob，如果参考序列和训练序列的 token 顺序不一致（差一个 token），
惩罚就打在错误的 token 上，防崩直接失效。47 个单元测试里有一半在测对齐。

## 2.8 train_grpo_local.py：本地训练入口（858 行）

### 2.8.1 配置常量与显存策略

```python
BASE_MODEL = "wang072266/qwen3.5-4b-search-r1-sft"
LORA_R=16, LORA_ALPHA=32, LORA_TARGET=["q_proj","k_proj","v_proj","o_proj",
                                      "gate_proj","up_proj","down_proj"]
LEARNING_RATE = 1e-5;  GRAD_ACCUM = 4
MICRO_BATCH_MAX_TOKENS = 4000;  MAX_TRAIN_SEQ_LEN = 4096
MAX_SEARCH_CALLS=4, MAX_ASSISTANT_TURNS=6, MAX_TRAJECTORY_TOKENS=8192,
MAX_ASSISTANT_TOKENS=1024, MAX_TOOL_RESPONSE_TOKENS=1024
```

4-bit QLoRA + 目标模块全家桶（Q/K/V/O + 三个 MLP 投影）。
`MICRO_BATCH_MAX_TOKENS=4000` 和 `MAX_TRAIN_SEQ_LEN=4096` 是 12GB 卡的
OOM 防线（§4.4 讲了 8000→4000 的调参史）。

### 2.8.2 `_batch_generate`：批量生成 + 左 padding 陷阱

```python
max_len = max(len(p) for p in prompts)
input_ids = torch.full((n, max_len), pad_id, ...)   # 左 padding
for i, p in enumerate(prompts):
    input_ids[i, -len(p):] = torch.tensor(p, ...)   # prompt 靠右对齐
...
new_ids = outputs[i, max_len:].tolist()             # ← 新 token 从 max_len 开始切
```

**左 padding 下 prompt 不在位置 0**：`input_ids[i, max_len:]` 才是新生成的
token。早期版本用 `outputs[i, prompt_len:]` 切片（prompt_len=原始长度），
左 padding 时把 prompt 尾部的 token 当成了 completion——**prompt 被重复训练**。
这是本文件最重要的 bug 修复之一（§4.4）。

`StopOnTokens` 在 eos 上停；`do_sample=True + temperature + top_p` 采样。

### 2.8.3 `_compute_completion_logprobs`：LLDS 的参考来源

rollout 后对 prompt+completion 做一次**无梯度前向**，在位置
`prompt_len + j − 1` 的 logits 上取 token_j 的 logprob（第 j 个 completion
token 由第 prompt_len+j−1 个位置的输出预测）。这就是 `Turn.logprobs`，
LLDS 的 `log π_ref`。代价：每轮多一次前向（本地管线不在乎这点时间）。

### 2.8.4 `rollout_question`：单题多轮状态机

```
for turn_idx in range(MAX_ASSISTANT_TURNS):
    prompts = [build_prompt(tokenizer, msgs[i]) for 未完成的轨迹 i]   # 并发批生成
    completions = _batch_generate(...)
    logprobs_batch = _compute_completion_logprobs(...)
    for 每条轨迹:
        parsed = parse_assistant(completion_text)
        if parsed.kind == "tool" and 还有搜索额度:
            result = search_client.search(parsed.query)   # 同步搜索
            tool_msg = {"role": "user", "content": f"<tool_response>{tc}</tool_response>"}
            # ⚠️ Qwen3.5 不支持 role:"tool" 消息，搜索结果包成 user 消息
        elif parsed.kind == "answer":
            finished = True
```

三个设计点：
1. **同题 group_size 条轨迹共享初始 prompt**，用不同随机种子分叉
   （rollout.py 里更明显，见 §2.9）；
2. **搜索同步执行**：本地版单卡跑，线程池是 search.py 客户端内部的事；
   实际上 GPU 大部分时间在等搜索返回（§4.4「本地训练瓶颈」）；
3. **`<tool_response>` 包 user 消息**：Qwen3.5 chat template 不支持
   `role:"tool"`，这是踩坑踩出来的兼容方案（veRL 版同款）。

### 2.8.5 异常检测 `is_anomalous`

```python
def is_anomalous(traj):
    if not traj.final_text or not traj.final_text.strip(): return True   # 空白
    if traj.search_calls >= MAX_SEARCH_CALLS and not traj.valid_format:  # 搜满没答
        return True
    if traj.search_calls == 0 and not traj.valid_format:                 # 没搜没答
        return True
    return False
```

三类异常：空白输出、搜索用尽仍无答案、完全没搜且格式错误。异常轨迹
`advantage=0` 且不进 `build_training_sequences`——**loss mask 排除**。
10 步验证：160 条轨迹检出 4 条（2.5%）。为什么异常必须排除：
异常轨迹的 logprob 极低，若参与训练，负 advantage 会把"崩溃输出"的
似然拉高（loss 最小值方向是模仿它），污染信号。

### 2.8.6 `compute_group_advantages`：GRPO 核心

```python
rewards = [t.reward for t in group if not t.anomalous]  # 先排除异常
mean_r = np.mean(rewards); std_r = np.std(rewards) + 1e-8
for t in group:
    t.advantage = 0.0 if t.anomalous else (t.reward - mean_r) / std_r
```

注意两个细节：**排除异常后重算组均值/标准差**（异常轨迹不污染基线）；
std 加 1e-8 防除零（全组 reward 相同 → advantage 全 0 → 这组无学习信号，
正确行为）。

### 2.8.7 `build_training_sequences`：mask 与对齐

逐 turn 拼 `all_ids`，同步拼 `loss_mask`（prompt 部分 0.0、completion
部分 1.0）和 `ref_logprobs`（completion 位置填 rollout logprob，长度不符
时截断/补零）。超 4096 截断（OOM 防线）。返回的 dict 同时给 grpo_loss
和 LLDS 用。跳过 anomalous 和 advantage=0 的轨迹。

### 2.8.8 `pack_micro_batches` 与 `_chunked_log_softmax_gather`

micro-batch 打包：按长度排序 + `max_padded_tokens` 限制（长序列短序列
分组，减少 padding 浪费）。`_chunked_log_softmax_gather` 按 512 token
分块算 log_softmax——**大词表 OOM 修复**：Qwen3.5 词表 151936，
`[batch, seq, vocab]` 全量 log_softmax 的峰值 = 4000×151936×2B ≈ 1.2GB/条，
分块后峰值降到 512×151936×2B ≈ 155MB（§4.4 有完整账）。

### 2.8.9 `grpo_loss`：GRPO + LLDS 的数学落地

```python
shift_logits = logits[:, :-1, :]          # 位置 t 的输出预测 token t+1
shift_labels = input_ids[:, 1:]
token_log_probs = _chunked_log_softmax_gather(shift_logits, shift_labels)
mean_lp = (token_log_probs * shift_mask).sum(1) / n_loss_tokens   # 每条轨迹均值
loss = -(advantages * mean_lp).mean()                              # GRPO
# LLDS：
token_penalty = torch.clamp(shift_ref_lp - token_log_probs, min=0) # max(0, ref-cur)
if llds_variant == "R":  # 响应级门控
    gate = (cur_total < ref_total).float()
    token_penalty *= gate
llds_penalty = (token_penalty * shift_mask).sum() / n_active
loss = loss + llds_lambda * llds_penalty
```

`shift_mask = loss_mask[:, 1:]` 完成 next-token 对齐。R 变体先比较轨迹
总似然（只在总似然下降时激活）——与 §1.4 的 gate 设计一致。

### 2.8.10 训练主循环

每步：取题 → `model.eval()` rollout（8 题 × 8 轨迹 = 64 条）→ 异常检测 →
组 advantage → PRM-Lite 叠加（`t.reward += process_reward`）→ LATA 重算 →
`build_training_sequences` → micro-batch → **`gc.collect() + empty_cache()`**
（rollout 的 KV cache 碎片不还给 CUDA 的修复）→ `model.train()` →
逐 micro-batch `grpo_loss` + `scaled_loss.backward()`（grad accum）→
clip_grad_norm(1.0) → step → 指标打印 → 每 25 步存 LoRA adapter。

指标行示例：
```
step=  3/50  loss=0.0452  reward=0.812  correct=52/64(81.2%)  search=1.9
anomalous=2 llds=0.0013  tokens=51200  time=960s
```

## 2.9 rollout.py：PyTRIO 平台版（446 行）

与 2.8.4 相同协议、不同采样机制（PyTRIO 的 `sample_async` 并发采样）。
值得讲的三个设计：

**根轨迹分叉**：首轮每题只发一个请求，`num_samples=group_size` 一次采样
8 条分支（共享 prompt 前缀，KV cache 复用），`copy.deepcopy` 后各分支
独立维护 messages/搜索历史。首轮分叉后 prompt 已不同，后续每轨迹单独采样。

**`PendingSearch` 解耦**：解析出的搜索先收集（`consume_assistant` 返回
PendingSearch），一轮生成全部结束后 `resolve_searches` 用
ThreadPoolExecutor 并发执行（`search_concurrency=16`），再按原顺序接回——
**生成与搜索批量交错**，GPU 不空等单个搜索。

**`fit_tool_content` 预算控制**：按完整结果条目逐个试拼（`\n\n` 连接），
超 `max_tool_response_tokens` 或总轨迹 token 预算就停——搜索结果按条目
截断而不是硬切字符，保证证据完整性。

## 2.10 eval_local.py：评估脚本（524 行）

与训练同构的 rollout（`evaluate_question`），差异点：

1. **greedy 解码**：`do_sample=False`——评估要可复现（面试高频：为什么
   评估用 greedy 而训练用采样？因为要消除随机性、测模型"最可能的"行为）；
2. 单轨迹/题（不做组采样）；
3. **多 checkpoint 扫描**：`--checkpoint-dir` 自动发现 `step-N/` 和 `final/`，
   逐个加载评估（每个 checkpoint 后 `del model + empty_cache` 防显存累积），
   最后打印跨 checkpoint 对比表；
4. **分源 EM**：`em/nq`、`em/hotpotqa`、`em/triviaqa`... 7 个来源各自报 EM；
5. **PRM 规则命中率**：对每条轨迹跑一遍 22 条规则，统计每条规则的触发率
   （>5% 的打印）——评估时白嫖一次"行为分析"；
6. `--no-adapter`：全参数 checkpoint（veRL 产物）直接当 base 加载。

输出 JSONL：每行一条轨迹（含 final_text/search_calls/em）+ 末尾 summary
行（全部指标）。`analyse.py` 消费这些 summary 画 EM/format 曲线。

## 2.11 蒸馏与 SFT 管线

### 2.11.1 distill_trajectories.py：教师蒸馏（632 行）

**为什么蒸馏**：RL 从零开始教模型"工具调用格式"非常低效（v1 训练证明纯 RL
格式收敛慢且不稳）；先让强模型（DeepSeek Chat）示范正确轨迹，SFT 几小时
就教会格式与推理模式，RL 再负责优化策略。这是 DeepSeek-R1 同款
"冷启动"思路的迷你版。

**Prompt 设计**（本文件最有价值的部分）：

```
1. 协议规定：每轮先推理再动作（tool call 或 Answer，绝不两者同轮）
2. 标记语言：---BEGIN ASSISTANT--- / ---BEGIN TOOL--- / ---END TOOL---
   三段式节标记，解析器用 SECTION_SPLITTER 正则切分
3. 让教师【模拟】搜索结果：用自身知识生成带来源的伪搜索结果
   （无真实检索——见 README 已知限制 4）
4. 两个 few-shot 示例：简单题（1 次搜索）与多跳题（2 次搜索），
   示范了 query 迭代（先查成立年份再查另一本杂志）
```

工程细节：断点续传（`load_completed_ids` 扫输出文件跳过已完成的 id）、
失败原样落盘 `*.failures.jsonl`（debug 用）、双层重试（SDK 内置 1 次 +
外层指数退避 5 次）、请求间隔 0.5s 防限流。抽样 NQ 200 + HotpotQA 200
交错排列，最终成功 397 条（3 条解析失败）。

### 2.11.2 fix_distill_tags.py：格式修复

DeepSeek 生成的 tool call 标签有各种坏法：`</call>`、`</talk>`、`</tool>`、
`</search>` 错关、`query=` 前缀…… 修复策略是 **tag-stripping**：
`TOOL_CALL_BLOCK_RE` 匹配以 `<tool_call>` 开头的整个块（容错各种闭合标签），
`XML_TAG_RE.sub("", inner)` 剥掉所有 XML 标签拿到纯 query 文本，再处理
`query=` 前缀残留。Answer 行和 tool 消息 content 同步转成目标格式。

**教训**：真实 LLM 输出的格式噪声远超想象，解析器必须"对输入宽容、
对输出严格"——宽容解析（多级容错）+ 严格重建（输出干净格式）。

### 2.11.3 sft_train.py：QLoRA SFT（313 行）

```python
LORA_R=32, LORA_ALPHA=64, LORA_DROPOUT=0.05   # rank 比 GRPO 阶段的 16 大
BATCH_SIZE=2, GRAD_ACCUM=4, LR=2e-4, 3 epochs, MAX_LENGTH=2048
bnb: 4-bit nf4 + double_quant + bf16 compute
optim: paged_adamw_8bit
```

**rank 含义**：LoRA 在冻结权重旁加低秩旁路 `ΔW = B·A`（B: d×r, A: r×d），
r 是旁路的秩。r=32 相对 4096 维的 hidden 是很小的子空间，但对"学会格式 +
模仿轨迹"足够；alpha=64 使缩放因子 alpha/r = 2（更新幅度 ×2）。SFT 阶段
r=32 比 RL 阶段 r=16 大是合理的：SFT 要学的分布变化（从不会工具调用到会）
比 RL 的（策略微调）大。

数据构造：每条蒸馏轨迹拼成
`<|im_start|>system...<|im_end|>\n<|im_start|>user 问题<|im_end|>\n<|im_start|>assistant 轨迹<|im_end|>`
单一序列，标准 CLM loss（next-token prediction 覆盖整条轨迹）。

**loss 曲线解读**（150 steps / 3 epochs）：1.6515 → 0.1143（min 0.085）。
1.65 起点 = 基座对"工具调用+搜索轨迹"这种分布完全陌生；0.11 = 模型几乎
逐 token 复现轨迹。**SFT loss 低只说明"会模仿"，不说明"会答题"**——
评估证据：SFT 模型在 dev 集 EM 只有 1.43%、搜索 0 次。SFT 教的是格式，
能力要靠 RL 逼出来。这个反差本身就是面试好素材。

### 2.11.4 merge_sft.py：合并与上传

`PeftModel.merge_and_unload()` 把 LoRA 旁路权重加回基座 → 保存 fp16 完整
模型（7.9GB）→ 上传 HF（`wang072266/qwen3.5-4b-search-r1-sft`）。
合并是为了 veRL 全参数训练不需要 adapter 加载路径。

## 2.12 单元测试：test_llds.py（47 项）与 test_reward_lite.py

test_llds 覆盖：对齐逻辑（prompt/observation 补零、右移）、三变体门控、
mask_answer 交互、长度不匹配报错、空轨迹报错。
test_reward_lite 覆盖：22 条规则逐一构造正/负样例、互斥逻辑、cap 边界、
LATA 分组与 sqrt(L)。

**为什么要给启发式规则写这么多测试**：规则是"业务逻辑"，边界情况
（空 query、全停用词、单结果、超长文本）极易静默算错；错一条规则 =
奖励偏置 = RL 学歪，而且训练 50 步后才暴露。测试在改动规则时是回归网。
---

# 第 3 章 veRL 管线：生产级多卡全参数训练

> veRL（volcengine/verl）是字节开源的 LLM 强化学习框架，Search-R1 官方
> 实现就是在 veRL fork 上做的。我们把本地管线验证过的整套协议迁移到
> veRL main（0.10），实现**全参数 FSDP 训练 + vLLM 推理 + 自定义 AgentLoop**。

## 3.1 veRL 架构总览

### 3.1.1 角色分工（Ray 分布式）

```
                        ┌─────────────────────────────────────────────┐
                        │              Ray Cluster（单节点 2×4090）      │
                        │                                             │
  DataLoader ──────────▶│  Trainer（driver 进程，hydra 入口）          │
  parquet                │    │                                        │
                         │    │ 下发 batch（DataProto）                │
                         │    ▼                                        │
                         │  ActorWorker ×2（FSDP，每卡一个）            │
                         │   - 权重分片（shard），全参数训练            │
                         │   - 前向/反向/优化器（Adam 状态可 CPU offload）│
                         │    ▲ update_weights（每步训完同步权重）      │
                         │    │                                        │
                         │  vLLM RolloutWorker ×2（每卡一个推理引擎）   │
                         │   - PagedAttention + continuous batching    │
                         │   - 执行 AgentLoop 里的 generate 调用        │
                         │    │                                        │
                         │    ▼                                        │
                         │  AgentLoopWorkerTQ（transfer_queue 框架）    │
                         │   - 持有 SearchR1AgentLoop 实例 ×N          │
                         │   - 多轮循环：generate → parse → search      │
                         │   - 输出 response_ids + response_mask       │
                         │    │                                        │
                         │    ▼                                        │
                         │  RewardLoopWorker ── 调 reward_fn（EM+PRM）  │
                         └─────────────────────────────────────────────┘
```

关键概念：

- **HybridEngine**：同一批 GPU 上训练引擎（FSDP）与推理引擎（vLLM）
  分时复用显存——rollout 阶段 vLLM 用、训练阶段 FSDP 用，
  `free_cache_engine=true` 在训练前释放 vLLM 的 KV cache。
- **DataProto**：veRL 的数据容器 = `batch`（tensor 字典：input_ids、
  attention_mask、response_mask、token_level_scores...）+ `non_tensor_batch`
  （python 对象字典：data_source、reward_model、tool_extra_fields...）。
- **AgentLoopBase 接口契约**：`__init__(trainer_config, server_manager,
  tokenizer, processor, dataset_cls, data_config, **kwargs)` +
  `async run(sampling_params, priority, **kwargs) -> AgentLoopOutput`。
  `AgentLoopOutput`（pydantic）= `prompt_ids / response_ids / response_mask
  / response_logprobs / num_turns / metrics / extra_fields`。
- **TransferQueue（TQ）**：Ascend/TransferQueue 框架，把 agent loop 的
  多轮 generate 请求排队调度给 vLLM——**本项目最后卡住的瓶颈就在这里**
  （§4.4）。

### 3.1.2 一次训练 step 的完整时序

```
时间轴（理论值，4090 上实际受 TQ 影响见 §4.4）

① 数据准备（~1s）
   DataLoader 取 8 题 → 每题重复 n=4 次 → 32 条 prompt（GRPO 组）
   → 切 DataProto → 交给 AgentLoopWorkerTQ

② Rollout 多轮循环（设计目标 3-5 min，TQ 瓶颈下 50-70 min）
   for turn in 1..max_assistant_turns:
     AgentLoopWorkerTQ 收集活跃轨迹的 prompt_ids
       → TQ 调度 → vLLM 批量生成（PagedAttention, ~几秒/轮）
       → parse_assistant 解析动作
       → 是 search：调搜索后端（0.7-14s/次，线程池并发）
          → tool token 拼入 response（mask=0）
       → 是 answer：结束该轨迹
   → 产出 AgentLoopOutput × 32（含 response_mask）

③ old_log_prob 计算（~8% 步时）
   FSDP 引擎对 32 条轨迹做一次无梯度前向，存 π_old（GRPO ratio 分母）

④ 奖励计算（~几秒，CPU）
   RewardLoopWorker 解码轨迹 → reward_fn：
   EM+Format（reward.py）→ +PRM-Lite（可选）→ 输出 token_level_scores
   → GRPO 组内归一化 advantage（adv_estimator=grpo）

⑤ 训练（10-20% 步时）
   FSDP 引擎逐 micro-batch：前向（unshard → 计算 → reshard）
   → GRPO clipped loss（仅 response_mask=1 的 token）
   → 反向 → 梯度 all-reduce → clip → optimizer.step()
   （optimizer_offload 时 Adam 状态在 CPU，逐步流式进出）

⑥ update_weights 权重同步（3-5% 步时，小模型）
   FSDP shard →（unshard / 分层）→ vLLM 引擎权重更新
   （本项目 OOM 战场，§4.4 详述）

⑦ checkpoint（save_freq=25）+ 指标落盘 → 下一 step
```

## 3.2 search_agent_loop.py 逐段讲解

这是 veRL 迁移的核心文件（425 行）。`@register("search_r1_agent")` 把类
注册进 veRL 的 agent loop 注册表，训练配置里
`default_agent_loop=search_r1_agent` 即可启用。

### 3.2.1 接口契约与防御性导入

```python
try:
    from verl.experimental.agent_loop.agent_loop import (AgentLoopBase, AgentLoopOutput, register)
except ImportError:
    AgentLoopBase = object   # veRL 未安装时允许 inspect 导入
```

`__init__` 完整接收 veRL 注入的依赖（trainer_config/server_manager/
tokenizer/processor/dataset_cls/data_config/hf_model_type + **kwargs）。
veRL main 0.10 把 `dataset_cls/data_config` 变成必填参数、用
`DictConfigWrap` 包配置——这些 API 漂移都是 probe 抓出来的（§3.5）。

### 3.2.2 类级单例：为什么 64 个 agent loop 共享一个搜索客户端

```python
_shared_client: SearchClient | None = None
_shared_cache: dict[str, Any] = {}
_shared_config_key: str = ""
```

veRL 给每个并发 sample 实例化一个 AgentLoop。如果不共享：
- 64 个实例各自建 HTTP 连接池、各自读 .env；
- 缓存互不可见——同 step 内 32 条轨迹搜同一个 query（比如都在搜
  "GGP shopping mall Texas"）要打 32 次 API；
- 限速各自为政，知乎后端的并发上限被架空。

共享后：`config_key`（backend:model:timeout）变了才重建客户端；
运行时缓存同 step 内自然命中（真实观察命中率约三成，随 batch 增大上升）。

### 3.2.3 run()：TITO 协议的落地

```python
async def run(self, sampling_params, priority=0, **kwargs) -> AgentLoopOutput:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(kwargs["raw_prompt"])
    prompt_ids = await self.loop.run_in_executor(None, self._apply_chat_template, messages)
    ...
    for _turn_idx in range(max_assistant_turns):
        current_prompt = prompt_ids + all_response_ids
        budget = min(max_model_len - len(current_prompt) - 16,
                     response_length - len(all_response_ids) - 16)   # 双预算
        output = await self.server_manager.generate(request_id=uuid4().hex,
                    prompt_ids=current_prompt,
                    sampling_params={**sampling_params, "max_tokens": max_tokens})
        completion_ids = list(output.token_ids)
        all_response_ids.extend(completion_ids)
        all_response_mask.extend([1] * len(completion_ids))    # 模型 token → 训练
        parsed = parse_assistant(self._decode_safe(completion_ids))
        if parsed.kind == "tool" and parsed.query:
            result = cache 命中 or await asyncio.to_thread(search)
            tool_ids = tokenizer.encode(tool_text, add_special_tokens=False)
            all_response_ids.extend(tool_ids)
            all_response_mask.extend([0] * len(tool_ids))      # 工具 token → 不训练
        elif parsed.kind == "answer": break
        else: break    # invalid 输出直接终止（交给 reward 判 −0.1）
    return AgentLoopOutput(prompt_ids=prompt_ids, response_ids=all_response_ids,
        response_mask=all_response_mask, num_turns=search_calls+1,
        metrics=metrics, extra_fields={"response_text": ..., "search_calls": ...})
```

逐段要点：

1. **双预算**：`max_model_len`（vLLM 引擎上限 6144）与
   `response_length`（veRL 数据字段 4096）取 min，再留 16 token 安全边距。
   多轮上下文每轮都在涨（工具 token 变下轮 prompt），不控预算就会
   中途 OOM 或超长报错——**预算要同时约束"当前轮生成"和"整条轨迹"**。
2. **mask 边生成边填**：模型 token 全 1、工具 token 全 0，这就是 TITO
   的 token 级实现，比论文版 `<information>` 标签的 state_masking
   更精确（不依赖框架对标记字符串的特判）。
3. **搜索结果直接 encode 进序列**：`tokenizer.encode(tool_text)`，
   不用 chat template 重渲染（避免 template 改写历史——与 §2.3.3
   同一哲学）。所以结果包成 `<tool_response>` 文本，因为 Qwen3.5
   的 template 不支持 `role:"tool"`。
4. **extra_fields 是给 reward 的通道**：`response_text`（最终答案文本）、
   `search_calls` 从这里流向 reward worker（§3.3）。

### 3.2.4 `_apply_chat_template`：tokenize=False 的原因

veRL 环境 transformers 5.10 下 `apply_chat_template(tokenize=True)` 返回
一个"像 dict 但不是 dict 子类"的对象，`isinstance(out, dict)` 归一化会
静默迭代它的 key 名——产出垃圾 id。所以**先用 tokenize=False 拿纯字符串
（跨版本永远返回 str），再手动 `tokenizer(text, add_special_tokens=False)`
编码**——与 veRL 自家 vLLM rollout 同款模式。`enable_thinking` 是
Qwen3 家族扩展，装不上就 try/except 降级。

## 3.3 reward_fn.py：veRL 奖励侧的字段流转

veRL 0.10 调用约定（keyword-only）：

```python
def search_r1_reward(data_source=None, solution_str=None,
                     ground_truth=None, extra_info=None, **kwargs) -> dict:
    # 返回 {"score": float, **per-metric} → 额外键进 reward_extra_info
```

字段来源（面试常考"reward 函数怎么拿到数据"）：

| 字段 | 来源 |
|---|---|
| `solution_str` | veRL 解码 response_ids 的完整文本 |
| `ground_truth` | parquet 的 `reward_model.ground_truth` 列（prepare_data.py 写入）|
| `data_source` | parquet 的 `data_source` 列 |
| `extra_info` | dataset 的 `extra_info` 列 + **agent loop 的 `extra_fields`**（合并后经 `non_tensor_batch["tool_extra_fields"]` 传入）|
| `enable_prm_lite` 等 | config 的 `reward.custom_reward_function.reward_kwargs.*` |

**API 漂移的教训**（2026-08-30 修复）：veRL 0.10 要求自定义 reward 挂在
`reward.*` 键下，额外参数走 `reward_kwargs.*`——**平铺在顶层的键会被
静默忽略**，导致 PRM-Lite 开着开关却没生效（reward 没变，训练照跑，
只有看 reward_extra_info 才露馅）。probe 的"wiring assert"就是为这个
设计的（§3.5）。

`_PRMTrajectoryAdapter`：把 reward worker 拿到的扁平字段
（final_text/question/turns/search_calls）适配成 `PRMLiteScorer` 要的
鸭子类型（turns/final_text/example.question/messages）——本地版 782 行
的 PRM-Lite 零改动复用。

## 3.4 grpo_config.yaml + e6b_run.sh：每个 override 键的含义

e6b_run.sh 以 veRL 自带 `ppo_trainer.yaml` 为基础，用 ~60 个 CLI override
覆盖。按类别逐一解释（**这是本教程最实用的查表**）：

### 数据类

| 键 | 值 | 为什么 |
|---|---|---|
| `data.train_files` | parquet 路径 | veRL 标准输入 |
| `data.max_prompt_length=2048` | | 问题 prompt 很短，2048 足够 |
| `data.max_response_length=4096` | | 多轮轨迹长度上限 |
| `data.train_batch_size=32` | 每步轨迹总数 | = 8 题 × n=4（GRPO 组）|
| `data.return_raw_chat=True` | | 把 parquet 的 prompt 列（消息列表）原样给 agent loop 的 `raw_prompt` |
| `data.truncation=left` | | 超长保留**尾部**——答案在尾部，不能切 |
| `data.shuffle=True` | | |

### 奖励类（2026-08-30 API 漂移修复区）

| 键 | 值 | 为什么 |
|---|---|---|
| `reward.reward_model.enable=false` | | 关内置 RM，用自定义函数 |
| `reward.custom_reward_function.path=pkg://verl_search_r1.reward_fn` | | **`pkg://` 前缀必须**：无前缀时 veRL 把 module 当文件路径 → FileNotFoundError |
| `+reward.custom_reward_function.reward_kwargs.enable_prm_lite=$PRM` | | `+` 表示向 config 添加新键（hydra 语法）；**reward_kwargs 下的键才会传进 reward 函数**，平铺键被忽略 |

### 算法类

| 键 | 值 | 为什么 |
|---|---|---|
| `algorithm.adv_estimator=grpo` | | 无 critic |
| `algorithm.use_kl_in_reward=false` | | KL 惩罚移到/关闭（配合下一条）|

### Actor（训练引擎）类——显存账的核心

| 键 | 值 | 为什么 |
|---|---|---|
| `actor_rollout_ref.model.path` | SFT 合并模型 | 全参数训练起点 |
| `model.use_remove_padding=True` | | 序列打包去 padding，省算力 |
| `model.enable_gradient_checkpointing=true` | | 重算激活换显存 |
| `actor.optim.lr=1e-6` | | 全参数训练 LR 必须比 QLoRA（1e-5）低一个量级——更新的参数量大 50 倍 |
| `actor.ppo_mini_batch_size=32` / `ppo_micro_batch_size_per_gpu=1` | | 24GB 卡的显存约束 |
| `actor.use_dynamic_bsz=true` + `ppo_max_token_len_per_gpu=4096` | | 动态 batch：按 token 预算而非条数打包 |
| `actor.use_kl_loss=false` | | **OOM 修复**：veRL 0.10 对 ref 强制 CPUOffload 的路径在 torch 2.13 下不生效（ref 常驻 GPU 8.46GB）→ 关 KL 则 ref 不创建。Search-R1 官方 GRPO 配方本来无 KL 项 |
| `actor.fsdp_config.optimizer_offload=true` + `param_offload=true` | | **核心省显存**：Adam 状态（8.4×3=25.2GB）放 CPU，参数流式进出。代价：每步 +30-60s |
| `ref.log_prob_use_dynamic_bsz=true` | | 0.10 的键名是 `log_prob_use_dynamic_bsz`（老文档写 use_dynamic_bsz，已漂移）|

显存账（e6b_run.sh 注释里的完整推算，24GB/卡）：

```
无 offload：每卡 = 权重 8.4/N + 梯度 8.4/N + Adam 25.2/N + ref 8.4/N + 激活 1.5
  N=1 → 51.9GB ❌    N=2 → 26.7GB ❌（还没算 vLLM 的 8.4GB 权重）
开 optimizer+param offload 后训练侧只剩 梯度 8.4/N + 激活 ~1.5：
  N=1：8.4+1.5+ vLLM(0.55 池≈10.8GB) ≈ 20.7GB ✓（余 ~2GB）
  N=2：4.2+1.5+ vLLM(0.75 池≈13.2GB) ≈ 18.9GB ✓（余 ~4GB）
```

### Rollout（vLLM 引擎）类

| 键 | 值 | 为什么 |
|---|---|---|
| `rollout.n=4` | GRPO 组大小 | 32/4=8 题/步 |
| `rollout.temperature=1.0 / top_p=1.0` | | 训练探索 |
| `rollout.gpu_memory_utilization` | 1 卡 0.55 / 2 卡 0.75 | vLLM 的 KV cache 池上限；训练侧 offload 后 GPU 空出来，2 卡可给 vLLM 更多 |
| `rollout.max_num_batched_tokens` | 32768 / 65536 | 在飞 token 上限（防 KV 撑爆）|
| `rollout.max_model_len=6144` | | prompt ~2048 + 3 轮×~1024 + 工具文本，6144 够用；KV cache 减半提升并发 |
| `rollout.max_num_seqs=229` | | **Mamba 适配**：Qwen3.5 是 Transformer+Mamba 混合架构，vLLM 需要 Mamba cache block；报错原文 "max_num_seqs (256) exceeds available Mamba cache blocks (229)"。设 229 以下 |
| `rollout.checkpoint_engine.update_weights_bucket_megabytes=512` | | 默认 2048MB bucket 恰好超过 torch 2.13 CUDA IPC 上限（INT_MAX=2GB−1）→ 静默走 shm 文件路径 → rebuild_ipc 越界。512MB 走正常 IPC |
| `rollout.layered_summon=true` | | 权重同步分层 unshard（summon_full_params），避免 all-gather 全量+分片并存（峰值 12.6→~5GB）|
| `rollout.enforce_eager=false` | | 开 CUDA graph 提速（推理无动态形状）|
| `rollout.free_cache_engine=true` | | 训练前释放 vLLM KV cache |
| `rollout.agent.default_agent_loop=search_r1_agent` | | 启用自定义 loop |
| `rollout.agent.agent_loop_config_path=...agent_config.yaml` | | loop 配置（YAML **列表**格式：name + _target_ + 额外 kwargs，veRL main 0.10 约定）|
| `rollout.multi_turn.enable=true, max_assistant_turns=3` | | 多轮开关（注意：与 agent_config.yaml 里的 max_assistant_turns 是两个层级的限制，loop 自己再限一次）|

### Trainer 类

`n_gpus_per_node=$N_GPUS`（1 或 2，环境变量切降级方案）、`total_training_steps=50`、
`save_freq=25`、`logger=['console']`、`default_local_dir=checkpoint 目录`。

### 环境变量（e6b_run.sh 尾部）

| 变量 | 为什么 |
|---|---|
| `PYTHONPATH=$PROJECT_DIR` | Ray worker 不继承 shell 环境 → **必须显式 export**（reward_fn 曾被 FileNotFoundError）|
| `VLLM_ATTENTION_BACKEND=FLASHINFER` | 无 FlashAttention2 时的后端（Qwen3.5 编译不了 FA2，见 §4.4）|
| `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` | AutoDL 网络对 huggingface.co 只解析出 IPv6 无路由 → from_pretrained 卡死几十分钟；模型已缓存，强制离线 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.05` | 去碎片 + GC 回收 reserved 未分配块（update_weights 峰值缺口 ~400MB，GC 可释放 ~700MB）|
| `TQ_DATA_UPDATE_RESPONSE_TIMEOUT=120` | TQ ACK 超时从 30s 放宽（controller 启动期 thundering herd）|

## 3.5 probe_verl_api.py：训练前体检的方法论（47 项检查）

**核心思想：probe 失败的成本是 0，训练失败的成本是 GPU 小时**。
47 项检查按 7 节组织，从"必然炸"到"概率炸"排序：

1. **agent-loop API 检查**：veRL 的 AgentLoopBase 签名、AgentLoopOutput
   字段、注册表里有没有 `search_r1_agent`——**API 漂移第一道防线**
   （veRL main 每天在变，你 8 月初写的代码 8 月底大概率已经不兼容）；
2. **worker 式实例化**：用真实 tokenizer + DictConfigWrap 按
   AgentLoopWorker 的方式实例化 loop + chat template 冒烟（>50 token）——
   probe 直接跑 driver 进程，而 Ray worker 的导入环境不同（§4.4 的
   .pth 坑）；
3. **hydra 合成 + key-drift 扫描**：`compose(ppo_trainer, overrides)`——
   hydra 对不存在的键 hard-fail，**带完整 override 列表的 compose 本身就
   是漂移检查**；再用 wiring assert 确认合成结果（reward_kwargs 真的
   传进去了、rollout.name 真是 vllm）；
4. **reward 函数**：import + 签名（必须有 **kwargs）+ 假 DataProto 冒烟
   （non_tensor_batch + tool_extra_fields 流转一次）;
5. **库版本 + GPU + flashinfer**（transfer_queue 缺失会精确告诉你 pip 装什么）;
6. **数据文件 + 列名 + e6b_run.sh 陈旧参数扫描**（如残留的 --config-path）;
7. **--live-search 实弹搜索**：真的发一次 query（"Baybrook Mall"），
   输出延迟/条数/错误——**API key 有效性在花 GPU 钱之前验证**。

**0 FAIL 才启动训练**，这是项目里成本收益比最高的一条纪律。

## 3.6 E6B_DEPLOY.md：部署清单的设计思想

该文档是"部署的止损思维"范本，四个亮点：

1. **路线决策前置**：原 `repo/` veRL fork 锁死 transformers<4.48 +
   vllm<=0.6.3，而 Qwen3.5-4B 要 transformers≥4.57——fork 根本加载不了
   模型。决策：放弃 fork 走 veRL main + 自定义 AgentLoop（response_mask
   原生替代 fork 的 `<information>` state_masking）。**先把"哪条路根本
   走不通"排除，再谈优化。**
2. **显存账先于开机**（§1.1 表格）：1 卡/2 卡/4 卡的费用、步时、显存
   余量全算清楚再租机器，且明确"选 2 卡的理由"（预算内 + 真实多卡经历）。
3. **止损规则写进文档**：>12min/步停、连续崩溃停、format 连续 5 步
   <20% 停、EM 连续 10 步无提升停——训练跑起来之前就写好"什么时候
   认输"，避免沉没成本效应。
4. **诚实风险清单**（§6）：预计首次跑通概率 60-70%，probe 挡配置问题
   但 Ray/AgentLoop 运行时问题要第 1-2 步才暴露——把不确定性写进计划。

---

# 第 4 章 工程篇

## 4.1 数据工程全流程

```
ModelScope zhuangzhuang2023/nq_hotpotqa_train（固定 revision）
  → prepare_data.py：下载 parquet → 清洗（question/golden_answers）
  → datasets/train.jsonl (169615 行) / test.jsonl (51713 行)
  → select_dev：7 benchmark 各 10 条 → dev.jsonl (70 题，评估口径)
  → 蒸馏：distill_trajectories.py 抽样 NQ 200 + HotpotQA 200
  → DeepSeek Chat 生成轨迹 → fix_distill_tags.py 修复格式
  → distilled_trajectories_fixed.jsonl（397 条）
  → veRL：verl_search_r1/prepare_data.py JSONL → parquet
    （prompt 列 = [{"role":"user","content":question}]，
      reward_model 列 = {"ground_truth": answers, "style": "rule"}）
```

## 4.2 SFT 阶段操作手册

1. AutoDL 开机 4090D → `pip install transformers peft trl accelerate
   bitsandbytes datasets`
2. 下载基座（HF mirror）
3. `python sft_train.py --data distilled_trajectories_fixed.jsonl
   --epochs 3`（150 steps，~2h，¥8）
4. 看 loss：1.65 → 0.11。**但别被 loss 骗了**：SFT 模型 dev 集 EM
   1.43%、搜索 0 次（§2.11.3）——SFT 只教格式。
5. `python merge_sft.py` 合并 → `huggingface-cli upload` 上传
   `wang072266/qwen3.5-4b-search-r1-sft`

## 4.3 服务器部署全流程（AutoDL 2×4090）

1. **上传打包件**（deploy 包 = verl_search_r1/ + 数据 + eval 脚本），
   解压到 `~/autodl-tmp/search-r1`
2. **装 veRL main**：`bash verl_search_r1/setup.sh`（git clone + pip -e .
   + vLLM 按 pyproject pin 版本装）
3. **转数据**：`python verl_search_r1/prepare_data.py datasets/train.jsonl
   datasets/train.parquet`
4. **probe**：`bash verl_search_r1/e6b_run.sh --probe` → **0 FAIL 才继续**
5. **启动**：`cd ~/autodl-tmp/verl && nohup bash ~/autodl-tmp/search-r1/
   verl_search_r1/e6b_run.sh > e6b_launch.out 2>&1 &`
6. **盯盘**（前 2 步最关键）：`tail -f e6b-grpo-prmlite.log`，看
   `step:1`、`reward/em`、format rate；nvidia-smi 看显存
7. **训练完评估**：`bash verl_search_r1/cloud_eval.sh`（同一实例直接评
   全参数 checkpoint，不用下载 16GB 权重）
8. **立即关机**（AutoDL 实例持续计费）

远程服务器操作纪律（来自 HOW_TO_WRITE_TUTORIAL）：命令单行 <100 字符
（SSH 粘贴硬换行）、sed 锚点唯一 + 改完 ast.parse 验证、复杂内容
tar.gz 上传、pkill 必须锚定 `'^python -u e6b_main'`（否则把远程 shell
一起杀掉，exit 255）。

## 4.4 踩坑记录（每个坑：现象→根因→修复→教训）

> 这是本项目最值钱的一章。每个坑都真实花过钱/时间。

### 坑 1：API key 过期导致 GRPO 学到"不搜索"的捷径（¥34 教训）

- **现象**：v1 训练 15 步，search_calls 掉到 0.17（几乎不搜），correct 封顶 34%。
- **根因**：DeepSeek API key 全程过期（HTTP 401），所有搜索失败；
  GRPO 发现"不搜直接猜"平均 reward 反而高 → 强化了错误行为。
- **修复**：换有效 key 重训（v2：correct 81.2%）。工程上：DeepSeekSearchClient
  `__post_init__` 提前校验 key；probe 增加 --live-search 实弹检查。
- **教训**：**RL 学到的是你奖励的东西，不是你想要的东西**。环境噪声
  （搜索失败）本身就是训练信号的一部分，必须在训练前用 probe 排除。

### 坑 2：LLD 死亡螺旋（v1 固定 λ 失败 → v2 比例缩放成功）

- **现象**：E1-A 56 步 correct 3.1%（从 81.2% 崩）；E3-B v1 step 63 崩到 1.6%。
- **根因**：GRPO 无 KL 约束时，advantage 主导的梯度单向压低似然 → 正反馈
  螺旋（§1.4）。
- **修复**：LLDS 比例惩罚 `Σ max(0, log π_ref − log π_θ)`。v2 不再永久
  崩溃（低点 3.1% → 恢复 54.7%）。
- **教训**：正则化强度必须与被防护量同阶；「reward 上升」不等于
  「训练健康」，要同时监控 log-likelihood。

### 坑 3：大词表 log_softmax OOM

- **现象**：训练第一步 forward OOM，`Tried to allocate 5.35 GiB`。
- **根因**：Qwen3.5 词表 151936。`[batch=2, seq=8000, vocab]` 的
  logits×2B ≈ 4.85GB——全量 log_softmax 一把爆。
- **修复**：`_chunked_log_softmax_gather` 按 512 token 分块
  （峰值 5GB→300MB）+ MAX_TRAIN_SEQ_LEN=4096 截断 +
  MICRO_BATCH_MAX_TOKENS 8000→4000。
- **教训**：大词表模型（Qwen/Yi/DeepSeek 系）的 log_softmax 是显存杀手，
  算显存账时要单列这一项。

### 坑 4：左 padding 切片 bug（prompt 被重复训练）

- **现象**：训练 loss 异常低、模型退化；检查发现 completion 里混着
  prompt token。
- **根因**：左 padding 下 prompt 靠右对齐，新 token 从 `max_len`
  开始；用 `prompt_len`（原始长度）切片等于把 prompt 尾部当 completion。
- **修复**：`outputs[i, max_len:]`。
- **教训**：padding 方向决定切片锚点——左 padding 用 max_len，
  右 padding 用 prompt_len。写完切片代码先打印几个样本验证。

### 坑 5：SFT 格式 100% 无效（格式桥接）

- **现象**：SFT 模型输出 Qwen 原生 `<tool_call>` 格式，但解析器只认
  论文的 `<search>` 标签 → rollout 全 `action=None` → 全部轨迹 reward
  −0.1 → advantage 全 0 → **GRPO 没有任何学习信号**。
- **根因**：格式协议不匹配（repo/ 版历史；最终版直接用 Qwen 原生协议
  从根上规避）。
- **修复**（repo/ 版）：4 级格式桥接：`<search>` → `<tool_call>` →
  `Answer:` → 宽松匹配。最终版设计原则：**解析器接受多格式，训练
  目标单一格式**——让模型"有路可走"，RL 自己收敛到最优格式。
- **教训**：RL 训练对格式极度敏感。接入任何新模型先检查"它的默认输出
  格式"与"解析器"是否匹配。

### 坑 6：veRL main API 漂移（PRM 静默失效）

- **现象**：PRM-Lite 开关打开但 reward 没变化（reward_extra_info 里
  prm 恒 0）。
- **根因**：veRL 0.10 把自定义 reward 挂到 `reward.*` 下，额外参数走
  `reward.custom_reward_function.reward_kwargs.*`；**平铺在顶层的键被
  静默忽略**。另有 `pkg://` 前缀、`log_prob_use_dynamic_bsz` 键名等漂移。
- **修复**：全部 override 对齐 0.10 键路径；probe 增加 wiring assert。
- **教训**：依赖上游 main 分支 = 签了"API 会变"的协议。probe 的
  key-drift 检查是必须品不是可选品。

### 坑 7：torch 混合安装 → torch.compile ImportError

- **现象**：`ImportError: cannot import name 'unpack_mixed_mm'`（torch
  compile 路径）。
- **根因**：多次 pip 安装 torch 留下陈旧的
  `torch/_inductor/kernel/unpack_mixed_mm.py` 与新版本库混用。
- **修复**：删除陈旧文件。
- **教训**：升级深度学习库时，旧文件残留是常见事故源；报错信息里
  的文件路径要去看一眼文件本身。

### 坑 8：FlashAttention2 缺失 → sdpa 注入

- **现象**：Qwen3.5（Mamba 混合架构）编译不了 FA2，训练启动崩。
- **修复**：在 veRL transformer_impl.py 注入
  `attn_implementation="sdpa"`（PyTorch 原生高效注意力）。
- **教训**：混合架构模型（Mamba/新注意力）经常不在 FA2 支持列表里，
  sdpa 是稳妥兜底；性能损失在 4B 小模型上可接受。

### 坑 9：Ray worker 不继承 PYTHONPATH

- **现象**：driver 进程能 import `verl_search_r1.reward_fn`，但
  RewardLoopWorker 报 FileNotFoundError。
- **根因**：Ray worker 进程不继承 shell 的 PYTHONPATH；probe 跑在
  driver 进程测不到这个问题。
- **修复**：site-packages 下写 `.pth` 文件（对所有 Python 进程生效）。
- **教训**：分布式框架里"driver 能跑"≠"worker 能跑"。probe 要模拟
  worker 的导入环境。

### 坑 10：FSDP OOM 六轮调试（update_weights 缺口 400MB）

- **现象**：`update_weights` 阶段 CUDA OOM——申请 1.19GB、free 795MB、
  峰值 20.67GB（本进程 21.94GB），缺口仅 ~400MB。
- **定位过程**（六轮，每轮一个观测点）：参数已 CPU offload（allocated
  0.00）→ OOM 发生在 load 之后、state_dict 之间 → 排除项逐个验证
  （ref 已关、layered_summon 只对 LoRA 生效、GC threshold 0.05 只能
  回收 ~3%）。
- **修复**（v6）：`get_per_tensor_param_shard()` 取本地 FSDP 分片 +
  FULL_STATE_DICT + offload_to_cpu 流式传输，峰值 12.6GB→~5GB；
  配合 visual 权重过滤、512MB IPC bucket、layered_summon。
- **教训**：① OOM 调试 = 逐点观测 allocated/reserved 的二分定位，
  不是猜；② 框架的"官方 offload"在新 torch 版本下可能不生效
  （veRL 0.10 ref CPUOffload 在 torch 2.13 失效，实测常驻 8.46GB），
  信日志不信文档；③ 缺口 400MB 的 OOM 比缺口 10GB 的难修——大缺口
  换方案，小缺口抠细节。

### 坑 11：Mamba cache block 限制

- **现象**：vLLM 启动报
  `max_num_seqs (256) exceeds available Mamba cache blocks (229)`。
- **根因**：Qwen3.5 的 Mamba 层在 vLLM 里用固定数量 cache block，
  256 并发超上限。
- **修复**：`max_num_seqs=229`。
- **教训**：混合架构模型的并发上限由**最受限的组件**决定；报错信息
  直接把上限数字告诉你了，先读报错再调参。

### 坑 12：TransferQueue 调度瓶颈（每步 50-70 分钟）

- **现象**：每步耗时 50-70 分钟（预期 3-5 分钟的十几倍）；
  TQ_STORAGE ACK 超时日志 12 次。
- **定位方法**：统计每个 agent loop 产生的 TQ 子操作数（~400/loop），
  128 个 loop/步 ≈ 51K 操作/步；TQ 统计显示 SimpleStorageUnit
  PUT_DATA 吞吐 ~736/min——**计算得出纯调度时间 51K/736 ≈ 70 分钟**，
  与实测步时吻合。
- **关键交叉验证**：搜索后端从知乎（0.68s）换 DeepSeek（1-3s）换
  MiMo（6-14s），**步时几乎不变** → 证明瓶颈在 TQ 调度而非搜索 API。
- **修复尝试**：加大 ACK 超时（TQ_DATA_UPDATE_RESPONSE_TIMEOUT=120）
  缓解启动期 thundering herd；架构级优化方向（按性价比排序）：
  批量提交调度 / 减少子操作数 / 搜索异步化 / 绕开 TQ，详见
  INTERVIEW_QA Q21。
- **教训**：① 瓶颈定位的黄金标准是**交叉验证**——换掉嫌疑组件看
  指标动不动；② 框架自带的吞吐统计（TQ 的 req_count/avg）是免费
  的性能剖析器，先看它再上 profiler；③ 框架选型时"多轮 agent 支持
  的调度架构"应该是评估项之一；④ 调度吞吐这类数据部署前 10 分钟
  就能 benchmark，先测再跑。

### 坑 13：预生成缓存命中率 0%

- **现象**：预生成 2000 条搜索缓存，训练时命中率 = 0%。
- **根因**：缓存 key 是训练问题原文（"total number of death row
  inmates in the us?"），运行时查找用的是**模型生成的搜索 query**
  （"death row inmates statistics"）——key 空间根本不重合。
- **修复**：放弃预生成，改运行时缓存（同 step 内相同 query 自然命中）。
- **教训**：**缓存 key 必须和运行时查询 key 一致**，先小样本验证命中率
  再全量部署。这个错误 ¥0 就能在本地发现，却花在了服务器上。

### 坑 14：vLLM 0.28 把 Qwen3.5 当多模态模型

- **现象**：`OSError: Can't load image processor ... containing a
  preprocessor_config.json file`。
- **根因**：vLLM 0.28 把 `Qwen3_5ForConditionalGeneration` 路由到
  Qwen3-VL 系多模态实现，初始化无条件要图像处理器配置；合并仓库只有
  纯文本文件。日志 `model.visual.merger.norm.weight | MISSING` 说明
  transformers 的 Qwen3_5 类自带视觉塔（随机初始化、纯文本训练用不到）。
- **修复**：从上游 Qwen3.5-4B 拿 preprocessor_config.json 补齐。
- **教训**：架构名路由是"看起来无关"的失败源；报错链（encoder_budget
  → qwen3_vl.py）追到最底层才知道是路由问题。

### 坑 15：chat template tokenize=True 返回"假 dict"

- **现象**：transformers 5.10 下 `apply_chat_template(tokenize=True)`
  返回一个不是 dict 子类的 dict-like，`isinstance(out, dict)` 归一化
  静默迭代 key 名 → 生成垃圾 token id。
- **修复**：tokenize=False + 手动 encode（§3.2.4）。
- **教训**：跨版本 API 的返回类型**类型检查要严格**；"像 dict"的
  对象不一定是 dict。probe 的 chat-template 冒烟检查（len>50）就是
  为这类问题设计的。

### 坑 16：本地训练 GPU 空转等搜索

- **现象**：本地管线每步 ~16min，GPU 利用率极低。
- **根因**：同步搜索，GPU 大部分时间在等 API 返回（搜索延迟 1-14s/次
  × 每条轨迹 1-4 次）。
- **结论**：本地版瓶颈是搜索 I/O 不是显存——**换便宜卡即可**（3060
  ¥1/h 和 4090 效果一样）。
- **教训**：I/O bound 和 compute bound 的优化方向完全不同；先用
  nvidia-smi / 计时日志分清是哪种再花钱。

## 4.5 评估与结果解读

### 4.5.1 评估口径（数字的测量方法，面试必背）

| 数字 | 测量方法 |
|---|---|
| dev 集 70 题 | prepare_data.py 从 test 集按 7 benchmark 各抽 10 条（seed=42）|
| EM | `eval_local.py` greedy 解码 + `normalize_answer` 字符串精确匹配 |
| 81.2% correct_rate | **训练过程指标**：PyTRIO 训练时每步批次内 64 条轨迹的
  EM 比例（step 3 峰值），不是留出集评估 |
| SFT 基线 1.43% | eval_local.py 在 dev 70 题上对 SFT 合并模型（zero_adapter）greedy 评估 |
| format rate | 恰好一行 `Answer:` 的轨迹比例 |
| 异常率 2.5% | 10 步验证，160 条轨迹中 is_anomalous 命中 4 条 |
| TQ 51K 操作/步、736/min | TQ 内部统计（req_count/avg）× loop 数推算 + 交叉验证（换后端步时不变）|

### 4.5.2 怎么读训练曲线

- **reward 上升 + correct 上升**：健康。
- **reward 上升 + correct 下降**：危险（LLD 前兆，看 log-likelihood）。
- **search_calls → 0**：模型在学"不搜"（检查搜索 API 健康度——坑 1）。
- **format rate 崩**：协议输出退化，检查 KL/LLDS 是否失效。
- **loss 爆炸（100+）**：梯度爆炸，查 clip 和 LR。

---

# 第 5 章 附：关键命令速查

```bash
# 数据
python prepare_data.py                          # ModelScope → train/test/dev.jsonl
python verl_search_r1/prepare_data.py datasets/train.jsonl datasets/train.parquet

# 蒸馏 + SFT
python distill_trajectories.py                  # 需要 .env DEEPSEEK_API_KEY
python fix_distill_tags.py
python sft_train.py --model Qwen/Qwen3.5-4B --data distilled_trajectories_fixed.jsonl
python merge_sft.py

# 本地 GRPO 训练（不依赖 PyTRIO）
python train_grpo_local.py --max-steps 50 --questions-per-batch 8 --group-size 8 \
    --search-backend deepseek --llds-lambda 0.05 --prm-lite --lata

# 本地评估
python eval_local.py --checkpoint-dir grpo_checkpoint --base-model <模型路径>
python analyse.py                                # 画 EM/format 曲线

# 单元测试
python -m pytest test_llds.py test_reward_lite.py -q

# veRL（服务器）
bash verl_search_r1/e6b_run.sh --probe           # 0 FAIL 才继续
N_GPUS=2 bash verl_search_r1/e6b_run.sh          # 2 卡训练
bash verl_search_r1/cloud_eval.sh                # 云端评估全部 checkpoint
```

---

## 结语：这个项目教会我的三件事

1. **RL 训练的是奖励定义的行为，不是你的意图**——奖励设计、环境健康度
   （API key）、格式协议，任何一个环节的偏差都会被 GRPO 放大成行为。
2. **probe 和止损规则是省钱的最高杠杆**——每花 1 分钟写检查，省的是
   数十小时的 GPU 时间。
3. **瓶颈要用交叉验证证明**——"搜索太慢"的直觉被"换后端步时不变"的
   实验推翻，真正的瓶颈是框架调度。直觉给方向，实验给结论。
