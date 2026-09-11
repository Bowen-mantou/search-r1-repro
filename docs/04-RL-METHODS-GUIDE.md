# LLM 强化学习方法全景指南

> 覆盖：PPO / GRPO / DPO 三大家族 + RLOO / REINFORCE++ / ReMax / PRIME / DAPO /
> Dr.GRPO / VAPO / GSPO / EXO / SPPO / SPIN / Best-of-N 等，含数学推导、代码实现、
> 优缺点、出处与演变脉络；另附《GPU 显存计算专题》教会你算显存。
> 配套项目代码：`C:\new\intern\plan\projects\search-r1\`（Search-R1 复现项目，
> GRPO 全流程落地）。

---

## 第 0 章 导读：RL 家族地图

### 0.1 两大家族，一条主线

LLM 时代的强化学习方法本质上在回答同一个问题：

> **在只有"好/坏"反馈（甚至只有最终对错）的情况下，怎么让语言模型的输出
> 分布朝好的方向移动？**

所有方法可以按"怎么利用反馈"分成两大家族：

```
                        ┌── 在线策略优化族（on-policy RL）──────────────┐
                        │  现场采样轨迹 → 算优势 → 梯度更新              │
   RLHF 主线 ──────────┤                                              │
  (2017 起)            │  PPO(2017) → GRPO(2024) → RLOO/REINFORCE++    │
                        │            → DAPO/Dr.GRPO/VAPO(2025)          │
                        │                                              │
                        └── 直接偏好优化族（DPO 系，offline）───────────┘
                           偏好数据直接算 loss，不采样不训 critic
                           DPO(2023) → IPO/KTO/ORPO/SimPO/SPPO
```

两大家族的本质区别，一句话记：

- **在线族**：模型**自己下场**（rollout），用自己生成的轨迹学——贵（要采样要 GPU），
  但信号永远新鲜，能学多轮交互（工具调用、搜索）；
- **DPO 族**：用**别人录好**的偏好对学——便宜（纯监督式 loss），
  但数据分布会过时，且表达不了"环境交互"。

### 0.2 演变主线（叙事版）

1. **2017 PPO**：Schulman 等人把 TRPO 的 KL 约束换成 clip，RL 第一次"好训"了。
2. **2017-2022 RLHF**：Christiano 用人类偏好训 RM → InstructGPT 把
   SFT→RM→PPO 三步流水线跑通 → ChatGPT 证明了这条路能做产品。
   代价：PPO 需要 4 个模型（actor/critic/ref/reward），贵、脆、难调。
3. **2023 减负运动**：RLOO 证明"没有 critic 也能训"（REINFORCE + 留一基线）；
   DPO 釜底抽薪——把 RM 和采样全干掉，偏好对直接算 loss。
4. **2024 GRPO 时代**：DeepSeekMath 的 GRPO 把"组内比较"标准化
   （同一题采样 N 条互相比），DeepSeek-R1 用 GRPO + 规则奖励做出了推理模型，
   引爆推理 RL。
5. **2025 精修时代**：GRPO 的长度偏差、方差问题被逐一开刀
   （Dr.GRPO 去偏、DAPO 四大补丁、VAPO 把 value 请回来）；PRIME 把过程奖励
   做成了"只靠结果标签"的在线更新；GSPO/EXO 等继续推低成本路线。

### 0.3 每个方法的一句话卡片（面试速记）

| 方法 | 一句话本质 |
|---|---|
| PPO | clip 住的策略梯度 + critic 给优势 |
| GRPO | 同一题采 N 条互相比，干掉 critic |
| RLOO | 留一均值当基线，单条也能算优势 |
| REINFORCE++ | REINFORCE 缝上 PPO 的 clip，工程最简 |
| ReMax | 用 batch 内最大 reward 当基线 |
| DPO | 偏好对直接算 loss，RM 和采样全省 |
| IPO | DPO 的过拟合修复版（平方损失） |
| KTO | 只要好/坏标签，不要偏好对 |
| ORPO | SFT 和偏好对齐一次做完，不要 ref |
| SimPO | 长度归一 + 无 ref 的 DPO |
| SPPO | 自己和自己博弈到 Nash |
| PRIME | 过程奖励只用结果标签在线学 |
| DAPO | GRPO 的四大工程补丁 |
| Dr.GRPO | 把 GRPO 的长度偏差从数学上拿掉 |
| VAPO | 价值模型路线回归，长短自适应 GAE |
| GSPO | 组序列 + 序列级 KL，无 ref |
| Best-of-N | 采样 N 条挑最好的（不是训练） |

### 0.4 文档地图

- **第 1 章**：数学基础（策略梯度定理 / REINFORCE / GAE）——所有方法的地基；
- **第 2 章**：PPO 与 RLHF 流水线；
- **第 3 章**：GRPO 完整推导 + 本项目落地 + 2025 变体（Dr.GRPO/DAPO/VAPO）；
- **第 4 章**：DPO 家族（含推导与五个变体）；
- **第 5 章**：其他方法与边界（RLOO/REINFORCE++/ReMax/PRIME/EXO/GSPO/
  Best-of-N/RLAIF）；
- **第 6 章**：全方法对比速查表 + 面试高频问答；
- **附录**：《GPU 显存计算专题》（独立文档 01-GPU-MEMORY-GUIDE.md）。

### 0.5 使用建议

1. **第一遍**：只读每章的"一句话本质 + 核心公式 + 直觉"段，建立地图；
2. **第二遍**：手推一遍第 1 章的策略梯度定理和第 3 章的 GRPO 公式（面试必考）；
3. **第三遍**：对着第 6 章速查表自问自答，每题 2 分钟内讲完；
4. 显存专题配合你项目的三个真实案例（2×4090 训 4B、3060 QLoRA、7B/70B 估算）
   反复算到能默写。

---
# LLM 强化学习方法全景指南 · Part 1：数学基础与 PPO

> 系列定位：为准备算法岗面试、学过概率论但不会运用、做过 GRPO 项目但想系统补理论的同学而写。
> 本文是 Part 1，覆盖策略梯度定理 → REINFORCE → baseline/advantage → GAE → TRPO → PPO → PPO 在 LLM RLHF 中的形态 → InstructGPT，最后给时间线总表和面试速查。
> 公式全部用行内 unicode 或代码块书写，不出现 LaTeX。每个推导的每一步都标注「为什么」。

---

## 0. 读前说明：符号、设定、三条主线

**符号约定（贯穿全文）**

| 符号 | 含义 |
|---|---|
| `s_t, a_t, r_t` | t 时刻的状态、动作、奖励 |
| `πθ(a|s)` | 策略：参数为 θ 的网络，输出在状态 s 下选动作 a 的概率 |
| `τ = (s_0, a_0, r_1, s_1, a_1, ..., s_T)` | 一条轨迹（episode） |
| `P(τ|θ)` | 策略 θ 下这条轨迹出现的概率 |
| `R(τ) = Σ_{t=0}^{T-1} r_{t+1}` | 轨迹总回报（先不折扣，最后统一加回 γ） |
| `G_t = Σ_{t'≥t} γ^{t'-t} r_{t'+1}` | 从 t 起的折扣回报（reward-to-go） |
| `V(s), Â_t` | 状态价值函数、优势函数 |
| `∇θ` | 对策略参数 θ 求梯度 |
| `E[·]` | 期望 |

**三条主线**（面试时用这三句话串起整个 Part 1）：

1. **策略梯度定理**告诉我们"往哪走"：`∇J(θ) = E[Σ_t ∇log πθ(a_t|s_t) · Â_t]`——把每个动作的对数概率往优势为正的方向推。
2. **TRPO/PPO**告诉我们"走多远"：纯策略梯度没有步长概念，一次更新太大策略就崩（性能悬崖）；TRPO 用 KL 硬约束管步长，PPO 把这个约束软化成 clip 项。
3. **RLHF 就是把 PPO 搬到 token 级**：动作变成"下一个 token"，奖励稀疏（只在回答末尾给），KL 惩罚防止模型跑出 SFT 基线的可信范围。

---

## 1. 策略梯度定理：全文的地基

**一句话本质**：把"最大化期望总回报"这个不能直接求导的目标，改写成一个可以用采样估计的梯度公式。

**出处**：Sutton et al. 1999《Policy Gradient Methods for Reinforcement Learning with Function Approximation》；Williams 1992 的 REINFORCE 是其单样本特例。

**推导前的直觉**：监督学习里 loss 对参数可微，直接反向传播。强化学习里"奖励"来自环境，我们不知道奖励对动作的导数，甚至不知道环境转移概率。策略梯度绕开这一切——**不要求奖励可微，也不要求知道环境模型**，只要求能采样轨迹、能算每个动作的 log 概率梯度（`∇log πθ(a|s)`，由神经网络自动微分给出）。

### 1.1 目标函数

```
J(θ) = E_{τ~πθ}[R(τ)] = Σ_τ P(τ|θ) · R(τ)          (1)
```

- **为什么**：期望的定义。轨迹是随机变量，其分布由策略 θ 决定；把"概率 × 回报"对所有轨迹求和，就是期望总回报。这里用 Σ 是离散写法，连续情形换成积分完全同理。

### 1.2 对 θ 求导：梯度进入概率

```
∇J(θ) = Σ_τ ∇P(τ|θ) · R(τ)                          (2)
```

- **为什么**：`R(τ)` 是环境给出的数字，不含 θ，∇ 只作用在 `P(τ|θ)` 上。麻烦在于 `∇P(τ|θ)` 没法直接估计——我们采样得到的是轨迹本身，不是概率的梯度。

### 1.3 似然比技巧（likelihood ratio trick）：整个推导唯一的"魔法"

```
∇P(τ|θ) = P(τ|θ) · ∇log P(τ|θ)                     (3)
```

- **为什么**：链式法则。`d(log P)/dθ = (1/P)·dP/dθ`，两边乘 P 移项即得。一步纯代数，作用巨大：等式右边是"**可采样量 P(τ|θ) 的期望形式 × 可计算的 ∇log P(τ|θ)**"。代入 (2)：

```
∇J(θ) = Σ_τ P(τ|θ) · ∇log P(τ|θ) · R(τ)
      = E_{τ~πθ}[∇log P(τ|θ) · R(τ)]                (4)
```

- **为什么**：把 Σ P(·)·(·) 重新读成期望 E[(·)]。现在梯度是"对 log 概率的梯度 × 回报"的期望，可以用采样轨迹做蒙特卡洛估计。`log` 出现在这里不是巧合，下一小节看到它的结构作用。

### 1.4 轨迹概率分解：log 把乘积变成求和

```
P(τ|θ) = ρ(s_0) · Π_{t=0}^{T-1} [ πθ(a_t|s_t) · P(s_{t+1}|s_t, a_t) ]    (5)
```

- **为什么**：马尔可夫性。轨迹联合概率 = 初始状态分布 ρ × 每步"策略选动作" × 每步"环境转移"，每项只依赖当前时刻。取 log：

```
log P(τ|θ) = log ρ(s_0) + Σ_t [ log πθ(a_t|s_t) + log P(s_{t+1}|s_t, a_t) ]   (6)
```

- **为什么**：log 把连乘变成连加。这就是策略梯度用 log 概率的深层原因——**求和形式让 ∇ 只落在依赖 θ 的策略项上**。

### 1.5 关键证明②：环境项梯度为零

```
∇log P(τ|θ) = Σ_t ∇log πθ(a_t|s_t)                 (7)
```

- **为什么**：`ρ(s_0)` 和 `P(s_{t+1}|s_t,a_t)` 是环境的，不依赖 θ，对 θ 求导恒为零，直接消失。**环境动力学（哪怕未知）不需要建模，这正是 model-free 的来源。** 代入 (4)：

```
∇J(θ) = E[ Σ_t ∇log πθ(a_t|s_t) · R(τ) ]           (8)
```

### 1.6 因果性裁剪：过去的奖励与当前动作无关

先给一个反复要用到的恒等式：

```
E_{a~πθ}[ ∇log πθ(a|s) ] = Σ_a πθ(a|s) · ∇log πθ(a|s) = Σ_a ∇πθ(a|s) = ∇(Σ_a πθ(a|s)) = ∇1 = 0   (9)
```

- **为什么**：`∇log π = ∇π/π`（链式法则），乘上前面的 π 消掉；概率求和恒为 1，常数的梯度为 0。**"score function 期望为零"——凡是不依赖动作的量乘上 ∇log π，期望都是 0。**

现在看 (8) 里 `R(τ) = Σ_{t'} r_{t'+1}` 中 t' < t 的项：奖励 r_{t'} 只依赖 t' 之前的历史，与 a_t 无关，期望可因式分解，内层正是 (9) 的零。所以：

```
∇J(θ) = E[ Σ_t ∇log πθ(a_t|s_t) · G_t ]，  G_t = Σ_{t'≥t} γ^{t'-t} r_{t'+1}    (10)
```

- **为什么**：动作只能影响"之后"的奖励，拿"之前"的奖励当权重是无用噪声。把 (8) 的全轨迹回报换成 reward-to-go，**方差更小、期望不变**。

### 1.7 关键证明③：baseline 梯度为零

```
E[ ∇log πθ(a_t|s_t) · b(s_t) ] = 0（b 是任意只依赖状态的函数）                  (11)
```

逐步展开：

```
E[ ∇log πθ(a_t|s_t) · b(s_t) ]
= Σ_{s_t} μ(s_t) Σ_{a_t} πθ(a_t|s_t) · ∇log πθ(a_t|s_t) · b(s_t)   # 全期望展开（μ 为访问分布）
= Σ_{s_t} μ(s_t) · b(s_t) · [ Σ_{a_t} πθ · ∇log πθ ]                # b(s_t) 不依赖 a_t，提出内层
= Σ_{s_t} μ(s_t) · b(s_t) · 0                                       # 内层就是 (9) 的零
= 0
```

- **为什么（每步）**：第一步是全概率公式展开期望；第二步把与 a_t 无关的因子提到内层求和外面；第三步套用恒等式 (9)。**结论：在梯度估计里减任何"只看状态、不看动作"的函数，期望不变。** 这就是后文 baseline 的全部数学根据。

### 1.8 策略梯度定理（最终形态）

```
∇J(θ) = E[ Σ_t ∇log πθ(a_t|s_t) · Â_t ]
```

其中 `Â_t` 是任意"关于动作的、期望非零"的加权项：可以是 `G_t`（REINFORCE）、`G_t − V(s_t)`（advantage）、`G_t − b(s_t)`（baseline）、或 GAE 估计（第 4 节）。

**直觉总结**：策略梯度 = 对每个"决策点"上的 log 概率求梯度，加权方向由 Â_t 决定——Â_t > 0 的动作（比平均好）概率被抬高，Â_t < 0 的被压低。权重不参与反向传播，只做方向缩放。**这就是 RLHF 里"好的生成 token 概率上升、坏的下降"这句话的数学本体。**

---

## 2. REINFORCE：最朴素的策略梯度

**一句话本质**：用整条轨迹的总回报当权重，做一次蒙特卡洛采样的策略梯度。

**出处**：Williams 1992《Simple statistical gradient-following algorithms for connectionist reinforcement learning》。名字是 "REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility" 的缩写。

**核心公式**

```
θ ← θ + α · Σ_t ∇log πθ(a_t|s_t) · G_t
```

**直觉**：跑完整条轨迹 → 拿到每个时刻的折扣回报 G_t → 对每个动作，把它的 log 概率沿 G_t 方向做一步梯度上升。G_t 大 → 这条轨迹做得好 → 里面所有动作都加分；G_t 小 → 都减分。

**算法步骤**

1. 用当前策略 πθ 采样一条完整轨迹 τ；
2. 对每个时刻 t 计算折扣回报 `G_t = Σ_{t'≥t} γ^{t'-t} r_{t'+1}`；
3. 计算梯度 `g = Σ_t ∇log πθ(a_t|s_t) · G_t`；
4. 参数更新 `θ ← θ + α·g`（可选：减 baseline，见第 3 节）。

**伪代码**

```python
def reinforce(env, policy, optimizer, num_episodes, gamma=0.99):
    for ep in range(num_episodes):
        tau = []                                   # (s, a, r) 列表
        s = env.reset()
        done = False
        while not done:
            a, logp = policy.act(s)                # 采样动作 + 记录 log πθ(a|s)
            s_next, r, done, _ = env.step(a)
            tau.append((s, a, r, logp))
            s = s_next
        G, loss = 0.0, 0.0
        for s, a, r, logp in reversed(tau):        # 倒序算 reward-to-go
            G = r + gamma * G
            loss += -logp * G                      # 负号：梯度上升 = loss 下降
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**方差大的三个原因**（面试必问）：

1. **G_t 本身方差大**：G_t 是 T 个随机奖励之和，随机性逐项累积，轨迹越长波动越大；奖励本身的量级（比如 RLHF 里 0/1 奖励 vs 1000 分奖励）直接放大梯度波动。
2. **单样本估计**：一条轨迹只给一个样本点去估计期望，蒙特卡洛误差 ∝ 1/√N，而 N=1。
3. **credit assignment 粗糙**：一条好轨迹里也有坏动作、坏轨迹里也有好动作，G_t 给所有动作贴同一个标签，信号被互相抵消。

**优缺点**：优点是几行代码、无偏、无需 value 网络；缺点是方差大到几乎不能训练长轨迹任务，且必须"跑完一条再更新"，样本效率极低。**它的意义是理论基石——所有后续方法都是在"保持期望不变"的前提下给它降方差。**

---

## 3. baseline 与 advantage：免费的降方差

**一句话本质**：从权重里减掉一个"只与状态有关"的基准，期望不变、方差大降。

**出处**：baseline 思想见 Williams 1992 与 Sutton & Barto 教材；advantage 估计系统化于 GAE 论文（arXiv 1506.02438）。

### 3.1 为什么减 baseline 期望不变

第 1.7 节已证：`E[∇log πθ(a_t|s_t)·b(s_t)] = 0`。所以：

```
E[ Σ_t ∇log πθ(a_t|s_t) · (G_t − b(s_t)) ] = E[ Σ_t ∇log πθ(a_t|s_t) · G_t ] − 0
```

**直觉**：baseline 是"常数偏移"，∇log π 的期望是零（恒等式 9），偏移乘零还是零。

### 3.2 为什么能降方差

粗略方差分析：估计项 `∇log π·(G−b)` 的方差关于 b 是凸的，最优 `b* = E[(∇log π)² G] / E[(∇log π)²]`（对 b 求导令方差导数为零即得，b* 是 G 的加权平均）。实践中取 `b(s_t) = V(s_t) = E[G_t|s_t]` 就已消掉方差的最大来源——**回报的"整体平移"**。

- **直觉**：不同初始状态的回报均值天差地别（下棋先手 vs 后手、RLHF 里简单题 vs 难题），这些差异与动作好坏无关，纯属噪声。V(s) 恰好吸收"这个状态平均能拿多少"，剩下的 `G_t − V(s_t)` 只含"这条轨迹比该状态的平均水平好多少"——这正是我们想学的信号。类比：全班平均分 60，小明考了 70，"70"和"高出平均 10 分"包含同样信息，但后者的方差小得多。

### 3.3 Â_t 的三种估计

| 估计方式 | 公式 | 偏差 | 方差 | 依赖 |
|---|---|---|---|---|
| 蒙特卡洛（MC） | `Â_t = G_t − V(s_t)` | 无偏 | 高 | 需跑完轨迹 |
| TD(0)（一步 bootstrap） | `Â_t = r_{t+1} + γV(s_{t+1}) − V(s_t)` | 有偏（V 不准） | 低 | 只需一步 |
| n-step | `Â_t = Σ_{k=0}^{n-1} γ^k r_{t+1+k} + γ^n V(s_{t+n}) − V(s_t)` | 随 n 增大而减小 | 随 n 增大而增大 | n 步 |

- **直觉**：MC 用真实奖励全量算，准但波动大；TD 用 V 网络"猜未来"，稳但受 V 误差污染（bootstrapping 偏差）；n-step 是两者的插值，n 就是旋钮。**MC 相当于 n=∞，TD(0) 相当于 n=1。**

**一句话本质 + 优缺点**：advantage = "做了比平均好多少"。优点：期望不变、方差显著降低、概念清晰（面试高频名词）；缺点：需要额外训练 V(s)（多一个网络、多一份显存），且 V 不准时 advantage 有偏。

---

## 4. GAE：用 λ 在偏差和方差之间调旋钮

**一句话本质**：把所有 n-step 优势估计按指数权重混合成一个，只用一个参数 λ 调节"信多少真实奖励、信多少 V 网络"。

**出处**：Schulman et al. 2015《High-Dimensional Continuous Control Using Generalized Advantage Estimation》，arXiv 1506.02438。

**核心定义**

```
δ_t   = r_{t+1} + γV(s_{t+1}) − V(s_t)                      # TD 误差（一步优势）
Â_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}                # 指数加权和
递推式：Â_t = δ_t + γλ · Â_{t+1}                             # 一次倒序扫描 O(T) 算出全部
```

**为什么 GAE 是"合法的优势估计"**（telescoping 证明，纯代数）：k-step 估计可以写成 TD 误差的前 k 项和——

```
Â_t^(k) = Σ_{i=0}^{k-1} γ^i δ_{t+i}
        = (r_{t+1} + γV(s_{t+1}) − V(s_t))
          + γ(r_{t+2} + γV(s_{t+2}) − V(s_{t+1}))
          + γ²(r_{t+3} + γV(s_{t+3}) − V(s_{t+2}))
          + ...
        = Σ_{i=0}^{k-1} γ^i r_{t+1+i} + γ^k V(s_{t+k}) − V(s_t)      # 中间 ±γV 项全部抵消
```

最后一行就是"k 步回报减 baseline"，说明每一层都是合法优势。GAE 再对所有 k 做加权平均：

```
Â_t^GAE = (1−λ) · [ Â_t^(1) + λ·Â_t^(2) + λ²·Â_t^(3) + ... ]          # 权重 (1−λ)λ^{k-1}，和为 1
```

把每层 Â_t^(k) 按 δ 展开、按 δ 的脚标合并同类项，即得 `Â_t = Σ (γλ)^l δ_{t+l}`。

**λ 的含义**（面试高频）：

- `λ=0` → `Â_t = δ_t`，TD(0)：只用一步真实奖励，其余全靠 V 兜底 → 低方差、高偏差；
- `λ=1` → `Â_t = Σ γ^l δ_{t+l} = G_t − V(s_t)`，蒙特卡洛：全靠真实奖励 → 无偏、高方差；
- 中间的 λ 是**指数衰减窗口**：权重 `(γλ)^l` 每往未来推一步乘以 γλ，λ 控制"往前看多远"。
- 一个关键洞察：**γ 控制回报本身的折扣（任务定义），λ 控制估计器的折扣（算法选择），两者解耦**——这正是 GAE 比传统 TD(λ) 高明的地方。

**直觉**：V(s) 是"快但可能错"的猜测，真实奖励是"慢但保真"的事实。GAE 让远未来的贡献被 λ 逐层打折，最终以 V 的预测收尾——既用了最近的可靠事实，又不至于等整条轨迹。倒序递推式实现极简：一个 for 循环，三行代码。

**伪代码**（一次倒序扫描）

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advs = [0.0] * len(rewards)
    last_adv = 0.0
    next_val = 0.0                      # 终止后 V=0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_val - values[t]
        last_adv = delta + gamma * lam * last_adv
        advs[t] = last_adv
        next_val = values[t]
    return advs                         # returns = advs + values
```

**优缺点**：优点——一个超参换一个 bias-variance 旋钮、实现 O(T)、是 TRPO/PPO 的标准配件；缺点——λ 仍需调（常用 0.95），且 V(s) 的误差会通过 bootstrap 扩散到所有时刻（只是被 λ 抑制）。

---

## 5. TRPO：给更新套上信任域

**一句话本质**：在"新策略与旧策略的 KL 距离 ≤ δ"的约束下最大化代理目标，保证每次更新都不出"可信范围"，用二阶方法（自然梯度）精确解这个约束问题。

**出处**：Schulman et al. 2015《Trust Region Policy Optimization》，arXiv 1502.05477。

**为什么需要它**：策略梯度只回答"往哪走"（一阶方向），没回答"走多远"（步长）。步长太大，新策略与采样策略分布差异过大，重要性采样权重 `πθ/πθ_old` 爆炸，一次大更新直接毁掉策略——TRPO 论文开篇的"性能悬崖"现象。普通梯度下降对学习率极敏感且无理论保障，TRPO 用理论保证：**每次更新都单调不降**（论文的核心定理，借助 KL 界 + 重要性采样界）。

**约束优化形式**

```
max_θ  E_{s~ρ_old, a~π_old}[ r(θ)·Â(s,a) ]              # 代理目标，r(θ) = πθ(a|s)/πθ_old(a|s)
s.t.   E_s[ KL( πθ_old(·|s) ‖ πθ(·|s) ) ] ≤ δ           # 信任域约束
```

- **为什么用 ratio r(θ)**：重要性采样。数据是旧策略采的，评估新策略要用 `E_{a~πθ}[f] = E_{a~π_old}[ (πθ/πθ_old)·f ]` 修正；θ=θ_old 时 r=1，且 `∇r|θ_old = r·∇log π = ∇log π`，所以**在起点处代理目标的梯度与真实策略梯度完全一致**——它只是管住了"走远之后"的行为。

**为什么难算（四层困难）**：

1. **带约束的优化**不能直接梯度下降，要先解约束问题；
2. 对约束做泰勒展开：`KL(πθ_old‖πθ_old)=0` 是全局极小，所以**约束的一阶项恒为零**（KL 在两个分布相等处梯度为 0），必须展开到二阶：

```
KL ≈ (1/2)·(θ−θ_old)^T F (θ−θ_old)，  F = E[ ∇log πθ · ∇log πθ^T ] = Fisher 信息矩阵
```

（**为什么 F 是 Fisher**：score function 的外积期望正是 Fisher 信息矩阵的定义，且由恒等式 (9) 知 score 均值为零，F 也是其协方差。）目标一阶展开 `L ≈ g^T(θ−θ_old)`，`g = E[∇log π·Â]`。解这个二次约束优化得：

```
θ − θ_old = α · F⁻¹ g     ← 这就是"自然梯度"
```

3. **F⁻¹ 算不动**：F 是 |θ|×|θ| 矩阵（7B 参数模型就是 7B×7B，显存天文数字）。TRPO 用**共轭梯度（CG）**迭代解 `F·x = g`，每步只需要 Fisher-vector 乘积 `Fv = E[∇log π · (∇log π^T v)]`（两个向量内积，可自动微分），**从不显式构造 F**；
4. 解出方向后还要**回溯线搜索**：步长 α 从大往小试，直到 KL 约束真正满足且目标不降。

**自然梯度一句话**：普通梯度是"参数空间里走 1 单位"，自然梯度是"分布空间（KL 几何）里走 1 单位"——`F⁻¹` 把陡峭方向压短、平缓方向拉长，使更新在"分布变化"的意义上等距。

**伪代码（流程）**

```
1. 用 π_θold 采样一批轨迹，用 GAE 算 Â_t
2. 算梯度 g = E[∇θ log πθ(a|s) · Â]            # 在 θ_old 处
3. 共轭梯度迭代解 F·x = g（只算 Fv，不建 F）     # 得到自然梯度方向 x
4. 回溯线搜索：α ← 1, 0.5, 0.25, ...，θ = θ_old + α·x
   直到 KL(π_old‖π) ≤ δ 且代理目标不下降，否则拒绝更新
```

**优缺点**：优点——理论保证单调改进、几乎不用调学习率（线搜索自动定步长）、样本效率高；缺点——实现复杂（CG + 线搜索 + FVP 三层套娃）、二阶计算昂贵、与 dropout/参数共享等技巧有兼容性麻烦、**很难规模化到几十亿参数模型**。所以工程界几乎全员转向了它的"一阶平替"——PPO。

---

## 6. PPO：一阶优化逼近信任域

**一句话本质**：把 TRPO 的 KL 硬约束"软化"进目标函数——用 clip 把重要性采样比钳在 `[1−ε, 1+ε]`，让代理目标对"走太远"的更新梯度归零，一阶梯度下降即可近似信任域效果。

**出处**：Schulman et al. 2017《Proximal Policy Optimization Algorithms》，arXiv 1707.06347。

### 6.1 为什么用 ratio（重要性采样）

与 TRPO 相同：数据由旧策略 π_old 采样，评估新策略需要权重 `r_t(θ) = πθ(a_t|s_t) / πθ_old(a_t|s_t)`。不 clip 的朴素目标：

```
L_CPI(θ) = E_t[ r_t(θ) · Â_t ]
```

**为什么必须约束 ratio**：L_CPI 对 r 是线性的，A>0 时梯度会无限把 r 推离 1（把动作概率推上天），KL 爆炸 → 策略崩坏。TRPO 用硬约束挡，PPO 用 clip 挡——只改一行。

### 6.2 为什么 clip、clip 后的梯度行为（分区间表）

```
L_CLIP(θ) = E_t[ min( r_t(θ)·Â_t ,  clip(r_t(θ), 1−ε, 1+ε)·Â_t ) ]     (ε = 0.2)
```

**min 的含义**：L_CLIP ≤ L_CPI，永远取保守的下界——clip 只会在"ratio 越界"时起作用，把目标截成常数。

**四个区间的梯度行为**（设 ε=0.2，A 的正负决定哪一侧 clip 生效）：

| 情形 | r 的位置 | L_CLIP 取值 | 梯度行为 | 直觉 |
|---|---|---|---|---|
| A > 0（好动作，想加概率） | r < 1−ε | `r·A` | 正常，继续推高概率 | 概率还不够高，放心加 |
| A > 0 | 1−ε ≤ r ≤ 1+ε | `r·A` | 正常 | 在安全区 |
| A > 0 | **r > 1+ε** | `(1+ε)·A`（常数） | **梯度 = 0** | 加够了，别再推了 |
| A < 0（坏动作，想减概率） | r > 1+ε | `r·A` | 正常，把概率拉回 1 | 概率不该这么高，往回拽 |
| A < 0 | 1−ε ≤ r ≤ 1+ε | `r·A` | 正常 | 在安全区 |
| A < 0 | **r < 1−ε** | `(1−ε)·A`（常数） | **梯度 = 0** | 罚够了，别再压了 |

- **为什么 A>0 时 clip 上限、A<0 时 clip 下限**：A>0 时更新把概率往上推，危险方向是 r 过大 → clip 在 `1+ε` 封顶；A<0 时更新把概率往下压，危险方向是 r 过小 → clip 在 `1−ε` 封底。**clip 永远只剪断"继续走远"的方向，绝不阻挡"往回走"的方向**（比如 A>0 而 r<1−ε 时，clip 不生效，因为往回走是安全的）。这个不对称性是理解 PPO 的关键。
- **为什么梯度为零是特性不是 bug**：clip 截断后目标是常数，梯度归零意味着"这一轮到此为止"。旧数据采样自 π_old，ratio 离 1 越远，importance weight 越不可信——clip 恰恰在数据开始不可信的地方停手。TRPO 需要二次规划 + 线搜索才能做到的"够远即止"，PPO 用一个 `min` + 一个 `clamp` 做到。

### 6.3 完整损失

```
L_t(θ) = E_t[ L_CLIP_t(θ) − c1·L_VF_t(θ) + c2·S[πθ](s_t) ]
L_VF_t = ( V_θ(s_t) − V_t^target )²        # value 网络回归，V_t^target = Â_t^GAE + V_old(s_t)
S[πθ]  = 策略熵                               # 熵奖励，防止过早收敛到确定性策略
论文超参：ε=0.2，c1=0.5，c2=0.01，K=3~4 轮更新，γ=0.99，λ=0.95
```

- **为什么减 c1·L_VF**：critic 要贴近 GAE 给出的 target 回报，advantage 才准；**为什么加 c2·S**：鼓励探索，防止策略变"死板"。

### 6.4 PyTorch 伪代码（可运行的简化版，约 30 行）

```python
# PPO（经典单智能体版；接口为示意，可逐行落地）
import torch

gamma, lam, eps = 0.99, 0.95, 0.2          # 折扣、GAE-λ、clip 宽度
K_epochs, c1, c2 = 4, 0.5, 0.01            # 更新轮数、value 系数、熵系数

# 1) 采样：当前策略 π_θold 与环境交互 T 步
states, actions, rewards, old_logp, dones = rollout(env, policy, T)

# 2) GAE：一次倒序扫描得优势 + 回报
vals = critic(states)
advs = torch.zeros(T); last_adv = 0.0
for t in reversed(range(T)):
    next_val = vals[t+1] * (1 - dones[t]) if t < T-1 else 0.0
    delta = rewards[t] + gamma * next_val - vals[t]      # TD 误差
    last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
    advs[t] = last_adv
returns = advs + vals
advs = (advs - advs.mean()) / (advs.std() + 1e-8)        # 优势归一化（实践标配）

# 3) 多轮小批量更新
for _ in range(K_epochs):
    for mb in minibatch(states, actions, old_logp, advs, returns):
        logp = policy.log_prob(mb.states, mb.actions)
        ratio = (logp - mb.old_logp).exp()                # 重要性采样比 r(θ)
        surr1 = ratio * mb.advs
        surr2 = ratio.clamp(1 - eps, 1 + eps) * mb.advs   # clip 封顶/封底
        loss_clip = -torch.min(surr1, surr2).mean()       # 负号：做梯度上升
        loss_vf = ((critic(mb.states) - mb.returns) ** 2).mean()
        loss = loss_clip + c1 * loss_vf - c2 * policy.entropy(mb.states).mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)   # 梯度裁剪，防爆
        optimizer.step()
```

**优缺点**：优点——一阶优化、实现几十行、与任意网络结构/RNN/LSTM 兼容、超参鲁棒（论文展示一组超参通吃多任务）、样本效率远好于 REINFORCE；缺点——clip 是启发式（理论保障弱于 TRPO）、ε/K/c1/c2 仍需调、对奖励量纲敏感（故实践中优势常归一化）、同一条数据被重复利用 K 轮会放大偏置（off-policy 程度随 K 增大）。

---

## 7. PPO 在 LLM RLHF 中的形态

**一句话本质**：把"状态"换成"提示词 + 已生成的 token 前缀"，把"动作"换成"下一个 token"，奖励稀疏（只在整段回答末尾给一次），再用 KL 惩罚把模型拴在 SFT 基线附近。

### 7.1 token 级实现

| 经典 RL | LLM RLHF |
|---|---|
| 状态 s_t：环境状态 | s_t = 提示词 x + 已生成 token y_<t |
| 动作 a_t：从动作空间选 | a_t = 下一个 token y_t（词表大小 ~10⁵ 的动作空间） |
| 每步都有即时奖励 r_t | 生成期 r_t = 0，整段回答结束后给一次 r = reward_model(x, y) |
| V(s_t)：状态价值 | per-token value head（在 LM 最后一层隐藏态上接线性头，每个 token 位置输出一个标量） |

- **loss_mask**：提示词 token 不算策略 loss，只对模型生成的 token 算——实现上就是逐 token 的 0/1 mask 乘到 log-prob 上。工程细节：本地 Search-R1 复现项目的 `train_grpo_local.py::build_training_sequences` 里，prompt tokens 的 mask 为 0、completion tokens 为 1，训练时 `masked_lp = token_log_probs * shift_mask` 再按有效 token 数归一化——这就是 token 级策略梯度在真实代码里的样子。
- **稀疏奖励 + GAE**：即时奖励全 0 时，`δ_t = γV(s_{t+1}) − V(s_t)`，credit assignment 完全由 value 网络承担——这正是 RLHF 里 critic 训练不稳的根源之一，也是后来 GRPO 想甩掉 critic 的动机。
- **每 token 一个 KL 惩罚**：`β·( log πθ(y_t|x,y_<t) − log π_ref(y_t|x,y_<t) )`，沿生成序列求和得序列级 KL。

### 7.2 KL 惩罚的两种做法

**做法 A：在奖励里减（reward shaping）**

```
r'(x,y) = r(x,y) − β·KL( πθ(·|x) ‖ π_ref(·|x) )
```

- **直觉**：把"偏离 SFT 基线"直接定义为负奖励，模型为了拿高分会主动靠近 ref。InstructGPT 用这种。
- **优点**：与 RL 算法解耦，任何 RL 后端都能用；**缺点**：KL 项混进奖励后会经由 GAE/bootstrap 影响 advantage 的长期信号，且 β 调起来和奖励量纲耦合（奖励是 0/1 时 β 取 0.01~0.1 量级，奖励是 0~10 时又要重调）。

**做法 B：在 loss 里加（直接给梯度）**

```
L = L_PPO + β·KL( π_ref ‖ πθ )   （或 KL(πθ‖π_ref)，方向有讲究）
```

- **直觉**：不污染奖励，直接在优化目标上拉住策略。DeepSeekMath（GRPO）即用此法，KL 用 Schulman 的 k3 估计器：`KL(π_ref‖πθ) ≈ π_ref/πθ − log(π_ref/πθ) − 1`，恒非负（由 `e^u ≥ 1+u` 保证）。⚠️ k3 估计器的出处为 Schulman 2020 博客《Approximating KL Divergence》，细节在 Part 2 展开。
- **方向问题**：`KL(πθ‖π_ref)` 是前向 KL（mean-seeking，惩罚"ref 没覆盖"的区域），`KL(π_ref‖πθ)` 是反向 KL（mode-seeking，惩罚"πθ 比 ref 少"的区域）。RLHF 常用反向 KL——直觉是"宁可收缩也不瞎扩张"。⚠️ 两种方向的实践选择存在争论，Part 2 结合 GRPO 源码细讲。
- **优点**：梯度直接、不污染优势估计；**缺点**：只在更新时约束，采样（rollout）时模型可能已经跑远，ratio 会很大。

### 7.3 PPO-ptx：混入预训练梯度防遗忘

```
L_total = L_PPO + γ · L_pretrain，   L_pretrain = −log P(预训练语料的 token)
```

- **一句话本质**：在 RL 更新里按比例 γ 混入预训练数据的语言模型梯度，防止"对齐税"（alignment tax——RLHF 后模型在公开 NLP 基准上的能力下降）。
- **出处**：InstructGPT（arXiv 2203.02155），论文取 γ=27.8（混合系数，RL 目标为单位 1）。
- **直觉**：RL 奖励只关心"有没有答好这个问题"，长期只用它更新，模型会漂移出自然语言分布（输出变得刻板、格式化、甚至语法退化）。预训练梯度像"锚"，把模型往回拉。它和 KL 惩罚是两种不同的锚：KL 锚在**输出分布**上，PPO-ptx 锚在**数据分布**上。

### 7.4 四模型架构与显存账

| 模型 | 角色 | 是否训练 | 用途 |
|---|---|---|---|
| Actor（策略模型） | 生成回答 | 训练 | 被 PPO loss 更新 |
| Critic（价值模型） | 给每个 token 打价值分 | 训练 | GAE 优势估计 |
| Reference（参考模型） | 冻结的 SFT 基线 | 否 | 算 KL 惩罚 |
| Reward Model（奖励模型） | 冻结的偏好打分器 | 否 | 给整段回答打标量奖励 |

**显存账（以 7B 参数、FP16 为例，⚠️ 数量级估算，实际取决于框架与并行策略）**：

- 一份权重：7×10⁹ × 2 字节 = **14 GB**；
- 训练态开销（actor）：梯度 14 GB + Adam 一阶/二阶矩（FP32）每参数 8 字节 = **56 GB**，即一个 7B actor 训练约 **84 GB**（不含激活），单卡 80G 放不下 → 必须 ZeRO 切分或 offload；
- critic 与 actor 同结构：再来约 84 GB（或减半，若 critic 用小模型）；
- ref 与 reward model 只推理：各 14 GB 级权重（reward model 通常 1B~7B，InstructGPT 用 6B）；
- **真正的显存大头是激活**：transformer 前向的中间激活 ≈ 层数 × 隐藏维 × 序列长 × batch × 2 字节，长上下文（8k+ token）时动辄几十上百 GB → 工程标配 gradient checkpointing（本地 `train_grpo_local.py` 也开了 `model.gradient_checkpointing_enable()`，并做按 token 数的 micro-batch 打包 `pack_micro_batches` 防 OOM）。

**一句话账**：4 个模型全参训练在单卡上基本不可行，主流框架（veRL、trl）用 ZeRO-2/3 + vLLM 分离式 rollout 把权重/优化器状态/激活摊开。**这也是 GRPO 的卖点之一：删掉 critic，训练态显存直接砍掉近一半**——Part 2 详述。

---

## 8. InstructGPT：RLHF 的工程蓝本

**一句话本质**：SFT 学会"照着人写的答案模仿"，奖励模型学会"什么是好答案"，PPO 学会"照着奖励模型喜欢的风格生成"——三步流水线 + 动态 KL 系数。

**出处**：Ouyang et al. 2022《Training language models to follow instructions with human feedback》，arXiv 2203.02155，OpenAI，基座为 GPT-3 家族。

### 8.1 三步流水线

```
GPT-3(175B) ──①SFT──> SFT 模型 ──②训练RM──> Reward Model(6B)
                  │                            │
                  └──③PPO（actor=SFT 初始化，ref=SFT 冻结，reward=RM 冻结）──> PPO/PPO-ptx 模型
```

**第 1 步 SFT**：把 GPT-3 在 **~13k 提示-回答对**上监督微调（其中 11,295 条提示来自 API 用户真实请求、1,514 条标注员手写，合计约 12.8k，通称 ~13k），回答由标注员手写。损失就是标准语言模型交叉熵。

**第 2 步 RM**：用 SFT 模型（去掉最后的词表映射层，换成输出标量的线性头）初始化，在 **~33k 人类对比数据**上训练。每条数据 = 一个提示 + 若干回答 + 标注员排序（4~9 个回答排序）。损失为 Bradley-Terry 成对排序损失：

```
L_RM = −E[ log σ( r_θ(x, y_w) − r_θ(x, y_l) ) ]
```

- **直觉**：σ 是 sigmoid。`P(标注员认为 w 优于 l) = σ(r_w − r_l)`——打分差越大，好回答胜出的概率越高；训练就是最大化"人类排序"在这个模型下的对数似然。用 6B 而非 175B 的公开理由是 175B RM 训练不稳定（⚠️ 论文给出的解释细节待核实，但 6B 是论文明确选择）。

**第 3 步 PPO**：actor 从 SFT 初始化，ref 冻结为 SFT，reward = RM 分数，在 **~31k 提示**上做第 7 节的 token 级 PPO。论文还训了 1.3B/6B actor 做消融。论文内结论（非跨论文对比）：**1.3B 的 PPO 模型在人类偏好上即可超过 175B 的 SFT 模型**——说明对齐质量与参数量解耦，RL 环节才是关键。

### 8.2 KL 系数怎么调（动态调整规则）

InstructGPT 不固定 β，而是**在线动态调整**（大意，⚠️ 具体阈值细节见论文附录，此处为规则主干）：

1. 设定 KL 目标：希望 `KL(π_θ ‖ π_SFT)` 稳定在约 **6 nats**（衡量"新策略偏离 SFT 多远"）；
2. 每轮 RL 更新后测量当前 KL：
   - 若 `KL > 1.5 × 目标` → 模型跑太远，**β × 2**（惩罚加倍，拉回来）；
   - 若 `KL < 目标 / 1.5` → 约束太紧，**β ÷ 2**（放松，给优化留空间）；
3. 早期停止：监控反向 KL `KL(π_SFT ‖ π_θ)`（衡量"ref 覆盖不到的生成区域"），若它与优化初期相比的比值变化超过 **1.3 倍**，停止该提示的训练（防止跑出分布）。
- **直觉**：β 是"缰绳的松紧"，目标 KL 是"允许马跑多远"。KL 太小 → 模型原地踏步（只会复读 SFT）；KL 太大 → 模型胡言乱语（奖励黑客）。动态调整让 β 始终维持在"既不锁死也不失控"的区间。⚠️ 初始 β 取值（复现笔记常见 0.02）待核实。
- **为什么 RLHF 里 KL 如此关键**：RM 只在训练分布内有意义，模型一旦跑出 RM 见过的分布，RM 打分即失效（reward hacking / over-optimization，Goodhart 定律的实例）——KL 惩罚是防止模型"骗过"RM 的第一道防线。

### 8.3 数据规模小结

| 阶段 | 数据 | 规模 |
|---|---|---|
| SFT | 提示-回答对（API 提示 + 标注员手写） | ~13k 提示 |
| RM | 提示 + 4~9 个回答的标注员排序 | ~33k 对比 |
| PPO | RL 提示（含预训练数据混合） | ~31k 提示 |

**为什么数据这么少也能训**：三步流水线每步都在"已有能力上做分布偏移"——GPT-3 已含海量世界知识，SFT 只教格式与意图，RM 只学排序偏好，PPO 只做微调式的偏好对齐。⚠️ 此解释为社区共识性解读，非论文原话。

---

## 9. 时间线总表：2017–2026 LLM RL 大事记

| 时间 | 事件 | 论文 / arXiv | 一句话 |
|---|---|---|---|
| 2017-06 | 人类偏好强化学习（RLHF 原型） | Christiano et al., 1706.03741 | 用人类对轨迹片段的偏好训练奖励模型，Atari/机器人任务验证 |
| 2017-07 | PPO 发表 | Schulman et al., 1707.06347 | clip 一阶近似信任域，至今仍是 LLM RL 主力算法 |
| 2017-09 | TRPO（2015-02 首版）的实践生态形成 | 1502.05477 | 信任域理论奠基，PPO 的直接前身 |
| 2022-03 | InstructGPT 发表 | Ouyang et al., 2203.02155 | SFT→RM→PPO 三步流水线成为行业蓝本 |
| 2022-11 | ChatGPT 发布 | （产品发布，无论文） | RLHF 从实验室走向十亿级用户 |
| 2023-05 | DPO 提出 | Rafailov et al., ⚠️ 2305.18290 | 隐式奖励 + 直接偏好优化，甩掉 RL 循环 |
| 2024-02 | RLOO 提出 | Ahmadian et al., ⚠️ 2402.14740 | REINFORCE 留一法 baseline，⚠️ 注：RLOO 实为 2024-02（常与 DPO 并称 2023，系误记） |
| 2024-02 | GRPO 提出（DeepSeekMath） | DeepSeekMath, 2402.03300 | 组内相对优势，删除 critic，LLM RL 大幅降显存 |
| 2025-01 | DeepSeek-R1 发布 | DeepSeek-R1, 2501.12948 | 纯 RL 激发推理能力，"aha moment"出圈 |
| 2025-03 | DAPO 提出 | ⚠️ 2503.14476 | 开源规模化 RL：clip-higher、动态采样等四项修复 |
| 2025-03 | Dr.GRPO 提出 | ⚠️ 2503.20783 | 修正 GRPO 的长度/难度偏置 |
| 2025-04 | VAPO 提出 | ⚠️ 2504.05118 | 价值增强 PPO，长链推理（32k 上下文）训练稳定 |
| 2025-07 | GSPO 提出 | ⚠️ 2507.18071 | 组序列策略优化，统一 GRPO 的 ratio 与优势估计 |
| 2026（本文写作时） | Agentic RL 规模化 | — | 多步工具调用 + 推理链的 RL 训练成为工程热点（本地实践观察，非论文事实） |

⚠️ 说明：任务给定的 7 个 arXiv 编号已核实直接使用；其余编号（DPO/RLOO/DAPO/Dr.GRPO/VAPO/GSPO）为笔者记忆，**使用前请核实**。

---

## 10. 本地实践对照：Search-R1 复现项目里的这些概念

（项目代码在 `C:\new\intern\plan\projects\search-r1\`，GitHub: Bowen-mantou/search-r1-repro）

**GRPO 训练脚本（`train_grpo_local.py`）与本文理论的对应关系**——读真实代码是最好的复习：

1. **组内相对优势**（第 3 节 baseline 的变体）：`compute_group_advantages()` 对同一问题的 group_size 条轨迹做 `(r − mean) / (std + 1e-8)` 标准化，异常的轨迹（格式错误、超搜索上限）优势置 0——baseline 不是 V(s) 而是"组内均值"，这是 GRPO 与本文 PPO 的最关键差异（Part 2 详述）。
2. **token 级 loss_mask**（第 7.1 节）：`build_training_sequences()` 给 prompt token 打 mask 0、completion token 打 mask 1；`grpo_loss()` 里 `loss = −(advantages * mean_lp).mean()`——每条轨迹一个标量优势，广播到它所有生成 token 的平均 log 概率上，这就是"把 token 级策略梯度按轨迹聚合"的最小实现。
3. **KL 的替代品 LLDS**：该脚本用自定义的 LLDS 惩罚 `λ·Σ max(0, log π_ref − log π_θ)`（只罚"新策略比旧策略差"的 token），是本项目对 KL 惩罚的一次工程探索——对照第 7.2 节，它属于"loss 里加"一路，但用了单侧截断而非对称 KL。
4. **显存工程**（第 7.4 节的现实版）：4-bit QLoRA + 梯度检查点 + 按 token 数打包 micro-batch，让 4B 模型的 GRPO 跑在 12GB 消费级显卡上；`_chunked_log_softmax_gather` 分块算 log-softmax 防词表维 OOM。
5. **veRL 框架侧**（`verl_search_r1/`）：在 veRL main 0.10 上以自定义 AgentLoop（`SearchR1AgentLoop`，经 `rollout.agent.agent_loop_config_path` 注册）把多轮"思考→搜索→回答"的 agent loop 接入 GRPO 训练管线，环境已调通；并对 rollout 侧的 **TQ（TaskQueue，rollout worker 与 agent loop worker 之间的调度队列）** 做了瓶颈定量分析：单步约 51K 个子操作、每个 agent loop 约 400 个子操作、内部存储单元 PUT_DATA 吞吐约 736/min（测量方法与数据见项目 `docs/07-INTERVIEW_QA.md`）。这正对应 Part 2 要讲的"agentic RL 工程瓶颈"主题。

---

## 11. 面试速查卡片

**必背公式（按依赖顺序）**

```
① J(θ) = E_{τ~πθ}[ Σ r_t ]                                   目标
② ∇P(τ) = P(τ)·∇log P(τ)                                    似然比技巧
③ ∇J(θ) = E[ Σ_t ∇log πθ(a_t|s_t) · Â_t ]                   策略梯度定理（环境项∇=0，baseline∇=0）
④ Â_t^GAE = Σ_l (γλ)^l δ_{t+l}，δ_t = r + γV(s') − V(s)      GAE（λ: bias-variance 旋钮）
⑤ L_CLIP = E[ min(r·Â, clip(r,1−ε,1+ε)·Â) ]                  PPO（ε=0.2）
⑥ r'(x,y) = r(x,y) − β·KL(πθ‖π_ref)                         RLHF 奖励（InstructGPT）
```

**高频问答（一句话答案）**

- **Q：为什么减 baseline 期望不变？** score function 期望为零：`Σ_a π·∇log π = ∇Σπ = 0`。
- **Q：GAE 的 λ 是什么？** 指数衰减窗口：λ=0 是 TD(0)（低方差高偏差），λ=1 是 MC（无偏高方差）；γ 是任务折扣、λ 是估计器折扣，解耦是 GAE 的洞察。
- **Q：PPO 为什么 clip 不会把梯度剪坏？** clip 只截断"继续走远"的方向（A>0 时 r>1+ε、A<0 时 r<1−ε 梯度归零），"往回走"永远放行；且 min 保证目标是保守下界。
- **Q：RLHF 为什么需要 reference model？** 提供 KL 锚点，防 reward hacking——RM 只在训练分布内有效，跑出分布即失效。
- **Q：PPO-ptx 的 γ 是什么？** 预训练梯度混合系数（InstructGPT 取 27.8），防对齐税/灾难性遗忘。

**Part 2 预告**：GRPO 数学与实现（组内优势、ratio 的 token 级形式、k3 KL 估计器）、DPO/RLOO 与 PPO 的等价/差异、DAPO/Dr.GRPO/VAPO/GSPO 的演化主线、以及 agentic RL（多轮工具调用）下的工程瓶颈。
# LLM 强化学习方法全景指南 · Part 2：GRPO 及其变体

> 目标读者：做过 Search-R1 复现项目（基于 GRPO 训练 Qwen3.5-4B 学会搜索）、正在准备面试的人。
> 阅读方式：公式全部逐步推导并配直觉；第 2.4 节直接引用复现项目的真实代码。
> 本篇涉及的论文编号（已核实）：DeepSeekMath = arXiv 2402.03300，DeepSeek-R1 = arXiv 2501.12948，
> Dr.GRPO = arXiv 2503.20783，DAPO = arXiv 2503.14476，VAPO = arXiv 2504.05118。

---

## 2.1 出处与动机：为什么 GRPO 要干掉 critic

### 2.1.1 PPO 的两个"重资产"

强化学习微调 LLM 的标准配方是 PPO。PPO 的策略梯度形式是：

    g = E[ Σ_t A_t · ∇_θ log π_θ(o_t | q, o_<t) ]

其中 A_t 是每一步的优势（advantage）。为了得到 A_t，PPO 需要：

1. **价值函数（critic）**：A_t = r_t + γ·V(s_{t+1}) − V(s_t)，V 是学习出来的"从这一步往后期望还能拿多少奖励"。对 LLM 来说，critic 是和策略**同规模**的第二个网络——7B 策略就要一个 7B 价值的 critic，显存几乎翻倍。
2. **每 token 的奖励**：GAE（Generalized Advantage Estimation）要把序列末尾的奖励沿时间轴折现回传到每个 token，这要求奖励模型（RM）逐 token 打分。但实际 RM 往往只对整条回答打一个分——"序列级奖励 vs token 级信用分配"的错配。

为后面对比，先把 PPO 的目标写全（带 clip 的代理目标）：

    L_PPO = −E[ min( ρ_t·A_t, clip(ρ_t, 1−ε, 1+ε)·A_t ) ]
    A_t  = Σ_k γ^k · δ_{t+k}（GAE）,  δ_t = r_t + γ·V(s_{t+1}) − V(s_t)

两个观察：clip 约束的是**新旧策略的概率比**（与优势无关）；A_t 的每一步都依赖 V——V 学偏了，整个梯度就偏了。GRPO 要做的就是把 A_t 的来源从"学习型 V"换成"同组蒙特卡洛样本"，其余骨架（clip、KL）保留。

DeepSeekMath（Shao et al., 2024, arXiv 2402.03300）在数学题强化学习场景提出：**题目有标准答案、奖励可以自动判定（对/错）**，那么"这条回答好不好"其实只需要跟**同一道题的其他几条回答**比一比就知道，根本不需要维护一个逐状态的价值函数。这正是 GRPO 成立的隐含前提：**奖励可由规则自动判定**。你的 Search-R1 复现完全满足这一点（答案 EM 匹配 + 格式规则），而开放域的偏好优化（人类偏好、LLM-as-judge）则更适合 PPO。

### 2.1.2 GRPO 的核心思想：组内比较替代价值函数

GRPO（Group Relative Policy Optimization）的采样方式：

- 对每个问题 q，用当前（旧）策略 π_old 采样 **G 条完整回答** o_1, …, o_G（DeepSeekMath 用 G=64）；
- 每条回答拿到一个**序列级**奖励 r_i = R(q, o_i)；
- 用这 G 个奖励的**组内均值 μ 和标准差 σ̂** 构造每条回答的优势；

    Â_i = (r_i − μ) / (σ̂ + ε),   μ = (1/G)Σ_j r_j,   σ̂ = sqrt((1/G)Σ_j (r_j − μ)²)

- 优势 **广播到这条回答的每一个 token**，进入带 clip 的策略梯度目标。

一句话总结两者的分界线：

    PPO：用「神经网络学出来的基线 V(s)」做相对比较 → 需要 critic
    GRPO：用「同题同组蒙特卡洛样本的均值 μ」做相对比较 → 不需要 critic

之后 DeepSeek-R1（arXiv 2501.12948）把 GRPO 用在了带推理链的 RL 上：R1-Zero 从基座模型直接 GRPO 训练，奖励全部来自可验证规则（答案对错 + 格式），展示了"纯 RL 也能涌现长思维链"。你复现的 Search-R1 正是这条技术路线在"搜索 agent"场景的延伸——奖励换成"最终答案 EM 匹配 + 搜索行为格式"，GRPO 教会模型自己决定何时搜索、搜索几次。

### 2.1.3 一步 GRPO 更新的完整管线

面试经常被要求"把 GRPO 训练流程从头讲一遍"，标准答案分五步：

1. **Rollout**：π_old 对每个问题采样 G 条回答（带温度采样；agent 场景里包括工具调用的多轮交互），每条记录 token 序列与 π_old 的逐 token logprob（后文 KL/LLDS 要用）；
2. **Reward**：对每条回答按规则打分 r_i（结果奖励；可叠加过程奖励，见 2.4.4 的 PRM-Lite）；
3. **Advantage**：按问题分组算 Â_i = (r_i − μ)/(σ̂ + ε)（异常轨迹先排除，见 2.4.1）；
4. **Loss**：把每条回答的 token 拼成训练序列，算 2.2.2 的 clipped loss + KL 项，loss_mask 只盖生成 token；
5. **Update**：反向传播、更新 π_θ（旧策略 π_old 在本步内冻结，下轮 rollout 再更新）。

面试加分细节：第 1 步存的 π_old logprob 与第 4 步用 π_θ 重算的 logprob 必须**按同一 token 对齐**（shift 一位），否则重要性比 ρ 全是错的——这就是 2.4.2 里项目代码反复强调 shift 对齐的原因。

---

## 2.2 完整数学推导

### 2.2.1 组内 advantage：为什么它是 sample-mean baseline 的蒙特卡洛估计

**第 1 步：组统计量。** 一组 G 条回答的奖励 {r_1, …, r_G}：

    μ = (1/G) · Σ_{j=1..G} r_j                      （组内样本均值）
    σ̂² = (1/G) · Σ_{j=1..G} (r_j − μ)²             （组内样本方差）

**第 2 步：优势定义。**

    Â_i = (r_i − μ) / (σ̂ + ε)                       （ε 防除零，项目实现里 ε = 1e-8）

**手算一遍（面试最爱考的数字题）。** 组内 3 条回答，奖励 {0, 0, 1}：

    μ = 1/3,  σ̂² = (1/3)[(−1/3)² + (−1/3)² + (2/3)²] = 2/9,  σ̂ ≈ 0.471

    两条错误回答: Â = (0 − 1/3)/0.471 ≈ −0.71
    一条正确回答: Â = (1 − 1/3)/0.471 ≈ +1.41

三个要点：正负优势之和恰好为 0（组内是零和比较）；正确回答的优势量级是错误回答的 2 倍（因为只有 1/3 概率被"抬"、2/3 被"压"，梯度守恒）；σ̂ 越大优势越小——同组对错混杂时信号被稀释。换个组 {0, 0, 0}：σ̂=0，靠 +1e-8 保底，优势全为 0，本组不产生梯度——这正是 DAPO Dynamic Sampling（2.6.2）要过滤的情形。

**第 3 步：为什么 μ 是"sample-mean baseline 的蒙特卡洛估计"。**

PPO 里的 advantage 是 r − V(q)：V(q) 是"这个状态平均能拿多少奖励"的**学习型基线**。把 V(q) 换成它的蒙特卡洛估计——从当前策略对同一道题采 G 个样本、取样本均值 μ——就得到 r_i − μ。μ 依大数定律收敛到 E_{o~π_old}[R(q, o)]，所以 r_i − μ 是"这条回答比当前策略平均水平好多少"的蒙特卡洛估计，这正是 advantage 的定义本身。G 就是蒙特卡洛的样本数：G 越大，基线估计越准，方差越低——这就是为什么 DeepSeekMath 要把 G 开到 64。

**第 4 步：除以 σ̂ 为什么"等价"（排序不变）。** σ̂ 对组内所有轨迹是**同一个公共常数**：把 r_i − μ 换成 (r_i − μ)/σ̂，组内轨迹的优势排序完全不变，梯度方向只差一个组级标量因子，等价于给每一组自动调整学习率。好处是奖励尺度无关（reward scale-invariance）：奖励是 0/1 还是 0~100，优势的量级被标准化到组内相对离散度，超参数可以跨任务迁移。代价见第 2.5 节。

**第 5 步：无偏性分析。**

不加 σ̂ 时，组均值基线的无偏性来自 score function 恒等式：

    E_{o~π}[ ∇_θ log π_θ(o) ] = Σ_a π_θ(a) · ∇_θ log π_θ(a) = ∇_θ Σ_a π_θ(a) = ∇_θ 1 = 0

因此对任何**与采样无关**的常数基线 b：

    E[ (r_i − b) · ∇ log π(o_i) ] = E[ r_i · ∇ log π(o_i) ] − b · E[ ∇ log π(o_i) ] = ∇_θ E[r]

基线不影响梯度的期望，这就是"减基线无偏"的证明。注意一个细节：组均值 μ 里包含 r_i 自己（1/G 的自项）。展开：

    E[ (r_i − μ) ∇ log π(o_i) ]
      = E[ r_i ∇ log π(o_i) ] − (1/G) Σ_j E[ r_j ∇ log π(o_i) ]
      = ∇E[r] − (1/G) · ∇E[r]                          （j≠i 的项因独立性 = E[r_j]·E[∇logπ] = 0）
      = (G−1)/G · ∇_θ E[r]

即组均值基线把整个梯度**统一缩放 (G−1)/G**——方向无偏，幅度差一个常数，等价于学习率乘 (G−1)/G（用 leave-one-out 均值 1/(G−1)Σ_{j≠i} r_j 则严格无偏）。除以 σ̂ 后严格说**不再无偏**：E[1/σ̂] ≠ 1/σ（Jensen 不等式），且 σ̂ 与 r_i 相关，σ̂→0 时数值爆炸——所以项目实现里要 +1e-8。实践中大家接受这个偏差，换取尺度不变性。

补充一个方差视角的定量直觉：无基线时梯度项方差 Var[r_i·∇logπ] 的量级由 r_i 的绝对大小决定（奖励是 0~100 还是 0~1 直接改变梯度噪声）；减掉 μ 后残留的是"组内相对波动"，与奖励的**绝对尺度无关**——这正是 baseline 减方差的具体含义。组内基线还会带来一个额外的统计福利：因为 μ 吸收了题目难度，不同难度题目之间的梯度尺度自动对齐，无需按题目难度调学习率。

**第 6 步：方差分析。** 基线减方差：Var[(r−b)∇logπ] 在 b ≈ E[r] 附近最小，μ 正是 E[r] 的估计。组内基线还消掉了**题目难度造成的公共偏移**：难题上所有人都低分，减 μ 后信号保留；不分组的全局基线则做不到。剩余方差来源：G 小 → μ̂、σ̂ 估计噪声大；0/1 奖励时 σ̂ ≈ sqrt(p(1−p))，组内"全对/全错"时 σ̂→0。对比 GAE：GAE 用学习到的 V 把方差压得很低但引入**函数近似偏差**；GRPO 用蒙特卡洛无偏但方差高，靠增大 G 补偿——这就是"GRPO 更省显存但采样更费 token"的数学根源。

### 2.2.2 Loss 逐项拆解

GRPO 的目标函数（DeepSeekMath 式，忽略期望符号）：

    L = −(1 / Σ_i |o_i|) · Σ_{i,t} min( ρ_{i,t}·Â_i ,  clip(ρ_{i,t}, 1−ε, 1+ε)·Â_i )  −  β · KL(π_θ || π_ref)

**项 1：ρ_{i,t} —— token 级重要性比。**

    ρ_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) / π_old(o_{i,t} | q, o_{i,<t})

当前策略与采样时旧策略在该 token 上的概率比，用于修正 off-policy 偏差。**第一次回放时 π_θ = π_old，ρ = 1**——这正是你项目里简化实现可以去掉 clip 的理论依据（见 2.4.2）。多 epoch 复用同一批 rollout 数据、或异步训练（采样与更新并行）时 ρ ≠ 1，clip 才起作用。

**项 2：Â_i —— 序列级优势，逐 token 广播。** 奖励只在序列结尾给出（结果奖励），组内所有 token 共享同一个 Â_i。这意味着 GRPO **不提供 token 级信用分配**——整条回答的"好/坏"被均匀摊到每个 token 上。

**项 3：min + clip —— 悲观代理（pessimistic surrogate）。**

- Â_i > 0（回答比组内平均好，要抬升概率）：clip 把 ρ 封顶在 1+ε，min 取更小的那支 → 一次更新最多把好 token 的概率放大 (1+ε) 倍，防止过度自信。
- Â_i < 0（回答比平均差，要压低概率）：Â_i 为负时，(1−ε)·Â_i > ρ·Â_i（负得更少），min 取 clip 支 → 惩罚力度被封顶，防止一次更新把某 token 概率砸穿。
- 直觉：clip 构造了一个**隐式信任域**——不允许单步更新让新旧策略在概率空间偏离太远。ε 是信任域半径（DeepSeekMath 取 ε = 0.2）。

**数值检验**（ε = 0.2）：某 token 的 ρ = 2.5。若 Â = +1：min(2.5·1, clip(2.5,0.8,1.2)·1) = min(2.5, 1.2) = 1.2——涨幅被按在 20%；若 Â = −1：min(2.5·(−1), 1.2·(−1)) = min(−2.5, −1.2) = −2.5——该 token 概率翻倍反而是坏消息（回答整体差），惩罚不设上限，只保护"概率上升"方向。再看 ρ = 0.3（概率跌了 70%）、Â = −1：min(0.3·(−1), 0.8·(−1)) = min(−0.3, −0.8) = −0.8——跌过头时惩罚被 clip 封顶在 20%，防止一次更新把 token 概率砸穿。**记忆法：clip 只在"变化方向与优势方向一致"时生效。**

**项 4：1/Σ|o_i| —— token 均值归一。** Σ_i |o_i| 是本批所有回答的 token 总数。除以它的意图：让不同长度的回答对总梯度贡献可比，防止长回答（token 多、梯度项多）天然主导。**代价**：它同时稀释了长正确回答的正梯度、也稀释了长错误回答的负梯度——长度偏差的另一面，DAPO 和 Dr.GRPO 后来都把它移除（见 2.5、2.6）。

**项 5：−β·KL(π_θ || π_ref) —— 显式信任域。** clip 是隐式约束，KL 是显式约束：让当前策略别离参考策略（初始 SFT 策略，冻结）太远，防止熵坍缩与奖励黑客。β 是系数，DeepSeekMath 取 β = 0.04 ⚠️（以原论文为准）。KL 的具体估计器见 2.2.3。

### 2.2.3 KL 惩罚：k3 无偏估计器

KL(π_θ || π_ref) = E_{o~π_θ}[ log π_θ(o) − log π_ref(o) ] 需要从 π_θ 采样（在线训练时 o 确实来自 π_old ≈ π_θ），DeepSeekMath 采用 Schulman 博客里的 k3 估计器。记

    x = log π_ref(o_t) − log π_θ(o_t)

则单 token 估计：

    k3 = exp(x) − x − 1

**无偏性**（对 π_θ 采样取期望）：

    E_πθ[ exp(x) ] = Σ_a π_θ(a) · (π_ref(a) / π_θ(a)) = Σ_a π_ref(a) = 1
    ⇒  E_πθ[ k3 ] = 1 − E_πθ[x] − 1 = E_πθ[ log π_θ − log π_ref ] = KL(π_θ || π_ref)

**直觉**：exp(x) 这一项把 ref 的密度"重要性采样"进 θ 的期望；e^x − x − 1 是 e^x 在 x=0 的凸展开剩余，恒非负且仅在 x=0 取 0——所以估计值天然满足 KL 非负性，而且**不需要 π_ref 的前向/梯度**：ref 是冻结的，rollout 时存好它的 logprob 即可，训练期零额外显存。这就是"GRPO 的 ref 模型为什么便宜"的原因。

**数值感受**：若某 token 上 π_θ = 0.3、π_ref = 0.1，则 x = log 0.1 − log 0.3 ≈ −1.099，k3 = e^−1.099 + 1.099 − 1 ≈ 0.333 + 1.099 − 1 = 0.432（正的，合理——θ 偏离了 ref）。若 π_θ = π_ref，x = 0，k3 = 0。若 π_θ = 0.9、π_ref = 0.1（θ 严重偏离），k3 = e^−2.197 + 2.197 − 1 ≈ 1.31，惩罚随偏离加速上升。注意 k3 只从 π_θ 采样即可无偏，这个性质让它在**在线采样**的 GRPO 里天然适用。

### 2.2.4 小结：GRPO 的三个设计决策

1. **用什么做基线**：组内样本均值 μ（蒙特卡洛）而非学习型 V(s)——省 critic，代价是方差；
2. **信任域怎么构造**：clip 在概率比上（隐式）+ KL 对 π_ref（显式，k3 无偏估计）——双保险防漂移；
3. **奖励怎么分配**：序列级奖励 → 组内标准化优势 → 广播到 token——不提供逐 token 信用分配，靠"整条回答同生共死"来引导行为。

---

## 2.3 代码实现：可运行的简化 GRPO（约 40 行）

```python
import torch
import torch.nn.functional as F

def compute_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """组内标准化优势。rewards: [G]，同一问题的 G 条轨迹奖励。"""
    mu = rewards.mean()
    sigma = rewards.std() + 1e-8            # 防除零
    return (rewards - mu) / sigma           # Â_i = (r_i - μ) / σ̂

def grpo_loss(logits, input_ids, advantages, loss_mask,
              old_log_probs=None, ref_log_probs=None,
              clip_eps=0.2, beta=0.04):
    """简化 GRPO 损失。logits: [B, L, V]；advantages: [B]（响应级）。"""
    # 1) 因果 LM 的 shift：位置 t 的 logits 预测 token t+1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:]
    shift_mask = loss_mask[:, 1:]           # 只在生成 token 位置为 1

    # 2) 当前策略的 token 级 log prob
    logp = F.log_softmax(shift_logits, dim=-1) \
             .gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)   # [B, L-1]

    # 3) 重要性比 ρ（首轮回放 ρ=1：old 就是 rollout 策略）
    if old_log_probs is None:
        old_log_probs = logp.detach()
    ratio = torch.exp(logp - old_log_probs)  # ρ_{i,t}

    # 4) 序列级优势广播到每个 token
    A = advantages[:, None] * shift_mask     # [B, L-1]

    # 5) PPO 式 clip：min(ρA, clip(ρ)A) 取悲观支
    surr1 = ratio * A
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A
    per_token = torch.min(surr1, surr2) * shift_mask
    n_tokens = shift_mask.sum().clamp(min=1)
    policy_loss = -per_token.sum() / n_tokens        # token 级平均（=1/Σ|o_i| 归一）

    # 6) KL 惩罚：k3 无偏估计器 exp(x) - x - 1, x = log π_ref - log π_θ
    if ref_log_probs is not None:
        x = ref_log_probs - logp
        k3 = torch.exp(x) - x - 1
        kl = (k3 * shift_mask).sum() / n_tokens
        return policy_loss + beta * kl
    return policy_loss
```

对应关系速查：第 5 块实现 min/clip，分母 n_tokens 实现 1/Σ|o_i|，第 6 块实现 β·KL。三个工程细节值得记住：(a) **shift**——LM 前向在位置 t 输出的是对 t+1 的预测，mask 必须同步右移一位，错位是复现事故高发区；(b) **log_softmax 一次性算**，不要对每个 token 单独 softmax；(c) **loss_mask 只盖生成 token**——prompt 和工具返回 token 不参与 loss。

把这个简化版接上真实的采样循环（第 2.1.3 节五步管线），就是一个能跑的最小 GRPO：rollout 时用 `model.generate` 采样并记录 `old_log_probs`（与训练期 `logp` 同 shift 对齐）与 `ref_log_probs`（冻结 ref 的 logprob，本简化版里 ref 就是 SFT 初始化权重），组内算完 advantage 后调用 `grpo_loss`。若想让多组问题共享一个 batch，把 advantages 按样本索引排列（不是按问题排列）即可——group 只在 advantage 计算时存在，loss 计算时不关心分组。

---

## 2.4 真实项目落地：Search-R1 复现的 GRPO 实现

项目路径：`C:\new\intern\plan\projects\search-r1\`（管线主体 `train_grpo_local.py`）。背景：在 12GB 3060 上用 4-bit QLoRA + LoRA(r=16) 训练 Qwen3.5-4B，SFT 学会协议格式后，GRPO 教它搜索。以下是真实代码片段与讲解。

**先看奖励设计**（`reward.py` 的 `score_answer`，三档规则）：

- **+1.0**：恰好一行 `Answer:` 且归一化后与参考答案精确匹配（小写化、去标点、去冠词）；
- **0.0**：格式有效但答案错误；
- **−0.1**：没有合法的 `Answer:` 行（格式惩罚）。

三档奖励对 GRPO 的意义：比 0/1 二值奖励**多了一个"格式负反馈"信号**，且让组内 σ̂ 更不易为 0（对错混杂时 {1,0,−0.1} 天然有离散度），优势信号更稳定。注意答案匹配用了**归一化 + 多参考列表**（nq/hotpotqa 类数据每个问题有多个等价答案），这是 QA 任务奖励可判定的关键工程细节。

### 2.4.1 compute_group_advantages：异常轨迹排除后重算 mean/std

`train_grpo_local.py` 中的真实实现（节选）：

```python
def compute_group_advantages(trajectories: list[Trajectory], group_size: int) -> None:
    n_questions = len(trajectories) // group_size
    for qi in range(n_questions):
        group = trajectories[qi * group_size:(qi + 1) * group_size]
        rewards = [t.reward for t in group if not t.anomalous]     # ① 先排除异常轨迹
        if not rewards:
            for t in group:
                t.advantage = 0.0
            continue
        mean_r = np.mean(rewards)                                  # ② 用干净子集重算 μ
        std_r = np.std(rewards) + 1e-8                             # ③ σ̂ + 1e-8 防除零
        for t in group:
            if t.anomalous:
                t.advantage = 0.0                                  # ④ 异常轨迹优势归零
            else:
                t.advantage = (t.reward - mean_r) / std_r
```

四个教学点：

1. **异常轨迹不参与统计量的计算**——否则一条格式坏掉的轨迹会把 μ 拉偏，全组优势失真。这是"先清洗、后比较"的纪律。
2. **mean/std 是在排除异常后的干净子集上重算的**，不是先算好再剔除。
3. **std + 1e-8** 对应 2.2.1 里的 ε：组内奖励全相等（σ̂=0）时优势退化为 0 而不是除零爆炸。
4. 异常轨迹优势置 0，且后续 `build_training_sequences` 会跳过 `advantage == 0.0` 的轨迹——**双重过滤**：异常轨迹既不污染统计量、也不产生梯度。

**为什么必须"排除后重算"**（数值直觉）：假设组内 7 条正常轨迹奖励 {1, 1, 0, 0, 0, 0, 0}，另 1 条异常轨迹 −0.1。若不过滤：μ = 0.24 → 正确轨迹优势 ≈ (1−0.24)/σ̂；过滤后：μ = 0.29。差别不大，但真正的风险在**异常轨迹自己**——它格式都坏了，还拿 −0.1 去跟别人比，会得到一个大负优势并把它的"坏 token"样本送进 loss，教模型学一堆垃圾行为。先滤掉，统计量和梯度两处都干净。

### 2.4.2 grpo_loss：shift 对齐 + 分块 log_softmax + token-mean

```python
def grpo_loss(model, batch, pad_token_id, llds_lambda=0.0, llds_variant="A"):
    ...
    outputs = model(input_ids)
    logits = outputs.logits                                    # [n, seq_len, vocab]

    # Shift: predict token t+1 from token t
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous()                 # ① mask 同步右移

    # Log probabilities (chunked to avoid OOM on large vocab)
    token_log_probs = _chunked_log_softmax_gather(shift_logits, shift_labels)  # ② 分块

    # GRPO loss: -advantage * mean(log_prob over loss tokens)
    masked_lp = token_log_probs * shift_mask
    n_loss_tokens = shift_mask.sum(dim=1).clamp(min=1)
    mean_lp = masked_lp.sum(dim=1) / n_loss_tokens             # ③ 每条序列 token-mean
    loss = -(advantages * mean_lp).mean()                      # ④ 批内平均
```

教学点：

1. **shift 对齐**（①）：这是 2.3 里强调过的因果 LM 前向错位问题，mask 不右移会漏掉最后一个生成 token 的梯度。
2. **chunked log_softmax**（②）：`_chunked_log_softmax_gather` 把 log_softmax 按 512 位置分块，避免 [n, seq, vocab] 全量 logits 的峰值显存——12GB 卡能跑 RL 的显存工程细节，面试讲 OOM 治理时是好素材。
3. **token-mean**（③④）：先按序列归一、再按批平均，对应公式里的 1/Σ|o_i|。
4. **没有 clip、没有 KL 项**——这是合法的简化：每条 rollout 数据只回放一次（单 epoch），ρ = 1，min/clip 恒等于 1，加不加无区别；KL 的信任域角色由可选的 LLDS 正则替代。**面试能主动讲出"我的实现为什么合法地省掉了 clip"是强力加分点**。

`_chunked_log_softmax_gather` 值得单独看一眼：它对 `logits[:, start:end, :]` 按 512 位置分块做 log_softmax 再 gather——log_softmax 的中间张量 [n, seq, vocab]（本项目 vocab 15 万、seq 可达 4096）不落显存，峰值从"seq × vocab"降到"512 × vocab"。同类的 OOM 治理手段在主循环里还有：micro-batch 打包（`pack_micro_batches` 按 padded tokens 上限 4000 切批）、rollout 后 `gc.collect() + torch.cuda.empty_cache()`、序列硬截断 4096、梯度累积 ×4。**这些细节在 12GB 卡上不是可选项，是训练能跑通的前提**——面试讲"小显存做 RL"时按这个清单展开即可。

主循环的骨架也值得背下来（与 2.1.3 的五步管线一一对应）：

```python
for step in range(args.max_steps):
    trajectories = rollout_batch_local(...)        # ① 采样 G=8 × 8 题
    for t in trajectories: t.anomalous = is_anomalous(t)
    compute_group_advantages(trajectories, G)       # ② 组内优势
    if prm_lite:  t.reward += PRMLiteScorer().score(t).process_reward
    if lata:      apply_lata(trajectories)          # ③ 优势/奖励修正
    sequences = build_training_sequences(...)       # ④ 组装序列+mask+ref logprob
    for mb in pack_micro_batches(sequences, 4000):
        loss, _ = grpo_loss(model, mb, pad_id, llds_lambda)
        (loss / n_mb / GRAD_ACCUM).backward()       # ⑤ 更新
```

### 2.4.3 is_anomalous：训练数据入口的异常过滤

```python
def is_anomalous(traj: Trajectory) -> bool:
    if not traj.final_text or not traj.final_text.strip():
        return True                                   # 没有产出答案
    if traj.search_calls >= MAX_SEARCH_CALLS and not traj.valid_format:
        return True                                   # 搜索次数耗尽且格式无效
    if traj.search_calls == 0 and not traj.valid_format:
        return True                                   # 一次没搜且格式无效
    return False
```

三条规则覆盖 agent 训练里最常见的三类垃圾轨迹：无输出、搜索死循环、完全不搜。它们的共性是**对 GRPO 的组内比较毫无信息量**（甚至有害），所以要在 advantage 计算前剔除——与 DAPO 的 Dynamic Sampling（2.6.2）思路同源，只是作用在轨迹级而非题目级。

### 2.4.4 配套改进各改哪一环

**LLDS（`llds.py`）——改的是 loss。** 在 GRPO loss 上叠加单向似然保持正则：

    L_llds = λ · Σ max(0, log π_ref(token) − log π_θ(token))

只在当前策略似然**低于**参考策略时激活、只惩罚似然下降的 token（max(0,·) 单向）。三个变体：R（轨迹总似然下降才罚）、A（只罚参与训练的 token，推荐）、MA（只罚搜索/工具 token，保护搜索行为不被压缩）。作用：防 **LLD 死亡螺旋**——GRPO 训练中模型发现"少搜/不搜"能贪到格式奖励，搜索行为的似然被懒惰地位移走，能力崩塌（出处见 llds.py 引用的 Deng et al., arXiv 2512.04220）。λ 推荐 0.05~0.1。

死亡螺旋的机制值得展开讲：结果奖励是稀疏的（对/错），早期训练里"格式正确但答案错误"的轨迹也能靠格式部分拿到部分奖励；模型很快发现放弃搜索（把似然从"搜索 token"位移到"直接编答案 token"）能更快刷高奖励——GRPO 的优势只认奖励高低，不管似然从哪来。KL 惩罚本该拦这个，但 k3 是**对称**的：似然上升同样被罚，会拖慢正常学习；LLDS 用 max(0,·) 做成**单向**（只罚下降），且 A 变体只罚参与训练的 token、MA 变体专门盯住工具调用 token——精准打击"搜索行为被懒惰位移"这件事，而不妨碍正常的能力提升。项目实践里 LLDS v2 的批次方差大、正确率低点能自行恢复（3.1%→54.7%），说明它阻止了永久性崩溃。

**PRM-Lite（`reward_lite.py`）——改的是 reward。** 12 条惩罚规则 + 10 条奖励规则（共 22 条）的过程奖励启发式（如：空查询 −0.05、复制粘贴 −0.03、关键词查询 +0.03、综合多结果 +0.05），惩罚 cap −0.2、奖励 cap +0.2，最终 process_reward ∈ [−0.2, +0.2]，**叠加在结果奖励上**（t.reward += process_reward）。本质是廉价 PRM（过程奖励模型）：不训神经网络，用规则给"搜索过程是否合理"打分，缓解结果奖励稀疏。

**LATA（`reward_lite.py`）——改的是 advantage。** 长度自适应优势归一：

    advantage_lata = (reward − group_mean) / sqrt(L),   L = 该轨迹的 completion token 总数

除以 sqrt(L) 保护长推理链：长轨迹 token 多、按 token-mean 摊薄后每个 token 分到的优势变小，长链被系统性低估；sqrt 阶（而非线性阶）是温和的折中。⚠️ 管线顺序细节：主循环里 PRM-Lite 叠加发生在 compute_group_advantages **之后**，单独开 --prm-lite 时过程奖励只进入打印指标；开 --lata 时 advantage 被 LATA 重算（此时 reward 已含 PRM），过程奖励才真正进入 loss——面试时能指出这个顺序关系说明你真读过训练循环。

### 2.4.5 veRL 方向的工程结论与结果口径

项目同时完成了 veRL（verl）全参数训练管线的**环境调通**：agent loop 以 YAML 注册进 veRL 0.10 的 AgentLoopWorker（`verl_search_r1/agent_config.yaml`，deploy 配置收紧为每轮最多 1 次搜索、最多 3 个 assistant 轮次、搜索超时 15s，对比本地 QLoRA 管线的 4 次/6 轮——**收紧交互上限是控制 agentic rollout 时长的直接手段**），Search-R1 奖励函数按 veRL 0.10 的 keyword-only 调用约定封装（`verl_search_r1/reward_fn.py`，可开关 PRM-Lite/LATA）。并产出了 **TQ（tool-queue）瓶颈的定量分析**：每训练步约 51K 工具队列操作，处理能力 736 次/分钟，折算单步时 50-70 分钟。这是一个值得面试讲述的工程发现——**agentic RL 里工具调用吞吐（而非 GPU 计算）会成为训练的主瓶颈**，对应解法是 rollout 与训练解耦、工具调用异步化/批处理。

结果口径（诚实表述的范例）：本地 QLoRA 管线**训练批次内 correct_rate 峰值 81.2%（step 3 出现）**；同一管线、同一评估脚本下，**SFT 基线 dev 集 70 题 Macro EM 仅 1.43%**。这个"1.43% → 81.2%"的前后对比证明的是：RL 训练让模型**真正学会了搜索**（SFT 阶段只会背格式、不会调用搜索解决问题），而非宣称任何跨论文/留出集 SOTA。面试中主动交代"81.2% 是训练批次内指标、不是留出集成绩"，比报一个裸数字可信得多。

### 2.4.6 从复现中学到的 GRPO 调试顺序

项目经历沉淀出的排查顺序（面试问"GRPO 训练崩了怎么办"就按这个答）：

1. **看 anomalous 比例**：垃圾轨迹占比高 → 先修采样/协议解析，别急着调超参——垃圾进组，advantage 全失真；
2. **看组内奖励分布**：σ̂=0 的组占比（全对/全错）越大，有效梯度越少——对应 DAPO 的 Dynamic Sampling 思路；
3. **看 loss 与 reward 的共舞**：reward 涨、loss 涨是正常（优势加权下负 logprob 被抬）；reward 掉、loss 也掉才危险——可能正在坍缩到奖励黑客行为（对应 LLDS 要防的死亡螺旋）；
4. **看搜索行为统计**（项目里打印 mean_search）：搜索次数均值单调下滑是 LLD 的早期信号；
5. **看 KL/LLDS 惩罚趋势**：惩罚项持续为 0 说明信任域形同虚设（β 太小），持续暴涨说明学不动（β 太大）。

---

## 2.5 GRPO 的已知问题

**问题 1：长度偏差（length bias）。** 机制链条：

1. 奖励与长度**统计相关**：带推理链的正确回答天然更长，r 与 |o| 正相关；
2. 组内优势只依赖 r：长而对的回答拿正优势，短而错的拿负优势；
3. 优势**广播到该回答的每个 token**：长回答接收正优势的 token 总量远大于短回答，梯度总质量 ∝ |o_i| · Â_i；
4. σ̂ 归一化**放大这个效应**：优势量级取决于组内构成而非奖励绝对尺度——组内 {0, 0, 1} 时唯一的正确轨迹优势 ≈ +1.41（μ=0.33, σ̂=0.47），它有多长，就有多少 token 被 +1.41 抬升。

结果：策略被系统性推向"更长"——R1-Zero 训练后期著名的**长度爆炸**，Dr.GRPO 的诊断是它在相当程度上是优化伪影，而非纯粹的能力涌现。另一个方向的偏差同样存在：标准 GRPO 的 1/|o_i| token-mean 归一，对**长而错**的回答稀释了惩罚（负优势被 |o_i| 摊薄），策略学会了"错了就啰嗦"。**修法谱系**：Dr.GRPO 与 DAPO 都直接去掉 1/|o_i|（2.6.1/2.6.2）；DAPO 额外用 Overlong Reward Shaping 在奖励侧软惩罚超长；你项目里的 LATA 则走第三条路——保留归一化但在 advantage 侧除以 sqrt(L) 做温和补偿。三种修法分别改 loss、reward、advantage 三个环节，正好是理解"RL 栈有哪些可调旋钮"的好例子。

**问题 2：方差与 σ̂ 的脆弱性。** 组内比较的方差随 G 减小而急剧增大（μ̂、σ̂ 都是小样本估计）；0/1 奖励下 σ̂ ≈ sqrt(p(1−p))，组内**全对或全错**时 σ̂→0——优势要么全 0（无梯度）、要么靠 1e-8 硬撑导致微小奖励差异被放大成巨大优势。Dr.GRPO 还指出：除以 σ̂ 会把优化目标扭曲成**难度加权的成功率**（σ̂ 小的题目——太简单或太难——被过度加权），偏离"直接最大化成功率"。

**问题 3：KL 系数 β 敏感。** β 过大 → 策略被钉在 π_ref 上，学不动；β 过小 → clip 又只在 ρ 上有约束（单次回放 ρ=1 时 clip 完全失效），策略漂移、熵坍缩、奖励黑客接踵而至。DeepSeekMath 的 β=0.04 是数学任务上的经验值，换任务通常要重调——这也是 DAPO、VAPO 等变体各自改造 KL 的原因之一。实践中常见替代：你项目里用**单向的 LLDS** 替代对称 KL（2.4.4）；也有工作干脆去掉 KL、只靠 clip + 少量回放（每个 rollout 只训练 1~2 epoch）控制漂移——**"回放次数"本身就是最有效的信任域参数**。

---

## 2.6 2025 年的三个重要变体

### 2.6.1 Dr.GRPO（arXiv 2503.20783, Liu et al., Sea AI Lab）

**出处**：《Understanding R1-Zero-Like Training: A Critical Perspective》，对 R1-Zero 式训练做批判性剖析，指出 GRPO 目标里藏着的两类偏差，并给出"做对了的 GRPO"（Dr. = done right）。

**诊断的两个偏差**：

- **响应级长度偏差**：1/|o_i| 归一使正优势短回答获得更大 per-token 更新、负优势长回答惩罚被稀释——推着策略往"长且错"走；
- **问题级难度偏差**：除以组内 σ̂ 后，σ̂≈0 的题目（全对/全错）优势被过度放大，中等难度题被冷落。

**核心改动**：

1. **去掉 1/|o_i|**：token 级损失改用全局常量归一（如最大生成长度预算），不再除以每条回答自身长度；
2. **去掉 std 归一化**：优势回到纯均值中心化 Â_i = r_i − μ，恢复"带基线的蒙特卡洛回报"估计器；
3. **减去 token 级基线**（方差削减）：

       b_t = (1/G) · Σ_j log π_old(o_{j,t} | q, o_{j,<t})

   即组内所有回答在第 t 个位置上的旧策略 logprob 均值。由于 b_t 只依赖前缀（组内其他轨迹的同位置 token 与当前 token 独立），由 score function 恒等式 E[b_t · ∇ log π_θ(o_{i,t})] = 0，把它从优势中减去**不破坏无偏性**，同时削减逐 token 梯度的方差。

**公式对比**（忽略 clip/KL 的骨架）：

    标准 GRPO :  L = −(1/G) · Σ_i Â_i · (1/|o_i|) · Σ_t log π_θ(o_{i,t}),   Â_i = (r_i − μ)/σ̂
    Dr.GRPO   :  L = −(1/G) · Σ_i Σ_t (Â_i − b_t) · log π_θ(o_{i,t}),        Â_i = r_i − μ

**解决了什么**：梯度对"回答长度"不再有系统性偏置，长度爆炸显著缓解、token 效率提升，且优势不再被 σ̂ 扭曲。⚠️ 注意：不同实现（如 TRL 的 dr_grpo）在聚合方式上有出入（TRL 用全局常量归一、不实现 b_t），面试答公式建议以原论文 Eq. 为准并说明这两条改动主线。

### 2.6.2 DAPO（arXiv 2503.14476, ByteDance Seed）

**出处**：《DAPO: An Open-Source LLM Reinforcement Learning System at Scale》。开源复现报告：**Qwen2.5-32B 在 AIME 2024 上达到 50 分（论文自报成绩）**。四大技术：

1. **Clip-Higher**——改的是 loss 的 clip 参数。对称 clip [1−ε_low, 1+ε_high] 改成**非对称**，上界放宽（论文设定 ε_low=0.2, ε_high=0.28 ⚠️）。动机：训练后期**熵坍缩**——策略对某些 token 已经极度确定，概率极低的好 token 要涨回去需要大更新步，而 1+ε 上界把单步涨幅卡死，多样性永久丧失；放宽上界让低概率 token 有恢复机会。机制细节：概率要从 0.001 涨到 0.01 需要 ρ=10，但对称 clip 只允许每步 ρ≤1.2，要几十步才回得来；ε_high 放宽到 0.28 也只是缓解，熵坍缩的根治还要靠奖励设计与采样温度配合。
2. **Dynamic Sampling**——改的是采样/数据。组内所有奖励相等（σ̂=0，全对或全错）时该组优势全 0、不产生梯度，**把这题从批次中过滤掉**、把采样预算让给有信号的题。直接对应 2.5 问题 2 里"σ̂=0 无梯度"的浪费。
3. **Token-Level Loss**——改的是 loss 的归一化。**去掉 1/|o_i|**：L = −(1/Σ_i 1) · Σ_{i,t} min(ρ_t Â_i, clip(ρ_t, 1−ε_low, 1+ε_high) Â_i)，每个 token 权重相同，长链不再被摊薄。与 Dr.GRPO 同一判断、不同解法（DAPO 不做 token 基线、保留 clip）。
4. **Overlong Reward Shaping**——改的是 reward。不再对超长回答做硬截断（截断会把半成品奖励当完整回答判分，噪声大），改为软惩罚：

       R ← R + min(0, (L_max − L)/K)      （L > L_max 时按超出量线性扣分，K 控制软硬 ⚠️ 具体值以原论文为准）

   并额外配套过滤"生成截断 + 奖励却达标"的噪声样本（论文的 overlong filtering）。

一句话记忆：**Clip-Higher 救熵坍缩，Dynamic Sampling 省无信号样本，Token-Level Loss 去长度摊薄，Overlong Shaping 替硬截断**。

### 2.6.3 VAPO（arXiv 2504.05118, ByteDance Volcano Engine）

**出处**：《VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks》，代表 **value-based 路线的回归**——GRPO 与 critic 不是二选一，可以各取其长。四个组件：

1. **价值模型预训练**：VAPO 重新引入价值模型，但**从策略模型初始化**并在 RL 前做一段 value warmup 预训练（用 rollout 数据把 V 先训到位），缓解 critic 冷启动的方差与不稳定 ⚠️（warmup 步数等细节以原论文为准）。
2. **Decoupled GAE**：把 GAE 的 λ 拆成两个——**策略侧 λ_policy = 1.0**（纯蒙特卡洛、无偏但高方差，策略梯度要无偏性）；**价值侧 λ_value = 0.95**（低方差 GAE，价值函数要学得稳）。一个 λ 无法同时满足两个需求，解耦后各得其所。
3. **Length-Adaptive GAE**：λ_t 不再全程常数，而是**随 token 位置自适应**（靠前的 token 多用 value 引导、越接近答案越趋向纯 MC）⚠️（λ_t 的具体闭式以原论文为准）。直觉：结尾的 token 离奖励近，MC 回报可信；开头 token 离奖励远，靠学习到的价值函数做信用分配更稳。
4. **Hybrid advantage**：最终优势取 **GRPO 式组内优势与 GAE 优势的混合**：Â_t = c · Â_GAE + (1−c) · Â_GRPO ⚠️（混合权重与调度以原论文为准），把"组内比较的无偏"与"价值函数的低方差"叠加。

一句话记忆：**VAPO = 预训练过的轻量 value 模型 + 策略/价值 λ 解耦 + λ 随位置衰减 + GRPO/GAE 混合优势**。

补充 VAPO 的动机一句话：推理任务的长链（几百上千 token）让纯蒙特卡洛的方差和纯 value 的偏差都不可接受——MC 把奖励责任平均摊给每个 token（责任稀释），value 的复合误差沿链累积。VAPO 的答案是**让价值函数接管"长距离信用分配"**（低方差），**让蒙特卡洛接管"近奖励端的无偏信号"**（Length-Adaptive λ_t 的直觉所在），再用 GRPO 式组内信号做正交的方差削减。它证明了 2025 年的收敛方向：**GRPO 与 PPO 不是对立流派，而是同一 RL 栈上可拆卸的组件**。

**三个变体速查表**：

| 变体 | 改的环节 | 核心改动 | 解决的问题 |
|---|---|---|---|
| Dr.GRPO | advantage + loss | 去 σ̂ 归一、去 1/|o_i|、加 token 基线 b_t | 长度偏差、难度偏差、方差 |
| DAPO | loss + 采样 + reward | Clip-Higher、Dynamic Sampling、Token-Level Loss、Overlong Shaping | 熵坍缩、无信号组、长链摊薄、超长噪声 |
| VAPO | advantage + value | value 预训练、Decoupled GAE、Length-Adaptive λ_t、混合优势 | 无偏与低方差不可兼得、长链信用分配 |

---

## 2.7 PPO vs GRPO 对比表

| 维度 | PPO | GRPO | 说明 |
|---|---|---|---|
| Critic | 需要（同规模价值网络） | 不需要 | GRPO 的核心卖点 |
| Ref 模型 | 无（KL 对旧策略，靠 clip 约束） | 需要 π_ref（冻结，存 logprob 即可） | GRPO 的 ref 零梯度、显存开销小 |
| 显存 | 策略 + critic + ref≈3 份 | 策略 + ref≈2 份 | GRPO 常能上一档模型尺寸 |
| 优势估计 | GAE：低方差、有函数近似偏差 | 组内 MC：无偏、高方差 | 方差靠 G 补偿（G=64） |
| 偏差 | value 偏差 + GAE λ 偏差 | 长度偏差 + σ̂ 难度偏差 | 见 2.5、2.6 |
| Token 级信用分配 | 有（价值函数沿序列折现） | 无（序列优势广播） | GRPO 假设序列级奖励够用 |
| 适用任务 | 开放任务：人类偏好、LLM-as-judge，奖励稀疏、需要信用分配 | 可验证任务：数学/代码/搜索问答，规则可判对错 | Search-R1 属于后者 |
| 奖励形态 | 偏好模型逐 token 打分 | 结果奖励（EM/格式规则） | 自动判分是 GRPO 成立前提 |

一句话结论：**自动判分任务选 GRPO（省一半显存、免训 critic）；开放偏好任务 PPO/GAE 仍有价值；2025 年的 DAPO/VAPO 说明两条路线正在互相借鉴**。

---

## 面试常见追问（答好即拉开差距）

**Q1：GRPO 的 ref 模型和 PPO 的 critic 有什么区别？** ref 是**冻结的参考策略**，只在 KL 项里提供 logprob（零梯度、零优化器状态）；critic 是**要训练的第二个网络**，有独立的 value loss 和优化器。所以 GRPO 省掉的显存主要来自 critic 的参数+梯度+优化器状态（约省 1/3 训练显存）。

**Q2：为什么 GRPO 里 advantage 要除以 std？** 两个理由：奖励尺度无关（超参可迁移）；组内对错混杂时按离散度自动调节更新强度。代价是 E[1/σ̂]≠1/σ 的 Jensen 偏差 + σ̂≈0 时的数值问题 + 难度偏差。Dr.GRPO 的选择是干脆不除。

**Q3：clip 和 KL 都限制策略变化，重复吗？** 不重复。clip 限制**新旧策略概率比**（数据回放视角，ρ≠1 时才有效）；KL 限制**当前策略与初始参考策略**的距离（覆盖整个训练期）。单次回放时 clip 退化为无约束，KL 是唯一的信任域——这就是为什么很多实现"去掉 clip 但保留 KL（或其替代品 LLDS）"是自洽的。

**Q4：GRPO 能处理序列级奖励之外的东西吗？** 能，但需要额外设计。过程奖励（PRM）可以叠加进 r_i（你项目的 PRM-Lite）；多目标奖励（正确性+格式+搜索次数）通常在 reward 侧加权求和。但 token 级信用分配 GRPO 本身不给——这是 VAPO 重新引入 value 模型的原因。

**Q5：G 怎么选？** G 越大基线越准、方差越低，但 rollout 成本线性上升（token 数 × G）。DeepSeekMath G=64（配合其计算规模）；你项目在 12GB 卡上 G=8、每步 8 题共 64 条轨迹，是小算力下的折中。实践中 G=8~16 是小模型常见档位 ⚠️（具体取值随任务与算力而定）。

## 面试速记卡（10 条）

1. GRPO 出处：DeepSeekMath（2402.03300）；动机：PPO 的 critic 太贵，用组内比较替代价值函数。
2. 优势公式：Â_i = (r_i − μ)/(σ̂ + ε)，μ 是 sample-mean baseline 的蒙特卡洛估计；组均值基线梯度无偏（差 (G−1)/G 常数缩放）；除以 σ̂ 排序不变=组级自适应学习率，但引入 Jensen 偏差。
3. Loss 骨架：−(1/Σ|o_i|) Σ min(ρÂ, clip(ρ,1±ε)Â) − β·KL；ρ 首轮回放=1（clip 失效）；clip 是隐式信任域，KL（k3 估计器）是显式信任域。
4. k3 无偏：E[e^x − x − 1] = KL，靠 E_πθ[π_ref/π_θ] = 1 这条重要性采样恒等式。
5. 三大已知问题：长度偏差（优势广播 × |o|）、σ̂ 脆弱（全对/全错无梯度）、β 敏感。
6. Dr.GRPO：去 1/|o_i|、去 σ̂、加 token 基线 b_t=(1/G)Σ_j log π_old(o_{j,t})——无偏 + 低方差。
7. DAPO 四件套：Clip-Higher（ε 非对称救熵坍缩）、Dynamic Sampling（滤 σ̂=0 组）、Token-Level Loss（去归一）、Overlong Shaping（软惩罚替硬截断）；Qwen2.5-32B AIME 50（论文自报）。
8. VAPO：value 回归 + Decoupled GAE（策略 λ=1 / 价值 λ=0.95）+ Length-Adaptive λ_t + 混合优势。
9. 项目实践：异常轨迹排除后重算 mean/std、std+1e-8、shift 对齐、chunked log_softmax、单次回放时合法省略 clip/KL；LLDS 改 loss、PRM-Lite 改 reward、LATA 改 advantage。
10. 诚实口径：训练批次内 correct_rate 峰值 81.2%（step 3），同管线 SFT dev EM 1.43% 作前后对比；veRL 侧结论是环境调通 + TQ 吞吐瓶颈定量分析（51K 操作/步、736/min、步时 50-70min）。

## 参考文献

- DeepSeekMath（GRPO 出处）：arXiv 2402.03300
- DeepSeek-R1：arXiv 2501.12948
- Dr.GRPO（Understanding R1-Zero-Like Training）：arXiv 2503.20783
- DAPO（ByteDance Seed）：arXiv 2503.14476
- VAPO（ByteDance Volcano Engine）：arXiv 2504.05118
- LLDS（项目 llds.py 引用的论文）：Deng et al., "On GRPO Collapse in Search-R1: The Lazy Likelihood-Displacement Death Spiral", arXiv 2512.04220
- Schulman's blog "Approximating KL Divergence"（k3 估计器出处）
- 项目代码：C:\new\intern\plan\projects\search-r1\（train_grpo_local.py / llds.py / reward_lite.py）
# LLM 强化学习方法全景指南 · Part 3
# DPO 家族：直接偏好优化

> 目标读者：准备算法岗面试、学过概率论的同学。
> 前置知识：RLHF 三阶段（SFT → 奖励模型 → PPO）的基本概念；KL 散度；logistic 回归。
> 本章约定：πθ 表示待训练的策略（语言模型），πref 表示参考模型（一般冻结的 SFT 基座）；
> y_w 表示偏好对中"更受偏好"的回答，y_l 表示"不受偏好"的回答，x 表示 prompt；
> σ(z) = 1/(1+e^(−z)) 为 sigmoid；log 一律为自然对数。

---

## 3.0 导读：为什么需要 DPO

先回顾 RLHF 的两个痛点，这是理解 DPO 动机的前提。

**痛点 1：显式奖励模型 + 在线强化学习的工程成本高。**
经典 RLHF（PPO 路线）需要在显存里同时维护 4 个模型（策略、ref、奖励模型、价值模型），
要在训练中不断用当前策略采样 rollout 来估计梯度，还要处理优势估计、clip、
奖励归一化、KL 惩罚调度等一堆不稳定因素。复现难、超参敏感、算力贵。

**痛点 2：奖励模型的"奖励黑客"风险。** RM 是有误差的模型，PPO 会钻 RM 的空子，
产出 RM 给高分但人类觉得差的回答。

**DPO 的思路**（Rafailov et al., arXiv 2305.18290, NeurIPS 2023）：
RLHF 第二阶段那个带 KL 约束的优化问题，其实有闭式解——最优策略可以直接用奖励函数写出来。
把它反过来用：**奖励函数也可以用策略写出来**。既然如此，为什么还要显式地训练奖励模型？
直接把"策略"参数化，用偏好数据训练它，一步到位。

DPO 出现后，几乎所有开源模型团队（Zephyr、Mistral、Llama-3 系列等）都改用 DPO 系方法做对齐，
本章是面试最高频考点之一。

**本章路线图**：BT 偏好模型（3.1）→ RLHF 闭式解（3.2）→ 反解奖励（3.3）→ DPO 损失（3.4）
→ 与 RLHF 的关系（3.5）→ 代码（3.6）→ 六大变体（3.7）→ 优缺点与对比表（3.8–3.9）→ 面试速记（3.10）。

---

## 3.1 前置模型：Bradley-Terry 偏好模型

### 3.1.1 形式

人类标注通常给不出绝对分数，但能给**成对比较**："对问题 x，回答 A 比回答 B 好"。
Bradley-Terry（BT）模型（Bradley & Terry, 1952；RLHF 场景最早由 Christiano et al. 2017、
InstructGPT 采用）用一个隐式分数 r(·)（在 RLHF 里就是奖励模型输出）来描述比较概率：

```
P(y1 ≻ y2 | x) = exp(r(y1)) / ( exp(r(y1)) + exp(r(y2)) ) = σ( r(y1) − r(y2) )
```

中间那步：分子分母同除以 exp(r(y1))，得到 1 / (1 + exp(r(y2) − r(y1))) = σ(r(y1) − r(y2))。

### 3.1.2 为什么是这个形式

**(a) 偏好天然是相对的，BT 只依赖相对差。**
给所有 r 同时加一个常数 c：exp 形式分子分母同乘 exp(c)，概率不变。
也就是说"绝对有多好"不可观测、也不该影响模型，"A 比 B 好多少"才是数据里真正含有的信息。
这正是后面 DPO 里"Z 项被约掉"的伏笔。

**(b) 对称性 / 互补性。** 因为 σ(−z) = 1 − σ(z)：

```
P(y2 ≻ y1) = σ(r(y2) − r(y1)) = 1 − σ(r(y1) − r(y2)) = 1 − P(y1 ≻ y2)
```

两个方向的概率互补——不是 y1 赢就是 y2 赢，符合直觉。

**(c) 饱和性刻画人类比较行为。** 分差 → +∞ 时 P → 1（悬殊时几乎确定）；
分差为 0 时 P = 1/2（五五开）。sigmoid 的形状正好表达"越悬殊越确定、越接近越犹豫"。

**(d) 指数得分的占比解释。** 令 w(y) = exp(r(y))，则 P(y1 ≻ y2) = w(y1)/(w(y1)+w(y2))，
即"候选的指数得分占两候选总得分的比例"——一个天然的软 max 比较机制。

### 3.1.3 偏好数据就是二元分类数据

对一条偏好对 (x, y_w, y_l)，BT 模型的对数似然是：

```
log P(y_w ≻ y_l | x) = log σ( r(y_w) − r(y_l) )
```

这就是一个 logistic 回归的对数似然：把 (r(y_w), r(y_l)) 当作特征，标签是"y_w 赢"。
**RLHF 的第一阶段（训练 RM）本质上就是在解这个分类问题。**
DPO 的洞察是：这个分类问题的"特征"不一定非要用显式 RM 提供——3.3 节会看到，
策略本身就能给出这个特征。

---

## 3.2 RLHF 最优策略的闭式解

### 3.2.1 问题设定

RLHF 第二阶段（强化学习阶段）要解的问题：给定奖励 r(y)（RM 输出或环境反馈），
在"不要偏离 πref 太远"的约束下最大化期望奖励：

```
max_π   E_{y~π}[ r(y) ] − β · KL( π ‖ πref )

其中  KL( π ‖ πref ) = Σ_y π(y) · log( π(y) / πref(y) )
```

β > 0 是 KL 惩罚系数：β 大 → 偏离 ref 的代价高 → 策略更保守；β 小 → 更激进地追逐奖励。

### 3.2.2 拉格朗日推导（逐步骤注释）

策略 π 是一个概率分布，满足归一化约束 Σ_y π(y) = 1。用拉格朗日乘子法：

```
第一步：写出拉格朗日函数（λ 惩罚违反归一化的程度；λ 前的正负号是约定，不影响结果）

  L(π, λ) = Σ_y π(y)·r(y) − β·Σ_y π(y)·log(π(y)/πref(y)) + λ·( 1 − Σ_y π(y) )

第二步：对固定的某个 y，对 π(y) 求偏导并置零

  ∂L/∂π(y) = r(y) − β·[ log(π(y)/πref(y)) + π(y)·(1/π(y)) ] − λ
           = r(y) − β·[ log(π(y)/πref(y)) + 1 ] − λ  =  0

  （用了乘积函数求导：d/dπ [ π·log(π/πref) ] = log(π/πref) + π·(1/π) = log(π/πref) + 1）

第三步：整理出 π(y)

  log( π(y)/πref(y) ) = ( r(y) − λ − β ) / β
  π(y) = πref(y) · exp( r(y)/β ) · exp( −(λ+β)/β )

第四步：exp(−(λ+β)/β) 与 y 无关，记作 1/Z，Z 由归一化约束确定

  π*(y) = πref(y) · exp( r(y)/β ) / Z
  Z = Σ_{y'} πref(y') · exp( r(y')/β )        ← 分区函数（配分函数）
```

### 3.2.3 解的解读

- **指数倾斜（exponential tilt）**：以 πref 为"先验"，按 exp(r/β) 逐项放大；
  奖励越高放大越多。等价地写：π* = softmax( log πref + r/β )——就是一个带先验的 softmax。
- **β 是温度**：β → 0 时 π* 收敛到"只取 r 最高的 y"（贪婪）；β → ∞ 时 π* → πref（完全不动）。
- **能量模型同构**：把 −r/β 看作能量，π* 就是玻尔兹曼分布。

### 3.2.4 第二种推导：信息论视角一眼看出答案（变分法）

把目标函数"配方"成 KL 散度：

```
定义  p(y) := πref(y)·exp(r(y)/β) / Z ，其中 Z = Σ_{y'} πref(y')·exp(r(y')/β)

KL( π ‖ p ) = Σ_y π(y)·log( π(y)·Z / (πref(y)·exp(r(y)/β)) )
            = Σ_y π(y)·[ log(π(y)/πref(y)) − r(y)/β + log Z ]
            = KL( π ‖ πref ) − E_π[r]/β + log Z

两边乘 β 再移项：

E_π[r] − β·KL( π ‖ πref ) = β·log Z − β·KL( π ‖ p )
```

β·log Z 与 π 无关，所以最大化原目标 ⟺ 最小化 KL(π ‖ p)。
KL ≥ 0，且取 0 当且仅当 π = p。**结论：最优策略就是 p 本身**，一行证完。

```
π*(y) = πref(y) · exp( r(y)/β ) / Z
```

这个证明在面试里说出来很加分：它把"带 KL 约束的奖励最大化"翻译成了
"向目标分布 πref·exp(r/β) 做 KL 投影"。

---

## 3.3 反解奖励函数：策略即奖励

### 3.3.1 从最优策略反解 r

3.2 的闭式解两边取对数、乘 β、移项：

```
π*(y) = πref(y) · exp(r(y)/β) / Z

log π*(y) = log πref(y) + r(y)/β − log Z

r(y) = β · log( π*(y) / πref(y) ) + β · log Z          （*）
```

注意 β·log Z 与 y 无关——奖励被反解到了"一个与 y 有关的项 + 一个常数平移"。
常数平移不影响任何偏好比较（回忆 3.1.2 的性质 a）。

### 3.3.2 关键引理：任意策略都对应一个奖励（重参数化）

上面是从"最优解"反解。更妙的观察是：**对任意策略 π（不必最优），
都存在一个奖励 r_π，使 π 恰好是以 r_π 为奖励的 RLHF 目标的最优解。**

```
对任意 π，定义  r_π(y) := β·log( π(y)/πref(y) ) + β·log Z
验证：π 是不是 r_π 下 RLHF 目标的最优解？
按 3.2 的闭式解，r_π 的最优策略应为  πref·exp(r_π/β) / Z'
其中 Z' = Σ πref·exp(r_π/β) 是新奖励下的分区函数。

  exp( r_π(y)/β ) = ( π(y)/πref(y) ) · Z          （把 r_π 的定义代回去）
  Z' = Σ_y πref(y)·( π(y)/πref(y) )·Z = Z·Σ_y π(y) = Z

  πref(y)·exp(r_π(y)/β) / Z' = πref(y)·(π(y)/πref(y))·Z / Z = π(y)   ✓

所以 π 确实是 r_π 的最优策略，验证完毕。
```

**含义（DPO 的总开关）**：策略空间与奖励空间（差一个常数平移）一一对应。
"你是什么样的策略，就等价于你在优化什么样的奖励"——策略自己"携带"奖励信息，
不需要单独的奖励模型去学。这就是 DPO 能把奖励模型整个删掉的根基。

---

## 3.4 DPO 完整推导（Rafailov et al., arXiv 2305.18290）

### 3.4.1 三步总览

1. BT 模型把偏好概率写成奖励差的 sigmoid。
2. 用 3.3 的反解把奖励换成"策略的对数似然比"，分区函数 Z 相减抵消。
3. 把最优策略直接参数化成 πθ，对偏好数据做最大似然。

### 3.4.2 代入与 Z 的约掉（逐步）

```
第一步（BT）：      p*(y_w ≻ y_l | x) = σ( r(y_w) − r(y_l) )

第二步（反解）：    r(y) = β·log( π*(y|x)/πref(y|x) ) + β·log Z(x)
                  （同一 prompt x 下，Z(x) 只依赖 x、不依赖 y）

第三步（代入）：

  p*(y_w ≻ y_l | x)
  = σ( [ β·log(π*(y_w)/πref(y_w)) + β·log Z(x) ] − [ β·log(π*(y_l)/πref(y_l)) + β·log Z(x) ] )
  = σ( β·log( π*(y_w)/πref(y_w) ) − β·log( π*(y_l)/πref(y_l) ) )

Z 约掉的原因：两项里是同一个 Z(x)，做差为 0。
直觉：偏好只取决于相对差；Z 是常数平移量，平移不影响比较（呼应 3.1.2(a)）。
```

### 3.4.3 最终损失

把上式里的最优策略 π* 直接换成待训练的参数化策略 πθ，对数据集 D 取负对数似然：

```
L_DPO(πθ; πref) = −E_{(x,y_w,y_l)~D} [ log σ( β·log(πθ(y_w|x)/πref(y_w|x))
                                              − β·log(πθ(y_l|x)/πref(y_l|x)) ) ]
```

三个视角理解它（面试都要能说）：

1. **隐式奖励**：r_θ(x,y) := β·log( πθ(y|x)/πref(y|x) ) 就是 πθ 隐含的奖励。
   DPO 训练 = 让隐式奖励在偏好数据上把 BT 分类做好。
2. **监督损失**：没有采样、没有价值函数、没有奖励模型，纯梯度下降即可训练。
3. **RLHF 的等价重参数化**：损失最小化时 πθ → π*，而 π* 正是 RLHF 目标的最优解
   （论文 Theorem 1：在 BT 假设 + πref 全支撑 + 容量足够的条件下，
   L_DPO 的全局最优解等于 RLHF 的最优策略）。

**严格性备注**：反解公式（*）严格说只在 πθ = π* 时精确成立。训练中 πθ ≠ π*，
此时按 3.3.2 的引理，πθ 仍对应一个隐式奖励 r_θ(y) = β·log(πθ/πref) + β·log Z_θ，
但 Z_θ 随 θ 变化——DPO 相当于在优化一个"移动靶"奖励。实践中它足够好用，
但这是 DPO 一切局限的根源（3.5 节细讲）。

### 3.4.4 梯度分析

```
记  Δ := β·[ log(πθ(y_w)/πref(y_w)) − log(πθ(y_l)/πref(y_l)) ]
则  L = −log σ(Δ)

dL/dθ = −( σ'(Δ)/σ(Δ) ) · dΔ/dθ
      = −(1 − σ(Δ)) · dΔ/dθ            （σ'(z) = σ(z)(1−σ(z))，所以 σ'/σ = 1−σ）

又  dΔ/dθ = β·[ ∇_θ log πθ(y_w) − ∇_θ log πθ(y_l) ]   （πref 与 θ 无关）

∇_θ L_DPO = −β · σ(−Δ) · [ ∇_θ log πθ(y_w) − ∇_θ log πθ(y_l) ]
```

两点解读：

- **权重 σ(−Δ) = 模型当前认为 "y_l 赢" 的概率**。模型已经强烈偏好 y_w 时权重 ≈ 0
  （该样本几乎零梯度，自动"课程"）；模型搞反了时权重 ≈ 1，全力纠错。
- **形式与 RLHF 策略梯度同构**：抬高好的、压低坏的，按"当前错多少"加权——
  这正是带优势函数的 policy gradient 的样子；区别是数据 offline、不需要采样。

另外注意：πref 的比值项不产生梯度，它只作为每条样本的常数截距，
把 sigmoid 的工作点移到合适位置——它起的是"锚点/正则"作用。

---

## 3.5 DPO 与 RLHF 的关系

### 3.5.1 什么时候等价

在以下理想条件下，DPO 与 RLHF 解的是**同一个优化问题**：

1. 偏好数据确实由某个奖励 r* 的 BT 模型生成（数据"诚实"地反映一个打分）；
2. πref 与 π* 满足支撑条件（πref 处处为正，π* 在 πθ 的函数类内）；
3. 数据量足够、模型容量足够。

此时 L_DPO 的全局最优 = RLHF 目标的最优策略 π*。
**一句话：DPO 不是换了个目标，而是换了个求解算法**——把"先学奖励再最大化"
换成了"直接参数化最优策略做分类"。等价性完全系在 BT 假设和"数据反映奖励"上。

### 3.5.2 什么时候不等价

**(1) 偏好数据来自 πref 之外（off-policy 漂移）——实践中最重要的不等价。**
DPO 的隐式奖励 r_θ = β·log(πθ/πref) 是**以 πref 为锚**的：它衡量的是"相对 πref 好多少"。
如果偏好数据是由另一个策略（比如更强/更早版本的模型，或别家的 API）生成的，
那么标注里隐含的"更好"基准与 πref 不一致。训练中 πθ 又离 πref 越来越远，
隐式奖励的读数越来越失真——损失还在降，优化的目标却已经偏了。
RLHF 的在线采样天然 on-policy，没有这个问题。

**这就是 iterative DPO / online DPO 存在的直接原因**：每轮用当前策略重新采样偏好对
（用外部 RM 或人工标注），并把 πref 更新为上一轮策略，让数据回到 on-policy、
锚点跟上策略。

**(2) 隐式奖励漂移（训练动态上的不等价）。**
如 3.4.3 所述，Z_θ 随 θ 变，DPO 优化的是移动靶；RLHF 的 RM 在一段训练内是固定目标。
当 πθ 与 πref 相差很远时，β·log(πθ/πref) 也不再是真实奖励的良代理。

**(3) BT 假设。**
人类偏好存在不可传递性（A≻B、B≻C 但 C≻A）时，BT 模型本身不成立，
DPO 的理论根基动摇（IPO 部分绕开 BT，SPPO 明确面向一般偏好）。

**(4) 探索能力。**
DPO 完全 offline，学不出数据分布之外的新行为；RL（PPO）可以通过试错发现新策略。
这是"天花板"差异，不是"下限"差异。

### 3.5.3 面试题："DPO 到底是不是 RL？"

推荐的三段式答案：

1. **目标层面：是。** DPO 优化的是与 RLHF 完全相同的 KL 约束期望奖励目标，
   它只是用"闭式解 + 重参数化"把 RL 问题改写成了监督学习损失；
   它的梯度与带优势的策略梯度同构。
2. **算法层面：不是经典 RL。** 无环境交互、无采样、无价值函数、无时间步——
   训练时它是一个排序/分类损失，用普通梯度下降。
3. **一句话总结：DPO 是"RL 目标的监督式求解器"。**
   为什么可以不用采样？因为 RLHF 唯一需要采样估计的量——KL 约束下的最优策略——
   恰好有闭式解；DPO 直接参数化这个闭式解，把采样"解析地做掉了"。
   等价性依赖 BT 假设与 on-policy 数据，违反时代价是目标漂移而非算法崩溃。

---

## 3.6 PyTorch 伪代码（可运行简化版）

```python
import torch
import torch.nn.functional as F

def dpo_loss(pi_logps_chosen, pi_logps_rejected,
             ref_logps_chosen, ref_logps_rejected, beta=0.1):
    """DPO 损失。四个输入的 shape 都是 (B,)，为每条回答逐 token 求和后的对数概率。

    公式: −log σ( β·[ (log πθ(yw) − log πθ(yl)) − (log πref(yw) − log πref(yl)) ] )
    """
    pi_ratio  = pi_logps_chosen  - pi_logps_rejected   # log [πθ(yw)/πθ(yl)]
    ref_ratio = ref_logps_chosen - ref_logps_rejected  # log [πref(yw)/πref(yl)]
    logits = beta * (pi_ratio - ref_ratio)
    # F.logsigmoid(x) = log σ(x)，数值上比 log(sigmoid(x)) 稳定
    loss = -F.logsigmoid(logits)
    return loss.mean()

# —— 可运行的玩具示例 ——
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4
    pi_w,  pi_l  = torch.randn(B), torch.randn(B)
    ref_w, ref_l = torch.randn(B), torch.randn(B)
    print("DPO loss =", dpo_loss(pi_w, pi_l, ref_w, ref_l).item())

    # 梯度方向检查：提高 yw 的对数概率应当降低 loss（∂L/∂logπθ(yw) ≤ 0）
    pi_w_grad = torch.randn(B, requires_grad=True)
    dpo_loss(pi_w_grad, pi_l, ref_w, ref_l).backward()
    print("∂L/∂logπθ(yw) 全部 ≤ 0:", bool((pi_w_grad.grad <= 0).all()))
```

实现备注：真实训练中 πθ 和 πref 的对数概率是对每个 token 求和（SimPO 用平均，见 3.7.4）；
πref 冻结不回传梯度；数值稳定用 logsigmoid（等价于 −softplus(−x)）。

---

## 3.7 变体

### 3.7.1 IPO：平方损失防过拟合

- **出处**：Azar et al., "A General Theoretical Paradigm to Understand Learning from
  Human Preferences"（arXiv 2310.12036，AISTATS 2024）。提出了统一的 Ψ-PO 框架，
  IPO 是其中 Ψ 取恒等函数（identity）的特例。
- **核心公式**：

```
h_π(y_w, y_l) := log( πθ(y_w|x)/πref(y_w|x) ) − log( πθ(y_l|x)/πref(y_l|x) )

L_IPO = E_(x,y_w,y_l)~D [ ( h_π(y_w, y_l) − 1/(2τ) )² ]
```

- **解决了 DPO 的什么问题**：
  1. **防过拟合**：DPO 的 −log σ 会把对数似然比往 +∞ 推——当偏好接近确定性时，
     sigmoid 饱和、梯度衰减慢，比值无上限地涨（chosen 概率发散、KL 失控）。
     IPO 用平方损失把 h 拉向**有限目标** 1/(2τ)，比值不会爆炸。
  2. **去掉 BT 假设**：Ψ = identity 时不需要假设偏好由 BT 模型生成（论文称
     "preference-model-free"），对偏好模型失配更鲁棒。
  3. τ 是 KL 正则系数：τ 越大目标 1/(2τ) 越小 → 允许偏离 πref 的幅度越小。
- **缺点**：目标与偏好强度无关——强偏好和弱偏好的 gap 都被拉向同一个值，
  可能欠拟合强信号；仍需 ref 模型；τ 需要调；DPO 论文之后有分析指出平方损失在
  接近最优时收敛较慢 ⚠️（此点属综述性转述，待核实）。

### 3.7.2 KTO：不需要偏好对，只要好/坏标签

- **出处**：Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic
  Optimization"（arXiv 2402.01306，ICML 2024）。理论来源是 Kahneman-Tversky
  前景理论（1979 年 Econometrica 论文，非 arXiv）。
- **前景理论一分钟**：人的效用不是绝对量的函数，而是**相对参照点**的损益的函数；
  价值函数在收益区凹（风险规避）、损失区凸（风险追求），且**损失比等量收益更痛**
  （损失厌恶，λ ≈ 2.25）。原始价值函数形式：v(z) = z^α（z ≥ 0）、−λ(−z)^α（z < 0），
  论文报告的拟合参数 α ≈ 0.88、λ ≈ 2.25。
- **KTO 的映射**：把隐式奖励 r_θ(x,y) = log( πθ(y|x)/πref(y|x) ) 当作"结果"，
  参照点 z_ref 当作"现状"，desirable 样本是收益、undesirable 样本是损失：
  收益要尽量高于参照点、损失要尽量低于参照点，且两边的权重可以不对称。
- **核心公式**：

```
L_KTO = E_(x,y)~D [ λ_y − v(x,y) ]

v(x,y) = λ_D · σ( β·( r_θ(x,y) − z_ref ) )        y 是"好"回答时
       = λ_U · σ( β·( z_ref − r_θ(x,y) ) )        y 是"坏"回答时

z_ref ≈ E_{x'~batch}[ KL( πθ(·|x') ‖ πref(·|x') ) ]
      （参照点：一个 batch 内无关样本上的平均 KL，不回传梯度；
       理论上是 π* 下的期望奖励，实践中用 KL 估计 ⚠️ 实现细节以原文/TRL 为准）
```

  λ_D、λ_U 分别是好/坏样本的权重（对应前景理论里损失权重大于收益的设定）；
  注意不同材料的符号约定不同（β 有时折进 r_θ），结构一致即可。
- **解决了 DPO 的什么问题**：**数据门槛**。只需要 (x, y, 好/坏) 二元标签——
  点赞点踩、评分阈值都能用，不需要昂贵的人工成对比较。论文报告在 1B–30B 规模上
  匹配或超过 DPO，且对噪声、不平衡标签更鲁棒（因为非对称权重可以按类平衡）。
  梯度特性：已"满意"的样本（好且已高于 z_ref、坏且已低于 z_ref）几乎零梯度。
- **缺点**：每条样本独立处理，丢掉了 DPO 偏好对里的相对信息（论文辩称影响不大，
  但直觉上信息量更少）；z_ref 的批量估计有方差、对 batch 组成敏感；λ_D、λ_U、β
  三个超参；对前景理论的借用偏启发式（sigmoid 替代了原始价值函数）。

### 3.7.3 ORPO：单阶段（SFT + 对齐合一），不需要 ref 模型

- **出处**：Hong, Lee, Thorne（KAIST AI），"ORPO: Monolithic Preference Optimization
  without Reference Model"（arXiv 2403.07691，EMNLP 2024）。
- **核心公式**：

```
odds_θ(y|x) := πθ(y|x) / ( 1 − πθ(y|x) )                （优势率：P 与"非 P"之比）

L_OR = −log σ( log( odds_θ(y_w|x) / odds_θ(y_l|x) ) )   （优势率比）

L_ORPO = L_SFT + λ · L_OR
L_SFT  = −log πθ(y_w|x)                                  （只在 yw 上的标准 NLL）
```

- **为什么用 odds ratio**：(a) odds 自带锚点——odds = 1 对应 P = 1/2，
  比值是与"自己"比较，因此**不需要 ref 模型**做基准（省一半前向/显存）；
  (b) 论文论证：概率比 πθ(y_w)/πθ(y_l) 与 SFT 损失叠加时会过度压低被拒回答、
  损害生成质量，odds ratio 的区分更温和。
  **单阶段**：SFT 项学"会说话"，L_OR 项学"分辨好坏"，一次训练完成，
  不需要先 SFT 再对齐的两段管线。论文报告在 7B 级模型上超过同规模 RLHF/DPO
  （AlpacaEval 2.0 达 12.20%、MT-Bench 7.32，论文报告值）。
  实现细节：论文建议 L_OR 只作用于模型的一小部分参数（"minor component"）⚠️ 待核实，
  以减少与 SFT 项的梯度冲突。
- **缺点**：log odds 在 P → 1 附近无界（饱和区梯度消失）；理论框架不如 BT 简洁
  （odds ratio 与 BT 的关系需要额外的桥接说明）；λ 与作用层数需要调；
  SFT 与对齐耦合，无法分别控制两阶段。

### 3.7.4 SimPO：长度归一化似然 + 目标边际，不需要 ref 模型

- **出处**：Meng et al.（普林斯顿，陈丹琦团队），"SimPO: Simple Preference
  Optimization with a Reference-Free Reward"（arXiv 2405.14734，NeurIPS 2024）。
- **核心公式**：

```
L_SimPO = −E [ log σ( (β/|y_w|)·log πθ(y_w|x) − (β/|y_l|)·log πθ(y_l|x) − γ ) ]
```

  论文默认 β ≈ 2、γ ≈ 1（实现中略有出入 ⚠️）。与 DPO 相比三处改动：
  ① 去掉 πref；② 似然除以长度取平均；③ 加目标边际 γ。

- **为什么除以 |y|**：log πθ(y|x) 是逐 token 对数概率之和，每项都 < 0，
  序列越长累积越负。以总和当奖励时，长回答天然"吃亏"——模型会学到长度偏差
  （倾向短回答，或反过来为了补偿学出冗长、重复的输出）。除以长度变成
  "每 token 平均似然"，奖励与长度解耦。论文消融显示这是最关键的一处改动：
  去掉长度归一化后输出质量显著退化。
  边际 γ 的作用：要求好回答的奖励比坏回答至少高出 γ，避免"略好也算赢"的
  无区分度区间（BT 变体：σ(r_w − r_l − γ)）。
- **解决了 DPO 的什么问题**：ref 模型的 2× 显存与计算开销；DPO 的长度偏差；
  以及 ref 引入的 Z(x) 漂移。
- **缺点**：没有 ref 就没有 KL 锚点，正则更弱、更容易过拟合/跑偏；
  γ 与 β 两个敏感超参（β 的尺度与 DPO 完全不同：无 ref 项衰减，所以是 2 而不是 0.1）；
  平均对数似然对极短输出噪声大；失去"别离初始模型太远"的显式保证。

### 3.7.5 SPPO：自博弈框架，收敛到 Nash 均衡

- **出处**：Wu, Sun, Yuan, Ji, Yang, Gu（UCLA/CMU），"Self-Play Preference
  Optimization for Language Model Alignment"（arXiv 2405.00675，2024）。
- **框架**：把对齐建模为策略与**自己**的两人**常和博弈**：
  payoff u(π, π') = E_{y~π, y'~π'} [ P(y ≻ y' | x) ]。
  常和性来自 P(y≻y') + P(y'≻y) = 1。目标不是 BT 的最大似然，而是逼近该博弈的
  **Nash 均衡**（von Neumann winner）——在一般（甚至不可传递的）偏好下也有定义，
  这是对 BT 可传递性假设的正面突破。
- **核心公式**（乘性权重更新的平方损失形式；用 log Z ≈ η/2 的近似 ⚠️ 原文推导）：

```
L_SPPO = E [ ( log( πθ(y_w|x)/πt(y_w|x) ) − 1/(2η) )²
           + ( log( πθ(y_l|x)/πt(y_l|x) ) + 1/(2η) )² ]

πt = 上一轮迭代的策略（自博弈对手）；实践中 η 常取 1（目标 ±1/2）⚠️ 以原文为准。
```

- **与 DPO 的目标函数区别**（面试重点）：
  1. DPO 只约束两侧的**差**（比值），单侧可以无界；SPPO 给两侧各自一个**有限绝对目标**
     （+1/(2η) 和 −1/(2η)），更新有界、稳定。
  2. DPO 一次性 offline；SPPO **迭代在线**：每轮用 πt 自己采样，逼近博弈均衡。
  3. DPO 假设 BT（偏好可传递）；SPPO 面向一般偏好收敛到 Nash。
  论文用小型成对 RM（PairRM，0.4B）估计偏好概率（win-rate），仅用 6 万 UltraFeedback
  提示，报告了当时 AlpacaEval 2.0 的 SOTA 长度控制胜率（论文报告）。
- **缺点**：多轮在线生成昂贵；自博弈稳定性（策略震荡）需要 η、轮数、早停调参；
  依赖偏好估计器（PairRM 类）质量；每轮只挪 1/(2η)，收敛慢。

### 3.7.6 一句话提及：iterative / online DPO

在 3.5.2(1) 的基础上：每轮用当前策略采样新偏好对并把 πref 更新为上一轮策略，
代表作 Xiong et al. 的 Iterative DPO（2023）等——本质是对抗 off-policy 漂移，
用"贵一点的在线数据"换"准一点的优化目标"。

---

## 3.8 DPO 家族的共同优缺点

**共同优点**

1. **无在线采样、无显式 RM（多数变体）**：训练便宜、稳定、易复现——
   这是工业界大规模采用的根本原因。
2. **目标直观**：梯度就是"升好降坏"，按当前错误程度加权，调试与解释都容易。
3. **直接吃现成偏好数据**：各大开源偏好数据集（UltraFeedback 等）即插即用。
4. **超参少**：核心就一个 β（再加各变体自己的 1–2 个）。

**共同缺点**

1. **offline 数据分布漂移**：πθ 偏离数据分布后梯度信号失真，单轮 DPO 无法自我纠偏
   （iterative/online 是补救而非根治）。
2. **天花板低**：没有探索，学不到数据分布之外的行为；上限被数据集质量锁死。
3. **偏好数据质量敏感**：标注噪声会被放大（DPO 尤其会把噪声对的比值推到无穷）；
   标签错误率直接影响对齐质量。
4. **依赖 BT 类假设**：偏好不可传递时理论失效（IPO/SPPO 部分缓解）。
5. **去掉 ref 的变体（ORPO/SimPO）正则更弱**：没有 KL 锚点，训练更易跑偏。

---

## 3.9 对比小表

| 方法 | 数据需求 | ref 模型 | 单阶段 | 主要改进 |
|------|----------|----------|--------|----------|
| DPO  | 偏好对 (y_w, y_l) | 需要 | 否（SFT 后对齐） | 隐式奖励，消掉 RM 与在线采样 |
| IPO  | 偏好对 | 需要 | 否 | 平方损失 + 有限目标 1/(2τ)，防过拟合、去 BT 假设 |
| KTO  | 好/坏二元标签（无需对） | 需要 | 否 | 前景理论非对称损失，数据更易得、抗噪声 |
| ORPO | 偏好对 | 不需要 | **是**（SFT+对齐合一） | odds ratio 温和对比，省 ref 与一个阶段 |
| SimPO | 偏好对 | 不需要 | 否（需 SFT 基座） | 长度归一 + 目标边际 γ，治长度偏差 |
| SPPO（参考行） | 偏好概率 win-rate + 自身采样 | 用上一轮 πt 作对手 | 迭代多轮 | 自博弈收敛 Nash，两侧有界更新 |

---

## 3.10 面试速记卡（高频问答）

1. **Q：DPO 里的 Z 为什么能约掉？**
   A：同一个 prompt x 下，y_w 和 y_l 的隐式奖励共享同一个分区函数 β·log Z(x)，
   做差抵消。本质是偏好只依赖相对差、平移不变（3.1.2(a)）。

2. **Q：β 怎么调、含义是什么？**
   A：β 继承自 RLHF 的 KL 系数。β 小 → 隐式奖励权重高 → 激进拟合偏好、偏离 ref 大；
   β 大 → 保守、贴近 ref。DPO 常见 0.01–0.5；SimPO 无 ref 项，尺度不同（≈2）。

3. **Q：DPO 和 SFT 的区别？**
   A：SFT 只抬高 y_w、完全不压 y_l，且各样本等权；DPO 做对比（升好降坏），
   并按当前错误程度加权。SFT 表达不了"y_l 是错的"这条信息。

4. **Q：偏好数据有 20% 噪声标签怎么办？**
   A：DPO 会把噪声对的比值推到无穷 → 过拟合噪声。换 IPO（有限目标）、
   KTO（非对称权重、更鲁棒），或做数据清洗/正则。

5. **Q：训练中 πθ(y_w)、πθ(y_l) 都在降，loss 还在降，正常吗？**
   A：这是 DPO 的已知病理：损失只约束比值，两个概率可以同时塌缩。
   IPO 论文对此有专门分析，平方损失缓解。

6. **Q：为什么要 iterative DPO？**
   A：off-policy 漂移（3.5.2）：锚点 πref 与数据分布双双过时，
   重新采样 + 更新 ref 使优化目标回到 on-policy。

7. **Q：DPO 是不是 RL？** 见 3.5.3 三段式答案。

---

## 3.11 参考资料

1. DPO：Rafailov et al., "Direct Preference Optimization: Your Language Model is
   Secretly a Reward Model", arXiv 2305.18290, NeurIPS 2023。
2. IPO：Azar et al., "A General Theoretical Paradigm to Understand Learning from
   Human Preferences", arXiv 2310.12036, AISTATS 2024。
3. KTO：Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization",
   arXiv 2402.01306, ICML 2024。
4. ORPO：Hong et al., "ORPO: Monolithic Preference Optimization without Reference
   Model", arXiv 2403.07691, EMNLP 2024。
5. SimPO：Meng et al., "SimPO: Simple Preference Optimization with a Reference-Free
   Reward", arXiv 2405.14734, NeurIPS 2024。
6. SPPO：Wu et al., "Self-Play Preference Optimization for Language Model Alignment",
   arXiv 2405.00675, 2024。
7. Kahneman & Tversky, "Prospect Theory: An Analysis of Decision under Risk",
   Econometrica 47(2), 1979。
8. Bradley & Terry, "Rank Analysis of Incomplete Block Designs: I. The Method of
   Paired Comparisons", Biometrika, 1952。
# LLM 强化学习方法全景指南 · Part 4：其他方法 + 对比速查 + 面试问答

> 本部分定位：把面试常考、但前面各部分没覆盖完的方法补齐——第一节是 critic-free 单样本族（RLOO / REINFORCE++ / ReMax），第二节是过程奖励与 verifier 路线（PRM / PRIME），第三节谨慎介绍 2025 新方向（EXO / GSPO），第四节画出「不是 RL 但面试常一起问」的边界（Best-of-N / 迭代 SFT / SPIN / RLAIF），第五节给一张 18 行全方法速查大表，第六节给 8 道面试高频问答（含 Search-R1 项目的标准答法）。
> 使用建议：速查表用于考前最后一遍，问答部分建议出声练 2-3 遍；所有 ⚠️ 标注处均属「待核实」，面试被追问时以原论文为准，不要背本文细节。
> 公式全部用行内 unicode 或代码块书写，不出现 LaTeX。

---

## 0. 本部分导读：地图与读法

**本部分在系列中的位置**：Part 1 讲了数学基础与 PPO（actor-critic 家族的代表），前面各部分覆盖了 GRPO、DPO 两大家族；本部分补上三块拼图——GRPO 家族的其他成员、过程奖励路线、以及「不是 RL 但面试常一起问」的边界方法。

**三条记忆主线**（面试时用它串起全文）：

1. **优势估计的范式分野**：PPO 一族「用学出来的 critic 算优势」（GAE），GRPO 一族「用组内采样统计量算优势」（RLOO 留一均值、ReMax 组内最大、GRPO 均值+标准差）——后者省显存、省超参，代价是方差/偏差换一换。
2. **奖励信号的密度分野**：outcome（结尾一个分）→ 过程奖励（逐步给分）→ 隐式过程奖励（PRIME 用 log-ratio 免费造过程分）——密度越高，训练信号越足，但标注/实现成本也越高。
3. **「是不是 RL」的边界**：Best-of-N、迭代 SFT、SPIN 都朝「高奖励」方向移动，但不动「策略对奖励的梯度」——面试先画边界、再谈联系，是得分的姿势。

**阅读顺序建议**：急着面试 → 直接看第五节速查表 + 第六节问答；要系统补课 → 按节顺序读，每节末尾的「面试记忆点」是背诵锚点。

---

## 1. critic-free 单样本族：GRPO 的亲戚们

这一类方法的共同点：**不训练价值网络（critic），直接用 rollout 得到的 reward 构造优势**。省显存、少一个不稳定源；代价是优势估计靠组内统计量，方差或偏差比 GAE 大。面试官若问「除了 GRPO 还有哪些不用 critic 的方法」，本节就是答案。家族树：RLOO（留一均值，2024）→ ReMax（组内最大，2023，最早）→ GRPO（均值+标准差，2024）→ REINFORCE++（加 PPO 工程件，2025）。

四种基线的速记对比（面试前扫一眼）：

| 方法 | 基线形式 | 是否除 std | 特殊之处 | 一句话定位 |
|---|---|---|---|---|
| ReMax | max_j r_j（或贪心样本 reward） | 否 | 几乎全负优势，只有冠军正 | GRPO 式基线最早的变体 |
| RLOO | (1/(k−1))·Σ_{j≠i} r_j | 否 | 留一：基线不含自己 | 教科书 REINFORCE + 分组基线 |
| GRPO | mean_k(r) | 是 | 组内归一化 + KL + 逐 token 复制 | DeepSeekMath 标配，生态最成熟 |
| REINFORCE++ | REINFORCE 优势（可归一化） | 可选 | PPO clip + mini-batch + token KL | 最小代码量的「准 PPO」 |

### 1.1 RLOO（Back to Basics）

【出处】Ahmadian et al., "Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs", arXiv 2402.14740, 2024。

【一句话本质】REINFORCE + 留一均值基线：每个样本的基线 = 同 prompt 下其余 k−1 条样本 reward 的平均值。

【核心公式】

```
同一 prompt x 采样 k 条 {y_1, ..., y_k}（k ≥ 2），第 i 条的基线：
b_i = (1/(k−1)) · Σ_{j≠i} r_j

优势：Â_i = r_i − b_i
梯度：∇_θ L = −E[ Â_i · ∇_θ log π_θ(y_i | x) ]
```

【为什么能降方差】（这题面试常追问，务必讲清）
- REINFORCE 的梯度是 ∇ = E[ R(τ) · ∇log P(τ|θ) ]。R 的绝对水平里混着大量与策略无关的公共噪声：这批 prompt 整体偏难、解码温度、batch 采样波动——这些不反映「当前策略好不好」，却会让梯度乱跳。
- 任何只依赖 x（不依赖样本自身动作）的函数 b(x) 都可以当基线：E[ b(x)·∇logπ ] = b(x)·∇E[1] = 0，梯度期望不变，不引入偏差。
- 留一的关键：基线用「同 prompt 其他 k−1 条样本」的均值，既保证与 r_i 不相关（留一后不含自己，避免「自己给自己当参照」的偏差），又恰好贴近「这批 prompt 的公共水平」——减掉它等于减掉公共噪声。k 越大基线估计越稳，但采样成本 ×k。

【与 GRPO 的异同】
- 相同：都 critic-free；都按 prompt 分组采多条；都拿组内 reward 统计量当基线；同属「REINFORCE with baseline」家族。
- 不同：GRPO 用全组均值 + 标准差做归一化 Â = (r − mean_k)/std_k；RLOO 只用留一均值、不除以标准差。另外 GRPO 的原始形式（DeepSeekMath）自带 KL 惩罚项和逐 token 复制，RLOO 基础版就是一条干净的 REINFORCE。

【解决了什么 / 代价是什么】
解决了 PPO 的显存与复杂度问题：无 critic、无 GAE，实现只需几十行，是最「教科书」的 LLM RL 路线。代价：基线只减均值、不利用样本间顺序信息，方差仍高于 critic 方法；每组至少采 2 条（实践常用 4-8），采样成本 ×k；方差控制不如带标准差归一化的 GRPO。

【面试记忆点】「GRPO 除标准差，RLOO 只减留一均值；RLOO = 教科书 REINFORCE + 分组留一基线。留一的目的：基线不含自己 → 无偏 + 减公共噪声。」

### 1.2 REINFORCE++

【出处】Jian Hu, "REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models", arXiv 2501.03262（⚠️ 编号待核实），2025。

【一句话本质】把 PPO 的工程稳定器（ratio clip、mini-batch、token 级 KL 修正）全部搬到 REINFORCE 上、去掉 critic——「PPO 的工程，REINFORCE 的骨架」。

【核心要点】（工程化清单，⚠️ 细节以原论文为准）

```
1. REINFORCE 式梯度：∇_θ L = −E[ Â · ∇_θ log π_θ(y|x) ]
2. PPO 式 ratio clip：用 π_θ/π_old 裁剪更新幅度（1±ε）
3. mini-batch 多 epoch：rollout 数据分成小批、多轮更新，提高数据利用率
4. token 级 KL 修正：把 KL 惩罚拆到每个 token，对偏离 ref 的 token 施加修正
```

【四件套逐个说】
- 梯度本体是 REINFORCE：无 critic、无 GAE，优势来自 rollout 的 reward（可选组内归一化）；
- clip 是 PPO 借来的「步长保险」：REINFORCE 没有步长概念，一条异常高奖励的样本能把策略掀翻，ratio clip 把单步更新幅度锁死在 1±ε；
- mini-batch 是数据利用率件：REINFORCE 传统上一条样本用完即弃，mini-batch 多 epoch 让同一批 rollout 反复参与更新（等价于 PPO 的多 epoch 训练循环）；
- token 级 KL 是「语言能力保险」：对每个 token 单独计算与 ref 的 KL 并施加修正/惩罚，防止为刷奖励把语言模型训成复读机。

【解决了什么】
回答「不想写 critic，但舍不得 PPO 的稳定性」这一真实工程需求。三件稳定器都是 PPO 里被验证过的，组合起来是一条实现成本极低的 RL 路线，适合教学、复现与快速原型。

【代价】
仍是样本级优势，没有 critic 的逐 token 状态估计；token 级 KL 修正需要 ref 模型，显存比 RLOO 多一个冻结模型；reward 归一化、KL 修正的具体做法对效果敏感，实现细节比表面复杂。⚠️ 该方法社区复现较少，面试建议只讲「四件套」思想，不背超参。

【面试记忆点】「REINFORCE 的梯度 + PPO 的 clip + mini-batch + token 级 KL = 四件套；它回答的问题是：怎么用最小代码量获得 PPO 级的训练稳定。」

### 1.3 ReMax

【出处】Li et al., "ReMax: A Simple, Effective, and Efficient Method for Aligning Large Language Models", arXiv 2310.10505, 2023。

【一句话本质】用 batch 内最大 reward 当基线：Â_i = r_i − max_j r_j。可以看作「GRPO 式 baseline」的早期变体——它发表于 GRPO（2024）之前。

【核心公式】

```
b = max_j r_j （原论文用贪心解码样本的 reward 作基线；社区实现常简化为 batch 最大值）
Â_i = r_i − b
```

每批只需为基线多算一条贪心解码样本（多一次采样/前向），无需 critic、无需 ref。注意一个细节：原论文的基线来自贪心解码样本，与采样样本独立，无「自己当自己基线」的偏差问题；社区简化成 batch 内 max 时，严格说应留一（b_i = max_{j≠i} r_j），否则最大样本自己的基线含自己。

【解决了什么】
在 PPO 四模型之外给出第二条省显存的路：只训练策略网络。max 基线比均值基线更「苛刻」——均值基线让组内一半正、一半负，max 基线几乎全负、只有冠军为正，策略被持续推向「超越当前最好」。原论文结论：在 RLHF 上以一半的模型数接近 PPO 效果。

【代价】
max 对异常值极敏感：一条 reward 虚高的样本会把全组压成负优势，一次噪声污染一整步更新；负优势样本占比大，需要 clip 配合；方差控制不如归一化基线。今天基本被 GRPO/RLOO 取代，但它是最早把「reward 统计量当基线」做进 LLM RLHF 的工作之一，历史地位面试可提。

【面试记忆点】「ReMax = 用组内冠军当基线，只有超过冠军才算进步；是 GRPO 式 baseline 的早期变体（早于 GRPO），敏感于异常值，已被取代。」

【本节小结】四个方法共享同一个决策原点：用什么替代 critic。答案依次是「组内最大」（ReMax）→「留一均值」（RLOO）→「均值+标准差」（GRPO）→「均值+PPO 工程件」（REINFORCE++）。记忆顺序按「苛刻程度」：max 最苛刻、mean 温和、mean+std 有标度、加 clip 最稳。面试被问「为什么不直接用 REINFORCE」，就用这条线回答：纯 REINFORCE 方差太大 → 加基线（本族）→ 加步长控制（clip）→ 需要更细的状态估计才轮到 critic（PPO 族）。

---

## 2. 过程奖励与 verifier 路线

导读：outcome reward 只在最后一个 token 有信号，多步推理中大部分 token 学不到任何东西。过程奖励把信号铺到每一步，支持步骤级搜索与错误定位；代价是标注极贵，且「过程对 ≠ 答案对」。面试常问「PRM 值不值得用」，本节给你两面。三种「奖励粒度」先扫一眼：

| 奖励粒度 | 信号位置 | 信号密度 | 标注成本 | 代表 |
|---|---|---|---|---|
| outcome RM | 序列结尾一个分 | 稀疏 | 低（自动判分或答案标注） | 通用 RLHF |
| 显式 PRM | 每个中间步骤一个分 | 密集 | 高（逐步人工标注/大量采样） | PRM800K、Math-Shepherd |
| 隐式 PRM | token 级 log-ratio | 密集 | 零（outcome 标签在线驱动） | PRIME |

### 2.1 PRM（过程奖励模型）

【出处】
- Lightman et al., "Let's Verify Step by Step", OpenAI, arXiv 2305.20050（⚠️ 编号待核实），2023——提出 PRM800K 数据集与过程监督（process supervision）。
- Math-Shepherd（Wang et al., 2024）——自动过程标注方法：对每个中间步骤做随机补全（MC 估计），用「从该步出发最终能做对的频率」当该步的过程分，绕开人工逐步标注。

【一句话本质】把「整条答案对不对」拆成「每一步对不对」：PRM 对每个中间步骤 y_t 输出得分 r_t，总奖励由步骤得分组合而成，提供密集的过程信号。

【核心对比：与 outcome RM 的三点差异】
- 信号密度：ORM 只在结尾给一个分，中间步骤零信号——多步推理里训练慢、错误定位靠猜；PRM 逐步给分，可支持步骤级搜索（对候选步骤打分剪枝）与错误溯源。
- 成本：ORM 只需最终答案标注（甚至自动判分）；PRM 需要逐步人工标注——PRM800K 的第一教训就是过程标注成本比结果标注贵一个数量级（数据集约 8 万条解答、约 80 万条步骤级标注，数据集名的由来，⚠️ 数字待核实）。Math-Shepherd 用自动补全缓解：对每个中间步骤做若干次随机补全到结尾，把「能推导出正确答案」的比例作为该步过程分——无需人工，但每步要大量采样。
- 可靠性：PRM800K 的第二教训——PRM 的步骤评分精度确实高于 ORM（论文核心结论），但后续实践发现「过程分涨了、最终答案没涨」：步骤对 ≠ 推理方向对，把 PRM 接进 RL/搜索管线后，最终答案准确率的增益远小于过程分的增益。⚠️ 这是社区对后续应用的观察，面试表述为「过程监督的实用价值有争议」即可。

【解决了什么 / 代价是什么】
解决多步任务的稀疏信号问题；代价是标注/采样成本与「过程-答案解耦」风险。经验法则：任务是数学证明、长链推理、agent 轨迹评估时再考虑 PRM；只要最终答案对就行时，先问一句「真的需要过程分吗」——这本身就是面试加分句。

【面试记忆点】「PRM 的两条教训：标注贵一个数量级；过程分涨了最终答案没涨。所以要么自动标注（Math-Shepherd），要么隐式奖励（PRIME）。」

### 2.2 PRIME（Implicit PRM）

【出处】Cui et al.（清华）, "Process Reinforcement through Implicit Rewards", arXiv 2502.01456, 2025。

【一句话本质】不训练显式 PRM，直接用「当前策略 vs 冻结参考策略」的逐步 log-ratio 当过程奖励，靠 outcome 标签在线更新——让过程奖励「免费」出现。

【核心公式】

```
第 t 步的隐式奖励：
r(x, y_t) = log π_θ(y_t | x, y_<t) − log π_ref(y_t | x, y_<t)

整条序列的隐式奖励 = Σ_t r(x, y_t) = log π_θ(y|x) − log π_ref(y|x)
```

【直觉（两层）】
- 第一层：π_ref 是 SFT 模型，代表「初始平均水平」。如果当前策略在某个 token 上比 ref 更「确信」——即训练中该步概率被上调过——说明策略正朝着「更可能拿高分」的方向移动。把这种逐步概率漂移当成过程奖励，就绕开了显式 PRM 的训练与标注。
- 第二层：序列级 log-ratio 正是 DPO 推导里的隐式奖励 r̂ = β·log(π/π_ref)。PRIME 的贡献是把「整条序列一个分」拆成「逐步给分」，并把隐式奖励接进在线 RL 循环：rollout → 隐式 PRM 给 token 级信号 → 用 outcome 标签在线更新隐式 PRM（原论文用偏好式目标驱动，⚠️ 细节待核实）→ GRPO 式更新策略。
- 动态视角：训练初期 θ 与 ref 几乎相同 → 隐式奖励几乎为 0；训练推进、好 token 的概率差拉大 → 过程信号自然变密集。奖励是「长出来的」，不是「标出来的」。

【解决了什么 / 代价是什么】
想要过程信号的密集性（训练更快、可定位错误），又付不起 PRM 标注成本——PRIME 用 outcome 标签 + 参数复用同时满足。代价：需要 ref 模型（显存 +1）；隐式奖励本质是「与自己过去的差距」，概率涨 ≠ 步骤对，与真实过程质量可能解耦；依赖 outcome 信号足够强。原论文在数学任务上报告超过显式 PRM 路线的效果（⚠️ 具体数字以论文为准）。

【面试记忆点】「PRIME 的公式就是 DPO 隐式奖励的逐 token 版：token 级 log-ratio 当过程分，outcome 标签在线驱动，不需要 PRM 标注。代价：需要 ref，且概率涨 ≠ 步骤对。」

【本节小结】过程奖励三条路的成本递减：显式 PRM（标注贵）→ Math-Shepherd（采样换标注）→ PRIME（outcome 标签换一切）。信号真实性也递减：人工步骤分最真实、MC 补全次之、log-ratio 是「概率漂移」的代理。面试表述：「要真实过程信号就付标注成本，付不起就用隐式代理，并接受它与真实质量解耦的风险。」

---

## 3. 2025 新方向（谨慎表述）

⚠️ 本节两个方法都比较新，细节请一律以原论文为准。面试策略：只背「思想 + 与已知方法的关系」，不背数字；被追问细节时直接承认「细节没核实」，比硬答安全。

### 3.1 EXO（Explore-Exploit Contrastive Pre-training）

【出处】2025 年提出（arXiv 号 ⚠️ 待核实）。引用时务必注意：2018 年有一篇同名旧论文 EXO（arXiv 1812.00116），与 LLM 无关，两者不是一回事。

【一句话本质】把 RL 的「探索-利用」思想搬到预训练阶段：以高质量完成（利用）为正例、当前策略的探索性采样（探索）为负例，用对比目标做预训练，给后续 RL 一个更好的起点。

【核心思想】（⚠️ 细节待核实）
- 正例：参考/教师策略产出的高奖励完成（exploit 侧）；
- 负例：学生策略自己采出的低质量探索样本（explore 侧）；
- 目标：对比损失——拉近正例、推远负例，先于（或伴随）RL 训练。

【解决了什么 / 代价是什么】
RL 的冷启动问题：策略初期采样质量差、奖励噪声大、训练易崩。先用对比预训练把策略「掰向」高奖励区域，再上 RL，宣称可改善训练稳定性与最终性能。代价：需要参考策略与成对采样，多一段预训练开销；方法较新，泛化性仍在验证中（⚠️）。

【与已有方法的关系（面试定位用）】和 DPO 的区别：DPO 吃人工/模型标注的偏好对，EXO 的「正负例」来自策略自身采样 + 判分（⚠️ 细节待核实）；和迭代 SFT 的区别：迭代 SFT 只学正例，EXO 用对比目标同时推远负例。

【面试记忆点】「EXO = RL 风格预训练：正负例对比、给 RL 稳定冷启动。引用前先确认编号，别混进 2018 年同名旧论文。」

【面试答法】被问到 EXO 时，标准三句话：「它把 RL 的探索-利用思想前移到预训练阶段，用对比目标让策略先靠近高奖励区域，再上 RL 会更稳（⚠️ 效果数据以论文为准）；它和 DPO 的区别在于正负例来自策略自身采样而非标注偏好对；引用时注意与 2018 年同名旧论文区分。」三句话讲完，不展开数字，最安全。

### 3.2 GSPO（Group Sequence Policy Optimization）

【出处】arXiv 2507.18071（⚠️ 编号与细节待核实），2025。

【一句话本质】以「组」为单位优化整条序列，用 sequential KL（逐步 KL）约束偏离，配合 IMED 目标估计 KL——全程不需要 ref 模型。

【核心要点】（⚠️ 细节待核实）
- 组序列优化：同一 prompt 的一组 rollout 作为一个整体做序列级优化（区别于 PPO 的逐 token 更新）；
- sequential KL：逐步 KL 约束比整条 KL 更严格，逐 token 限制偏离初始策略的幅度；
- IMED 目标：用组内样本估计 sequential KL（互信息下界），取代显式 ref 模型——省掉 ref 的显存与同步开销。

【解决了什么 / 代价是什么】
ref 模型在 GRPO/PPO 族里「不训练但占显存、占通信」，GSPO 试图把它从方程里消掉，同时把 KL 约束细化为逐步版本。代价：组级优化与 IMED 估计引入新近似误差；方法新、社区验证少。面试中若被问到，承认「了解思想、细节未核实」比硬答更安全。

【与 GRPO 的关系（面试定位用）】GRPO 是「组内统计量优势 + ref 做 KL」；GSPO 是「组级序列优化 + sequential KL + IMED 无 ref」。一句话记：「GRPO 还要 ref，GSPO 连 ref 都不要了」（⚠️ 表述以原论文为准）。

【面试答法】被问到 GSPO 时：「三个关键词：组序列优化、sequential KL、IMED 无 ref 估计。它的卖点是省掉 ref 模型；代价是新方法、社区验证少（⚠️）。如果项目里有 ref 显存压力，这值得跟踪，但面试不会要求细节。」主动承认「细节未核实」是应对新方法的正确姿势。

---

## 4. 「不是 RL 但面试常一起问」

导读：面试官爱用这几个概念测你「会不会画边界」。统一口径：它们优化的是数据分布或推理过程，不是策略对奖励的梯度；但和 RL 共享「朝高奖励方向移动」的目标，常作为 RL 的前置或兜底。答题套路：先画边界（「严格说这不是 RL」），再讲联系（「但它是 RL 的特例/前身/兜底」）。

### 4.1 Best-of-N / rejection sampling

【一句话本质】推理时采样 N 条候选，按奖励/判分器挑最好的一条输出。**不训练任何参数**。

【为什么不是训练】策略分布 π 完全没变，只是从它的 N 次采样里做了一次 max。它移动的是「输出分布」（结果分布右移），不是「参数」。

【为什么推理成本高】每次请求都要付 N 倍生成成本，且不可摊销。RL 是把「采样 N 条找最好」的过程蒸馏进参数：训练时贵一次，推理时免费。生产上 N 通常只能给到 4-8，再多推理成本不可接受。

【关键结论】
- 收益律：在 reward 近似正态的假设下，BoN 期望奖励 ≈ r̄ + σ·√(2 ln N)——随 N 对数增长，边际收益锐减；要吃一个 σ 的增益，N 要涨近一个数量级。
- KL 代价：BoN 的输出分布偏离原策略，且 KL 随 N 近似按 log N 增长（KL(π_BoN‖π) ≈ log N − (N−1)/N，⚠️ 公式待核实）——没有 KL 控制，等于无约束地朝奖励最大方向挤。
- 风险：大 N 下会选中「骗过判分器」的答案（reward overoptimization / boomerang 效应：RM 分数涨、真实质量跌）。

【正确用法】低成本兜底（来不及训练时）；数据过滤（给 expert iteration 供数据）；充当 RL 效果的上界参照（RL 收益若低于 BoN 上限，说明训练有问题）。

【面试记忆点】「BoN 动输出分布不动参数：收益对数增长、成本线性增长、无 KL 控制会 hack 判分器。它是测试时搜索，RL 是训练时搜索。」

### 4.2 Expert iteration / 迭代 SFT

【一句话本质】π 采样 → 按 reward 过滤 → 只留高奖励样本 → SFT 更新 π → 重复。每轮 π 变强，采样质量变高，过滤阈值水涨船高。代表工作：STaR（Self-Taught Reasoner）、ReST（Reinforced Self-Training）、ReST^EM（⚠️ 编号略，面试报名字即可）。

【与在线 RL 的关系】
- 相同：都从「当前策略」在线采样，数据分布随策略一起演化——所以它是「半在线」的；
- 不同：迭代 SFT 对负样本是「硬过滤」（只学正例），RL 是「软加权」（优势加权，负样本也贡献梯度）。可以说 RL 是 expert iteration 的连续化（软过滤版），expert iteration 是 RL 在「权重二值化」下的特例。工程上常见组合：先迭代 SFT 冷启动，再上在线 RL 精修。

【为什么它有用但不充分】只学正例意味着「负信号全丢」——模型不知道「什么错、错在哪、错多严重」；多样性随迭代坍缩（都在抄上一轮的「最优解」）；对判分器稳定性要求高（判分器一漂，过滤出的数据就脏）。

【面试记忆点】「expert iteration = BoN 过滤 + SFT 的循环 = 硬过滤版 RL。RL 的软加权版保留了负样本梯度，这是两者最本质的区别。」

【与 DPO 的区别（易混淆）】DPO 吃的是「人工/模型标注的偏好对」，迭代 SFT 吃的是「自己采样 + 判分器过滤」的正例——前者离线、靠标注，后者半在线、靠判分器。两者都能涨分，但迭代 SFT 的多样性和负信号都更弱。

### 4.3 SPIN（Self-Play Fine-Tuning）

【出处】Chen et al., "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models", arXiv 2401.01335（⚠️ 编号待核实），2024。

【一句话本质】把 SFT 变成自我博弈：对手 = 上一轮迭代的模型，本轮模型学习「在与上一轮的对比中胜出」，全程无需外部奖励。

【核心思想】每一轮：当前模型 θ_t 在 SFT 数据上生成响应，与 θ_{t−1}（上一轮）的响应配对，用对比损失让 θ_t 的输出「更像人类标注、更不像 θ_{t−1} 的输出」——即区分「现在的自己」与「昨天的自己」。损失大体形如 −log σ( log π_θ(y|x) − log π_{θ_{t−1}}(y|x) ) 的变体（⚠️ 具体形式以原论文为准）。对手随迭代同步变强，形成自我博弈。

【为什么不是 RL】没有环境、没有奖励信号，只有自我对弈的对比目标；更新形式是最大似然 + 对比正则，不是策略梯度。与 SPPO 的区别：SPPO 用偏好/胜率信号做自博弈偏好优化，SPIN 只靠「区分新旧自己」。

【解决了什么 / 代价是什么】在没有偏好数据、没有判分器的场景下，靠「超越昨天的自己」继续提升（原论文报告多轮迭代持续提升，⚠️ 数字以论文为准）。代价：依赖 SFT 种子数据质量；自我博弈可能陷入「自己骗自己」的退化。

【为什么自博弈能涨分（直觉）】SFT 数据是「人类答案」分布，第一轮模型生成的是「模型答案」分布；让模型学会区分这两个分布（人类 vs 自己），等价于把模型分布拉向人类分布。每迭代一轮，对手更强，区分任务更难，模型必须更接近人类分布才能赢——对手变强的过程就是模型变强的过程。这是自我博弈范式的通用直觉（AlphaGo 同理：对手越强，自己必须更强）。

【面试记忆点】「SPIN = 自我博弈 SFT：对手是上一轮的自己，对比损失让今天的输出更像人、更不像昨天。没有奖励信号，所以不是 RL。」

### 4.4 RLAIF / Constitutional AI

【出处】Bai et al., "Constitutional AI: Harmlessness from AI Feedback", arXiv 2212.08073, Anthropic, 2022。

【一句话本质】用 AI 反馈替代人类反馈；Constitutional AI = RLAIF + 一本「宪法」（成文规则），让模型按规则自评、自修。

【流程（三段式）】
1. 自评自修：模型按宪法条款（如「不得支持违法活动」「不得鼓励暴力」等成文原则）批评自己的初稿 → 按批评自我修改；
2. SFT：用修改后的响应做监督微调；
3. RLAIF：用 LLM 按宪法生成偏好判断（两个回答哪个更符合宪法），替代人类标注，跑偏好优化（原论文用 RL，现在也常用 DPO）。

【与 RLHF 的关系】RLAIF 替换的是「人类反馈」这一环（标注员 → AI 裁判），优化范式不变；宪法解决的是「反馈标准从哪来」——从标注员直觉变成可审计的成文规则。两个概念的范围：RLAIF 是泛指（任何 AI 生成的反馈），Constitutional AI 是 Anthropic 的具体实现（宪法 + 自评自修 + AI 偏好）。

【代价】AI 裁判有系统性偏差（自偏好、长度偏好、对弱模型误判）；要求裁判模型足够强；宪法写得不好，效果上不去；「自己批评自己」的环节依赖模型的自我认知能力。

【面试加分点】RLAIF 的意义是「可扩展性」（成本、隐私、可审计），不是「比人类更准」——人类标注不可扩展，AI 反馈可以随模型一起长大。这句话是 Anthropic 原论文的核心动机，答出来很加分。

**§4 边界小结（背这一句）**：Best-of-N 动输出不动参数、迭代 SFT 只学正例、SPIN 无奖励信号、RLAIF 换的是反馈来源——四个都不是策略梯度；但它们与 RL 共享「朝高奖励移动」的目标，是 RL 的前置、特例或兜底。

---

## 5. 全方法对比速查大表

读表提示：「显存」指训练同规模模型的相对量级（低 = 1 个可训练模型 + 可选冻结 ref；很低 = 单模型；中 = 多模型/多卡框架；高 = 完整四模型栈）。「数据需求」只列最严格的一种。⚠️ = 待核实，— = 不适用。

| 方法 | 年份 | arXiv | critic | ref | 数据需求 | 在线/离线 | 显存 | 主要风险 | 适用场景 |
|---|---|---|---|---|---|---|---|---|---|
| PPO | 2017/2022 | 1707.06347 | 需要 | 需要 | 仅问题+判分器 | 在线 | 高 | critic 不稳、显存大 | RLHF 标准基线 |
| GRPO | 2024 | 2402.03300 | 不需要 | 需要 | 仅问题+判分器 | 在线 | 低 | 长度/格式偏差 | 可判分任务、单卡 |
| RLOO | 2024 | 2402.14740 | 不需要 | 基础版不需要 | 仅问题+判分器 | 在线 | 低 | 方差仍偏大 | 干净 REINFORCE 基线 |
| REINFORCE++ | 2025 | 2501.03262 ⚠️ | 不需要 | 需要（KL 修正） | 仅问题+判分器 | 在线 | 低 | 实现细节敏感 | 要 PPO 工程、不要 critic |
| ReMax | 2023 | 2310.10505 | 不需要 | 不需要 | 仅问题+判分器 | 在线 | 低 | max 对异常值敏感 | 历史简化方案 |
| DPO | 2023 | 2305.18290 | 不需要 | 需要 | 偏好对 | 离线 | 低 | 分布外脆弱、长度偏差 | 有偏好数据的场景 |
| IPO | 2023 | 2310.12036 | 不需要 | 需要 | 偏好对 | 离线 | 低 | 收敛慢 | 防 DPO 过拟合 |
| KTO | 2024 | 2402.01306 | 不需要 | 需要 | 单标签（好/坏） | 离线 | 低 | 理论假设强 | 只有好坏标签 |
| ORPO | 2024 | 2403.07691 | 不需要 | 不需要 | 偏好对（SFT 数据内构成） | 离线 | 很低 | 与 SFT 耦合、超参敏感 | 一步对齐 |
| SimPO | 2024 | 2405.14734 | 不需要 | 不需要 | 偏好对 | 离线 | 很低 | 长度处理粗糙 | DPO 轻量替代 |
| SPPO | 2024 | 2405.00675 ⚠️ | 不需要 | 需要（上一轮自身） | 偏好对/胜率 | 迭代自博弈 | 低 | 自我博弈退化 | 无绝对奖励的自博弈 |
| PRIME | 2025 | 2502.01456 | 不需要（隐式 PRM） | 需要 | 仅问题+outcome 判分器 | 在线 | 低-中 | 隐式奖励与真实质量解耦 | 数学/代码、要过程信号 |
| VAPO | 2025 | 2504.05118 ⚠️ | 需要 | 需要 | 仅问题+判分器 | 在线 | 中-高 | 长链训练不稳 | 竞赛级长推理 |
| DAPO | 2025 | 2503.14476 | 不需要 | 需要 | 仅问题+判分器 | 在线 | 中（多卡框架） | 工程复杂 | 大规模开源 RL 复现 |
| Dr.GRPO | 2025 | 2503.20783 ⚠️ | 不需要 | 需要 | 仅问题+判分器 | 在线 | 低 | 修偏差引入新超参 | 修 GRPO 长度偏差 |
| GSPO | 2025 | 2507.18071 ⚠️ | 不需要 ⚠️ | 不需要（IMED 替代）⚠️ | 仅问题+判分器 ⚠️ | 在线 ⚠️ | 低 ⚠️ | 新方法、验证少 ⚠️ | 想去 ref 的 RL |
| Best-of-N | — | — | — | — | 仅问题+判分器（推理时） | 不训练 | 0（推理 ×N） | 无 KL 控制、reward hack | 兜底/数据过滤 |
| 迭代 SFT | 2022+ | STaR 2203.14465 ⚠️ | — | — | 仅问题+判分器（过滤） | 半在线 | 低 | 丢负信号、多样性坍缩 | 冷启动、数据增强 |

三行速记：
- 在线 + 无 critic = GRPO 族（GRPO/RLOO/REINFORCE++/ReMax/DAPO/Dr.GRPO）；在线 + 有 critic = PPO/VAPO。
- 离线 + 偏好对 = DPO 族（DPO/IPO/ORPO/SimPO）；离线 + 单标签 = KTO；自博弈 = SPPO/SPIN。
- 「判分器能自动打分」是选在线 RL 的第一前提；「有没有现成偏好数据」是选 DPO 族的第一前提。

**选型决策树（从数据出发，面试可直接背）**：
1. 有现成偏好对 → DPO 族（怕过拟合选 IPO；没 ref 预算选 ORPO/SimPO；只有好坏单标签选 KTO）。
2. 没有偏好对、但有自动判分器 → 在线 RL（单卡、无 critic 预算 → GRPO 族；有 critic 预算且要长链推理 → PPO/VAPO；要过程级信号又不想标 PRM → PRIME）。
3. 有判分器、但没时间/设施做在线 rollout → 迭代 SFT / BoN 过滤兜底。
4. 什么都没有、只有 SFT 数据 → SPIN 自我博弈 / ORPO 一步对齐。
5. 预算与工程成熟度：优先选有成熟开源实现（veRL/TRL/OpenRLHF）支撑的方法——这常比方法本身的纸面优势更重要。

**易混淆概念对（面试高频送命题）**：

| 概念对 | 一句话区分 |
|---|---|
| RLOO vs GRPO | 都组内统计量；RLOO 留一均值不除 std，GRPO 全组均值+除 std |
| GRPO vs DAPO | DAPO = GRPO + 四个工程修复（clip higher/动态采样/token-level loss/overlong shaping） |
| DAPO vs Dr.GRPO | 都修 GRPO 偏差：DAPO 修奖励塑形与采样，Dr.GRPO 修归一化统计量 |
| PRM vs PRIME | PRM 显式训练（要过程标注），PRIME 隐式奖励（只要 outcome 标签） |
| SPPO vs SPIN | 都自博弈：SPPO 用偏好/胜率信号，SPIN 只用「区分新旧自己」 |
| DPO vs 迭代 DPO | 迭代 DPO 每轮用当前策略重新采样构造偏好对，离线变准在线 |
| Expert iteration vs RL | 硬过滤只学正例 vs 优势软加权（负样本也有梯度） |
| Best-of-N vs RL | 测试时搜索（动输出不动参数）vs 训练时搜索（蒸馏进参数） |
| RLAIF vs Constitutional AI | RLAIF 泛指 AI 反馈，CAI 是 Anthropic 的具体实现（宪法+自评自修） |
| GRPO vs GSPO | GRPO 组内统计量优势+ref 做 KL；GSPO 组级序列优化+sequential KL+IMED 无 ref（⚠️ 待核实） |

---

## 6. 面试高频问答

每题按「核心论点（30 秒）→ 展开（60-90 秒）→ 加分点/追问预案（30 秒）」组织，共 2-3 分钟。先给结论、再给理由，被追问时往下钻一层。

### Q1｜PPO 和 GRPO 的本质区别？GRPO 省了什么？

【核心论点】区别在优势估计的来源：PPO 用「学出来的 critic」（GAE），GRPO 用「组内 reward 统计量」。GRPO 省掉：一个网络、一批超参、一类不稳定源。

【参考展开】
- PPO：Â_t = Σ_l (γλ)^l δ_{t+l}，其中 δ_t = r_t + γV(s_{t+1}) − V(s_t)（GAE）。需要一个价值网络 V 与策略同步训练，整套是 actor-critic 架构，加上 ref 和 RM 共四模型。
- GRPO：同一 prompt 采样 k 条 rollout，Â = (r − mean_k)/std_k，组内归一化直接当优势，全程不碰价值函数。
- 所以 GRPO 省的：① critic 网络——显存约省一半；② GAE 的时序估计误差与价值函数不收敛的风险；③ γ、λ、value clip 等一批超参；④ critic 训练与 actor 训练的同步复杂度。

【加分点/追问预案】
- 族谱定位：「PPO 是 actor-critic 家族，GRPO 是 REINFORCE-with-baseline 家族（和 RLOO、ReMax 同族）」——一句话展示你掌握的是范式而非名字。
- 雷区：别答成「GRPO 就是省内存的 PPO」——省内存是结果，本质区别是优势估计的范式（学出来的价值函数 vs 采样统计量）。
- 追问「GRPO 的代价」→ 接 Q3：组内归一化带来长度/格式偏差；k 太小时统计量噪声大；没有绝对价值信息（组内相对值）。

【一句话总结】「PPO 靠学出来的 critic 算优势，GRPO 靠组内采样统计量算优势——省一个网络、一批超参、一类不稳定源。」

### Q2｜DPO 和 RLHF 的关系？DPO 算 RL 吗？

【核心论点】同一优化目标（KL 约束下最大化奖励）的两种解法：RLHF 显式两步走，DPO 一步代数合并。「算不算 RL」取决于定义：不算策略梯度，但算同一 RL 问题的隐式解。

【参考展开】
- 推导链（建议背熟）：Bradley-Terry 偏好模型 P(y₁ ≻ y₂) = σ(r(y₁) − r(y₂))；带 KL 的 RLHF 最优策略有闭式解 r*(y) = β·log(π(y)/π_ref(y)) + Z(x)；把它代入 BT 消掉显式奖励 r，就得到 DPO 损失——直接优化 log-ratio：log(π_θ(y_w)/π_ref(y_w)) − log(π_θ(y_l)/π_ref(y_l)) 的 sigmoid 负对数似然。
- DPO 算 RL 吗：标准答——它不是策略梯度、不做环境交互、是 offline 的，因此「不是 RL 算法」；但它解的是 RLHF 的同一优化问题，是它的代数重写/隐式解，因此「对齐目标与 RLHF 等价」。两个说法都对，先下定义再回答是加分的姿势。
- 追「DPO vs PPO 谁好」→ 看数据与环境：DPO 受离线数据覆盖上限约束；PPO 在线采样能持续发现新错误。两者不是新旧替代关系，是数据可得性决定选型。
- 追问「β 是什么」→ β 是 KL 强度的倒数：β 大 → 隐式 KL 约束弱 → 允许偏离 ref 更远（但易过拟合偏好、变分布外）；β 小 → 贴紧 ref（但可能欠拟合偏好）。这也是 DPO 的隐式奖励 r̂ = β·log(π/π_ref) 的标度。

【加分点】DPO 的隐式奖励 r̂ = β·log(π/π_ref) 可以当样本级奖励估计用——这是 DPO 与 RL 之间的桥梁，提到它说明你读懂了推导而非背公式。PRIME 的隐式 PRM 就是把它逐 token 化（呼应 2.2 节）。

【一句话总结】「DPO 是 RLHF 的代数重写：不是策略梯度，但是同一 KL 约束奖励最大化问题的隐式解。」

【追问预案】「DPO 和 IPO 的区别」→ IPO 指出 DPO 的损失在偏好概率趋近 1 时会无界地推 log-ratio 到无穷，改用平方损失（回归到真实偏好强度）替代 sigmoid 对数损失，牺牲收敛速度换稳健性。一句话：DPO 怕过拟合，IPO 是它的保守版。

### Q3｜GRPO 的长度偏差怎么来的？怎么修？

【核心论点】两个放大器：① 组内归一化把「长度-正确率相关」放大成优化信号；② 稀疏奖励逐 token 均摊，长答案摊薄、短答案放大。修法两派：Dr.GRPO 改归一化，DAPO 改奖励塑形与采样。

【参考展开】
- 机制一：reward 只在最后一个 token 出现，GRPO 把组级优势复制给序列中每个 token。序列越长，越容易覆盖正确答案与格式要求 → 长序列组内 reward 偏高 → 组均值被抬高 → 「写长」被系统性奖励，长度与质量在优势里混在一起。
- 机制二：组内归一化 (r − mean)/std 让每个样本的优势取决于同组其他样本的分布；长样本多的组里，长度信息直接进入优势计算。另一个隐患：组内 std 趋近 0 时（全组 reward 相同）优势爆炸——这就是「全对组/全错组」问题，DAPO 的动态采样（过滤全零优势组）针对它。
- Dr.GRPO 的修法：去掉组内 std、用全局统计量归一化、移除与长度直接相关的奖励项——目标是把「长度 → 优势」这条通道剪断。
- DAPO 的修法（工程向）：overlong reward shaping（超长截断样本不丢弃、给惩罚，保留负信号）；token-level loss（防长序列梯度被稀释/爆炸）；clip higher（限上不限下，防熵坍缩）；动态采样（过滤全零优势的无效组）。

【加分点】补一句辩证：长度增长不全是坏事——R1-Zero 的反思能力依赖长链；要修的是「长度驱动的奖励黑客」，不是长度本身。再补一句诊断方法：「训练中谁的输出越来越长、谁的正确率虚高，说明奖励函数在奖励长度」。

【一句话总结】「长度偏差 = 组内归一化 + 稀疏奖励均摊两个放大器；Dr.GRPO 剪归一化通道，DAPO 修奖励塑形与采样。」

### Q4｜为什么用 RL 而不只做 SFT？什么任务必须 RL？

【核心论点】SFT 学「复制好答案」，RL 学「从坏答案走向好答案」。SFT 是分布内模仿，RL 是用奖励在解空间里搜索——多出负反馈和探索两种信息。

【参考展开】
- SFT 的信息上限：标注数据里「好答案的样子」。模型只能插值已知的好模式：①学不到「哪些错误不要犯」（负信号）；②无法超越数据里的最佳答案（探索）。
- RL 多给两样东西：负反馈（不只要像好的，还要不像坏的）与奖励引导的探索（尝试数据里没有的路径：多步推理、工具调用、搜索交互）。
- 必须 RL 的任务画像（四占其一，RL 收益就大概率大于成本）：① 可自动判分（代码执行、数学验证、游戏分数）——有廉价可靠的奖励；② 好答案不在现有数据里、需要探索（长时程 agent、开放域搜索问答）；③ 需要在线交互（环境反馈本身就是奖励的一部分）；④ 安全性（显式压低坏行为概率）。四个都不占，SFT/DPO 就够了。
- 反向也成立：没有判分器/环境、只有静态数据时，硬上 RL 是负收益——先解决奖励信号，再谈 RL。

【加分点】引用 R1-Zero 的经验证据：纯 RL（无 SFT 冷启动）也能涌现自我修正、反思等行为；SFT 冷启动的价值是稳定训练，不是能力上限。结论句：「SFT 决定起点，RL 决定上限。」

【一句话总结】「RL 多给两样 SFT 给不了的东西：负反馈与探索；可判分、需探索、需交互、要安全，四占其一就该上 RL。」

### Q5｜奖励模型 vs 规则奖励 vs verifier，怎么选？

【核心论点】按三个维度选：信号可得性（能不能自动打分）、信号密度（结尾还是逐步）、信号可靠性（会不会被 hack）。

【参考展开】
- 规则奖励：程序化判分（EM 匹配、代码执行、格式检查）。优点：免费、无偏、可无限采样。缺点：只覆盖可判定部分，模型会优化规则本身（格式分刷满、内容靠猜）。原则：能写规则就写规则，规则为主、RM 兜底。
- 奖励模型：偏好数据训练。优点：开放生成、语义判断。缺点：标注贵、有偏差、能力受训练数据上限约束；存在 reward overoptimization 曲线（Gao et al.）：RM 分数涨到一定点后真实质量下跌。RM 比策略弱时没有使用价值。
- verifier（PRM / learned verifier）：过程级信号，支持步骤搜索与错误定位。缺点：标注极贵（PRM800K 教训），过程分与答案分解耦；隐式 PRM（PRIME）用 outcome 在线驱动是折中方案。
- 决策树：任务可自动判分 → 规则奖励；开放任务且有偏好数据预算 → RM；多步推理需要过程引导 → verifier；工程上常见混合（规则为主 + RM 兜底）。混合时注意：不同来源的奖励要归一化到同一尺度，否则权重大的来源劫持训练。

【加分点】金句：「奖励是唯一的训练信号，奖励设计决定训练上界；无偏的弱规则往往优于可被 hack 的强 RM。」追问「怎么知道奖励被 hack 了」→ 看训练曲线：判分器分数涨、留出集真实质量不涨，就是 overoptimization 的经典信号。

【一句话总结】「按可得性、密度、可靠性三维选：能自动判分选规则，开放任务选 RM，要过程引导选 verifier，工程上规则为主 RM 兜底。」

### Q6｜在线与离线对齐的权衡？

【核心论点】本质是「数据分布与策略分布的匹配度」：离线便宜但不匹配，在线匹配但贵。工程上走折中：迭代离线 / 混合。

【参考展开】
- 离线（DPO/IPO/KTO/SimPO）：静态数据集，便宜、稳定、可复现。致命问题：策略每更新一次，数据里的反馈对「新策略」就失效一分（分布外），训练到平台期后可能越训越差。
- 在线（PPO/GRPO）：rollout 自当前策略，反馈永远匹配；新错误一出现就进下一批负样本，能被持续修复。代价：需要环境/判分器实时打分、采样与训练串行（慢）、基础设施复杂。
- 折中三招：迭代 DPO（每轮用当前策略重新生成偏好数据，把离线变「准在线」）；混合（离线冷启动 + 在线精修）；经验回放（旧数据按比例复用，稳定训练）。
- 什么时候该从离线切在线：离线训练进入平台期（loss 降、真实质量不涨）时——这是覆盖耗尽、需要新采样的信号。

【加分点】引出 offline RL 的两个理论概念：extrapolation error（在数据没覆盖的区域做乐观估计）与保守性约束；在线的核心价值是 coverage——覆盖策略真实行为空间。一句话总结：「离线对齐的上限是数据集的覆盖范围，在线对齐的上限是奖励信号的质量。」

【记忆锚点】「离线买便宜、在线买匹配；训练进入平台期，就是该从离线切在线的信号。」

### Q7｜你的项目为什么用 GRPO，不用 PPO/DPO？（结合 Search-R1 项目）

【核心论点】三个约束的交叉点：任务可自动判分（排除 DPO 的数据假设）、必须在线交互训练（排除 DPO 的离线范式）、无 critic 显存预算（排除 PPO）→ GRPO 是唯一交集。方法选择是约束求解，不是排行榜抄答案。

【参考展开】
- 先说任务：Search-R1 是检索增强的开放域问答，奖励由 EM 匹配 + 格式规则自动判分——廉价、无偏、可规模化的奖励信号，在线 RL 的前提成立。
- 为什么不是 DPO：① 没有现成偏好对数据；② 更关键——要训练的是「搜索-浏览-作答」的交互策略，奖励取决于在线搜索轨迹，静态偏好对表达不了这个闭环。任务需要在线交互式搜索训练，DPO 是离线偏好优化，被任务性质直接排除。
- 为什么不是 PPO：预算单卡/双卡。PPO 需要 critic（显存约翻倍）+ ref + RM 的完整栈，放不下；GRPO critic-free，一个可训练 actor + 冻结 ref 就能跑，与预算匹配。
- 效果口径（务必按此表述）：同管线前后对比，SFT 基线 1.43% → GRPO 81.2%。这是同一评测管线、同一判分器下的前后对比，不与任何外部论文做跨论文比较。
- 工程侧：生产化迁移用 veRL 框架，完成了环境调通，并对瓶颈做了定量分析——每步约 51K 个 TransferQueue 调度子操作、SimpleStorageUnit PUT_DATA 吞吐 ~736 次/分钟、单步时长 50-70 分钟；交叉验证（搜索后端从 0.68s 换 1-3s 再换 6-14s，步时几乎不变）证明瓶颈在框架调度层，不在搜索 API。

【加分点/追问预案】
- 「为什么不用 RLOO/ReMax？」→ 工程成熟度：GRPO 有成熟开源实现（veRL 等）与社区验证，RLOO/ReMax 要自己搭，额外风险换不来收益。
- 被问「你的结果和别人比怎么样？」→ 明确拒绝跨论文对比：「我只对比同管线的 SFT 基线，跨论文的数字没有可比性，我不做这种比较。」
- 被问「瓶颈为什么在调度层」→ 每步约 51K 个 TQ 子操作、PUT_DATA 吞吐仅 736 次/分钟；换搜索后端步时几乎不变证明了结论，这是量化过的数据，不是感觉。

【一句话总结】「GRPO 是被三个约束逼出来的交集：可自动判分、需在线交互、无 critic 预算——方法选择是约束求解。」

### Q8｜Best-of-N / rejection sampling 能替代 RL 训练吗？

【核心论点】不能。BoN 移动输出分布、不动参数：收益对数增长、推理成本线性增长、无 KL 控制会奖励黑客；RL 把「搜索最好」蒸馏进参数，训练贵一次、推理免费。BoN 的正确位置是兜底与造数据。

【参考展开】
- 数学直觉：reward 近似正态时，BoN 期望 ≈ r̄ + σ·√(2 ln N)。要多吃一个 σ，N 要涨近一个数量级——边际收益锐减；RL 的上界不受这个对数律约束。
- 成本：BoN 每次请求 ×N 生成成本、不可摊销；RL 训练成本被无限次推理摊薄。生产上 N 通常只能给到 4-8。
- 风险：无 KL 控制，大 N 时选中的是「骗过判分器」的答案（overoptimization）；RL 的 KL 项正是为此设计。
- 正确用法：数据不足时兜底；expert iteration 的过滤环节（BoN 造数据 → SFT → 迭代，等于「硬过滤版 RL」）；给 RL 效果当上界参照。

【加分点】金句：「BoN 是测试时搜索，RL 是训练时搜索，两者正交，现代系统常常同时用。」被追问「Rejection sampling + SFT 呢」→ 就是 expert iteration，是 RL 的退化特例，负信号全丢。

【一句话总结】「BoN 收益对数增长、成本线性增长、无 KL 控制；它是测试时搜索，替代不了训练时搜索，但可以兜底和造数据。」

---

## 附录 A：术语小词典（面试速记）

| 术语 | 一句话解释 |
|---|---|
| GAE | 广义优势估计：把多步 TD 误差按 γλ 加权求和，PPO 的优势来源 |
| critic | 价值网络：学「状态值多少钱」，PPO 用它算优势；GRPO 族不用它 |
| ref 模型 | 冻结的参照模型（通常 SFT 权重），KL 惩罚以它为锚，防策略跑飞 |
| rollout | 用当前策略在线采样一批（prompt, 回复, reward）三元组 |
| reward hacking | 模型优化奖励函数本身而非真实目标（格式刷满、迎合 RM 偏好） |
| overoptimization | RM 分数持续涨、真实质量见顶下跌的现象（Gao et al. 曲线） |
| 偏好对 | (prompt, chosen, rejected) 三元组，DPO 族的数据单元 |
| 单标签 | 只标「好/坏」不配对的数据，KTO 的数据单元 |
| 过程监督 | 对每个中间步骤给奖励信号（PRM）；对应 outcome 监督只给结尾 |
| 隐式奖励 | 不训练 RM、由 log(π/π_ref) 构造的奖励（DPO 推导、PRIME 逐 token 化） |
| sequential KL | 逐步 KL：逐 token 约束偏离幅度，比整条 KL 更严格（GSPO 关键词） |
| IMED | GSPO 中用于无 ref 估计 sequential KL 的目标（⚠️ 细节待核实） |
| expert iteration | 采样→过滤→SFT 循环，硬过滤版 RL |
| 自博弈 | 模型与自身（历史版本）对弈，对手同步变强（SPIN/SPPO） |
| 宪法 | Constitutional AI 中的成文原则，AI 反馈的评判标准 |
| boomerang | BoN/RL 中 RM 分数涨、真实质量反而回落的「回旋镖」现象 |
| 组内归一化 | GRPO 的 (r − mean_k)/std_k，长度偏差的来源之一 |
| clip higher | DAPO 的 ratio 裁剪：限上不限下，防熵坍缩 |
| 动态采样 | DAPO：过滤优势全零的无效组再更新 |
| overlong shaping | DAPO：超长截断样本给惩罚而非丢弃，保留负信号 |
| 优势均摊 | 组级优势复制给序列内每个 token（GRPO 默认做法） |
| 熵坍缩 | 策略输出分布过度尖锐化、多样性消失，训练崩坏的信号 |
| 性能悬崖 | 策略更新过大导致能力断崖下跌（TRPO/PPO 要防的事） |
| 留出集 | 不参与训练的评测数据，用于检测 overoptimization |

---

## 附录 B：方法演变时间线（面试「讲脉络」用）

```
2017  PPO（1707.06347）：actor-critic + clip，RL 的标准范式
2022  RLHF/InstructGPT：PPO 搬进 LLM，四模型栈成为默认
2023  DPO（2305.18290）：偏好优化一步到位，RLHF 的代数重写
2023  Let's Verify Step by Step（2305.20050 ⚠️）：PRM800K，过程监督登场
2023  ReMax（2310.10505）：最早的 critic-free reward 基线变体
2024  GRPO（2402.03300，DeepSeekMath）：组内归一化，critic-free 成为主流
2024  RLOO（2402.14740）：留一基线，教科书级简化
2024  SPIN（2401.01335 ⚠️）：自我博弈 SFT
2024  ORPO/SimPO/SPPO：DPO 族去 ref、自博弈变体
2025  PRIME（2502.01456）：隐式 PRM，outcome 驱动过程奖励
2025  DAPO/Dr.GRPO/VAPO：GRPO 修复（偏差、塑形）与 critic 路线反攻
2025  REINFORCE++（2501.03262 ⚠️）：PPO 工程件搬进 REINFORCE
2025  GSPO（2507.18071 ⚠️）/EXO（⚠️）：去 ref、RL 风格预训练
```

脉络三句话：2022 年四模型栈定标准 → 2023-2024 年两条省成本的路（DPO 去采样、GRPO 去 critic）→ 2025 年进入「修 GRPO」和「去 ref」的工程深水区。

---

## ⚠️ 标注汇总（引用前必核对）

1. REINFORCE++ arXiv 2501.03262（待核实）
2. Let's Verify Step by Step arXiv 2305.20050（待核实）；PRM800K 数据集规模数字（待核实）
3. EXO 的 arXiv 编号（待核实；勿与 2018 年论文 1812.00116 混淆）
4. GSPO arXiv 2507.18071 及核心细节（待核实）
5. SPIN arXiv 2401.01335（待核实）
6. BoN 的 KL 增长公式（待核实）
7. 速查表中 SPPO 2405.00675、VAPO 2504.05118、Dr.GRPO 2503.20783、STaR 2203.14465 四个编号（待核实）；GSPO 整行（待核实）

**结束语**：本部分到此收束。考前最后一遍建议顺序：速查表三行速记 → 选型决策树 → 八道问答的核心论点 → ⚠️ 汇总。祝面试顺利。

---

## 附录 C：Q7 完整话术示范（3 分钟版，可直接背诵）

> 面试官：「你这个 Search-R1 项目为什么用 GRPO，没用 PPO 或者 DPO？」

「我从任务特征和资源约束两个角度说。

第一，任务特征。Search-R1 是检索增强的开放域问答，评测靠 EM 匹配加格式规则自动判分——奖励信号廉价、无偏、可以无限采样，这是在线 RL 的前提。

第二，为什么排除 DPO。DPO 需要偏好对数据，我们既没有现成偏好对；更关键的是，这个任务训练的是『搜索-浏览-作答』的交互策略，奖励取决于在线搜索轨迹，静态偏好对表达不了这个闭环。任务天然需要在线交互式搜索训练，所以 DPO 这条离线路线被任务性质直接排除。

第三，为什么排除 PPO。我们预算是单卡到双卡。PPO 需要 critic 价值网络，显存量级接近翻倍，加上 ref 和奖励模型就是完整四模型栈，放不下。GRPO 是 critic-free 的，一个可训练 actor 加一个冻结 ref 就能跑，和预算正好匹配。

效果上我按同管线前后对比来表述：SFT 基线 1.43%，GRPO 训到 81.2%，是同一评测管线、同一判分器下的对比，我不做跨论文比较。

工程上我们用 veRL 框架完成了环境调通，并且对瓶颈做了定量分析：每步约 51K 个 TransferQueue 调度子操作，SimpleStorageUnit 的 PUT_DATA 吞吐约 736 次每分钟，单步 50 到 70 分钟；换搜索后端（0.68 秒、1 到 3 秒、6 到 14 秒三种）步时几乎不变，交叉验证了瓶颈在框架调度层，不在搜索 API。

总结一句：选 GRPO 是被约束逼出来的交集——可自动判分、需在线交互、无 critic 预算，三个条件取交集就是 GRPO。方法选择是约束求解，不是排行榜抄答案。」
