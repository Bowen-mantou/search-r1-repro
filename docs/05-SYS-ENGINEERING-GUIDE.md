# 系统工程教学指南：分布式训练与推理全栈

> 覆盖：DDP/FSDP/ZeRO/TP/PP 分布式训练、vLLM 推理引擎、Ray + veRL 框架架构、
> OOM 排查方法论、性能分析、云 GPU 工程实践，以及对应的面试问答。
> 所有内容以「2×4090 训 4B 模型」的 Search-R1 复现项目为主线串联
> （项目代码：`C:\new\intern\plan\projects\search-r1\`）。
> 配套阅读：《GPU 显存计算专题》（01-GPU-MEMORY-GUIDE.md）——本指南的显存账
> 都假设你已掌握那份专题的算法。

---

## 第 0 章 导读：一张图看懂"训练一个大模型要解决的全部系统问题"

```
                        ┌─────────────────────────────────────────────────┐
                        │           一个 LLM RL 训练系统的五个子系统       │
                        └─────────────────────────────────────────────────┘

 ① 数据流            ② 训练引擎            ③ 推理引擎          ④ 调度层
 数据加载/清洗   →   FSDP 分布式训练    →   vLLM 高效推理   →   Ray 角色编排
 parquet/JSONL       权重分片/梯度同步       PagedAttention       + TransferQueue
                     CPU offload/OOM         continuous batching   agent loop 请求调度

                          ⑤ 工程保障
               probe 体检 / 止损规则 / 环境治理 / 日志观测 / 成本控制
```

面试官问"系统工程"时，本质是在考你对这五个子系统的**分工与协作**的理解——
谁能把"一个 batch 从磁盘走到梯度更新再走回推理引擎"的完整旅程讲清楚，谁就赢。

### 0.1 本指南的读法

- **第 1 章（分布式训练）**：回答"一张卡放不下怎么办"。DDP→FSDP 的演进逻辑 +
  OOM 排查方法论（含六轮定位实战复盘）。
- **第 2 章（vLLM）**：回答"推理为什么能快 3-5 倍"。PagedAttention、continuous
  batching、混合架构（Mamba）适配。
- **第 3 章（Ray + veRL）**：回答"几千个小请求怎么组织起来"。角色分工、DataProto、
  AgentLoop 契约、权重同步、TransferQueue 调度、reward 字段流转。
- **第 4 章（工程实践）**：回答"怎么不浪费 GPU 小时"。环境治理、云 GPU 成本、
  probe 文化、止损规则、性能分析 + 10 道面试问答。
- **第 5 章**：速查表（术语/关键数字/报错三列表）。

### 0.2 五个子系统的一句话地图

| 子系统 | 核心问题 | 核心答案 | 你的项目证据 |
|---|---|---|---|
| 数据流 | 数据怎么变成 token 批 | parquet + chat template + padding/mask | 169615 行训练集、dev 70 题 |
| 训练引擎 | 显存不够怎么办 | FSDP 分片 + CPU offload + 梯度检查点 | 18.9GB/卡（余 4GB） |
| 推理引擎 | 生成为什么慢 | vLLM 分页 KV + 连续批处理 | 步时从 HF 的 ~16min 设计为 3-5min |
| 调度层 | 多轮请求怎么组织 | Ray actor + TransferQueue 队列 | TQ 定量分析：51K 操作/步 vs 736/min |
| 工程保障 | 怎么不烧冤枉钱 | probe + 止损规则 | 47 项检查 0 FAIL 才启动 |

### 0.3 学习顺序建议

1. 先读《GPU 显存计算专题》——所有分布式讨论的地基；
2. 第 1 章（训练侧）→ 第 2 章（推理侧）→ 第 3 章（两者怎么合起来）——
   这个顺序对应"先分开理解，再理解协作"；
3. 第 4 章当故事书读（16 个坑的底层机制），面试前再精读问答部分；
4. 每章结尾的「面试记忆点」攒起来就是你的面试速记卡。

---
# 《系统工程教学指南》Part 1：分布式训练基础 + OOM 排查方法论

> 适用读者：有过 2×4090 训 4B 模型实战、正在准备算法岗面试的人。
> 每个概念按固定结构展开：一句话定义 → 原理展开 → 图/表 → 联系读者项目 → 面试记忆点。
> 口径纪律：显存与吞吐数字优先引用本项目真实测量值；标 ⚠️ 处为推导值或记忆值，以官方文档/实测为准。

## 项目文件索引（本文案例的数据来源）

| 文件 | 内容 | 本文用到的部分 |
|---|---|---|
| `agentic-rl-lab/03-search-r1/verl_search_r1/e6b_run.sh` | veRL GRPO 启动脚本 | 显存账注释、offload 配置、vLLM 池配比 |
| `agentic-rl-lab/03-search-r1/train_grpo_local.py` | 本地 GRPO 训练脚本 | MICRO_BATCH_MAX_TOKENS、MAX_TRAIN_SEQ_LEN |
| `2026-08-30-OOM-HANDOFF.md` | OOM 六轮调试交接 | 观测数据、排除项、v6 修复结论 |

本文的"读者项目"指同一件事：在 2×4090（24GB/卡）上训练 Qwen3.5-4B 级模型（bf16 权重 8.4GB），
本地用 train_grpo_local.py 训练、云端用 veRL 全参数 GRPO 训练，后者在权重同步阶段触发过一场
六轮 OOM 排查战——这就是 Part B 的核心案例。

---

# Part A 分布式训练基础

## A.1 为什么需要分布式：显存墙与算力墙

**一句话定义**：分布式训练的动机只有两个——单卡显存放不下（显存墙）、单卡算得太慢（算力墙）。

**原理展开——先算显存账（4B 模型，本项目口径）**：

```
bf16 权重   = 4.2B 参数 × 2 字节 ≈ 8.4 GB
梯度        ≈ 8.4 GB
Adam 状态   ≈ 25.2 GB（本项目简化口径 = 3 × 权重字节数，⚠️ 见下）
─────────────────────────────
静态合计    ≈ 42.0 GB
+ 激活 ~1.5GB + 推理引擎权重副本 8.4GB ≈ 51.9 GB  >>  单卡 24 GB  ← 显存墙
```

⚠️ Adam 口径说明：上表沿用 e6b_run.sh 注释的简化账（Adam = 3 × 权重）。教科书口径里
Adam 两个矩估计与 fp32 master 权重各占 4 字节/参数，优化器态可能高达 12 字节/参数（本项目
规模约 50GB）；框架实现差异大，显存账务必以实测为准。

**算力墙**：训练每 token 约 6×参数量 次浮点运算（前向 2P + 反向 4P，P=参数量）。
4B 模型、单步 10 万 token：6 × 4.2e9 × 1e5 ≈ 2.5e15 FLOPs。单卡 4090 的 bf16 峰值算力
约 165 TFLOPS（⚠️ 以官方规格为准），按 30-50% 实际利用率算，单步几十秒量级起步；
batch 增大、步数增多后单卡训练时间是"天"级。多卡就是把这面墙拆掉。

**总图：多卡到底要解决什么问题**

```
                 ┌────────────── 两大墙 ──────────────┐
                 │ 显存墙：单卡 24GB < 42GB+ 静态需求  │
                 │ 算力墙：单卡算不完 / 算得太慢       │
                 └────────────────────────────────────┘
            多卡分工的三种答案（按"切什么"分类）:
  数据并行 DDP          状态分片 ZeRO/FSDP        层内/层间切 TP/PP
  每卡一份完整模型       每卡只持 1/N 状态          每卡部分矩阵 / 部分层
  梯度 all-reduce 同步   用时拼齐、用完即丢          每步同步激活 / 流水线
  不省显存，纯加速       省显存，通信量上升          省单卡显存，要求高带宽
```

**联系读者项目**：e6b_run.sh 的显存账注释把这件事算得很清楚——不开 offload 时
N=1 卡需 51.9GB、N=2 卡需 26.7GB（含 vLLM 权重副本），24GB 卡必爆。这就是 2×4090
这台"最小多卡实验室"被逼出来的原因：不是想炫技，是单卡根本装不下。

**面试记忆点**：分布式只解决两个问题——显存放不下与算得慢；动手选型前先列显存账
（权重 2 字节/参数 + 梯度 + Adam + 激活 + 推理副本），账算不清方案必错。

## A.2 DDP：数据并行

**一句话定义**：每张卡持有一份完整模型副本，各自前向反向，梯度 all-reduce 求平均后各自更新。

**原理展开**：DDP 只切数据不切模型。每卡拿不同 mini-batch，独立前向/反向得到梯度 g_i，
全卡对梯度做 all-reduce（求和/平均），更新后的权重各卡完全一致，进入下一步。
注意：优化器状态（Adam）也是每卡完整一份——所以 DDP 的显存和单卡训练一样大。

**ring all-reduce 原理（为什么比 gather+broadcast 好）**：

```
gather+broadcast（中心化）:            ring all-reduce（去中心化）:
  卡0 收齐 N 份梯度 → 平均            梯度切成 N 块，卡 i 持有第 i 块
  → 广播回 N-1 张卡                    第 1 步起，沿环传 (N-1) 轮：每轮各卡
  卡0 收发 2(N-1)G，成为带宽瓶颈       收一块、归约一块、传一块（scatter-reduce）
                                      再传 (N-1) 轮 all-gather，人人拿全
  总通信 2(N-1)G 压在一条链路上        每卡总收发 2(N-1)/N × G ≈ 2G，带宽分散到所有卡
```

ring 的妙处：把"主卡吞吐"瓶颈变成"全网卡带宽之和"，每个节点只用满自己的双向带宽——
这是通信量意义上接近最优的方案（每卡至少要把 G 发出去、把归约结果拿回来，≈2G 是下界）。

**通信量公式**：

```
每卡每步通信量 = 2 × (N-1)/N × G        （G = 梯度总字节数，N = 卡数）
N=2, G=8.4GB → 每卡 8.4GB；N=8 → 14.7GB；N→∞ 极限 → 2G（与卡数无关）
```

通信时间估算：2×4090 之间没有 NVLink（消费卡 4090 起取消了 NVLink 桥接），走 PCIe
Gen4 x16，理论带宽约 32GB/s（⚠️ 以硬件规格为准），8.4×2/32 ≈ 0.5s/步，相对几十秒的
计算可忽略；但若换 70B 模型（G≈140GB），约 8.8s/步，通信就不可忽略——这就是大模型
必须上 NVLink/InfiniBand 的原因。

**缺点**：显存完全不省（每卡仍是全量权重+梯度+Adam），DDP 是纯"算得快"方案。

**联系读者项目**：2×4090 场景下 DDP 两步就能被否掉：① 单卡 43.5GB+ 的需求远超 24GB，
模型根本放不下；② 4090 无 NVLink，梯度同步走 PCIe。所以 e6b 最终没走 DDP 而走了 FSDP。
但 DDP 的 ring all-reduce 是理解一切"分片方案通信代价"的基准尺。

**面试记忆点**：DDP = 每卡全量副本 + 梯度 all-reduce；每卡通信量 2(N-1)/N·G 与卡数几乎
无关（趋近 2G），它省的是时间不是显存。

## A.3 ZeRO 三阶段：分片进化史（DeepSpeed）

**一句话定义**：ZeRO 按"优化器态 → 梯度 → 权重"的顺序逐级分片，把每卡显存从"整份"压到"1/N 份"。

**原理展开——三个问题的分片账（4B 模型、2 卡、单位 GB，激活 ~1.5 不分片）**：

```
| 方案    | 权重 | 梯度 | Adam | 每卡小计 | 通信量(相对DDP) | 分片对象       |
|---------|------|------|------|---------|----------------|---------------|
| 无 ZeRO | 8.4  | 8.4  | 25.2 | 43.5    | 1×             | 无             |
| ZeRO-1  | 8.4  | 8.4  | 12.6 | 30.9    | 1×             | Adam 状态      |
| ZeRO-2  | 8.4  | 4.2  | 12.6 | 26.7    | 1×             | + 梯度         |
| ZeRO-3  | 4.2  | 4.2  | 12.6 | 22.5    | ≈1.5×          | + 权重         |
```

- **ZeRO-1 分片优化器状态**：Adam 是最大头（25.2GB），先把它切成 N 份，每卡只更新自己
  分片里的参数，更新完再通信同步。梯度仍全量 all-reduce，通信量与 DDP 相同。
- **ZeRO-2 再分片梯度**：梯度也切成 N 份，reduce-scatter 取代 all-reduce——每卡只保留
  自己分片对应的梯度，归约过程本身就完成了分片。
- **ZeRO-3 再分片权重**：权重也分片常驻，前向/反向需要全量权重时临时 all-gather 拼齐、
  用完即丢。这就是 FSDP 的蓝图。
- **通信代价**：ZeRO-1/2 通信量与 DDP 持平；ZeRO-3 因为权重要反复拼齐，通信量升到约
  1.5× DDP（⚠️ 近似推导，见 A.4）。

**联系读者项目**：e6b_run.sh 注释里"权重 8.4/N + 梯度 8.4/N + Adam 25.2/N"就是 ZeRO-3
口径的每卡账。N=2 时 22.5GB 看似贴着 24GB 能过，但 vLLM 推理引擎的权重副本 4.2GB 还没
算——26.7GB 仍然爆。结论：卡少时分片摊薄有限（N=2 只能摊一半），分片不够用，还得叠加
CPU offload（A.4.7）。这是"ZeRO 有天花板"的最直观例证。

**面试记忆点**：ZeRO 三步曲 = 先分 Adam（最大的 25.2GB）、再分梯度、最后分权重；每卡
显存按 1/N 逼近"单卡整份"，通信代价逐级上升——ZeRO-3 的权重分片是它与 DDP 的本质区别。

## A.4 FSDP 深度讲解（重点）

**一句话定义**：FSDP 是 PyTorch 官方的 ZeRO-3 实现——权重以分片形式常驻，前向/反向时
all-gather 拼齐、用完立刻丢弃。

### A.4.1 核心循环：unshard → compute → reshard

```
rank_i 常驻状态: 只持有本卡权重分片 8.4/N GB（可放 CPU，见 A.4.7）

每一层的生命周期:
  unshard : all-gather 从各卡收集分片 → 拼出本层全量权重（显存临时 +全量）
  compute : 用全量权重做前向，得到本层激活
  reshard : 前向用完，丢弃全量权重 ← 显存回落
  ── 反向时再来一遍 ──
  unshard : 再次 all-gather 拼齐（前向那次已丢弃，不存全量）
  backward: 用全量权重 + 本层激活算梯度
  reduce-scatter: 梯度规约并切回 1/N 分片，全量权重再次丢弃
```

任一时刻，GPU 上只有"一层全量权重 + 累积的分片 + 激活"，这就是 FSDP 能把 42GB 静态
需求压到 1/N 的机制。

### A.4.2 为什么反向算完立刻丢弃分片（逐层释放）

反向传播对权重的需求是"逐层且一次性的"：算第 l 层梯度时只需要第 l 层权重和第 l 层
激活；算完第 l 层，第 l 层全量权重对第 l-1 层反向毫无用处，立刻丢弃。若不丢，所有层的
全量权重会同时驻留，退化成"分片 + 全量并存"的峰值叠加（这正是 Part B 案例的 OOM 根因）。
"用完即丢"是 FSDP 省显存的全部秘密，代价是反向时每层要再拼一次（前向拼的那份不保留）。

**面试记忆点**：FSDP 的本质是"权重平时分片躺着，用时拼齐、用完即丢"；说得出"反向还要
再 all-gather 一次"说明你真的懂它。

### A.4.3 通信开销：与 DDP 的对比

```
每卡每步 ≈ 2 次 all-gather（前向拼、反向拼）+ 1 次 reduce-scatter（梯度）
          = 3 × (N-1)/N × G          （G = 全量权重字节数）
DDP 每卡每步 = 2 × (N-1)/N × G（梯度 all-reduce）
→ FSDP ≈ 1.5× DDP 通信量（⚠️ 近似推导，忽略激活通信与重叠优化）
```

```
| 维度       | DDP              | FSDP (FULL_SHARD)      |
|-----------|------------------|------------------------|
| 每卡权重显存 | G（全量）          | G/N（分片）             |
| 通信时机    | 每步 1 次（梯度）   | 每层 3 次（拼/拼/切）     |
| 通信量      | 2G               | ≈3G                     |
| 结论       | 零显存收益         | 显存换通信的经典 trade-off |
```

**面试记忆点**：FSDP 是"显存换通信"——显存降到 1/N，通信涨到约 1.5 倍；问通信量就答
"两遍 all-gather + 一遍 reduce-scatter，约 3G/步/卡"。

### A.4.4 sharding_strategy 四种策略

```
| 策略            | 等价物            | 分片对象          | 典型场景               |
|----------------|------------------|------------------|-----------------------|
| NO_SHARD       | DDP              | 无               | 单卡装得下，只要加速      |
| SHARD_GRAD_OP  | ZeRO-2           | 梯度 + Adam      | 权重装得下、优化器态爆    |
| FULL_SHARD     | ZeRO-3           | 权重+梯度+Adam    | 权重本身装不下（本项目）   |
| HYBRID_SHARD   | 节点内FSDP+节点间DDP | 节点内权重等      | 多节点大集群             |
```

**面试记忆点**：FULL_SHARD 才是 ZeRO-3，SHARD_GRAD_OP 是 ZeRO-2，NO_SHARD 是 DDP——
先分清这层对应关系，面试就不会张冠李戴。

### A.4.5 wrap_policy：分片粒度（min_num_params）

`wrap_policy` 决定"哪些参数组成一个 FSDP 单元"，单元就是拼齐/释放的最小粒度。
`min_num_params` 是常见开关：参数总量超过阈值的子模块各自成单元。标准做法是按
Transformer 层切——每层一个单元（4B 模型每层约 200MB 量级 ⚠️ 以模型 config 为准）。

```
| 粒度         | 效果                                             |
|-------------|--------------------------------------------------|
| 太细（单元过小） | 通信次数爆炸、all-gather 碎片化，调度开销吞掉收益       |
| 太粗（整模型一单元）| 无法逐层释放 → 全量权重一直并存 → 退化成 DDP 的显存     |
| 每层一个单元（标准）| 逐层拼齐/释放，峰值 ≈ 分片 + 单层全量 + 激活            |
```

**面试记忆点**：wrap_policy 决定"拼齐/释放"的最小单位，按 Transformer 层切是标准答案；
问"为什么不能整模型一个单元"——因为那样永远无法逐层丢弃全量权重。

### A.4.6 get_per_tensor_param_shard vs get_per_tensor_param（结合本项目 OOM 修复）

这是 FSDP 使用者最容易踩的接口坑，也是本项目六轮 OOM 的最终答案：

```
get_per_tensor_param()        → 返回"全量参数"视图
                                实现上会触发 unshard / all-gather
                                把 8.4GB 全量权重物化到 GPU
                                而本卡分片 4.2GB 还在 → 峰值 12.6GB

get_per_tensor_param_shard()  → 返回"本 rank 本地分片"视图
                                官方注释: "yields each rank's *local* FSDP shard"
                                不 unshard、不 all-gather → 只碰 4.2GB
```

本项目 update_weights 阶段的 OOM 正发生在走第一条路径时：全量物化 8.4GB 与分片 4.2GB
并存（峰值 12.6GB），再叠加 state_dict 构建的临时对象，在申请 1.19GB 时爆掉。v6 修复
就是换成 shard 接口 + FULL_STATE_DICT + offload_to_cpu 流式，峰值 12.6→约 5GB
（细节见 B.3）。

**面试记忆点**：名字带 shard 的接口拿"分片视图"，不带的是"全量物化视图"——调用任何
FSDP 接口前先问一句"这会不会触发 all-gather 物化全量"，OOM 往往就在这一行。

### A.4.7 CPU offload：搬到 CPU 什么、什么时候搬、代价多少

```
optimizer_offload: Adam 状态 + fp32 master 权重搬 CPU 常驻
                   优化器更新时，把"本卡分片 + 对应 Adam 状态"经 PCIe 搬回 GPU
                   算完更新再搬回 CPU —— GPU 上从不常驻优化器态（省下 25.2/N GB）
param_offload:     权重分片本身也搬 CPU 常驻
                   unshard 需要拼齐时，把分片经 PCIe 搬回 GPU，用完放回 CPU
```

代价是 PCIe 带宽：Gen4 x16 理论约 32GB/s（⚠️ 以硬件规格为准），对比 4090 显存带宽
1TB/s 量级（⚠️ 以官方规格为准）差一个数量级。所以 offload 的代价不是显存而是步时——
e6b_run.sh 注释实测口径："参数每步流式进出，步耗时 +30-60s"。

**联系读者项目（e6b_run.sh 开 offload 后的显存账，N=2）**：

```
训练侧: 梯度 8.4/N + 激活 ~1.5 ≈ 4.2 + 1.5 = 5.7GB（权重与 Adam 已上 CPU）
vLLM 池: 24GB × 0.55 ≈ 13.2GB（offload 后训练侧几乎不占 GPU，池配比从 0.45 提到 0.55）
合计 ≈ 18.9GB/卡 ✓ 余 ~4GB（N=1 降级方案 ≈ 20.7GB ✓ 余 ~2GB）
```

offload 生效的铁证来自第 6 轮观测：`before load, allocated 0.00`——参数已全部 CPU
offload，PyTorch 分配器视角归零。这个数字本身也成了后续定位 OOM 的锚点（见 B.3）。

**面试记忆点**：CPU offload 是拿 PCIe 慢带宽当租金换显存——只适合显存实在不够、
步时不敏感的场合；说代价时答"步时 +30-60s 量级"比背带宽数字更接地气。

### A.4.8 梯度检查点（activation checkpointing）：用重算换激活显存

**一句话定义**：前向不存中间激活，反向时把对应层的前向"再算一遍"来恢复激活。

```
无检查点: 前向存下每层激活
          激活显存 ≈ 2 × S × B × H × L 字节（S=序列长, B=微批, H=hidden, L=层数）
          （⚠️ 简化公式，忽略注意力中间量，bf16 口径）

有检查点: 前向只存"每层输入"（checkpoint 边界）
          反向到第 l 层 → 用 checkpoint 重算第 l 层前向 → 得到激活 → 算梯度 → 丢弃
          激活显存 ≈ 2 × S × B × H（单层量级）→ 从 O(L) 降到 O(1) 层
          计算代价：前向总量多算约 33%（每层重算一遍前向）
```

**联系读者项目**：e6b_run.sh 开了 `enable_gradient_checkpointing=true`，配合注释里的
"激活 ~1.5GB"口径。本地脚本 train_grpo_local.py 是"三保险"的完整示范：

```
MICRO_BATCH_MAX_TOKENS = 4000   # 按 token 数打包微批（pack_micro_batches），限制同时存活的激活
MAX_TRAIN_SEQ_LEN = 4096        # 注释原文: "hard cap per training sequence to prevent OOM"
```

截序列、限 token、重算激活——三道闸分别控制激活显存的"单样本长、样本数、存储方式"。

**面试记忆点**：梯度检查点是时间换空间——激活显存从 O(L) 降到 O(1) 层，多付约 33%
前向算力；问"为什么多算不亏"——因为显存才是硬约束，算力多等 30% 好过根本跑不起来。

## A.5 TP/PP 概览：切矩阵与切层

### 张量并行（TP）

**一句话定义**：把单个矩阵乘法切到多卡并行，每层前向需要 all-reduce 同步激活。

**原理**：以 MLP 两块矩阵为例——第一块按列切（column-parallel）：W=[W1|W2]，两卡各算
X·W1 与 X·W2，输出 all-reduce 拼回；第二块按行切（row-parallel）：输入天然是两半，各算
各的，输出 all-reduce 相加。注意力层同理。每层至少 2 次 all-reduce，通信发生在每一次
前向的每一层——频率极高，因此只在机内高带宽互联（NVLink/NVSwitch）下可行。

**何时需要**：单卡连"单层激活"都放不下（超长序列、超大 hidden），或单层权重都放不下时。
4B 模型单层激活很小，用不上 TP。

### 流水线并行（PP）

**一句话定义**：按层把模型切成 P 段，每卡跑一段，microbatch 像流水线一样推进。

**原理与气泡**：第 1 个 microbatch 进入时，后面的 stage 全在等；流水线填满前存在
"气泡"（空闲）。气泡比例公式：

```
气泡比例 = (P-1) / (P-1 + m)      （P = 段数，m = microbatch 数）
m 越大气泡占比越小；1F1B 调度可进一步减少气泡与显存峰值
```

PP 的通信只在段边界传激活，量小；但它**不省单卡权重显存**（每卡要放它那一段的完整
权重+优化器态），且段数少时气泡大、利用率低。

### 3D 并行

大集群（70B+）的标准组合：机内 TP（高带宽）、机间 PP（低通信）、数据维度 FSDP/DDP。
三种并行各治一种病，组合起来覆盖全部瓶颈。

**联系读者项目**：2×4090 场景下 TP 与 PP 各有一个硬伤——4090 无 NVLink，TP 每层
all-reduce 走 PCIe 会吃光收益；2 卡 PP 只有 2 段、气泡大且每卡仍要半张模型的权重。
所以 e6b 最终是 FSDP + CPU offload + 梯度检查点 + 训练/推理分时（vLLM sleep 机制 +
v1/v2 补丁的 load/offload 闭环）。选型不是"谁先进用谁"，是"谁没有硬伤用谁"。

**面试记忆点**：TP 治"单层都放不下"（通信换并行）、PP 治"层数太多"（有气泡代价）、
FSDP 治"状态放不下"（显存换通信）——三种并行治三种病，混合使用叫 3D 并行。

## A.6 并行选择决策树

```
① 算显存账: 权重(2B/参数) + 梯度 + Adam + 激活 + 推理副本 → 单卡放得下吗?
     ├─ 放得下 → DDP（要加速）/ 单卡直接训
     └─ 放不下 → 看"谁放不下"（瓶颈对象定位）
           ├─ 优化器态爆 → ZeRO-1 / optimizer_offload
           ├─ 梯度也爆   → ZeRO-2
           ├─ 权重也爆   → ZeRO-3 / FSDP（+ param_offload 兜底）
           ├─ 单层激活爆 → TP（前提: 机内高带宽）
           ├─ 层数太多、单卡装不下若干层 → PP（注意气泡）
           └─ 70B+/大集群 → 3D: DP×TP×PP 混合
② 带宽约束复核: 卡间无 NVLink（如消费卡 2×4090）→ TP 直接出局，
   分片方案优先；NVLink/InfiniBand 集群 → 通信方案选择空间更大
```

**联系读者项目落位**：4B + 2×4090 → 瓶颈是 Adam(25.2GB) 与推理副本 → 卡少、无 NVLink
→ FSDP(FULL_SHARD) + optimizer/param 双 offload + 梯度检查点 + 训练/推理分时——这正是
e6b_run.sh 的最终配置，代价是步时 +30-60s，换来的是一次能跑起来。

**面试记忆点**：并行选型是"瓶颈定位题"不是"背配置题"——先说出显存账里最大的那块
（本项目是 Adam 的 25.2GB），再给方案，最后复核带宽约束。

---

# Part B OOM 排查方法论（本项目六轮复盘为核心案例）

## B.1 显存观测工具箱

**一句话定义**：三个视角各管一段——nvidia-smi 看卡、allocator 看 PyTorch、峰值看历史。

### 三个工具的职责边界

```
nvidia-smi                    进程级、驱动视角：本进程向驱动"买断"的全部显存
torch.cuda.memory_allocated() PyTorch 分配器视角：tensor 实际占用的字节
torch.cuda.memory_reserved()  分配器从驱动整块买断的缓存池（含未使用的余量）
torch.cuda.max_memory_allocated() 峰值 allocated（事后复盘最有用的一行）
```

**allocated 与 reserved 的区别（缓存不归还）**：PyTorch 分配器释放 tensor 时把内存
还给自己的缓存池，而不是还给驱动——因为向驱动申请/归还显存很慢。所以 OOM 报错里的
"free 795MiB"是缓存池内的余量，不是整卡余量。本项目 OOM 报错原文：

```
torch.OutOfMemoryError: 申请 1.19 GiB, free 795 MiB, 本进程 21.94 GiB, allocated 20.67 GiB
```

解读：本进程向驱动买断 21.94GB，其中 20.67GB 被 tensor 实际占用，池内余量（free）仅
795MB，而新对象要 1.19GB——缺口 = 1.19 - 0.795 ≈ 400MB。

**观测点插入写法（本项目 v5 补丁，logger=None 直打）**：

```python
def obs(tag):
    import torch
    print(f"[E6B] {tag}: allocated {torch.cuda.memory_allocated()/2**30:.2f} GB, "
          f"reserved {torch.cuda.memory_reserved()/2**30:.2f} GB, "
          f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB", flush=True)
```

要点：训练框架的 logger 可能有异步缓冲、rank 过滤或级别过滤，排查时直接 print + flush，
绕过日志框架——这是第 6 轮能一次定位的前提。

**一个经典陷阱**：第 6 轮观测里 `before load, allocated 0.00` 但 `used 1.37/23.52`——
nvidia-smi 显示占了 1.37GB，PyTorch 视角却是 0。这 1.37GB 是 CUDA context + vLLM sleep
状态残留（约 788MB）等非 tensor 占用。两个视角打架时，以"哪个视角离出错代码近"为准。

**面试记忆点**：allocated 是"真实在用"，reserved 是"买断的缓存池"，OOM 报错里的 free
是池内余量——三者都背得出，面试官就知道你真的看过报错。

## B.2 OOM 分类学：五种 OOM 与各自的修法

**一句话定义**：OOM 不是一个病，是五种病——先分类再动手，"什么时候爆"决定"修哪里"。

```
| 类别          | 典型特征                          | 修法                              | 本项目对应                     |
|--------------|-----------------------------------|----------------------------------|-------------------------------|
| 前向 OOM      | forward 中申请激活失败               | 砍 micro batch / 序列长 / 检查点   | MICRO_BATCH_MAX_TOKENS=4000   |
|              |                                   |                                  | MAX_TRAIN_SEQ_LEN=4096        |
| 反向 OOM      | backward 中爆（梯度+激活并存）        | 梯度检查点 + ZeRO-2 分梯度         | enable_gradient_checkpointing |
| 优化器 OOM    | step/Adam 初始化时爆                | ZeRO-1 / optimizer_offload       | optimizer_offload=true        |
| 碎片化 OOM    | 申请小块失败但 reserved 总量大        | expandable_segments + GC 阈值     | GC 实测只能回收 ~3%           |
| 峰值叠加 OOM  | 两个大对象并存（临时峰值）            | 错峰 / 流式 / 拿 shard 不物化      | update_weights 12.6→~5GB      |
```

逐类一句话：

- **前向 OOM（激活爆）**：激活显存随序列长与微批线性涨，特征是在 forward 里的某次
  tensor 分配失败。修法是三闸：截序列、限 token、梯度检查点。
- **反向 OOM（梯度+激活）**：反向要同时持有本层激活与梯度缓存，比前向峰值更高。
  修法是检查点压缩激活、ZeRO-2 分掉梯度。
- **优化器 OOM**：step 里 Adam 要一次性实例化 m/v/master，25.2GB 级的大头通常在这
  一步爆。修法是 ZeRO-1 分片或 optimizer offload。
- **碎片化 OOM**：reserved 池总量够，但没有一块连续的放得下新 tensor——特征是申请
  的块不大却失败。修法是 `PYTORCH_CUDA_ALLOC_CONF` 的 expandable_segments（减少碎片）
  与 garbage_collection_threshold（回收池内余量）。本项目实测：GC 只能回收约 3% 的
  reserved，对 400MB 缺口杯水车薪。
- **峰值叠加 OOM**：平时显存够，某一步两个大对象同时在场——本项目 update_weights
  阶段全量物化 8.4GB 与分片 4.2GB 并存即属此类。修法不是"再挤一点"，而是"别让它们
  并存"：错峰（先卸载再加载）、流式、拿分片视图。

**面试记忆点**：报错信息先看三样——申请多大、free 多少、allocated 峰值多少——再对照
这张表分类；分类对了，修法就是查表。

## B.3 六轮复盘：update_weights 阶段的 400MB 缺口之战

**背景**：E6-B 在 veRL 0.10 上跑 4B 全参数 GRPO（2×4090），训练侧开双 offload 后显存
账约 18.9GB/卡本来够用，却在"把训练权重同步给 vLLM 推理引擎"（checkpoint 的
update_weights）时爆发 CUDA OOM。六轮调试，最终收敛到一行接口调用。

### 现象链（第 6 轮观测数据）

```
on_init_end → checkpoint_manager.update_weights
  → get_per_tensor_param() 物化全量权重
  → load_fsdp_model_to_gpu + state_dict() 期间
torch.OutOfMemoryError: 申请 1.19 GiB, free 795 MiB, 本进程 21.94 GiB, allocated 20.67 GiB
→ 峰值 allocated 20.67GB，缺口仅 ~400MB
```

### 观测定位法：两个观测点夹出的答案

```
行1117: E6B before load (get_per_tensor_param), allocated 0.00, used 1.37/23.52
        → 参数已 CPU offload ✓（offload 生效铁证，也是定位锚点）
"E6B after state_dict" 从未打印
        → OOM 发生在 load 之后、state_dict 之间
        → 二分锁定: 爆在"物化全量权重"这一步
        → 根因: get_per_tensor_param() 物化全量 8.4GB + 本卡分片 4.2GB 并存 = 12.6GB 峰值
```

两个 print 之间夹死出错区间——这就是"观测点二分法"的全部威力：不猜，量。

### 排除项清单（前五轮的成果，别再走回头路）

```
| 候选修复             | 结论   | 依据                                        |
|---------------------|--------|---------------------------------------------|
| 关掉 ref/KL 模型     | 已排除 | use_kl_loss=false，ref 不创建，与 OOM 无关      |
| layered_summon 分层 | 排除   | 只对 LoRA 分支生效，本项目全参数 FSDP1 无效       |
| GC 阈值 0.05         | 排除   | reserved 只有 ~3% 可回收，不够 400MB            |
| torch CPUOffload 替代 | 排除  | veRL 0.10 对 ref 的强制 CPUOffload 路径在        |
|                     |        | torch 2.13 下不生效（实测常驻 8.46GB）           |
```

### v6 修复：换接口而不是挤显存

```
v6 = get_per_tensor_param_shard()      # 拿本地分片，不 unshard、不 all-gather
   + FULL_STATE_DICT                   # state_dict 按分片口径构建
   + offload_to_cpu 流式               # 逐块经 CPU 中转，不与分片并存
→ update_weights 峰值 12.6GB → ~5GB
→ 第 12 轮日志实锤: "E6B after update_weights, memory allocated (GB): 0.00"
```

前五轮补丁链（v1-v5）也没白费——它们把"官方只做了一半"的 load/offload 闭环补齐
（v1/v2：构建后 offload + EngineTrainModeCtx 进出时 load/offload，验证 allocated
8.46→0.00）、把权重复制与 vLLM resume 错峰（v3：vLLM sleep 保持 788MB）、埋下观测点
（v4/v5）。每一步都有日志实锤，没有一步靠猜。

### 教训三条

1. **信日志不信文档**：veRL 0.10 官方文档说 ref 的 CPUOffload 可用，实测 torch 2.13 下
   不生效、常驻 8.46GB；e6b_run.sh 里"GC 可释放 ~700MB"的乐观估计也被实测 ~3% 推翻。
   文档与注释是假设，观测点是事实——两者冲突时以观测点为准。
2. **小缺口抠细节、大缺口换方案**：缺口 400MB 时，GC、错峰、碎片这些"抠细节"手段
   值得试（虽然都失败了）；若缺口是几个 GB，直接上架构级方案（offload/砍副本/换接口），
   别在细节里浪费时间。
3. **每轮留观测点**：v4/v5 补丁全是 print，成本一分钟；盲试一轮的成本是一小时+一张卡
   的租金。第 6 轮能一次定位，是因为前五轮把观测点布到了正确的位置。

**面试记忆点**：update_weights OOM 的根因是"两个大对象并存"（全量 8.4 + 分片 4.2 =
12.6GB），修复思路不是"再挤 400MB"而是"别物化全量"——接口名里的 shard 二字就是答案。

## B.4 通用排查 SOP：五步法

**一句话定义**：OOM 排查是标准的"二分法 + 排除法"，每一步都有本项目的对应动作。

```
① 先看错误信息数字
   申请 1.19 GiB / free 795 MiB / allocated 20.67 GiB
   → 缺口 = 1.19 - 0.795 ≈ 400MB → 判定量级: 小缺口抠细节 / 大缺口换方案
② 打观测点二分
   在"大对象物化/释放"的边界打点: before load / after load / before state_dict / after offload
   "after state_dict 从未打印" → 锁定出错区间
③ 排除法列清单
   候选假设列表化, 逐条找证据否决: ref 已关 / layered_summon 只 LoRA / GC ~3% / CPUOffload 失效
④ 读源码找接口
   从报错行往上读, 找官方注释: transformer_impl.py 的 get_per_tensor_param_shard
   "yields each rank's *local* FSDP shard" → 换接口
⑤ 验证后固化补丁
   v6 补丁 → 复验日志 allocated 0.00 → 同步本地 patch 脚本 → 结论写进启动脚本注释
```

**固化**这一步最容易被省略但最值钱：服务器上的补丁必须同步回本地补丁脚本，否则重装
即丢（本项目为此专门维护了补丁套件并逐版本升级 v1→v6）。排查成果要以"脚本/配置/注释"
的形式沉淀，而不是躺在聊天记录里。

**面试记忆点**：OOM 排查 = 看数字 → 二分观测点 → 排除清单 → 读源码 → 固化补丁；
面试讲案例时按这五步讲，每一步都带一个真实数字，比背八股有说服力得多。

---

## 结语：Part 1 的终点，Part 2 的起点

本项目显存问题清零后，瓶颈立刻从"显存"转移到"调度"——这正是系统工程的味道：修好
一层，下一层瓶颈自动浮现。veRL 环境调通后对 rollout 侧的 TQ（调度队列）做了瓶颈
定量分析：每步约 51K 个子操作、内部存储单元 PUT_DATA 吞吐约 736 次/分钟、单步时长
50-70 分钟；交叉验证更有意思——搜索后端从 0.68s 换到 1-3s 再换到 6-14s，步时几乎
不变，证明瓶颈在框架调度层而非搜索 API。Part 2 的主题就是这类 agentic RL 的系统
工程瓶颈（IO、调度、吞吐）与优化方法。

（Part 1 完）
# 系统工程教学指南 Part 2：vLLM 推理引擎与混合架构适配

> **配套项目**：Search-R1 复现（本地 HF 管线 → veRL + vLLM 迁移，Qwen3.5-4B 混合架构，
> 2×4090 推训同卡）。Part 1 是《GPU 显存计算专题》（`01-GPU-MEMORY-GUIDE.md`），
> 已经教会你"显存账 = 数数组 × 算大小"；本 Part 讲引擎：显存账算清楚之后，
> **推理引擎负责回答三个问题——显存放哪、谁先跑、怎么跑得快。**
> **读者画像**：用 vLLM 跑过 rollout、准备面试的人；能调通配置，但说不清
> "PagedAttention 和操作系统的关系""max_num_seqs 为什么被 Mamba 卡住"的人。
> **本文承诺**：每个概念走同一套流程——一句话定义 → 原理 → 图/数字例子 →
> 联系你的项目（`e6b_run.sh` 的真实 override 与真实报错）→ 面试记忆点。
> 凡估算必标 ⚠️；凡报错原文必给出处。
> **对照物**：所有 override 可核对 `verl_search_r1/e6b_run.sh`；
> 报错实录见 `2026-08-30-CONTINUE.md`、`SEARCH-R1-COMPLETE-REPORT.md`。

在开始之前，先记住一句心法：

> **推理引擎 = 三个问题的答案：**
> **显存放哪（PagedAttention）、谁先跑（continuous batching）、怎么跑得快（CUDA graph）。**
> 面试被问到任何 vLLM 问题，先判断它属于这三个问题中的哪一个。

---

## 第 A 章 为什么需要 vLLM：从 HF generate 的痛点讲起

### A.1 HF model.generate 为什么慢

**一句话定义**：`model.generate()` 是 transformers 提供的高层生成 API——你给它
prompt，它自回归地一个 token 一个 token 采样，直到停或到 `max_new_tokens`。
它"能用"，但它是为"单机单请求调通"设计的，不是为"每步要采几百条轨迹"设计的。

**原理**：慢在三个层面，逐层看。

第一层：**计算**——每生成 1 个新 token，模型都要做一次**全量前向**。生成第 t 个
token 时，全注意力层要对"历史全部 t 个 token"算一遍注意力。所以解码 500 个 token
= 500 次前向，第 500 次前向要处理 2500 个 token 的注意力。总计算量随生成长度
**平方级**增长。而"理想情况"（只关心新 token）应该是线性级。

第二层：**显存**——历史 token 的 K/V 要么每步重算（更慢），要么存下来每步重读。
HF generate 用 KV cache 存了，但：
- 短序列在 batch 里被 **padding 补齐**到最长序列（本地脚本用左对齐填充），
  padding 位置的 token 白算一遍注意力；
- KV 按"最大长度"连续预分配，一条 500 token 的回复按 2048 token 的额度占显存
  （这一条的解法就是 B.1 的 PagedAttention）。

第三层：**调度**——一个 batch 从进来到全部生成完，中间**不能插入新请求**：
batch 里谁先结束谁空等，GPU 没有"迭代级调度"这个概念（这一条的解法是 B.2）。

**联系你的项目**：本地管线 `train_grpo_local.py` 的 `_batch_generate()` 就是典型
HF generate 用法——每个问题的 group_size 个 prompt 手动左填充后 `model.generate()`
一次跑完；多轮搜索 agent loop 是**串行**的：每轮生成完 → 等搜索 API 返回 →
再生成下一轮，搜索等待期 GPU 完全空转。项目口径：本地管线每步约 **16 分钟**
（见 `verl_search_r1/READMD.md` 对比表），其中 rollout 生成占大头。迁移 veRL +
vLLM 的收益预期就是推理加速 3-5 倍——这是"为什么上 vLLM"的项目级答案。

**面试记忆点**：HF generate 慢的三层原因各对应 vLLM 一项技术——
计算浪费 → CUDA graph，显存浪费 → PagedAttention，调度浪费 → continuous batching。
面试问"为什么用 vLLM"，按这三层答，比背"吞吐高 2-4 倍"有说服力得多。

### A.2 自回归生成的 KV cache 基础（衔接 Part 1）

**一句话定义**：KV cache 是把每一层注意力算出的 K（钥匙）、V（值）两个中间矩阵
存进显存，生成下一步时直接取用、不重算——本质是**用显存换时间**。

**原理**：自回归生成第 t 个 token 时，注意力只新增"第 t 个 token 的查询 Q"，
历史 token 的 K/V 和第 t-1 步完全一样。所以历史 K/V 可以缓存复用，每步只算
新 token 的一份 K/V 追加进去。注意 **Q 不缓存**——每步只用到当前 token 的 Q，
用完就扔。

**数字例子**（七因子公式，Part 1 第 2.2 节的原版，这里只做衔接）：

```
KV 字节数 = 2 × n_layers × n_kv_heads × head_dim × seq_len × dtype_bytes × batch
          = 2（K和V）× 32 层 × 4 头 × 256 维 × seq_len × 2 字节 × 1
          = 131,072 字节/token = 128 KiB/token
```

一条 2048 token 的序列 ≈ 256 MiB KV。七个因子里，**seq_len 是唯一的动态项**——
这就是为什么"限制 max_model_len"能直接决定单条序列的 KV 上限（项目里 6144 的
由来见 B.1）。⚠️ 进阶口径（Part 1 §2.5 提示）：Qwen3.5-4B 是混合架构，32 层里
只有 8 层全注意力、24 层线性注意力，真正随 seq_len 线性增长的只有那 8 层——
这条"分层算账"的原则正是第 C 章所有坑的根源。

**面试记忆点**：KV cache 七因子公式必背（这是"算 KV 显存"的白板题）；
面试追问"Q 为什么不缓存"，答案：每步只用当前 token 的 Q，无复用价值。

---

## 第 B 章 vLLM 三大核心技术

### B.1 PagedAttention

**一句话定义**：PagedAttention 是把 KV cache 切成固定大小的 **block（页）**，
按需分配、非连续存储，用一张 **block table** 记录"逻辑序列 → 物理页"的映射——
照搬了操作系统的虚拟内存分页思想。

**原理**：传统做法为每条请求按**最大生成长度**预留一整块连续显存。两个后果：
- 短回复白占长额度：按 2048 预分配、实际只生成 500 token，中间的 1548 个
  token 位子是空的，碎片率文献口径高达 **60-80%**；
- 碎片无法合并回收：不同请求的预留区错落排列，释放后留出的空洞塞不进新请求。

PagedAttention 的做法：KV 池切成固定大小 block（vLLM 默认 **16 token/block**），
一条序列的 KV 被装进若干物理 block，block 之间**不需要连续**，逻辑顺序由 block
table 维护。生成到新 token 时，先看当前 block 有没有空位，满了就再分配一个新
block。副产品：beam search 分支时，多个候选共享同一批前缀 block，只有分叉的
block 复制（copy-on-write），几乎零成本。

**数字例子**（沿用 A.2 的 128 KiB/token 口径，单条序列）：

```
传统预分配（按 max_model_len = 2048 连续预留）：
    128 KiB × 2048 = 256 MiB         ← 预留额度
    128 KiB ×  500 =  62.5 MiB       ← 实际只用了这些
    浪费 = 256 − 62.5 = 193.5 MiB，浪费率 ≈ 75.6%

PagedAttention（block = 16 token，按需分配）：
    实际占用 block 数 = ceil(500 / 16) = 32 个
    实际占用 = 32 × 16 × 128 KiB = 64 MiB
    碎片只有最后一个 block 内的空位 ≤ 15 token ≈ 1.9 MiB
    64 MiB vs 256 MiB → 省下 75%
```

同一个 KV 池，PagedAttention 能服务的并发序列数约为传统方案的 **4 倍**
（256/64）——这就是 vLLM 吞吐高的第一根支柱。

**联系你的项目**：`e6b_run.sh` 里 `max_model_len=6144` 的注释原文是
"多轮上下文：prompt ~2048 + 6 turns × ~512 ≈ 5120，6144 够用且 **KV cache 减半
提升并发**"。逻辑链：max_model_len 决定单条序列在池里的**最大额度**，6144 相对
默认的更大值减半额度 → 同样的池子能容纳约 2 倍的并发序列。PagedAttention 让
"额度"变成按需分配，max_model_len 才敢放心地给每条序列设上限。

**面试记忆点**：三个词——**block（16 token）、block table、copy-on-write**；
再配一个数字——传统方案碎片率 60-80%，PagedAttention 把碎片压到 block 内部
（≤1 个 block）。面试问"vLLM 为什么显存利用率高"，就答这两句。

### B.2 Continuous batching

**一句话定义**：continuous batching（连续批处理/迭代级调度）指**每个 decode
step 都重新组装 batch**——谁生成了新 token 谁留下，谁结束谁出队，新请求随时
插队——而不是把 batch 锁死跑到底。

**原理**：传统 static batching 的 batch 一旦组好，要等**批内最慢的那条**结束
才整体释放，这是经典木桶效应。而自回归生成天然适合迭代级调度：每个 step 之后，
每条序列的状态都变了（新增 1 token / 结束 / 到达 prefill 阶段），于是每个 step
都可以从全局队列里重新选人。长序列不再拖死整个 batch，短序列的槽位立刻让给
排队中的新请求。

**数字例子**（示意，⚠️ 假设请求源源不断到达）：

```
同一时刻入队的 4 条请求，生成长度 100 / 200 / 400 / 1000 token：

static batching：
    整批锁死 1000 个 decode step，共产出 1700 token
    平均 1.7 token/step；第 100 step 后 4 个槽位只剩 1 条在跑

continuous batching：
    第 100 step 第 1 条出队 → 新请求补位；第 200 step 第 2 条出队 → 再补位……
    队列不空时槽位始终接近满载 → 平均接近 4 token/step
    吞吐 ≈ 4 / 1.7 ≈ 2.3 倍（示意值，实际增益取决于长短序列混合比例）
```

**联系你的项目**：这是多轮 agent loop 场景的**刚需**。veRL 的 rollout 一次在飞
128 条序列（train_batch_size=32 × n=4），每条都会中途停下来等搜索 API
（`max_search_calls=1`、`search_timeout=15.0`，见 `agent_config.yaml`）。搜索等待
期不占 GPU 的生成槽位，别的序列继续生成；搜索返回后请求再插回队列。没有
continuous batching，整个 rollout 就会被"等搜索"这个最慢环节卡死。
另外 `max_num_batched_tokens=65536`（2 卡档）限的就是**在飞 token 总数**——
这是给 continuous batching 的"油门"：批里拼多少条、每条拼多长，由它兜底，
防止一个 step 的 KV 需求超过池子。

**面试记忆点**：static batching 等最慢的，continuous batching **每个 step
重新组队**；它解决的是"长短序列混跑 + 请求异步到达"的吞吐问题。面试追问
"为什么能插队"，答案：因为每 step 的 batch 是动态拼出来的，批组成不是固定的。

### B.3 CUDA graph

**一句话定义**：CUDA graph 是把**一整个 decode step 的全部 GPU kernel 捕获成
一张图**，之后每个 step 一次 launch 重放整张图，省掉几百次逐个启动 kernel 的
CPU 开销。

**原理**：一个 decode step 不是"一个大 kernel"，而是几百个小 kernel——每一层的
QKV 投影、RMSNorm、注意力、输出投影、门控，层层都是独立 kernel 启动。
每个 kernel 启动有 CPU 侧开销（launch 指令、参数拷贝、CPU-GPU 同步，单次
约 5-10 μs 量级 ⚠️）。4B 模型一次 decode step 有上百到几百个 kernel，如果每个
kernel 本身只跑十几微秒，**CPU 启动开销和 GPU 计算量级相当**——GPU 在等 CPU
发活，这就是 launch-bound。CUDA graph 把这些 kernel 的启动序列整体捕获成一张
图（形状、指针、依赖全部固定），重放 = 一次 launch，CPU 开销被摊薄到接近零。

代价是**动态形状问题**：图里一切都固定了，batch 大小每 step 都在变怎么办？
vLLM 用分段捕获 + padding 应对——按 2 的幂把 batch 大小分桶（1/2/4/8/…），
每个桶一张图，多余槽位填 padding token 白算 ⚠️（vLLM piecewise CUDA graph 的
实现细节随版本演进，此处是原理口径）。`enforce_eager=true` 就是"我不信这张图，
每个 kernel 老老实实逐个启动"的开关——主要用于调试、内核不支持的模型、
或需要极短 prefill 的动态形状场景。

**联系你的项目**：`e6b_run.sh` 里 `actor_rollout_ref.rollout.enforce_eager=false`
（false = 开 CUDA graph），注释原文"vLLM 吞吐优化：启用 CUDA graph（推理阶段
无动态形状，安全）"。⚠️ 严谨口径：rollout 的 batch 大小其实每 step 都在变
（搜索请求进出队），"无动态形状"的准确意思是——vLLM 的分桶/padding 机制已把这
个动态性消化掉了，且推理不涉及训练侧真正破坏图捕获的动态结构（如可变序列打包
训练），所以这里开 graph 是安全的。配套的 `VLLM_ATTENTION_BACKEND=FLASHINFER`
指定注意力后端，图捕获要求 kernel 图稳定，后端选择是它的一部分。

**面试记忆点**：一句话——"**几百个 kernel 的启动开销 > 小模型单步计算**，
CUDA graph 把它们打包成一次 launch"。追问动态形状怎么办：分桶 + padding，
或者关掉（enforce_eager=true）。再追问"enforce_eager 默认值"：vLLM 默认开
graph，eager 是逃生舱。

### B.4 显存与 gpu_memory_utilization

**一句话定义**：`gpu_memory_utilization` 是 vLLM 启动时的**显存圈地比例**——
允许 vLLM 最多使用整卡显存的百分之几，用来一次性预留它的显存池。

**原理**：vLLM 启动时预留的池由三部分组成：

```
vLLM 显存池 = 模型权重 + KV cache 池（PagedAttention 的页池）+ 激活/CUDA graph 缓冲
```

权重先扣（4B bf16 = 8.4GB），激活缓冲是估算值 ⚠️，剩下的全部变成 KV 页池。
用项目 2×4090（24GB）的 0.75 档算（Part 1 §2.5 的原版账）：

```
0.75 × 24 GB = 18 GB          ← vLLM 总预算（不是全给 KV！）
18 − 8.4（权重）− ~1.5（激活缓冲 ⚠️ 估算）≈ 8 GB   ← KV 页池
8 GB ÷ 128 KiB/token ≈ 6.5 万 token               ← 池容量
```

**联系你的项目**：`e6b_run.sh` 按卡数给两档（显存账注释见脚本中部）：

```
1×4090: gpu_memory_utilization=0.55   max_num_batched_tokens=32768
2×4090: gpu_memory_utilization=0.75   max_num_batched_tokens=65536
```

逻辑：1 卡时训练侧（梯度 8.4GB + 激活 ~1.5GB，offload 后）与 vLLM 抢同一张卡，
vLLM 收敛到 0.55 并砍半在飞 token 上限；2 卡时训练侧分片（梯度 4.2GB/卡）且
rollout 阶段训练侧几乎全 offload，vLLM 能圈 0.75 并把油门加到 65536。
⚠️ 口径坑（Part 1 第 489 行专门警告过）：脚本显存账注释里的
"vLLM(0.45/0.55 池)"（10.8/13.2GB）和 override 的 0.55/0.75（13.2/18GB）
**不是同一个数**——前者是"vLLM 总占用的账"，后者是"vLLM 总预算口径"，
不同 vLLM/veRL 版本按整卡还是按剩余显存计算也不一致。**一律以 vLLM 启动日志
打印的 KV cache 容量为准。**

最后一件事预告：推训同卡时这个池子不能常驻——训练阶段要把它还回去，
机制叫 `free_cache_engine`，详见第 D 章。

**面试记忆点**：gpu_memory_utilization 圈的是**总预算**，KV 池 = 预算 − 权重
− 激活缓冲；纯推理服务常给 0.9，推训同卡要主动压低（本项目 0.55/0.75）。
面试问"KV 池多大"的完整答法：看这个公式，再看 vLLM 启动日志，不背常数。

---

## 第 C 章 混合架构模型适配（本项目踩过的真实坑）

### C.1 Transformer+Mamba 混合架构是什么

**一句话定义**：混合架构指模型的一部分层用**全注意力**（Transformer），另一部分
层用**线性注意力/类 SSM 层**（Mamba 族）——用少数全注意力层保精度，用多数线性
注意力层把长序列成本从平方级压到线性级。

**原理**：全注意力层每个 token 要看全部历史 token（计算 O(L²)）；线性注意力层
维护一个**固定大小的循环状态**，每来一个 token 用 O(1) 更新状态，看历史不靠
重算注意力矩阵（计算 O(L)）。代价是单层表达能力弱于全注意力，所以每隔几层
放一层全注意力"校准"。

**数字例子**（Qwen3.5-4B 的真实结构，`text_config.layer_types` 原文）：

```
32 层，full_attention_interval = 4 → 每第 4 层是全注意力：
  linear_attention ×3, full_attention, linear_attention ×3, full_attention, ...
  合计：8 层全注意力 + 24 层线性注意力
```

⚠️ 命名口径：本项目文档统一叫"Mamba 混合架构"（vLLM 报错也写 Mamba cache
blocks），但 Qwen3.5-4B 模型卡 README 的官方表述是 **Gated Delta Networks**
（GDN，一种门控线性注意力/Delta 规则族）+ 稀疏 MoE。GDN 不是经典 Mamba
（Mamba = 选择性 SSM），但同属"固定大小循环状态 + 线性复杂度"一族，
缓存管理逻辑相通。面试时如果对方较真，先承认"项目口径叫 Mamba，模型卡写作
GDN"再讲机制，不会翻车。

**联系你的项目**：Part 1 §2.5 的进阶提示就是这里的应用——混合架构下真正随
seq_len 线性增长的 KV 只有 8 层全注意力：128 KiB/token × (8/32) × 2048 ≈
**64 MiB**，而不是 256 MiB。**混合架构模型要分层算账**——这句话在下一节会
变成一条真实的 vLLM 报错。

**面试记忆点**：混合架构 = "多数线性注意力层管长序列成本 + 少数全注意力层
管精度"；算显存必须分层（线性层状态不随 seq_len 增长）；Mamba/GDN 是族名
不是同一模型。

### C.2 Mamba 的 recurrent state 与 cache block

**一句话定义**：线性注意力层的"缓存"是**固定数量、按序列预分配、不可分页**的
循环状态块（Mamba cache block）——这就是 vLLM 对并发序列数设上限的由来。

**原理**：全注意力层的 KV 随序列增长，可以用 PagedAttention 按 16 token 的页
动态切分；而线性注意力层每层的状态是**固定形状**的（卷积状态 + SSM 状态，
大小与序列长度无关），vLLM 在初始化时按"最多同时服务多少条序列"一次性预留
一份份完整状态。⚠️ 关键差异：KV 页池是"token 池"（容量算 token 数，序列短
用的少），Mamba 状态块是"**序列池**"（一条序列占一个固定块，与生成长度无关，
也无法切页复用）。所以 vLLM 初始化时会根据剩余显存算出"能预留多少个 Mamba
状态块"，并把它和 `max_num_seqs` 对账——请求的并发序列数超过块数就直接报错。

**报错原文解读**（本项目实录，`SEARCH-R1-COMPLETE-REPORT.md` §2.11）：

```
max_num_seqs (256) exceeds available Mamba cache blocks (229)
```

逐词拆：默认 `max_num_seqs=256`（vLLM 默认并发上限）> 当前显存池算出来的
Mamba 状态块 **229 个** → 启动即拒绝。修复是 `max_num_seqs=229`（e6b_run.sh
第 140 行）。⚠️ 229 这个数由 vLLM 内部的显存账推导（总预算 − 权重 − KV 页池
− 激活缓冲后剩下的钱能买几个状态块），确切公式随版本而变，此处不给出推导；
但方向是确定的：**给 vLLM 的池子越大（gpu_memory_utilization 越高），可预留
的 Mamba 块越多，max_num_seqs 上限越高**。

**为什么 229 对 GRPO 组采样够用**（数字例子）：

```
rollout 并发序列数 = train_batch_size × n = 32 × 4 = 128 条
128 < 229 ✓，还留了 101 条余量
多轮工具调用不增加并发——同一序列的搜索结果拼进同一条序列，占的还是 1 个块
```

⚠️ 注意上限的推论：若未来把 train_batch_size 提到 64（64×4=256），会再次
撞上 229 这堵墙——要么提高 gpu_memory_utilization 换更多块，要么改架构分配。
这就是"报错信息里的数字都值得算一遍"的意义。

**面试记忆点**：线性注意力层的状态 = **固定大小、按序列预分配、不可分页**
（三词定义），所以 max_num_seqs 被"可用 Mamba cache blocks"硬性卡住；
KV 页池卡的是 token 数（max_num_batched_tokens），Mamba 块卡的是**序列数**
（max_num_seqs）——两个上限管两种池子，别混。

### C.3 FlashAttention 2 缺失 → sdpa 注入

**一句话定义**：FA2 是 GPU 上融合、免物化注意力矩阵的高性能注意力 kernel；
sdpa（scaled_dot_product_attention）是 PyTorch 原生的融合注意力入口，内部自动
在 flash / 内存高效 / 数学实现三者中择优。

**原理与坑**：transformers 加载模型时默认优先用 flash-attn（装了就用
flash_attention_2）。⚠️ 混合架构的支持矩阵问题：FA2 的 kernel 覆盖的是标准
全注意力，Qwen3.5 这种"GDN 线性注意力 + 稀疏 MoE"的新结构在当时的 FA2/torch
版本组合下装不出可用的 flash-attn → 训练侧（veRL 的 FSDP engine 用 transformers
加载 actor）初始化时直接崩（项目实录 CRASH#2：FlashAttention2 缺失，见
`2026-08-30-CONTINUE.md` §2）。

修复实录（本项目真改过的两行）：

```
veRL 侧（生效）：verl/workers/engine/fsdp/transformer_impl.py 约 260 行
    注入 attn_implementation="sdpa" → ast.parse 验证 + 日志证实生效
模型 config.json 侧（无效）：塞 "attn_implementation": "sdpa" 键
    ⚠️ 无害但无效——transformers 5.10 的 Qwen3_5 类忽略 config.json 该键
```

vLLM 侧不需要 sdpa：e6b_run.sh 用 `VLLM_ATTENTION_BACKEND=FLASHINFER`
指定了 vLLM 自己的注意力后端，两个引擎的注意力实现是**两套独立体系**。

sdpa 为什么是可接受的降级：它一次 dispatch 完成"选 kernel + 免物化注意力矩阵"，
性能上 flash 实现 ≥ 内存高效实现 ≈ FA2 的 1/2 左右（⚠️ 理论量级，本项目
未实测 FA2 vs sdpa 的差距）。对 4B/2×4090 的场景，训练侧注意力只涉及 8 层
全注意力，且本项目步耗时的主导项经定量分析是 transfer queue（每步 51713 个
TQ 操作，见 D.1），注意力 kernel 差一档不会改变任何结论。

**面试记忆点**：sdpa = PyTorch 原生融合注意力，**三选一自动降级**；
新架构模型 FA2 没跟上的标准解法就是注入 sdpa；注入位置要对——改框架加载点
（transformer_impl.py），不是改模型 config.json（新版 transformers 可能忽略）。
vLLM 侧另有一套后端体系（本项目用 FLASHINFER），别和训练侧混为一谈。

### C.4 vLLM 把 Qwen3_5 路由到多模态实现的坑

**一句话定义**：vLLM 的**架构注册表**按模型 config.json 的 `architectures`
字段查表，把模型名路由到对应的实现类——路由错一张表，加载就崩。

**原理与坑**：本项目合并后的 checkpoint 的
`architectures = ["Qwen3_5ForConditionalGeneration"]`。⚠️ vLLM 0.28 把
`Qwen3_5ForConditionalGeneration` 路由到了 **Qwen3-VL 系的多模态实现**
（因为 transformers 的 Qwen3_5 类自带视觉塔），多模态实现初始化时**无条件**
向 HuggingFace 要图像处理器配置 → 我们的纯文本仓库没有
`preprocessor_config.json` → 直接炸：

```
OSError: Can't load image processor for 'wang072266/qwen3.5-4b-search-r1-sft'
... containing a preprocessor_config.json file
traceback: vllm/multimodal/encoder_budget.py:94
         → vllm/model_executor/models/qwen3_vl.py:896-903
```

同期日志里还有一行常被误读的：

```
model.visual.merger.norm.weight | MISSING
```

**解读**：transformers 的 Qwen3_5 类结构上自带视觉塔（visual.merger 等），
我们合并仓库只有纯文本权重，视觉塔权重自然 MISSING → 随机初始化。
⚠️ 纯文本训练用不到视觉塔，**这行本身无害**；真正的崩溃点是"路由到多模态
实现 → 强制要 preprocessor 文件"，不是权重缺失。诊断时别追着 MISSING 跑。

**修复思路三选一**（`2026-08-30-CONTINUE.md` §4，⚠️ 最终采用哪条以服务器
会话为准，三条思路都是通用方法论）：
- **A 补文件**：从上游 Qwen3.5-4B 仓库复制 `preprocessor_config.json`（KB 级）
  → 满足多模态实现的初始化检查，纯文本推理不受影响；
- **B 改路由**：若 vLLM 注册表有纯文本路由（别的 arch 名/开关），改路由；
- **C 改 arch 字段**：若上游 arch 本就是纯文本、被 mergekit 改成了
  ConditionalGeneration，把 config.json 的 architectures 改回上游值。
  ⚠️ 改 arch 会换加载类，必须核对权重形状。

**面试记忆点**：vLLM 按 `architectures` 字段查注册表路由实现类；
类名带 VL/ConditionalGeneration 而仓库是纯文本时，先怀疑"路由到多模态
实现 + 缺 preprocessor_config.json"；修复三件套：补 processor 文件 / 改路由 /
改 arch 字段。MISSING 权重 ≠ 崩溃原因，追 traceback 的第一行。

---

## 第 D 章 推训同卡：HybridEngine

### D.1 同一批 GPU 上 vLLM 与 FSDP 分时复用

**一句话定义**：veRL 的 HybridEngine 在同一批 GPU 上同时挂两个引擎——FSDP
actor 管训练、vLLM 管 rollout——靠**分时复用**和 CPU offload 在 24GB 的卡上
来回腾挪显存，每步走一个固定循环。

**原理与显存切换时序**（数字为 e6b_run.sh 显存账的 2 卡口径）：

```
第 n 步：
┌─ rollout 阶段（vLLM 活跃，训练侧让路）────────────────────┐
│ vLLM：权重 8.4GB + KV 池 ≈ 13.2GB 总账（0.75 预算档）     │
│ FSDP actor：参数已 offload 到 CPU → GPU 只留梯度 4.2GB    │
│ 128 条 agent loop 并发生成，中途搜索 API 往返              │
└──────────────┬───────────────────────────────────────────┘
               ▼ update_weights：把新权重从 FSDP 同步给 vLLM
┌─ update 阶段（FSDP 活跃，vLLM 让路）──────────────────────┐
│ free_cache_engine：KV 池释放 → 显存还给训练侧              │
│ FSDP：梯度 4.2 + 激活 ~1.5 ≈ 5.7GB（offload 后）          │
│ 算 log_prob → GRPO 更新 → optimizer step（参数流式进出）   │
└──────────────┬───────────────────────────────────────────┘
               ▼ 回到 rollout，下一轮权重已是第 n+1 步的模型
```

两笔同步账（e6b_run.sh 里的真实 override 与注释）：
- `checkpoint_engine.update_weights_bucket_megabytes=512`：默认 2048MB 的分桶
  恰好超过 torch 2.13 CUDA IPC 的 2GB−1 上限 → 静默改走 shm 文件路径 → veRL
  rebuild_ipc 越界崩溃；512MB 走正常 IPC 路径；
- `actor_rollout_ref.rollout.layered_summon=true`：权重同步时分层 unshard
  （summon_full_params），避免"全量 8.4 + 分片 4.2 并存"的 12.6GB 峰值，
  压到 ~5GB（注释口径）。

**联系你的项目（TQ 瓶颈定量分析，正面框架）**：环境调通（probe 47/47 全绿、
双引擎初始化成功）之后做的定量分析发现：每步 128 个 agent loop × ~400 个
TQ 子操作 ≈ **51713 个 transfer queue 操作/步**，而 TQ 的 PUT_DATA 吞吐上限
实测 ~736 次/min（总体 600-700 tasks/min）——即 rollout 阶段 GPU 其实大量
时间在**等 TQ 调度**，而不是在等生成或搜索 API（知乎 0.68s、DeepSeek 1-3s、
MiMo 6-14s 三种搜索后端的步耗时几乎相同，证明瓶颈不在搜索）。这就是为什么
"推理加速 3-5 倍"不等于"步耗时缩短 3-5 倍"：引擎层优化和调度层瓶颈是两本账，
**先测瓶颈再谈优化**。

**面试记忆点**：HybridEngine = 双引擎分时复用 + offload 腾挪；
一张卡上"rollout 谁活跃、update 谁活跃、显存怎么让"是高频面试题，
把上面的时序图默画一遍就是满分答案。同步路径两个坑：IPC 2GB 上限、全量
unshard 峰值（layered_summon 解法）。

### D.2 free_cache_engine 的机制与代价

**一句话定义**：`free_cache_engine=true` 指 rollout 结束后**释放 vLLM 的 KV
池**，把显存完整还给训练阶段；代价是下一轮 rollout 要**重新分配 + 预热**。

**原理**：KV 池是 vLLM 圈的最大一笔地（~5-8GB 量级）。若常驻（false），训练
阶段要同时养"FSDP 梯度 + 激活 + vLLM KV 池"，本项目 1 卡档（20.7GB 账、余
~2GB）直接爆卡。释放后训练侧拿到完整预算；但下一轮 rollout 开始时池子要
重新分配——预热代价包括重新建池、⚠️ 以及 CUDA graph 是否随池重建而重捕获
（依赖版本实现，此处不打包票；实践中 rollout 首步会因此多花若干秒到几十秒）。

**联系你的项目**：`e6b_run.sh` 的 `free_cache_engine=true` + `enforce_eager=
false` 是一对组合：池每轮重建，图（分桶/padding 后）形状规则不变，所以
开图仍安全。这是"显存换预热时间"的典型交易——4B/24GB 的场景里没有第二个
选项，池不还回去训练就 OOM。

**面试记忆点**：free_cache_engine = 用"每轮重新预热"换"训练侧显存够用"；
面试问"推训同卡显存怎么调度"，按"offload 管训练侧 + free_cache_engine
管推理侧"两层答。

### D.3 与推训分离（disaggregation）的对比与选择

**一句话定义**：推训分离指 rollout 和训练各占**独立的 GPU 池**，权重通过
网络/NVLink 在两组卡之间传输；推训同卡则共享一组 GPU、分时复用。

**对比**：

```
维度           推训同卡（本项目）        推训分离
卡数需求       2 卡即可                推理池 + 训练池至少 4+ 卡
KV 池          每轮释放重建（预热代价） 常驻（无预热、无争抢）
权重同步       同卡内存/IPC            PCIe/NVLink/网络传输
故障隔离       差（一卡挂全挂）        好（推理/训练独立存活）
适用规模       小模型、少卡、预算紧     大模型、多卡、高吞吐服务
```

**为什么 4B/2×4090 选同卡**（三条硬理由）：
1. **拆不出池**：只有 2 张卡，拆出独立推理池训练就只剩 1 卡，FSDP 分片失去
   意义；同卡 + offload 后每卡峰值 18.9GB（余 4GB），装得下；
2. **模型小、传输贵**：4B 权重 8.4GB，分离式每步要跨卡传一整份权重，4090
   无 NVLink、走 PCIe，传输时间可能比 4B 的 rollout 本身还疼（⚠️ 未实测，
   量级判断）；同卡走 IPC + layered_summon 已把同步峰值压到 ~5GB；
3. **预算红线**：分离式 = 更多卡时 = 超预算（项目红线 ~¥43），4B 模型用不上
   大模型的"推理训练双池并行"收益。

**面试记忆点**：决策框架一句话——**"模型大、卡多、KV 池要常驻 → 分离；
模型小、卡少、offload 装得下 → 同卡"**。面试问"什么规模该上 disagg"，
答卡数门槛（至少能拆出两个有意义的池）和权重传输成本两条，再补一句
"同卡的预热代价是 free_cache_engine 每轮重建 KV 池"。

---

## 附录：e6b_run.sh 真实 override 速查表

| override / 环境变量 | 值 | 作用 | 对应章节 |
|---|---|---|---|
| `gpu_memory_utilization` | 0.55（1卡）/ 0.75（2卡） | vLLM 显存圈地比例，KV 池 = 预算 − 权重 − 激活缓冲 | B.4 |
| `max_num_batched_tokens` | 32768 / 65536 | 在飞 token 上限，防 KV 池撑爆（管 token 数） | B.2/B.4 |
| `max_num_seqs` | 229 | Mamba cache blocks 上限（管序列数） | C.2 |
| `max_model_len` | 6144 | 单序列额度，KV 减半提升并发 | B.1 |
| `enforce_eager` | false | 开 CUDA graph（false = 开） | B.3 |
| `free_cache_engine` | true | 训练前释放 KV 池 | D.2 |
| `update_weights_bucket_megabytes` | 512 | 避开 CUDA IPC 2GB−1 上限 | D.1 |
| `layered_summon` | true | 权重同步分层 unshard，峰值 12.6→~5GB | D.1 |
| `tensor_model_parallel_size` | 1 | 每卡完整权重副本（无 TP 切分） | D.3 |
| `VLLM_ATTENTION_BACKEND` | FLASHINFER | vLLM 侧注意力后端（训练侧走 sdpa） | C.3 |
| `rollout.n` / `train_batch_size` | 4 / 32 | 并发序列 = 128，被 229 上限兜住 | C.2 |

最后一句收尾：本 Part 的三个心法句——**显存放哪问 PagedAttention，谁先跑问
continuous batching，怎么跑快问 CUDA graph；混合架构先分层算账；推训同卡先
画时序图再谈优化**。面试前把 e6b_run.sh 每个 override 都答得出"它管哪个坑"，
这一 Part 就毕业了。
# 系统工程教学指南 · Part 3：Ray + veRL 架构

> 目标读者：做过 veRL 迁移、准备面试的人。
> 本文每个概念按固定节奏展开：**一句话定义 → 原理 → 架构图（ASCII）→ 联系读者项目 → 面试记忆点**。
> 读者项目 = Search-R1 的 veRL 迁移（`agentic-rl-lab/03-search-r1/verl_search_r1/`，2×4090 训 Qwen3.5-4B，veRL main 0.10 / 0.10.0.dev0）。
> 文中标 ⚠️ 的是未逐行核实源码的 veRL 内部细节，面试引用前以源码/官方文档为准。
> 口径纪律：veRL 侧的成绩一律以「环境调通（probe 47/47 全绿）+ TQ 瓶颈定量分析（51K 操作/步、736/min、步时 50-70min）」为正面框架表述，不引用其他未完成/未验证的表述。

---

## A. Ray 基础（veRL 的地基）

### A.1 Ray 是什么：Python 原生的分布式运行时

**一句话定义**：Ray 把「Python 函数/类」变成可以跨进程、跨机器调用的远程对象；veRL 的全部角色（训练器、引擎、判分器）都是 Ray 上的进程。

**原理**：Ray = actor/task 编程模型 + 全局调度器 + 对象存储三件套。`ray.remote` 装饰后：普通函数 → **task**（无状态远程函数），类 → **actor**（有状态远程进程）。调用方进程叫 **driver**（veRL 里就是 Trainer），它不需要知道被调用方在哪台机器——全局调度器负责找位置、传参数、回收结果。

**架构图**：

```
+----------------+   submit task / call actor        +-----------------+
| Driver 进程     | --------------------------------> | 全局调度器       |
| (Trainer:      |            get(ObjectRef)         | (GCS + raylet)  |
|  加载 hydra    | <-------------------------------- |                 |
|  跑主循环)      |                                   +-----------------+
+----------------+                                        |        |
        |  ray.put(data) 返回 ObjectRef                    |        | 分发
        v                                                  v        v
+----------------+   plasma object store (共享内存)   +-------------+  +-------------+
| 对象存储        | <===============================> | Worker A    |  | Worker B    |
| (plasma)       |   零拷贝共享 / 引用计数回收          | (ActorWorker|  | (Rollout    |
+----------------+                                   |  FSDP rank0)|  |  Worker)    |
                                                      +-------------+  +-------------+
```

**联系读者项目**：看训练日志前缀——`(AgentLoopWorkerTQ pid=...)`、`(RewardLoopWorker pid=...)`，每个括号就是一个 Ray actor 进程；pid 各不相同说明它们真的分布在多个进程里，跨进程通信全靠 Ray 的对象存储与 actor 调用。

**面试记忆点**：「Ray = 分布式 Python 运行时：remote 函数是 task，remote 类是 actor，数据交换走对象存储；全局调度意味着任何进程可以调任何进程的函数，进程位置对业务代码透明。」

### A.2 Actor vs Task：有状态长驻进程 vs 无状态函数

**一句话定义**：Task 每次调用都可能在任意 worker 上「从零执行」；Actor 是绑定在某个进程里的「常驻对象」，成员变量在多次调用之间一直活着。

**原理**：
- **Task**：无状态。调度器为了负载均衡可把它放到任何空闲 worker；两次调用之间不保存任何东西。
- **Actor**：有状态。创建时确定进程位置，之后所有方法调用都路由到同一进程；成员变量（self.xxx）就是它的状态。
- 生命周期：Task 随调用结束而结束；Actor 需显式销毁（`ray.kill` 或引用归零）。

```
            Task（无状态）                          Actor（有状态）
  调用1 ──> [worker A] 执行完即忘        创建 ──> [worker B 长驻进程]
  调用2 ──> [worker C] 可能换进程          │ 方法调用都路由到 worker B
  调用3 ──> [worker A] 又换回来           │ self.state 持续存在
        （每次都从零开始）                  └ 销毁需显式
```

**RL 训练为什么需要 actor**：模型权重住在 GPU 显存里。训练器每步要调用「算 log_prob」成百上千次——若用 task，每次调用都得重新把 4B 模型加载进显存，开销不可接受。用 actor 把「模型 + 优化器 + GPU 上下文」作为进程状态长驻，方法调用只做一次前向。同理 vLLM 引擎（KV cache、调度状态）也必须是 actor。

**联系读者项目**：`search_agent_loop.py` 里 `self.server_manager.generate(...)` 就是对推理服务（veRL 管理的 vLLM 后端）的调用；而 agent loop 内部的同步函数（tokenize、搜索）用 `loop.run_in_executor` / `asyncio.to_thread` 扔进线程池——它们不是 Ray 任务，是 actor 进程内的线程，注意这个区别（面试常考）。

**面试记忆点**：「Actor = 有状态 + 单进程绑定 + 显式生命周期，装的是模型/引擎这类『加载一次、服务多次』的重资源；Task = 无状态、可随处调度、适合纯计算。RL 训练器一天调几千次前向，只有 actor 付得起加载成本。」

### A.3 对象存储：plasma object store 与零拷贝共享

**一句话定义**：Ray 的对象存储是共享内存数据层——同一节点上的进程之间传大对象（tensor、batch）只传「引用」，不复制数据。

**原理**：`ray.put(x)` 把 x 写进对象存储，返回 ObjectRef（引用）；任何进程拿到 ref 后 `ray.get(ref)` 取数据。
- 同节点：直接从共享内存读，**零拷贝**；
- 跨节点：走网络传输（分布式对象存储会缓存副本）；
- 生命周期：引用计数归零自动回收；小对象直接内联在 ref 里，不走存储。

```
  Worker A                 对象存储 (plasma)                 Worker B
ray.put(tensor) ──────>  [共享内存页]  <───── ray.get(ref)   零拷贝读取
        └──> ObjectRef ──── 传递的只是 ref（几十字节）
```

**联系读者项目**：veRL 里 DataProto 在 Trainer、AgentLoopWorker、RewardLoopWorker 之间传来传去（见 B.2），传的就是放在对象存储里的对象引用；权重同步（B.4）本质也是「actor 把权重 put 进对象存储 → vLLM 侧按 ref 取」。明白这条后，日志里「driver 和 worker 传 batch 却不慢」就不再神秘。

**面试记忆点**：「零拷贝 = 同节点共享内存 + 只传 ObjectRef；对象存储是 Ray 吞吐的关键——大对象传引用不传值，垃圾回收靠引用计数。」

### A.4 坑：Ray worker 不继承 shell 环境（读者项目的 .pth 方案）

**一句话定义**：Ray worker 进程由 raylet 拉起，继承的是「启动 Ray 时的环境快照」——你在 shell 里事后 export 的变量，worker 看不到。

**原理**：进程环境的继承链是 `shell → ray head/raylet → worker`。Ray 集群启动后，在 shell 里 export 的环境变量只影响当前 shell 和由它直接启动的进程（driver）；已经常驻的 raylet 并不会被修改，它拉 worker 时用的还是旧环境。这是「环境变量的时间性」问题：**worker 的环境由集群启动时刻决定，不由提交任务时刻决定**。

```
  启动 Ray 时:  shell(env v1) ──> ray head / raylet(env v1 快照)
  之后 export:  shell(env v2 = v1 + PYTHONPATH)
                    │
                    ├──> driver 进程: 继承 v2 ✓（probe 跑在这，测得到）
                    └──> raylet 拉起的 worker: 仍是 v1 ✗（PYTHONPATH 丢失）
```

**联系读者项目（真实现场）**：训练启动后 `RewardLoopWorker` 报错：

```
(RewardLoopWorker pid=37053) FileNotFoundError: Custom module file not found:
module_path='verl_search_r1.reward_fn'
```

- **根因**：`verl_search_r1` 靠 PYTHONPATH 可见，但 Ray worker 不继承 shell 的 PYTHONPATH；
- **为什么 probe 没拦住**：probe 47 项检查跑在 **driver 进程**（继承 v2），模拟不了 worker 的环境——「在 driver 里验证通过 ≠ 在 worker 里能跑」；
- **修法（.pth 文件，作用于所有 Python 进程）**：

```bash
echo /root/autodl-tmp/search-r1 > /root/miniconda3/lib/python3.10/site-packages/searchr1.pth
```

任何 Python 进程启动时都会读 site-packages 下的 .pth 文件并加入 sys.path，与父进程环境无关——把「环境问题」降维成「解释器问题」。

**driver/worker 环境差异的排查心法**：
1. **看日志前缀定位失败进程**——`(RewardLoopWorker ...)` 直接告诉你死在哪个 actor；
2. **区分 driver 视角与 worker 视角**——probe、直接 `python -c` 都是 driver 视角，worker 侧要用「真正由 worker 执行的代码」验证；
3. **修复要作用于所有 Python 进程**——.pth 文件、`ray.init(runtime_env={"env_vars": ...})`、或写进集群启动脚本；只改当前 shell 没用。

**面试记忆点**：「Ray worker 的环境 ≠ 启动脚本的 shell 环境——worker 继承的是集群启动时的快照。传环境变量用 runtime_env 或 .pth 文件，事后 export 只对 driver 生效。probe 跑在 driver 里，测不到 worker 的导入问题。」

### A.5 Ray.get 为什么阻塞、async actor 的注意事项

**一句话定义**：`ray.get(ref)` 是同步等待，会卡住调用线程；在 async actor（async def 方法）里用它，会卡住整个事件循环。

**原理**：asyncio 是单线程事件循环，靠「await 让出控制权」实现并发。`ray.get` 内部是阻塞等待，事件循环被它占住后，这个 actor 里的所有协程全部停摆——并发退化成串行。正确姿势是直接 `await ref`（Ray 的 ObjectRef 可 await，等价于「挂起当前协程等结果，把控制权还给事件循环」）。

```
  错误:  async def method(...):
           x = ray.get(ref)     # 阻塞：整个事件循环冻结，本 actor 所有任务停摆
  正确:  async def method(...):
           x = await ref        # 挂起当前协程，事件循环继续跑别的任务
```

**联系读者项目**：训练日志里出现过无害警告 `Using blocking ray.get inside async actor`——veRL 自己的部分代码路径也有这个现象（官方标记为性能警告）。项目里 `AgentLoopWorkerTQ` 就是 async actor：它的 `run()` 全程 async，搜索用 `asyncio.to_thread` 把同步的搜索客户端调用移出事件循环，与这里讲的是同一套纪律——**别在事件循环里做阻塞的事，阻塞的事交给线程池**。

**面试记忆点**：「async actor 里用 `await ref` 而非 `ray.get(ref)`；阻塞调用 = 事件循环饿死 = 并发变串行。CPU 密集/阻塞 I/O 一律 `asyncio.to_thread` / `run_in_executor` 扔线程池。」

**A 部分小结（一张图记全）**：Ray 提供进程（actor/task）与数据（对象存储）两层抽象；veRL 用 actor 装引擎、用对象存储搬 batch；两个高频坑——worker 环境继承（.pth 修）与事件循环阻塞（await 修）。

---

## B. veRL 架构总览

### B.1 角色分工：五种进程的职责地图

**一句话定义**：veRL 把一次 GRPO 训练拆成五种 Ray actor——编排者（Trainer）、训练者（ActorWorker）、生成者（vLLM RolloutWorker）、智能体（AgentLoopWorker）、判分者（RewardLoopWorker）。

**架构大图**：

```
                       +----------------------------------------------+
                       |  Trainer (driver 进程)                         |
                       |  · hydra 加载配置, ray.init 建集群              |
                       |  · 编排训练循环: rollout→reward→adv→train       |
                       |  · 算 advantage、切 micro-batch、发 checkpoint   |
                       +-----+-----------+-----------+---------+-------+
                             |           |           |         |
                DataProto (Ray 对象存储)   |           |         |
                             |           |           |         |
               +-------------+--+  +-----+------+  +-+-------+--+
               | ActorWorker    |  | AgentLoop  |  | RewardLoop |
               | (FSDP × N 卡)  |  | Worker × M |  | Worker     |
               | · 分片持有模型  |  | · 跑用户    |  | · 跑 reward |
               | · log_prob /   |  |   的 run()  |  |   函数      |
               |   loss / step  |  | · 多轮生成  |  | · 产出 score|
               +-------+--------+  |   与搜索    |  |  + 分项指标 |
                       |           +-----+------+  +-------------+
         update_weights|                 | TransferQueue (TQ)
         (B.4 权重同步) |                 | 生产者-消费者调度层
                       v                 v
               +--------------------------------------+
               | vLLM RolloutWorker (× K, 占用 GPU 池)  |
               | · 批量自回归生成 token                  |
               | · 持有权重副本, 随训练同步更新            |
               | · KV cache 分页 / Mamba cache block    |
               +--------------------------------------+
```

**各角色职责一句话**：
- **Trainer**：driver 进程，跑主循环——发起 rollout、收集轨迹、算 advantage、驱动训练、存档；
- **ActorWorker**：每个 GPU 一个，持有 FSDP 分片的可训练模型，执行 log_prob 计算、loss 反向、optimizer.step；
- **vLLM RolloutWorker**：推理引擎，只管「批量生成 token」，权重由训练侧定期同步；
- **AgentLoopWorker**：每个 agent loop 一个，运行用户的 `run()` 多轮逻辑（生成→解析→搜索→再生成）；
- **RewardLoopWorker**：跑自定义 reward 函数，把轨迹变成标量 score。

⚠️ 各角色类名/进程名随 veRL 版本有出入（读者项目日志里是 AgentLoopWorkerTQ、SimpleStorageUnit 等），面试描述用「职责」而非「类名」最稳。

**联系读者项目**：2×4090 = 2 个 ActorWorker（FSDP rank 0/1），vLLM 侧同样占用这两块卡；日志里 `(AgentLoopWorkerTQ pid=...)` 与 `(SimpleStorageUnit pid=...)` 同场出现，正是 D 部分要讲的调度层。

**面试记忆点**：「一个 step 里有五种进程：编排者、训练者、生成者、智能体、判分者。生成与训练分家，是因为训练吃权重+梯度+Adam 显存、推理吃 KV cache，两者峰值错开才能在同一批卡上共存（这就是 HybridEngine 的动机，见 B.5）。」

### B.2 DataProto：双通道数据容器

**一句话定义**：veRL 的数据载体叫 DataProto，分两个通道——`batch`（torch tensor 字典，走 GPU/集合通信）与 `non_tensor_batch`（Python 对象字典，走 Ray 对象存储）。

**原理**：为什么分两个通道——训练侧的数学运算（attention、loss、梯度）只需要 tensor，且 tensor 之间要做 GPU 传输、NCCL 集合通信、按 token 拼接切分；但奖励函数、agent loop 需要的是「原始字符串答案、搜索次数、异常标记」这类 Python 对象，塞进 tensor 既占显存又语义扭曲。于是：**tensor 进 batch（计算通道），Python 对象进 non_tensor_batch（上下文通道）**，两通道随 DataProto 一起流转，各取所需。

```
  DataProto
  +-----------------------------------------------------+
  | batch: dict[str, Tensor]            ← 计算通道        |
  |   input_ids / attention_mask / response_mask /       |
  |   old_log_probs / advantages / returns ...           |
  |   （参与 GPU 传输、micro-batch 切分、NCCL 集合通信）     |
  +-----------------------------------------------------+
  | non_tensor_batch: dict[str, object] ← 上下文通道       |
  |   raw_prompt / ground_truth / tool_extra_fields /    |
  |   data_source / uid ...                              |
  |   （经 Ray 对象存储序列化传递，进奖励函数当 extra_info） |
  +-----------------------------------------------------+
```

**`tool_extra_fields` 的流转路径（读者项目代码里的完整链路）**：

```
  SearchR1AgentLoop.run() 返回 AgentLoopOutput.extra_fields
        │   {"response_text": ..., "search_calls": 2}
        v
  veRL 打包进 non_tensor_batch["tool_extra_fields"]
        │
        v
  RewardLoopWorker 调用 search_r1_reward(..., extra_info=...)
        │   extra_info = 数据集 extra_info 列 + tool_extra_fields 合并
        v
  reward_fn 里 extra.get("response_text") / extra.get("search_calls")
```

**联系读者项目**：`reward_fn.py` 里 `final_text = solution_str or extra.get("response_text")`——answer 解析失败的轨迹靠 extra_fields 里的文本兜底；`search_calls` 从生成侧一路穿到奖励侧，实现「生成阶段的统计量变成奖励阶段的输入」。这条链路面试时能画出来，胜过背概念。

**面试记忆点**：「DataProto 双通道 = tensor 走计算、对象走上下文；agent loop 的 extra_fields → non_tensor_batch["tool_extra_fields"] → reward 的 extra_info，是自定义字段在 veRL 里的官方流转路径。」

### B.3 一次训练 step 的完整时序（每段标注 worker 与通道）

**一句话定义**：一个 GRPO step = 「采样 n 条轨迹 → 算旧概率 → 打分 → 算优势 → 梯度更新 → 权重同步」的流水线，由 Trainer 串行编排、五类 worker 各管一段。

**时序图**（↓ 时间方向；括号内为执行 worker；[ ] 为传输通道）：

```
  Trainer        ActorWorker      AgentLoopWorker(TQ)    vLLM Rollout   RewardLoop
    │                │                   │                   │              │
    │ ①repeat(n=4)×样本                   │                   │              │
    │────────────────> 分发 prompt        │                   │              │
    │   [Ray 对象存储] │ ───────────────> │                   │              │
    │                │                   │ ②多轮 agent loop  │              │
    │                │                   │ ←TQ: PUT prompt──>│ 生成 token   │
    │                │                   │ <─TQ: GET tokens─│ (多轮重复)    │
    │                │                   │ ③搜索(线程池,缓存) │              │
    │                │                   │  TITO: LLM token  │              │
    │                │                   │  mask=1 / 工具=0   │              │
    │ <── AgentLoopOutput(prompt/response/mask/extra_fields) ─│              │
    │ ④汇总成 DataProto(batch + non_tensor_batch)              │              │
    │────────────────>│ ⑤ old_log_prob(π_old, 训练前必算)     │              │
    │                │    [FSDP 前向, tensor 通道]            │              │
    │─────────────────────────────────────────────────────────>│ ⑥ reward    │
    │ <─────────────────────────────────────────────────────────│  score+指标  │
    │ ⑦advantage: GRPO 组内归一化 (r-mean)/(std+eps)  [Trainer 内存]         │
    │────────────────>│ ⑧训练循环: micro-batch 切分            │              │
    │                │    forward→loss→backward                │              │
    │                │    FSDP 集合通信→optimizer.step          │              │
    │ ⑨ update_weights│ ── 新权重 put 进对象存储 ─>│ 加载新权重   │              │
    │                │    [FULL_STATE_DICT 分片聚合, B.4]      │              │
    │ ⑩ checkpoint ──│ 保存模型/优化器状态                      │              │
```

**各段通道说明**：
- ① 分发轨迹采样请求：Python 对象 → Ray 对象存储；
- ②③ 生成与搜索：token 走 TQ（D 部分的主角），搜索走 HTTP（与 GPU 无关）；
- ④⑤ 汇总与 log_prob：tensor 进 `batch`，FSDP 前向在 GPU 上做；
- ⑥ 奖励：文本 + extra_info 走 non_tensor_batch 到 RewardLoopWorker；
- ⑦ advantage：纯 Trainer 内存计算，不占 GPU；
- ⑧ 训练：FSDP 集合通信 + micro-batch 显存峰值段；
- ⑨ 权重同步：训练引擎 → 推理引擎（同一批 GPU 上换引擎）；
- ⑩ checkpoint：保存模型/优化器状态，不阻塞下一步。

**联系读者项目**：n=4 意味着 ① 每道题发 4 条轨迹（32 样本 × 4 = 128 个 agent loop/步，这是 D.3 里 51K 子操作的基数）；② 里 agent loop 最多 3 轮（思考→搜索→回答，agent_config.yaml）；⑤ 必须在 ⑧ 之前算——PPO/GRPO 的 importance sampling 需要 π_old 做分母。

**面试记忆点**：顺序口诀「**采 → 估 → 奖 → 优 → 训 → 同步 → 存档**」；两个「为什么」：为什么 old_log_prob 在训练前算（importance ratio 的分母必须来自采样时的策略）、为什么 reward 不在 GPU 上算（打分是字符串处理，放 RewardLoopWorker 与训练并行，不占训练显存）。

### B.4 权重同步 update_weights：训练完把权重交给 vLLM

**一句话定义**：每个 step 训练结束后，把训练引擎（FSDP 分片）更新后的权重聚合、序列化、传给 vLLM 引擎，让下一轮 rollout 用新策略采样。

**原理**：训练侧是 FSDP——权重按 rank 分片躺在各卡上；vLLM 需要**完整的**权重字典才能推理。同步的本质是三步：**分片聚合（gather 成完整 state_dict）→ 传输（Ray 对象存储 / IPC）→ vLLM 加载（卸旧装新）**。`FULL_STATE_DICT` 就是「先聚合成全量再传」的策略名。同步发生在每次 optimizer.step 之后，是 rollout 与训练的唯一耦合点。

```
  ActorWorker rank0(分片0)  ┐
  ActorWorker rank1(分片1)  ┴─ gather ──> 完整 state_dict
                                            │ 序列化
                                            v
                            [Ray 对象存储 / IPC 通道]
                                            │
                                            v
                       vLLM RolloutWorker: load_weights(新权重)
```

**三个实战深坑（读者项目全部踩过，含修复）**：

1. **IPC 包大小上限（INT_MAX）**：`update_weights_bucket_megabytes=2048` 时，单 bucket 的字节数超出 32 位有符号整数上限（2^31−1）→ `IndexError: list assignment index out of range`（报在 rebuild_ipc）。修复：bucket 降到 **512MB** + 源码补丁防御。⚠️ 报错形态与修复位置随 veRL 版本变化，以源码为准。

2. **聚合瞬间的显存双副本 OOM**：`on_init_end → checkpoint_manager.update_weights → get_per_tensor_param() → load_fsdp_model_to_gpu + state_dict()` 这段把「分片参数搬到 GPU + 拼全量 state_dict」叠加在 vLLM 之上，实测峰值 20.67GB allocated，最后申请 1.19GB 时爆（free 仅 795MB，缺口 ~400MB）。读者项目 v6 修复：**FULL_STATE_DICT + offload_to_cpu 流式**——物化到 CPU 再传输，不占 GPU 显存；观测点日志 `after update_weights, memory allocated (GB): 0.00` 全链路通过，vLLM 侧保持 sleep 状态 ~788MB。

3. **分片聚合方式选错**：`layered_summon` 只对 LoRA 分支生效，全参数 FSDP1 下无效（读者项目已排除）；v6 改用 `get_per_tensor_param_shard()`——官方注释「yields each rank's *local* FSDP shard」，即不 unshard、不 all-gather，逐 shard 流式传输，峰值压到单分片大小。⚠️ 该接口的聚合/传输细节以 veRL 源码为准。

**联系读者项目**：这条链路的每一环都有日志实锤：OOM 定位靠「before load → 峰值 20.67GB → after state_dict 从未打印」的观测点二分；修复验证靠 `allocated 0.00`。面试讲 update_weights，讲的就是这个「观测点二分定位 → 流式方案」的过程。

**面试记忆点**：「权重同步 = 分片聚合 → 序列化传输 → 引擎加载；三颗雷：聚合时双副本显存峰值、IPC 包大小上限（bucket 别超过 INT_MAX）、聚合方式（layered_summon 只管 LoRA，全参用 shard 流式）。」

### B.5 混合引擎（HybridEngine）与 free_cache_engine

**一句话定义**：HybridEngine 让「训练引擎」与「推理引擎」在同一批 GPU 上分时复用显存——训练阶段 vLLM 让位，rollout 阶段训练侧让位。

**原理**：训练吃权重+梯度+Adam 状态，推理吃 KV cache，两类需求峰值天然错开。HybridEngine 用显式的「模式切换」管理复用：进训练模式时 vLLM 进入低显存状态（sleep、释放 KV cache），出训练模式（rollout 前）再把控制权交回。⚠️ 单进程混合 vs 分离进程的具体形态随 veRL 版本演进（0.10 的 ActorWorker 与 vLLM 的耦合方式），以源码为准。

```
  +---------------------- 同一批 GPU 显存，分时复用 ----------------------+
  |  rollout 阶段:  [ vLLM: 权重 + KV cache ]   训练侧 offload / sleep    |
  |  训练阶段:      [ 训练: 权重+梯度+Adam+激活 ]  vLLM sleep / free cache |
  +---------------------------------------------------------------------+
        切换协议: EngineTrainModeCtx（进出训练模式时 load/offload 闭环）
```

**联系读者项目**：`e6b_run.sh` 里 `free_cache_engine` 开启——vLLM 进入训练模式时释放 KV cache，把显存还给训练侧；读者项目补丁 v1/v2 发现 veRL 0.10 官方的 offload 闭环「只做了一半」（构建后 offload 有了，但 EngineTrainModeCtx 进出时的 load/offload 缺失），补齐后 allocated 从 8.46GB 降到 0.00。这条补丁证明：**混合引擎的显存账，一半在配置（free_cache_engine），一半在源码的切换闭环**。

**衔接 Part 2**：Part 2 的「GRPO 显存账」（权重 2B/参数、Adam 8B/参数、PPO 与 GRPO 的差距）在这里落地为「分时复用协议」；Part 2 的 PPO/GRPO loss 在这里执行于 ActorWorker 的训练循环。

**面试记忆点**：「HybridEngine = 显存分时复用，切换协议是 EngineTrainModeCtx；free_cache_engine 释放 vLLM KV cache 给训练侧。2×4090 训 4B 全参的可能，一半靠 FSDP offload，一半靠这个分时协议。」

---

## C. AgentLoop 接口契约（读者项目核心代码）

> 本部分逐段对应 `verl_search_r1/search_agent_loop.py`（`SearchR1AgentLoop`，注册名 `search_r1_agent`）。

### C.1 @register("search_r1_agent")：注册机制

**一句话定义**：`@register(name)` 把自定义 agent loop 类登记进 veRL 的注册表，让 YAML 配置里的一个字符串就能把它实例化出来。

**原理**：veRL 的 `verl.experimental.agent_loop.agent_loop` 模块提供 `register` 装饰器，本质是「名字 → 类」的全局字典。AgentLoopWorker 启动时：读 `rollout.agent.default_agent_loop` 拿名字 → 读 `agent_loop_config_path` 指向的 YAML 列表 → 用 hydra `instantiate` 按 `_target_` 建实例。配置与代码的耦合只通过「名字」这一个字符串。

```
  @register("search_r1_agent")                 agent_config.yaml (YAML 列表)
  class SearchR1AgentLoop(AgentLoopBase):        - name: search_r1_agent
      ...                                          _target_: verl_search_r1.search_agent_loop.SearchR1AgentLoop
                                                    search_backend: zhihu
  注册表: {"search_r1_agent": SearchR1AgentLoop}     max_search_calls: 1
              ▲                                           │
              └──────────── name 对上了 ──────────────────┘
  训练配置: actor_rollout_ref.rollout.agent.default_agent_loop=search_r1_agent
```

**联系读者项目**：`search_agent_loop.py` 文件头注释就写着两条接入指令（`default_agent_loop=search_r1_agent` + `agent_loop_config_path=path/to/agent_config.yaml`）；文件里对 veRL 的 import 有 try/except 兜底（不在 veRL 环境里也能被 import 检查）——这是「代码可离线审查」的工程习惯。

**面试记忆点**：「注册机制 = 字符串寻址：配置里的名字在注册表里找到类，hydra 再按 _target_ 实例化；用户代码与框架的耦合点只有名字这一个字符串。」

### C.2 __init__ 签名：注入参数各是什么

**一句话定义**：AgentLoopBase 的构造函数是一份「框架向用户代码注入依赖」的接口契约——tokenizer、推理服务、配置树都是 veRL 递进来的。

**原理**：AgentLoopWorker 创建 agent loop 实例时，按基类签名传参。读者项目签名：

```python
def __init__(self, trainer_config, server_manager, tokenizer, processor,
             dataset_cls=None, data_config=None, hf_model_type=None, **kwargs):
```

逐个拆解：
- **trainer_config**：veRL main 0.10 的 `DictConfigWrap`，包着完整的配置树（`actor_rollout_ref.rollout.agent` 就在里面）。⚠️ DictConfigWrap 的 getattr/字典语义随版本演进，以源码为准；
- **server_manager**：`AsyncLLMServerManager`，agent loop **唯一的 LLM 推理入口**（`self.server_manager.generate(...)`）——agent loop 不直接碰 vLLM，这是 C 与 D 的关键边界；
- **tokenizer / processor**：HuggingFace 的分词器/处理器（纯文本项目 processor 闲置）；
- **dataset_cls / data_config**：veRL main 0.10 把这两个变成基类必需参数，读者项目「接受并转发」以兼容版本漂移；
- **hf_model_type**：模型架构标识；
- ****kwargs**：`agent_config.yaml` 列表条目里除 name/_target_ 外的字段（search_backend、search_timeout、max_search_calls…）经 `hydra.utils.instantiate` 以 kwargs 形式传入。

**agent_config.yaml 的列表格式**（veRL main 0.10+）：必须是 YAML **列表**，每项 `name + _target_ + 其余 kwargs`——`name` 对注册表，`_target_` 对类路径，其余字段进 `**kwargs`。

**联系读者项目**：`__init__` 里用 `_cfg(key, default)` 实现了三级配置优先级——**kwargs > config 树节点 > 代码默认值**；这套 fallback 让同一份代码既能被 YAML 驱动，也能在探针脚本里裸构造（probe_verl_api.py 的 47 项检查就靠它）。

**面试记忆点**：「__init__ 的参数是框架注入的依赖清单：配置树、推理管理器、分词器、数据集类；自定义字段全部走 **kwargs（YAML 列表条目里的额外键）。server_manager 是唯一推理入口——这条边界决定了 agent loop 不直接控制引擎。」

### C.3 run() → AgentLoopOutput：七个字段的用途

**一句话定义**：`run(sampling_params, priority, **kwargs)` 处理一条轨迹（一个 prompt 的多轮生成+搜索），返回一个 `AgentLoopOutput`，veRL 用它拼出训练用的 DataProto。

**原理**：AgentLoopWorker 对每个样本调一次 `run()`（128 个 loop/步 = 128 次 run）。`sampling_params` 是采样参数（temperature/top_p…）；`priority` 是 veRL main 0.10 传的每样本优先级（读者项目未用）；`kwargs` 里必有 `raw_prompt`（role/content 消息列表）。返回值七个字段逐个说用途：

| 字段 | 内容 | 下游谁消费 |
|---|---|---|
| prompt_ids | prompt 的 token id | 拼进 batch 的 input_ids |
| response_ids | 全部生成 token（**含搜索结果 token**） | 拼进 response 段 |
| response_mask | 与 response_ids 等长，模型 token=1 / 工具 token=0 | loss 只算 mask=1 的 token（TITO，见 C.5） |
| num_turns | 轮次数（读者项目 = search_calls + 1） | 统计/诊断 |
| metrics | 计时/计数 dict（generate_sequences、search_execute） | 性能监控 |
| extra_fields | 自定义 Python 对象（response_text、search_calls） | → non_tensor_batch["tool_extra_fields"] → reward extra_info |
| response_logprobs / multi_modal_data | 可空占位 | 兼容框架接口 |

**联系读者项目**：`run()` 里 `extra_fields = {"response_text": final_answer_text or "", "search_calls": search_calls}` 是 B.2 那条字段流转链路的起点；`metrics` 用 `simple_timer("generate_sequences", metrics)` 计时，为 D.3 的瓶颈分析埋了测量点。

**面试记忆点**：「AgentLoopOutput 七个字段分三类：训练要的（prompt/response/mask）、监控要的（metrics/num_turns）、奖励要的（extra_fields）。response_mask 与 extra_fields 是自定义 loop 与框架交互的两个关键契约。」

### C.4 类级单例：多个 loop 实例共享搜索客户端与缓存

**一句话定义**：`_shared_client` / `_shared_cache` 是挂在类上的共享状态——所有 agent loop 实例共享同一个搜索客户端和同一份缓存，而不是各建各的连接。

**原理**：AgentLoopWorker 每处理一个样本就实例化一个 agent loop。128 个 loop/步意味着几十个实例同时活着（代码注释按 64 个实例预估）。如果每个实例都 `create_search_client`：① 建立几十条 HTTP 连接；② 缓存彼此不可见，相同 query 重复请求；③ 没有任何集中式速率控制，容易打爆搜索 API。类级单例把「连接 + 缓存 + 限速」变成进程级共享资源。`_shared_config_key`（backend:model:timeout 拼串）决定配置变更时是否重建客户端。

```
  实例1 ─┐                                ┌─ 实例1 查缓存
  实例2 ─┼─> _shared_client（唯一连接）───┼─ 实例2 命中共享缓存
  实例3 ─┘                                └─ 实例3 未命中 → 共享客户端发请求
  （类级属性，所有实例看到同一份）
```

**联系读者项目**：代码注释写明动机「避免 64 个实例各自创建连接 + 缓存不共享 + 无速率限制」；缓存 key 用 `parsed.query.strip().lower()`（模型生成的 query 本身），每 100 次 miss 打印命中率。**反面教训**：项目早先的「预生成缓存」用训练问题原文当 key，运行时查询却是模型生成的 query，key 对不上 → 命中率 0%——缓存 key 必须与运行时查询 key 一致，先验证再部署。

**面试记忆点**：「类级单例 = 进程内共享：唯一连接、共享缓存、集中限速。缓存设计第一定律：key 必须和运行时查询完全一致（大小写/空白/改写全要归一化），用『预想中的 key』造的缓存命中率是 0。」

### C.5 TITO 与 mask：边生成边填 extend 1 / extend 0

**一句话定义**：TITO（Token-In-Token-Out）协议要求多轮对话的**所有 token 都进序列**（模型下一轮能看到搜索结果），但只有**模型自己生成的 token 参与训练**（搜索结果 token 掩掉）。

**原理**：多轮 agent loop 的 response 是「模型 token、搜索 token、模型 token、搜索 token、…」的交错序列。两个需求冲突：① 下一轮生成必须看见搜索结果（所以要 extend 进序列）；② 搜索结果不由策略生成，给它梯度/优势没有意义且引入噪声（所以要掩掉）。解法就是 response_mask 与 response_ids 同步 extend：模型段 extend 1，工具段 extend 0。

```
  序列:  [prompt | 思考1 | 搜索query1 | 工具返回1 | 思考2 | Answer]
   mask: [   -   |   1   |     1      |     0     |   1    |   1  ]
                 └────── 模型生成（训练）─────┘ └── 工具（不训练但可见）──┘
  代码: all_response_ids.extend(completion_ids);  all_response_mask.extend([1]*len(...))
        all_response_ids.extend(tool_ids);        all_response_mask.extend([0]*len(...))
```

**联系读者项目**：`run()` 主循环里就是这两对 extend——LLM 输出 extend 1，`_format_search_result` 编码后的工具文本 extend 0；训练时 loss 按 mask 只落在模型 token 上。这也是为什么 advantage 计算（B.3 的 ⑦）必须与 response_mask 对齐——搜索 token 不进优势。

**面试记忆点**：「TITO = Token-In-Token-Out：全序列入、部分出。『搜索结果的 token 为什么不训练』——它不由策略 π_θ 生成，梯度无意义；但必须进序列，因为下一轮生成要读它。」

### C.6 tokenize=False 手动编码：transformers 5.10 的「假 dict」

**一句话定义**：`apply_chat_template(tokenize=False)` 永远返回纯字符串，再用 `tokenizer.encode` 手动编码——绕开 transformers 版本间 tokenize 返回类型的漂移。

**原理**：`apply_chat_template` 在不同 transformers 版本下行为不同：老版本 `tokenize=False` 返回 str、`tokenize=True` 返回 list；transformers 5.10 的 `tokenize=True` 返回一个「dict-like 但不是 dict 子类」的对象——读者项目此前的 `isinstance(out, dict)` 归一化分支静默地**迭代了它的 key 名**（拿到的是字段名而不是 token）。改用 `tokenize=False`（恒返回 str）+ 手动 encode，行为跨 4.x/5.x 稳定，与 veRL 自己的 vLLM rollout 同款模式。

```
  tokenize=True : 返回类型随版本漂移（5.10 是 dict-like 非 dict）→ 归一化代码踩雷
  tokenize=False: 恒返回 str → tokenizer(text) 手动编码 → list[int]
                  （唯一稳定的跨版本路径）
```

**联系读者项目**：`_apply_chat_template` 里还有两个防御：`enable_thinking=False` 是 Qwen3 系扩展，老 tokenizer 不接受就 try/except TypeError 回退普通模板；encode 结果做「tensor→list、嵌套 list→扁平、批量→取首」三连归一化——全是踩坑后补的防御。

**面试记忆点**：「chat template 的稳定用法：tokenize=False + 手动 encode；`isinstance(out, dict)` 在 transformers 5.10 会被 dict-like 假 dict 骗过。依赖库返回类型漂移的通用解法：选一个恒定返回类型的分支，自己控制后续归一化。」

---

## D. TransferQueue 深度（读者项目的瓶颈分析主场）

### D.1 TQ 是什么：agent loop 与 vLLM 之间的调度层

**一句话定义**：TransferQueue（TQ）是 rollout 阶段 agent loop 与 vLLM 引擎之间的调度层——一个生产者-消费者队列，把「生成请求」和「生成结果」变成排队搬运的子操作。

**原理**：为什么不能直连——agent loop 是多轮、异步、每个样本一条轨迹的视角；vLLM 是批量、连续、全局一张 KV cache 池的视角。两者之间的每次「送 prompt / 取 token」都需要调度（谁先谁后、何时批起来、结果放哪）。TQ 把这层调度做成显式组件：agent loop 作为生产者提交子操作，调度层排队处理，结果回传后 agent loop 继续下一轮。

⚠️ TQ 是 veRL 内部组件，其具体实现（队列模型、并发控制、批处理策略）随版本快速演进，面试描述以「职责」为准，细节以源码/官方文档为准。

```
  生产者                          队列/调度                       消费者
  AgentLoopWorker × M      TransferQueue (controller)       vLLM RolloutWorker
        │  PUT prompt ──────────> [排队] ──────────> 批量生成 ──┐
        │  GET tokens  <────────── [回传] <─────────────────────┘
        │  （每轮交互 = 若干子操作，多轮 = 数百子操作）
```

**联系读者项目**：rollout 阶段的步时（50-70 分钟/步）绝大部分花在 TQ 的排队调度上——这正是 D.3 要定量证明的结论；`AgentLoopWorkerTQ` 这个名字里的 TQ 后缀即「走 TransferQueue 的 agent loop worker」。

**面试记忆点**：「TQ = agent loop 与 vLLM 之间的生产者-消费者调度层；agent loop 不直接碰引擎（B/C 的 server_manager 边界），一切交互都拆成排队子操作。多轮 agent loop × 大规模采样时，子操作数量爆炸——这是 agentic RL 的特有瓶颈。」

### D.2 TQ 架构组件：存储单元、controller 与 ACK 协议

**一句话定义**：TQ 内部可以粗分为三件套——存储单元（SimpleStorageUnit，数据落点）、controller（调度与状态管理）、以及它们之间带 ACK 确认的子操作协议。

**原理**：一次数据搬运不是「发过去就完」，而是带确认的协议：
- **SimpleStorageUnit**：存储单元，PUT_DATA / GET_DATA 子操作的落点，日志里带自己的统计（req_count、avg 耗时）；
- **controller**：调度中枢，排队、路由、维护数据状态，并对状态更新回 ACK；
- **PUT_DATA / GET_DATA**：两类原子子操作——agent loop 侧每轮「送 prompt、取 token」都会展开成若干这类操作；
- **ACK 确认机制**：写入方 PUT 完数据后，要等 controller 确认「状态已更新」才算完成。日志里的 `TQ_STORAGE ACK 超时` 指的就是这一步：数据已经 PUT 成功，但等 controller 的状态更新确认超时了。

⚠️ 上述组件划分基于 veRL 日志与源码目录名（transfer_queue/storage/managers/...），具体类职责与协议细节以源码/官方文档为准。

```
  AgentLoopWorker ──PUT_DATA──> [SimpleStorageUnit] ──> controller 更新状态
       ▲                              │                        │
       └──────── GET_DATA <───────────┘◄─────── ACK 确认 ──────┘
                 （等 ACK 超时 = 数据在，确认没回来）
```

**联系读者项目（真实日志解读）**：读者项目 rollout 阶段出现过 12 次密集的

```
(AgentLoopWorkerTQ pid=138066) ERROR - transfer_queue.storage.managers.base -
[TQ_STORAGE_07f08a73]: Timeout waiting for data status update ACK from controller after 30s.
```

同时间窗口 SimpleStorageUnit 的 PUT_DATA 统计完全正常（req_count=10、avg 0.01s）——**存储没死、数据进去了，是 controller 没回 ACK**。这个「谁没回」的二分，就是 TQ 排障的第一问。

**面试记忆点**：「TQ 子操作是带确认的：PUT_DATA/GET_DATA 由 SimpleStorageUnit 落地、controller 管状态并回 ACK。ACK 超时的第一问：『数据在不在、谁没回』——存储统计正常 → 查 controller；存储统计也停 → 查队列是否饿死。」

### D.3 瓶颈定量分析方法论：分解 → 测量 → 交叉验证

**一句话定义**：定位「训练为什么慢」的标准三段式——把步时分解成阶段、测量每个阶段的工作量与吞吐、换掉嫌疑组件做交叉验证。

**读者项目完整方法（重点复述）**：

**第一步：分解**。把每步拆成五段：rollout（TQ 调度多轮生成 + 搜索）/ old log prob / reward / 训练 / 权重同步。观测到步时 50-70 分钟，先问「哪段吃的」。

**第二步：测量（两个数相除）**：
- 统计每个 agent loop 产生的 TQ 子操作数 ≈ **400 个/loop**（多轮交互 × 每轮若干 PUT/GET）；
- 每步 128 个 loop（32 样本 × n=4 轨迹）→ **51,713 个子操作/步**；
- TQ 内部统计：SimpleStorageUnit 的 PUT_DATA 吞吐 ≈ **736 次/分钟**（TQ 整体任务吞吐约 600-700 tasks/min）。

```
  纯调度时间 ≈ 51,713 操作/步 ÷ 736 操作/分钟 ≈ 70 分钟/步
  实测步时 50-70 分钟 —— 数量级吻合 ⇒ 调度开销解释了绝大部分步时
```

**第三步：交叉验证（黄金标准）**。直觉上「搜索太慢」是最大嫌疑——搜索后端从知乎（0.68s/次）换 DeepSeek（1-3s）再换 MiMo（6-14s），**步时几乎不变**。如果瓶颈在搜索 API，步时应随延迟线性变化；不变 ⇒ 瓶颈在 TQ 调度层，不在搜索 API。

```
  假设: 瓶颈在搜索 API    预测: 后端 0.68s→6-14s，步时应放大 ~10 倍
  实测: 步时几乎不变      结论: 假设被推翻 ⇒ 瓶颈在 TQ 调度
```

**为什么这套方法值得面试讲**：① 两个数字（51K、736/min）都是框架统计字段读出来的，可复现；② 结论不是「感觉慢」，而是「数量级吻合 + 对照实验不变」的双重证据；③ 它还顺带证明了一个工程判断——**在多轮 agentic RL 里，调度层的子操作开销可以盖过业务 API 的开销**。

**联系读者项目**：测量点就是 C.3 里埋的 metrics（generate_sequences / search_execute 计时）与 TQ 自带统计；搜索后端的可插拔设计（search.py 五个后端）让交叉验证「换后端」变成一行配置——**可插拔的架构是瓶颈分析的工具**。

**面试记忆点**：「瓶颈定位三段式：分解（五段）→ 测量（400 子操作/loop × 128 loop = 51K/步 ÷ 736/min ≈ 70min ≈ 实测）→ 交叉验证（换搜索后端步时不变）。『换掉嫌疑组件看指标动不动』是黄金标准；可插拔架构让这个实验便宜。」

### D.4 优化方向四层：从「改配置」到「改范式」

**一句话定义**：针对「子操作数量 × 单操作成本」这两个乘数，优化方向按改动成本从低到高排四层。

**原理**：调度时间 = 子操作数 × 单操作成本。四层优化分别攻击两个乘数：

```
  第1层 批量提交        攻击「单操作成本」：把多轮 generate 的逐条排队改成批量提交
                        （PUT/GET 一次搬一批），省掉排队/调度的固定开销
  第2层 减少子操作数     攻击「子操作数」：合并 metrics/extra 字段的多次 PUT；
                        降组内并发（n=4→2）直接砍一半 loop 数（牺牲多样性换吞吐）
  第3层 搜索异步化       攻击「关键路径」：搜索是 HTTP I/O，不占 GPU——
                        把搜索移出 TQ 关键路径，不阻塞 generate 队列
  第4层 调度范式改造     换范式：把多轮循环放进推理引擎内部（引擎侧一次调度跑完
                        多轮），需要框架级改动
```

⚠️ 各方向的实际收益、实现难度取决于 TQ 的具体实现版本，动工前应先在目标版本上 benchmark 验证（读者项目经验：先在本地验证逻辑再上云跑训练）。

**联系读者项目**：这四层来自对 51K 子操作/步 的归因——每 loop 400 个子操作里，纯粹的「生成往返」占大头；搜索本身（0.68-14s）在 70 分钟步时里只是零头。所以优化优先级是「调度 → 数量 → I/O → 范式」，而不是直觉上的「把搜索调快」。

**面试记忆点**：「优化从两个乘数入手：子操作数 × 单操作成本。先做便宜的（批量提交、合并 PUT），再做贵的（降并发、范式改造）。归因决定优先级——51K/步 的数据说明搜索延迟不是主要矛盾。」

### D.5 ACK 超时与 thundering herd、TQ_DATA_UPDATE_RESPONSE_TIMEOUT=120

**一句话定义**：ACK 超时是「数据已写入、确认没回来」的协议级故障；当大量子操作同时超时并在恢复后同时重试，就形成 thundering herd（惊群），把刚恢复的 controller 再次压垮。

**原理**：超时 → 重试是分布式系统的标准恢复手段，但「所有等待方用同一个超时窗口」会让恢复变成雪崩：controller 一恢复，几十个被憋住的等待方同时重试，瞬时请求量反而超过故障前。⚠️ 该故障模式的成因推断（等待方同步重试）基于日志现象与通用分布式理论，veRL TQ 的具体重试/退避实现以源码为准。

**联系读者项目（修复记录）**：
- **现象**：`Timeout waiting for data status update ACK from controller after 30s` 密集出现 12 次；同期 SimpleStorageUnit PUT_DATA 正常 → 存储活着，controller 没回 ACK；
- **修复**：补丁 v7 给 `_notify_and_wait` 加 3 次重试 + 环境变量 `TQ_DATA_UPDATE_RESPONSE_TIMEOUT=120`（超时窗口从 30s 拉长到 120s）——给 controller 更多时间回确认，同时用有限重试避免无限等待；
- **旁证**：同日志里 `Using blocking ray.get inside async actor` 是性能警告（A.5），与 ACK 超时无因果，排障时别被无关警告带偏。

**面试记忆点**：「ACK 超时第一问『谁没回』：存储统计正常 → 查 controller。雪崩型超时要防重试风暴：有限重试 + 拉长超时窗口（30s→120s）+ 重试退避/抖动。日志里的警告不等于根因——先做因果排除。」

---

## E. Reward worker 字段流转

> 本部分逐段对应 `verl_search_r1/reward_fn.py`（`search_r1_reward`）。

### E.1 keyword-only 调用约定：reward 函数的标准签名

**一句话定义**：veRL 0.10 以 keyword-only 方式调用自定义 reward 函数——固定传 `data_source / solution_str / ground_truth / extra_info` 四个关键字，返回值是 float 或含 score 的 dict。

**原理**：RewardLoopWorker 的 naive reward manager 按统一约定调 reward 函数：每个样本一次，文本类输入（数据源名、模型答案文本、标准答案列表、附加信息）全部按名字传入；函数返回 float 时直接当 score，返回 dict 时取 `"score"` 键当 score、其余键进 reward_extra_info（见 E.4）。⚠️ 该调用约定随 veRL 版本漂移（读者项目就经历过「旧签名 → 新签名」的破坏性变更），以当前版本 `experimental/reward_loop/reward_manager/naive.py` 源码为准。

```python
def search_r1_reward(
    data_source: str | None = None,          # 数据集名（如 "nq"/"hotpotqa"）
    solution_str: str | None = None,         # 解码后的模型回答文本
    ground_truth: list[str] | None = None,   # 标准答案列表
    extra_info: dict[str, Any] | None = None,# 数据集 extra_info + tool_extra_fields 合并
    **kwargs: Any,                           # reward_kwargs 注入（见 E.3）
) -> dict[str, Any]:
```

**联系读者项目**：读者项目第一版 reward 函数签名不符合 0.10 约定，训练启动即 `TypeError: search_r1_reward() missing 'data'`——旧版本传的是 `data`（整个数据行），新版改成四个 keyword-only 参数。修复就是按新约定重写 `reward_fn.py`。**教训：自定义 reward 是框架版本升级的重灾区，升级 veRL 后第一件事是对照 naive.py 的调用点核签名。**

**面试记忆点**：「veRL 0.10 的 reward 调用约定：四个 keyword-only 参数 + **kwargs；返回 float 或 {"score": ..., 分项指标}。签名漂移是版本升级第一雷——对着 naive.py 的调用点核对，别对着旧教程抄。」

### E.2 parquet 列 → 数据加载 → reward 函数的路径

**一句话定义**：训练数据的 parquet 列（`prompt`、`reward_model.ground_truth`）经 veRL 数据加载变成 DataProto 的 non_tensor_batch，再按名字喂进 reward 函数的对应参数。

**原理**：数据流三步——① `prepare_data.py` 把 NQ/HotpotQA 的原始 JSONL 转成 veRL 可读的 parquet（列名就是接口契约）；② veRL 数据加载器读 parquet，每行拆进 DataProto（prompt 进 non_tensor_batch 的 raw_prompt，ground_truth 留在同名字段）；③ RewardLoopWorker 按 E.1 的约定把对应字段填进 `ground_truth=...`、`extra_info=...` 调 reward 函数。

```
  NQ/HotpotQA JSONL
       │ prepare_data.py（列名即契约）
       v
  train.parquet: [prompt | reward_model.ground_truth | ...]
       │ veRL 数据加载
       v
  DataProto: batch(input_ids...) + non_tensor_batch(raw_prompt, ground_truth, ...)
       │ RewardLoopWorker 按名传参
       v
  search_r1_reward(ground_truth=..., solution_str=..., extra_info=...)
```

**联系读者项目**：`reward_fn.py` 里 `gt = ground_truth or []` 后交给 `score_answer(final_text, gt)` 做 EM 判分；`extra_info` 里既有数据集列的 extra_info，也有 C.3 那条链路送来的 `tool_extra_fields`（response_text/search_calls）——**两条数据源在 extra_info 里合流**，这是 E.2 与 B.2 的接头点。

**面试记忆点**：「parquet 列名 = 数据契约，reward_model.ground_truth 这种带点号的列名对应嵌套字段；数据集列与 agent loop 的 tool_extra_fields 在 reward 的 extra_info 里合并——奖励函数看到的是『数据 + 轨迹』的合流视图。」

### E.3 reward_kwargs 配置注入：平铺键被静默忽略的坑

**一句话定义**：自定义 reward 函数的额外参数只有一条合法通道——`reward.custom_reward_function.reward_kwargs.*`，写在别处的键会被静默忽略。

**原理**：veRL 配置里 reward 节点的结构是 `reward.custom_reward_function.{path, name, reward_kwargs}`。加载 reward 时框架只把 `reward_kwargs` 子树展开成 dict、以 **kwargs 形式传进函数（E.1 的 **kwargs 就是接这里的）。如果你把参数写成平铺键（如 `reward.custom_reward_function.enable_prm_lite`），配置树里有这个键、训练也照常启动，但函数永远收不到——**静默失效，没有报错**。

```
  ✓ 合法:  reward.custom_reward_function.reward_kwargs.enable_prm_lite: true
           → 出现在 search_r1_reward(**kwargs) 的 kwargs 里
  ✗ 平铺:  reward.custom_reward_function.enable_prm_lite: true
           → 配置树里存在, 训练正常启动, 但 kwargs 里没有 → PRM 静默关闭
```

**联系读者项目**：`reward_fn.py` 的用法注释就是合法写法（`reward_kwargs: enable_prm_lite: true / enable_lata: false`），函数内 `kwargs.get("enable_prm_lite", False)` 取值。项目更早还踩过「veRL main API 漂移导致 PRM 静默失效」——同一类问题的两个变体：**配置键位置错了、或版本迁移后键挪了，都不会报错，只会在指标里悄悄消失**。防御手段：reward 函数启动时把收到的 kwargs 打一条日志，配置对不对一眼可见。

**面试记忆点**：「自定义 reward 的参数只有 reward_kwargs.* 一条合法通道；平铺键静默失效、版本迁移挪键也静默失效——静默失效比报错更贵。防御：函数里 log 一次收到的 kwargs。」

### E.4 返回 dict 的额外键 → reward_extra_info：分指标监控

**一句话定义**：reward 函数返回 dict 时，`"score"` 键进优势计算，其余键全部进 `reward_extra_info`——变成每步可监控的分项指标。

**原理**：score 是标量裁决（进 GRPO 组内归一化），但训练曲线上的总分解释力有限：score 掉了，是格式崩了还是答案没匹配上？把判分过程的分项（格式对否、EM 对否、过程奖励多少）作为额外键返回，veRL 把它们收集进 reward_extra_info 并随训练日志/指标系统每步上报——**score 管训练，extra 管诊断**。

```python
out = {
    "score": float(reward_val),               # 裁决 → advantage
    "exact_match": float(result.exact_match), # ─┐
    "valid_format": float(result.valid_format), # │→ reward_extra_info
    "prm_process_reward": float(prm_val),       # ─┘  分项监控
}
```

**联系读者项目**：读者项目的四个键恰好覆盖三层信号——`score`（EM+格式+过程奖励的合成裁决）、`exact_match`/`valid_format`（答案正确性与格式合规拆开看）、`prm_process_reward`（PRM-Lite 22 条规则的过程奖励，开关打开时才出现）。每步日志里这些分项曲线能直接回答「奖励在涨，但涨的是格式分还是答案分」。

**面试记忆点**：「score 是标量裁决，extra 键是诊断维度——多指标监控是训练曲线的第二双眼睛。讲奖励设计时顺带讲『我为什么把 EM 和 format 拆开返回』，比只讲一个 score 函数可信得多。」

### E.5 pkg:// 前缀：importlib 动态导入 vs 文件路径

**一句话定义**：veRL 里引用自定义 reward 函数，`pkg://包.模块` 走 importlib 动态导入；不带前缀的裸路径会被当成文件路径去 open——后者必然 FileNotFoundError。

**原理**：`pkg://verl_search_r1.reward_fn` 让 veRL 用 importlib 按「包路径」加载模块（要求模块在 worker 进程的 sys.path 上）；没有 `pkg://` 前缀时，框架按「文件系统路径」解析——而 `verl_search_r1.reward_fn` 不是合法的文件路径。两条加载路径完全不同，前缀决定走哪条。

```
  path: pkg://verl_search_r1.reward_fn   → importlib.import_module("verl_search_r1.reward_fn")
  path: verl_search_r1.reward_fn         → 当文件路径 open → FileNotFoundError ✗
```

**联系读者项目**：这是读者项目踩坑清单第 3 条（「reward_fn 路径必须 pkg:// 前缀」）。注意它与 A.4 的闭合关系：pkg:// 走 importlib → importlib 用 sys.path → worker 的 sys.path 靠 **.pth 文件**保证。所以「pkg:// 前缀 + .pth 文件」是一对配套修复：前者决定**怎么加载**，后者决定**在哪找**——缺任何一个，reward 都导不到。

**面试记忆点**：「pkg:// = importlib 动态导入，裸路径 = 文件路径。自定义模块在 Ray worker 里要可见，必须同时满足两条：路径写法正确（pkg://）+ 模块在 worker 的 sys.path 上（.pth/runtime_env）。」

---

## 结尾：Part 3 面试提纲（十问自查）

1. Ray 三件套是什么？actor 和 task 的区别？RL 为什么用 actor？
2. Ray worker 的环境继承问题：现象、根因、.pth 修法、为什么 probe 测不到？
3. async actor 里 ray.get 会怎样？正确写法？
4. veRL 五种角色的职责一句话各是什么？（画图）
5. DataProto 双通道怎么分？tool_extra_fields 从生成到奖励的完整路径？（画图）
6. 一个 step 的七段时序：哪段哪个 worker、走哪条通道？（画图）
7. update_weights 的三颗雷（显存双副本 / IPC INT_MAX / 聚合方式）与读者项目的修复？
8. AgentLoopOutput 七字段、TITO 的 mask 规则、tokenize=False 的原因？
9. TQ 瓶颈怎么定位的：分解 → 51K/步 ÷ 736/min ≈ 70min → 换搜索后端交叉验证？
10. reward 的 keyword-only 约定、reward_kwargs 平铺键的坑、pkg:// 与 .pth 的配套关系？

**口径自查（写作纪律）**：veRL 侧成绩一律以「环境调通（probe 47/47 全绿；OOM/IPC/接口/DNS 等启动问题全部修复并验证）+ TQ 瓶颈定量分析（每步约 51K 子操作、PUT_DATA 吞吐约 736/min、步时 50-70min、换搜索后端交叉验证）」为正面框架；所有 TQ 内部机制与 veRL 版本行为均已用 ⚠️ 标注、以源码/官方文档为准。

**参考文件**：
- 读者项目代码：`agentic-rl-lab/03-search-r1/verl_search_r1/{search_agent_loop.py, reward_fn.py, agent_config.yaml, e6b_run.sh}`
- 调试记录：`2026-08-30-OOM-HANDOFF.md`（update_weights OOM 六轮调试）、`2026-08-30-R12-TQ-HANDOFF.md`（TQ ACK 超时）
- 瓶颈分析：`SEARCH-R1-COMPLETE-REPORT.md` §2.13、开源仓库 `docs/07-INTERVIEW_QA.md` Q16
- 衔接：Part 1（理论基础）/ Part 2（GRPO 算法与显存账）
# 系统工程教学指南 · Part 4：工程实践 · 面试问答 · 速查表

> 目标读者：准备算法岗/工程岗面试、有云 GPU 实操经历的人。
> 项目背景：Search-R1 复现（2×4090、Qwen3.5-4B、veRL main 0.10）。
> 数字出处：docs/02-TUTORIAL.md、E6B_DEPLOY.md、e6b_run.sh（2026-08-30 快照）。
>
> **数字口径纪律**（写作与面试通用）：
> - 跨口径数字不比大小；81.2% 仅作同管线前后对比（SFT 1.43% →
>   GRPO 81.2%），并注明是训练批次内指标，不是留出集评估。
> - veRL 阶段表述框架：环境调通（12 轮调试、probe 47/47 全绿）+
>   TQ 瓶颈定量分析（51K 操作/步、736/min、50-70min/步）。
> - 不确定的细节标 ⚠️，不编数字。

---

## 第 A 章 环境治理：九个坑的底层机制

每个坑按「现象 → 机制（原理级）→ 修复 → 教训」组织。表面现象谁都会描述，
面试场上的区分度在机制层：文件系统语义、进程模型、类型系统、网络协议——
坑在哪个层，修复就在哪个层。

### A.1 torch 混合安装 → torch.compile ImportError

**现象**：`ImportError: cannot import name 'unpack_mixed_mm'`
（torch.compile 的 inductor 路径）。

**机制**：pip 的安装语义是"覆盖写 + 增量删除"，不是"清空重装"——
- pip 卸载只删除当前发行版 RECORD 文件里登记的路径；旧版本独有的模块
  文件（如旧的 `torch/_inductor/kernel/unpack_mixed_mm.py`）不在新版本的
  RECORD 里，卸载时不会被动，site-packages 留下"僵尸文件"。
- 新版本安装只覆盖同名文件，不会删除"新版本里已不存在的模块"。
- import 时 Python 按 sys.path 找到文件即加载；僵尸文件加载成功，
  但它内部 import 的符号在新库里已改名 → ImportError。
- 本质：site-packages 是"文件堆"，版本一致性靠 RECORD 约定，不靠目录隔离
  （对比 conda/nix 的目录级隔离）。本项目触发路径：多次 pip install torch
  升级后新旧文件混装。

**修复**：按报错信息里的文件路径找到陈旧文件直接删除——报错里的路径
要去看文件本身，而不是只看错误消息。

**教训**：深度学习库升级后跑出 ImportError，第一个怀疑对象就是文件残留；
根治手段是重建虚拟环境（目录级隔离，一了百了）。

### A.2 FlashAttention2 缺失 → sdpa 注入

**现象**：Qwen3.5（Transformer + Mamba 混合架构）加载即崩，
FA2 路径编译失败，训练启动不了。

**机制**：
- FA2 不是纯 Python 库：它提供编译期按 GPU 架构（sm_*）与 head_dim
  特化的 fused CUDA kernel，且模型类必须显式实现 FA2 的调用路径。
  ⚠️ 具体 head_dim/架构支持矩阵随版本变化，但"编译型 kernel +
  白名单模型类"是恒定的约束结构。
- Qwen3.5 含 Mamba 层，其注意力路径不在 FA2 支持矩阵里 → 指定
  `attn_implementation="flash_attention_2"` 时找不到可用实现 → 启动崩。
- sdpa 的原理：`scaled_dot_product_attention` 是 PyTorch 原生的 fused
  入口，内部自动 dispatch 到 cuDNN attention / FlashAttention kernel /
  memory-efficient attention——后者用 online softmax（分块 tile，只保留
  running max 与 running sum，避免物化 [seq, seq] 的注意力矩阵）。
  它不要求模型类特殊适配，是"框架替你选融合 kernel"的兜底方案。

**修复**：在 veRL 的 transformer_impl.py 注入 `attn_implementation="sdpa"`。

**教训**：接入新模型先查"注意力后端支持矩阵"；sdpa 是通用兜底，
4B 小模型上性能损失可接受，别再为 FA2 单独折腾。

### A.3 Ray worker 不继承 PYTHONPATH → .pth 固化

**现象**：driver 进程能 import `verl_search_r1.reward_fn`，但
RewardLoopWorker 报 FileNotFoundError——同一个 Python 包，两个进程
两种命运。

**机制**：三层递进——
1. shell 环境 vs 进程环境：`export PYTHONPATH` 只作用于从该 shell 启动的
   进程；Python 解释器只在启动时读一次环境变量构造 sys.path，之后
   shell 再 export 对已运行进程无效。
2. Ray 的进程树：worker 不是 driver 的子进程，而是由 raylet 派生/管理
   的进程，其环境继承链与"你 export 过的那个 shell"不是一条线。
   ⚠️ Ray 不同版本对 worker 环境同步（runtime_env / env_vars）的行为
   有差异，但"driver 能 import ≠ worker 能 import"是分布式框架的
   通病，必须按通病设防。
3. .pth 机制：Python 解释器启动时 site 模块扫描 site-packages 下的
   .pth 文件，把每行当作 sys.path 追加项——它挂在"解释器"上，不挂在
   shell 或环境变量上，因此对任何用这个 Python 启动的进程生效，
   天然跨进程。

**修复**：在 site-packages 写 .pth 文件（一行一个项目路径）。

**教训**：probe 跑在 driver 进程，天然测不到 worker 问题——所以 probe 第二节专门做"worker 式实例化"模拟（见 B.4）。

### A.4 DNS 只解析 IPv6 无路由 → HF_HUB_OFFLINE

**现象**：服务器上 `from_pretrained` 卡死数十分钟，本地同样代码秒开。

**机制**：
- DNS 查询返回两类记录：A（IPv4 地址）与 AAAA（IPv6 地址）。AutoDL
  网络下 huggingface.co 只解析出 AAAA（IPv6），而实例没有可用的 IPv6
  默认路由——connect() 发往黑洞，TCP 在超时与重试上消耗数分钟。
- huggingface_hub 的加载路径：`from_pretrained` 默认先做网络探测（远端
  metadata/etag 检查），网络挂起时整个加载被拖死——即使模型已全部缓存。
- HF_HUB_OFFLINE=1 的机制：读到该环境变量直接短路所有网络路径，走本地
  缓存快照（~/.cache/huggingface 的 snapshot）——模型已缓存时这是最快
  也最稳的路径，连探测都不做。

**修复**：`HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`（e6b_run.sh 固化）。

**教训**：云环境网络是隐性变量——本地能跑 ≠ 云上能跑；部署前把模型
缓存打满 + 打开离线开关，把"网络"这个依赖直接从启动路径里删掉。

### A.5 chat template tokenize=True 返回"假 dict"

**现象**：transformers 5.10 下 `apply_chat_template(tokenize=True)` 的
返回值没过 `isinstance(out, dict)` 归一化分支，代码迭代出 key 名
（字符串）→ encode 出垃圾 token id——生成序列全乱，且没有任何异常
抛出（静默故障）。

**机制**：
- 该 API 的返回类型随 transformers 版本变化；5.10 下实测返回的是
  "dict-like 但不是 dict 子类"的对象。⚠️ 具体类型名随版本漂移，
  但"跨版本返回类型不稳定"是 transformers 这类快节奏库的常态。
- 防御性代码写 `isinstance(out, dict)` 才走 dict 归一化 → 判断为 False
  → 落到"直接当返回值用"的路径 → 迭代 dict-like 时拿到的是 key 名
  （"assistant" 这类字符串）→ tokenizer.encode("assistant") 得到字符串
  的 token id 而不是模板 token 的 id → 垃圾序列。
- 本质：Python duck-typing 的坑——"像 dict"≠"是 dict"。isinstance 严格
  检查在版本漂移下从"防御"变成"故障点"：类型一漂移，检查静默失效，
  且失效方向是"退回到更糟的分支"而不是报错。

**修复**：tokenize=False + 手动 encode——把模板渲染与 tokenize 解耦，
自己控制每一步的类型，不依赖上游的返回类型契约。

**教训**：跨版本 API 的返回类型要按"严格契约"处理；probe 的
chat-template 冒烟检查（输出 >50 token）就是为这类静默故障设计的——
垃圾 token 序列通常很短。

### A.6 左 padding 切片 bug（prompt 被重复训练）

**现象**：训练 loss 异常低、模型退化；检查发现 completion 里混着
prompt token。

**机制**：
- batch 内序列长度不一 → 必须 padding 对齐。padding 方向决定"有效
  内容"在张量里的锚点：
  - 左 padding：`[PAD PAD ... prompt tokens]`，内容靠右，有效末尾 =
    max_len，新生成的 token 从 max_len 位置继续追加；
  - 右 padding：`[prompt tokens ... PAD PAD]`，内容靠左，有效末尾 =
    prompt_len。
- 代码用 prompt_len 切片（右 padding 的锚点）取 completion——在左
  padding 下切出来的是 prompt 尾部 token。它们被当 completion 训练 →
  模型学"重复 prompt 尾部" → loss 异常低（目标 ≈ 输入，模型躺着拿
  低 loss）但能力退化。
- 本质：索引语义绑定 padding 方向，两个约定在同一张张量上不可混用。切片
  锚点选错的 bug 不报错、loss 反而更好看——这是最危险的一类 bug。

**修复**：`outputs[i, max_len:]`。

**教训**：写完切片代码先打印几个样本人肉验证；padding 方向写进配置，
并在切片处断言，让两种约定在代码里只出现一处。

### A.7 大词表 log_softmax OOM

**现象**：训练第一步 forward 就 OOM：`Tried to allocate 5.35 GiB`。

**机制**：
- Qwen3.5 词表 151936（大词表，tiktoken 级）。lm_head 输出 logits 形状
  = [batch, seq, vocab]；batch=2、seq=8000 时：
  2×8000×151936×2B(bf16) ≈ 4.85GB。
- 关键认知：logits 张量的显存与模型参数量无关，与 token 吞吐有关——
  vocab 维是"隐式放大镜"，seq、batch 越大这一项越凶；fp32 下还要翻倍。
  它不在"权重 + 梯度 + Adam"的常规账里，是独立的第四项开销，大词表
  模型必须单列。全量 log_softmax 要求 logits 一次性物化 → 峰值一把爆。

**修复**：`_chunked_log_softmax_gather` 按 512 token 分块——只在需要 gathered
log prob 的位置算完整 softmax，其余 token 逐块算完即释放（峰值 5GB → 300MB）；
配合 MAX_TRAIN_SEQ_LEN=4096 截断 + MICRO_BATCH_MAX_TOKENS 8000 → 4000。

**教训**：大词表模型（Qwen/Yi/DeepSeek 系）的 log_softmax 是显存杀手，
算显存账时要单列 logits 峰值这一项。

### A.8 预生成缓存命中率 0%

**现象**：预生成 2000 条搜索缓存，训练时命中率 = 0%。

**机制**：
- 缓存成立的前提：写入 key 的生成函数 == 查找 key 的生成函数。
- 预生成时 key = 训练问题原文（"total number of death row inmates in
  the us?"）；运行时查找 key = 模型生成的搜索 query（"death row
  inmates statistics"）——中间隔了一层不可控的模型改写映射，两个
  key 空间几乎不重合 → 命中率 0%。
- 这不是实现 bug 而是设计缺陷：key 空间设计错误，代码写得再对
  命中率也是 0。
- 对照：运行时缓存（同 step 内相同 query 自然命中）的 key 生成函数
  与查找函数是同一个，所以天然正确。

**修复**：放弃预生成，改运行时缓存。

**教训**：缓存 key 必须和运行时查询 key 一致；先小样本验证命中率再
全量部署。这个错误 ¥0 就能在本地发现，却花在了服务器上。

### A.9 服务器操作纪律（四条铁律）

远程服务器没有 IDE、没有撤销，每条纪律背后都是一次真实事故：

1. **命令单行 <100 字符**：SSH 终端粘贴多行命令时，换行符被当作回车执行
   ——命令在换行处被截断执行一半，剩下半条变成新命令。<100 字符 ≈ 一行放得下。
2. **sed 锚点唯一 + ast.parse 验证**：sed 是行级文本替换，锚点不唯一
   会误替换多处；改完 .py 文件用 `ast.parse` 验证语法——远程没有
   编辑器红线，ast.parse 是零成本的语法 CI。
3. **pkill 锚定防自杀**：`pkill -f` 按命令行模式匹配，模式太宽
   （如 'python'）会匹配到你正在执行该命令的远程 shell 自身 → 把自己
   杀掉，SSH exit 255。必须锚定唯一前缀，如 `'^python -u e6b_main'`。
4. **tar.gz 打包上传**：多文件/目录结构一次性上传（保留权限与目录
   结构），避免逐文件传输的遗漏与顺序问题；上传后 `tar xzf` 解压即
   完整工作目录。

**教训**：远程操作的成本不对称——本地写错按一下退格，远程写错可能
毁掉一个小时的安装成果。所有操作先本地演练、写成脚本再上传执行。

---

## 第 B 章 云 GPU 实操

### B.1 AutoDL 费用模型

三个概念必须分清（价格为 2026-08-30 快照；⚠️ 平台规则会更新，
以控制台为准）：

1. **按卡时计费**：价格单位是"每 GPU 每小时"（本项目 4090 为
   ¥2.18/卡/h），2×4090 实例 = ¥4.36/h。开机即计费，与利用率无关——
   GPU 空转等搜索结果时，钱照扣。
2. **关机 ≠ 释放**：关机停止计费、数据保留；但实例仍占用资源配额。"释放
   实例"才返还配额，代价是数据删除——关机与释放是两次决策：还要继续跑
   就关机留数据，项目结束才释放。
3. **无卡模式**：纯 CPU 实例，价格极低，用于传数据、预处理、打包——
   数据工作不需要 GPU，别在 GPU 实例上做。

**实操顺序**：无卡模式传数据/打包 → 开机 → 安装依赖（一次性）→ probe
→ 训练 → 评估 → 立即关机。每一分钟 GPU 时间都要有明确用途。

### B.2 预算先行：费用公式与止损规则设计

**费用公式**：步时 × 步数 × 单价 × 卡数 = 总费用。E6-B 实例：
¥2.18/卡/h × 2 卡，50 步 × 预估 4-6 min/步 ≈ 5-6.5 h + 安装 1.5 h →
总费用 ¥22-28。1 卡降级 ¥14-20；4 卡 ¥35-44 超预算不推荐。

**止损规则要在跑起来之前写进文档**——训练中的人有沉没成本效应，"再跑跑看"
是烧钱最强的一句话。每条规则背后是对应的失败模式：

1. **>12min/步 → 停**：预估 4-6 min，12min 意味着存在隐藏瓶颈（本项目
   TQ 瓶颈实测 50-70min/步）。步时异常通常伴随 bug，继续跑买不到任何
   信息；且 50 步 × 12min = 10h ≈ ¥40+，预算直接击穿。
2. **连续崩溃 → 停**：同位置反复崩溃 = 环境问题未解决，重启不改变
   任何变量，重试 = 重复花同样的钱买同样的崩溃。
3. **format 连续 5 步 <20% → 停**：格式是 Search-R1 的硬门槛——
   format 崩 → 解析器拿不到动作 → 轨迹 reward 全 −0.1 → advantage
   全 0 → GRPO 没有任何学习信号，后面步数纯属浪费。
4. **EM 连续 10 步无提升 → 停**：RL 收益递减，10 步无提升说明配方
   （LR/奖励/数据）无效，要改配方而不是加步数。

**教训**：止损规则不是悲观，是把"什么时候认输"从情绪决策变成事前决策。
预算 ~¥43 没有重试余量，止损规则就是预算的执行器。

### B.3 部署包设计：把"本地验证过的一切"完整搬运

**打包件结构**（deploy_e6b/ → 上传解压为 ~/autodl-tmp/search-r1/）：
- `verl_search_r1/`：代码核心——e6b_run.sh（启动 + 显存预设）、
  e6b_main.py（入口）、probe_verl_api.py（体检）、prepare_data.py、
  search_agent_loop.py、reward_fn.py、agent_config.yaml；
- `datasets/`：train/test.jsonl + dev.jsonl（70 题评估集）；
- `eval_local.py` / `cloud_eval.sh`（云端评估全参数 checkpoint，不用下载 16GB 权重）；
- 本地管线复用件（protocol/search/reward/reward_lite）、`deepseek_search/`、
  `.env`（API key）、E6B_DEPLOY.md。

**三个设计要点**：

1. **环境锁定**：模型权重在服务器用 HF mirror 下载并缓存；veRL/vLLM
   由 setup.sh 按 pyproject 的 pin 版本安装。⚠️ veRL main 无 tag 锁定
   ——这是权衡：原 fork 锁死 transformers<4.48 + vllm<=0.6.3，根本
   加载不了 Qwen3.5-4B（需要 transformers≥4.57）；走 main 则 API 会
   漂移，对策是"probe 守漂移"（见 B.4）。
2. **.pth 固化**：site-packages 写 .pth（A.3 的修复）写进部署流程——
   导入路径在环境层面固化，不依赖任何 shell。
3. **单一事实源**：e6b_run.sh 的 OVERRIDES 数组同时被 probe 与启动
   消费（probe 用同一份 overrides 做 hydra compose 漂移扫描）——
   检查的和跑的是同一份配置，杜绝"检查的配置 ≠ 运行的配置"。

**教训**：服务器只做三件事：解压、装依赖、跑 probe。所有"聪明"都在本地
完成——本地写错的成本是分钟，服务器写错的成本是 GPU 小时。

### B.4 probe/体检文化：47 项检查的 7 节设计方法论

**总原则**：probe 失败的成本是 0，训练失败的成本是 GPU 小时。一次
"跑 10 分钟才发现 import 错" = ¥0.7+，而这类错误本地 1 分钟就能
查出。

47 项检查按 7 节组织，从"必然炸"到"概率炸"排序——排序本身就是
方法论：

1. **agent-loop API 检查**：AgentLoopBase 签名、AgentLoopOutput 字段、
   注册表里有没有 search_r1_agent——API 漂移第一道防线（veRL main
   每天在变，8 月初写的代码 8 月底大概率不兼容）。
2. **worker 式实例化**：真实 tokenizer + DictConfigWrap，按 AgentLoopWorker
   的方式实例化 loop + chat template 冒烟（>50 token）——模拟 Ray worker
   的导入环境（A.3 的教训）。
3. **hydra compose + key-drift 扫描**：compose(ppo_trainer, overrides) 对
   不存在的键 hard-fail，带完整 override 列表的 compose 本身就是漂移检查；
   再用 wiring assert 确认合成结果（reward_kwargs 真的传进去了、
   rollout.name 真是 vllm）。
4. **reward 函数**：import + 签名（必须有 **kwargs）+ 假 DataProto
   冒烟（non_tensor_batch + tool_extra_fields 流转一次）。
5. **库版本 + GPU + flashinfer**（transfer_queue 缺失会精确告诉你
   pip 装什么）。
6. **数据文件 + 列名 + 启动脚本陈旧参数扫描**（如残留的 --config-path）。
7. **--live-search 实弹搜索**：真发一次 query（"Baybrook Mall"），输出
   延迟/条数/错误——API key 有效性在花 GPU 钱之前验证。

**0 FAIL 才启动训练**。probe 抓出过：reward_fn 导入路径、API 漂移 3 处、
配置键移动、数据列缺失。47 项里每一项背后都是一次真实失败——清单越长，
说明踩过的坑越多。

**边界**：probe 挡配置问题，但 Ray/AgentLoop 运行时问题要第 1-2 步才暴露
（项目文档预估首次跑通概率 60-70%）——把不确定性写进计划，而不是假装
probe 能挡一切。

---

## 第 C 章 性能分析实战

### C.1 分阶段计时方法论

**方法论**：把一步拆成阶段 → 每段埋计时点 → 占比排序 → 只优化最大头。
"每步耗时"是黑箱数字，必须拆开才能定位。

本项目（Search-R1 RL 一步）的设计目标五阶段占比：
1. rollout 内的 vLLM 生成：60-75%（多轮多轨迹的批量推理是大头）；
2. rollout 内的搜索调用：10-15%（每次 0.7-14s，线程池并发摊薄）；
3. FSDP 训练（前向/反向/优化器）：10-20%；
4. old log prob 计算（π_old 无梯度前向）：~8%；
5. update_weights 权重同步：3-5%（小模型占比低）。
⚠️ 以上是设计目标占比；TQ 调度瓶颈下 rollout 膨胀到 50-70min/步，占比
结构被彻底改写——这恰恰是分阶段计时的价值：它告诉你"时间花在哪个阶段"，
然后才能追进那个阶段继续拆。

**落地手段**（由便宜到贵）：日志时间戳切段 → 每阶段显式打印耗时 →
框架自带统计字段（TQ 的 req_count/avg 是免费 profiler）→ 最后才是
专业 profiler。

### C.2 瓶颈交叉验证方法论（可迁移的分析范式）

**范式**：怀疑组件 X 是瓶颈 → 换掉 X（换实现/换后端/换数据源）→ 看
指标动不动 → 不动则瓶颈不在 X，继续下一嫌疑。

本项目完整实例（TQ 瓶颈定位）：
1. 分解：每步 = rollout（TQ 调度多轮生成 + 搜索）+ old log prob + 奖励
   + 训练 + 权重同步五段；
2. 测量：每个 agent loop 产生 TQ 子操作 ≈400 个，128 个 loop/步 → ~51K
   子操作/步；TQ 内部统计 SimpleStorageUnit PUT_DATA 吞吐 ~736/min →
   纯调度时间 51K/736 ≈ 70 分钟，与实测步时 50-70 分钟吻合；
3. 交叉验证（关键一步）：搜索后端从知乎（0.68s/次）换 DeepSeek
   （1-3s）再换 MiMo（6-14s），**步时几乎不变**——若瓶颈在搜索 API，
   步时应随延迟线性变化；不变说明瓶颈在 TQ 调度，不在搜索。

**可迁移性**：这个范式适用于一切"慢在哪里"的问题——先列嫌疑清单
（搜索 API / 调度层 / 推理引擎 / 训练引擎），再用替换法逐个排除。
直觉给方向，实验给结论——"搜索太慢"的直觉被实验推翻了。

### C.3 I/O bound vs compute bound 的判断

**为什么重要**：两种 bound 的优化方向完全不同，选错方向的成本是
实打实的钱（买更贵的卡 vs 改代码）。

**判断工具**：
- nvidia-smi 的 Volatile GPU-Util：低且波动大 = GPU 在等（I/O bound）；
  高且平稳 = GPU 在算（compute bound）；
- 计时日志：哪段耗时占比最高；
- 观察 GPU 空转等搜索——本地管线每步 ~16min，GPU 大部分时间在等搜索
  API 返回（延迟 1-14s/次 × 每条轨迹 1-4 次，同步等待）。

**本项目结论**：本地管线是 I/O bound（同步搜索）——瓶颈不是显存
不是算力，换便宜卡即可（3060 ¥1/h 和 4090 效果一样）。veRL 管线
同样是 I/O bound，但 I/O 在调度层（TQ 51K 操作/步）——这层 I/O
换卡救不了，要改调度架构。

**教训**：先用 nvidia-smi + 计时日志分清 bound 类型，再决定是花钱
（升级硬件）还是改代码（异步化/批量化/换调度）。

---

## 第 D 章 系统工程类面试问答（10 题）

每题按「核心论点 → 展开 → 加分点/追问预案 → 一句话总结」组织，话术以
本项目（2×4090、Qwen3.5-4B、Search-R1 复现、veRL main 0.10）为背景。

### D.1 FSDP 原理？unshard/compute/reshard 是什么？通信量多少？

**核心论点**：FSDP 把参数、梯度、优化器状态**分片**到每张卡，前向时
按需 all-gather 拼出完整权重，算完立即释放回分片——它省的是 DDP 的
"复制"，代价是通信。

**展开**：
- 三阶段：unshard（all-gather，每卡发出自己的 1/P 分片、收到所有人分片，
  拼出完整权重）→ compute（正常前向）→ reshard（只保留自己分片，其余释放）。
- 通信量：前向每层一次 all-gather，每卡传输 (P−1)/P × 参数量；反向
  reduce-scatter 梯度，每卡传出 (P−1)/P × 梯度量。每步合计约
  2 × (P−1)/P × 参数量（⚠️ 不含 gradient checkpointing 重算带来的
  额外前向通信）。
- 与 DDP 对比：DDP 每卡一份完整模型副本，反向 all-reduce 梯度（每卡传
  全量梯度）；FSDP 每卡只常驻 1/N 参数，显存随卡数线性下降。

**结合项目**：2×4090 上 FSDP 分片 + optimizer_offload + param_offload——
Adam 状态（25.2GB）与参数（8.4GB）流式放 CPU；无 offload 每卡 26.7GB
必爆，offload 后 18.9GB 余 4GB。权重同步阶段用 layered_summon 分层 unshard，
避免 all-gather 全量 + 分片并存（峰值 12.6 → ~5GB）。

**加分点/追问预案**：
- 追问"通信会不会是瓶颈"：4B × 2 卡通信量小，PCIe 也够；本项目的
  步时代价（+30-60s）来自 offload 的 H2D/D2H，不是集合通信。
- 加分：ZeRO 谱系递进（ZeRO-1 只分片优化器状态 → ZeRO-2 加梯度 →
  ZeRO-3 加参数 = FSDP）；activation checkpointing 与 FSDP 正交。

**一句话总结**：FSDP 用通信换显存，分片粒度越细省得越多、通信也越多。

### D.2 CPU offload 省了什么、代价是什么？

**核心论点**：省的是 GPU 显存里的"低频使用状态"，代价是 PCIe 带宽
——纯时间换空间。

**展开**：
- 省了什么：Adam 状态（25.2GB/卡 → 0 常驻）与参数（8.4GB → 按需流式）。
  Adam 状态只在 optimizer.step 时被读写；参数只在对应层前向时需要——
  两者都是"低频按需访问"，放 CPU 用 PCIe 按需搬运，GPU 常驻只剩
  梯度 + 激活。
- 代价是什么：每步参数/状态流式进出，步时 +30-60s（本项目实测）；
  带宽瓶颈在 PCIe（32GB/s 级），比 NVLink 慢一个数量级。
- 为什么 4B 能接受：每步搬运量是参数量级（8.4GB 级），小模型步时
  增量可接受；70B 模型 offload 会让每步慢几倍，届时考虑 ZeRO-3 +
  更多卡或量化优化器。

**结合项目**：N=2 无 offload 每卡 26.7GB ❌（24GB 卡，vLLM 权重还没算）
→ offload 后 18.9GB ✓（余 ~4GB）。测量方法：公式推算 + 日志 allocated 验证。

**加分点/追问预案**：
- 追问"和 gradient checkpointing 的区别"：offload 省参数/状态显存
  （走 PCIe），checkpointing 省激活显存（走重算）——两者正交可叠加。
- 追问"offload 到 NVMe 呢"：比 CPU 内存再慢一个数量级，只在超大模型
  极限场景用。

**一句话总结**：offload 是"时间换空间"的经典权衡，模型越小越划算。

### D.3 2×4090 训 4B 显存怎么算？OOM 排查（峰值 20.67GB、缺口 400MB）

**核心论点**：显存账是四则运算，OOM 排查是逐点观测的二分定位——
两段式回答，先算后查。

**显存账**：bf16 全参数训练，每卡 = 权重 8.4/N + 梯度 8.4/N + Adam
25.2/N + 激活 ~1.5 + vLLM 推理池 8.4/N：N=2 无 offload → 26.7GB 必爆
（24GB 卡）；开 optimizer+param offload 后训练侧只剩梯度+激活 ≈ 5.7GB +
vLLM 池 13.2GB → 18.9GB ✓。

**OOM 排查（六轮，每轮一个观测点）**：
1. 现象：update_weights 阶段 CUDA OOM——申请 1.19GB、free 795MB、峰值
   allocated 20.67GB（本进程 21.94GB），缺口仅 ~400MB。
2. 观测点二分：参数已 CPU offload（allocated 0.00）→ OOM 发生在 load
   之后、state_dict 之间 → 定位到"权重复制进 CPU dict 前在 GPU 上
   物化了全量权重"这一步。
3. 排除项逐个验证：ref 已关（不成立）；layered_summon（只对 LoRA 生效）；
   GC threshold 0.05（只能回收 ~3%）。
4. 修复（v6）：get_per_tensor_param_shard() 取本地 FSDP 分片 +
   FULL_STATE_DICT + offload_to_cpu 流式传输，峰值 12.6GB → ~5GB。

**加分点/追问预案**：
- "信日志不信文档"：veRL 0.10 的 ref CPUOffload 在 torch 2.13 下
  实测不生效（常驻 8.46GB），官方文档与实测矛盾时信实测。
- 追问"400MB 缺口和 10GB 缺口哪个难修"：大缺口换方案，小缺口抠
  细节——碎片化/保留块级别的 400MB 最磨人。

**一句话总结**：OOM 调试 = 补丁打印 allocated/reserved 的二分定位，
不是猜。

### D.4 veRL 架构：Ray 角色分工、vLLM 与训练引擎怎么协作、TransferQueue 干嘛的？

**核心论点**：veRL 是 Ray 上的 RL 训练系统——五类角色 + 分时复用显存
的 HybridEngine + 一个调度层（TQ）。

**展开**：
- 角色（单节点 2×4090）：Trainer（driver，hydra 入口，编排整个 step）；
  ActorWorker ×2（FSDP 分片训练）；vLLM RolloutWorker ×2（推理引擎，
  PagedAttention + continuous batching）；AgentLoopWorkerTQ（持有多轮
  交互的 agent loop 实例——自定义 SearchR1AgentLoop 跑在这里）；
  RewardLoopWorker（调奖励函数）。
- 协作方式：HybridEngine——同一批 GPU 上训练/推理**分时复用显存**（rollout
  时 vLLM 用、训练时 FSDP 用，free_cache_engine 在训练前释放 vLLM 的
  KV cache）；每步训练完 update_weights 把新权重从 FSDP shard 同步进 vLLM。
- TransferQueue（Ascend/TransferQueue 框架）：agent loop 与 vLLM 之间
  的调度层——多轮 generate 请求的排队、分派、回传。它解决"几千个
  小请求怎么高效调度给推理引擎"的问题。⚠️ 框架设计动机以官方文档
  为准，此处以本项目观察到的调度语义描述。

**结合项目**：在 veRL main 0.10 上实现 SearchR1AgentLoop（TITO +
response_mask），对齐 0.10 API（reward.* 键路径、pkg:// 导入、DictConfigWrap），
用 probe 47 项检查守住 API 漂移，并对 TQ 调度瓶颈做了定量分析（51K 操作/步、
736/min，详见 D.5）。

**加分点/追问预案**：
- 加分：DataProto（batch 张量 + non_tensor_batch 对象）是 veRL 的
  数据容器核心；AgentLoopBase 接口契约（__init__ 签名 + run 返回
  AgentLoopOutput）。
- 追问"为什么多轮不写个简单 for 循环"：多轮请求需要统一排队、并发
  调度与异常隔离，TQ 提供的是生产级调度语义——本项目也证明这层
  调度的开销必须被量化。

**一句话总结**：框架迁移最有价值的产出不是"跑起来"，是读懂它的
调度架构并定位瓶颈。

### D.5 TQ 瓶颈怎么定位的？如果让你优化？

**核心论点**：分解 + 测量 + 交叉验证——三段式定位法。

**定位过程**：
1. 分解：每步 = rollout（TQ 调度多轮生成 + 搜索）+ old log prob + 奖励
   + 训练 + 权重同步五段；
2. 测量：每个 agent loop 产生 TQ 子操作 ≈400 个，128 个 loop/步 → ~51K
   子操作/步；TQ 内部统计 SimpleStorageUnit PUT_DATA 吞吐 ~736/min →
   纯调度时间 51K/736 ≈ 70 分钟，与实测步时 50-70 分钟吻合；
3. 交叉验证（关键）：搜索后端从知乎（0.68s）换 DeepSeek（1-3s）再
   换 MiMo（6-14s），步时几乎不变——瓶颈在 TQ 调度，不在搜索 API。

**优化方案（按性价比排序）**：
1. 批量提交调度：PUT_DATA 是每请求一次的小批量操作，把多轮 generate 的
   "逐条排队"改"批量提交"，省掉大部分调度开销；
2. 减少子操作数：合并 loop 内部状态搬运（metrics/extra 字段的多次 PUT），
   或降低组内并发（n=4→2，牺牲多样性换吞吐）；
3. 搜索 I/O 异步化：搜索不占 GPU，把它移出 TQ 关键路径（不阻塞 generate 队列）；
4. 调度范式级改动：多轮循环移进推理引擎内部——框架级改动，最后手段。

**加分点/追问预案**：
- 加分：调度吞吐这类数据"部署前 10 分钟就能 benchmark"——先测再跑
  是成本最低的习惯。
- 追问"为什么不直接换框架"：先量化再决策——迁移成本 vs 剩余预算，
  定量分析（51K/736）就是决策依据。

**一句话总结**：定位瓶颈的黄金标准是交叉验证——换掉嫌疑组件看指标
动不动。

### D.6 vLLM 的 max_num_seqs 和 Mamba cache block 什么关系？

**核心论点**：并发序列上限 = 最受限组件的容量——纯 Transformer 受限
于 KV cache 页数，混合架构还受限于 Mamba 状态块数。

**展开**：
- max_num_seqs 是 vLLM 的并发序列数上限：每个并发序列要预留一份
  状态缓存。纯 Transformer 的状态是 KV cache（按 token 分配，
  PagedAttention 分页管理，容量弹性）；
- Qwen3.5 是 Transformer + Mamba 混合架构：Mamba 层有固定大小的
  recurrent state（cache block），**按序列数预分配、不可分页**——
  并发上限 = Mamba cache block 总数。
- 报错 `max_num_seqs (256) exceeds available Mamba cache blocks (229)` 直接
  把上限数字告诉你：设 229 即通过。

**结合项目**：max_num_seqs=229 已写入 e6b_run.sh；229 的并发对 GRPO 组内
采样够用（batch 32）。

**加分点/追问预案**：
- 追问"为什么 Mamba 不可分页"：Mamba 的状态是定长状态空间卷积的
  recurrent state，不是逐 token 增长的 KV 序列，按序列预分配是实现选择。
- 加分：混合架构的通用解法——先读报错（数字都在里面），再查该组件的
  分配模型。

**一句话总结**：混合架构模型的约束由最受限的组件决定，先读报错
再调参。

### D.7 PagedAttention 和 continuous batching 原理？

**核心论点**：PagedAttention 管显存（空间），continuous batching 管
吞吐（时间）——两者都是"把 GPU 当共享资源池管理"的思想。

**展开**：
- PagedAttention：KV cache 按固定大小 block（如 16 token）分页存储，逻辑
  连续、物理离散，用块表（block table）索引。解决的问题：不同序列长度
  不一，连续预分配 KV cache 会产生大量内外部碎片；分页后按需申请 block，
  碎片率大幅下降——类比 OS 虚拟内存分页。
- continuous batching：传统 static batching 要等整批全部完成才能换入新请求
  （短序列空等长序列）；continuous batching 在每步 decode 后检查序列是否
  结束，结束即让出槽位给新请求——GPU 不空转，吞吐与延迟双赢。
- 与 RL 的关系：GRPO 组内采样（n=4）+ 多轮 agent 轨迹天然长短不一，
  两者缺一不可。

**加分点/追问预案**：
- 追问"和 prefix caching 什么关系"：prefix caching 复用相同前缀的
  KV block，是与分页正交的又一层优化。
- 加分：本项目 free_cache_engine=true——训练阶段把 vLLM 的 KV cache
  整池释放给 FSDP，是推训同卡方案里"池化"的延伸。

**一句话总结**：一个把显存切块管，一个把时间切片管——本质都是
资源池化。

### D.8 为什么训练前要跑 47 项检查？

**核心论点**：成本不对称——probe 失败的成本是 0，训练失败的成本是 GPU
小时。服务器 ¥4.36/h，一次"跑 10 分钟才发现 import 错"就是 ¥0.7+，
而这类错误本地 1 分钟就能查出。

**展开**（7 节，按"必然炸 → 概率炸"排序）：
1. agent-loop API 签名/注册表——API 漂移第一道防线（veRL main 天天变）；
2. worker 式实例化 + chat template 冒烟——模拟 Ray worker 环境，不止
   driver 进程（A.3 的教训）；
3. hydra 全量 compose + key-drift 扫描 + wiring assert——compose 对
   不存在的键 hard-fail，带完整 overrides 的 compose 本身就是检查；
4. reward 函数 import + 签名 + 假 DataProto 冒烟；
5. 库版本 + GPU + transfer_queue/flashinfer 存在性；
6. 数据文件 + 列名 + 启动脚本陈旧参数扫描；
7. --live-search 实弹搜索——API key 真的有效（花 GPU 钱之前验证）。

**结合项目**：probe 抓出过 reward_fn 导入路径、API 漂移 3 处、配置键移动、
数据列缺失；0 FAIL 才启动是铁律。

**加分点/追问预案**：
- 追问"probe 挡不住什么"：Ray/AgentLoop 运行时问题要第 1-2 步才暴露——
  项目文档预估首次跑通概率 60-70%，把不确定性写进计划比假装 probe 万能
  更专业。

**一句话总结**：probe 的本质是把"运行时才发现"的问题前移到"花第一分钱
之前"。

### D.9 推训同卡 vs 推训分离怎么选？

**核心论点**：同卡省钱（分时复用显存）、分离省心（互不干扰、独立
扩缩容）——选型判据是模型大小、预算与步时敏感度。

**展开**：
- 同卡（HybridEngine）：同一批 GPU 上 vLLM rollout 与 FSDP 训练分时复用
  ——rollout 时 vLLM 用、训练时 FSDP 用，free_cache_engine 释放 KV cache。
  显存账：训练侧 offload 后 ≈5.7GB + vLLM 池 13.2GB。
- 分离：推理引擎独立一组卡、训练引擎一组卡——没有显存争抢与切换开销，
  但推理卡在训练阶段全部空闲，RL 场景下利用率低。
- 判据：RL 训练与 rollout 天然交替，推理引擎在训练时本就空闲 → 小模型
  （4B）预算有限选同卡；大模型（70B）显存挤不下、或步时极度敏感选分离。

**结合项目**：2×4090 + 4B 选同卡；同卡方案的典型代价是 update_weights
阶段显存峰值叠加（D.3 的 OOM 战场正是同卡方案的调度点）。

**加分点/追问预案**：
- 追问"推理服务为什么通常分离"：服务场景流量持续、SLO 要求低延迟，与训练争抢显存不可接受——场景决定架构。
- 加分：同卡方案的显存账要按"峰值阶段"算——update_weights 与
  rollout 的叠加点才是真正的峰值。

**一句话总结**：同卡 = 分时复用换成本，显存调度是代价；模型越大
越该分离。

### D.10 显存碎片化 OOM 和真实 OOM 怎么区分？

**核心论点**：看 reserved 与 allocated 两个数——allocated 逼近上限是
真实 OOM，allocated 不高但找不到连续块是碎片化 OOM。

**展开**：
- 机制：PyTorch caching allocator 向 CUDA 申请大块（reserved），再切
  给张量（allocated）；张量释放后内存回到缓存池，但小块边界不能
  随意合并 → 碎片化：allocated 很低，却拼不出一个新分配的连续需求。
- 判据：报错时刻 allocated << 显存总量 且 reserved 高 → 碎片化；
  allocated 本身逼近上限 → 真实 OOM。
- 项目实例：update_weights 缺口 400MB——峰值 allocated 20.67GB，申请
  1.19GB 失败、free 795MB 后仍失败：释放的小块拼不出 1.19GB 的连续
  需求，是碎片/保留块问题而非真实容量不足。
- 修复三板斧：expandable_segments:True（可扩展段，减少块边界碎片）+
  garbage_collection_threshold:0.05（强制 GC 回收未分配缓存块）+ 减少大
  张量物化（本项目最终靠 get_per_tensor_param_shard + offload_to_cpu 流式
  传输，峰值 12.6GB → ~5GB）。⚠️ 本项目实测 GC threshold 收益有限
  （~3%），最终修复靠流式——env 变量是防御，不是解药。

**加分点/追问预案**：
- 追问"怎么观测"：torch.cuda.memory_allocated()/memory_reserved()
  补丁打印观测点——D.3 的六轮定位全在这个观测框架里。

**一句话总结**：先读 allocated/reserved 两个数，再谈修法——碎片化
与真实 OOM 的修法完全不同。

---

## 第 E 章 速查表

### E.1 分布式术语表（一句话版）

| 术语 | 一句话 |
|---|---|
| DDP | 每卡一份完整模型副本，反向 all-reduce 梯度——显存省不了，通信简单 |
| FSDP | 参数/梯度/优化器状态分片到每卡，前向 all-gather 拼全量——用通信换显存 |
| ZeRO-1 | 只分片优化器状态（显存最大头），参数与梯度仍全量 |
| ZeRO-2 | 分片优化器状态 + 梯度 |
| ZeRO-3 | 分片优化器状态 + 梯度 + 参数（= FSDP 的完整形态） |
| TP（张量并行） | 单层内按矩阵切分，每卡算一部分，逐层通信——适合高带宽互联 |
| PP（流水线并行） | 按层切分，每卡持若干层，batch 切微批流过——通信少但有流水气泡 |
| GSPMD | 把各类并行统一成"分片标注"的编译框架，由编译器推导通信 |
| unshard | all-gather 拼出完整权重的前向准备阶段 |
| all-gather | 每卡把自己的分片广播给所有卡，人人拼出全量 |
| all-reduce | 所有卡聚合（求和/均值）后人人拿到同一结果——梯度同步 |
| reduce-scatter | 归约结果按卡切分分发，每卡只拿自己那份——比 all-reduce 省流量 |
| offload | 把 GPU 显存放不下的状态搬到 CPU/磁盘，用时搬回——时间换空间 |
| gradient checkpointing | 前向不存中间激活，反向重算——重算换激活显存 |

### E.2 关键数字表（本项目实测，2026-08-30 快照）

| 数字 | 值 | 口径/出处 |
|---|---|---|
| 4B 模型权重（bf16） | 8.4GB | 显存账（e6b_run.sh 注释） |
| 梯度（bf16） | 8.4GB | 同上 |
| Adam 状态（fp32×2） | 25.2GB | 同上 |
| 无 offload 每卡（N=2） | 26.7GB ❌ | 24GB 卡必爆（vLLM 权重还没算） |
| offload 后每卡（N=2） | 18.9GB ✓（余 4GB） | 训练侧 5.7GB + vLLM 池 13.2GB |
| TQ 调度瓶颈 | 51K 操作/步、736/min | ≈70min 纯调度，与实测 50-70min/步吻合 |
| Mamba cache block | 229 | vLLM 报错数字，max_num_seqs 上限 |
| CUDA IPC 上限 | 2GB−1（INT_MAX） | torch 2.13 reduce_tensor；bucket 2048MB 越界 → 512MB |
| 搜索延迟 | 知乎 0.68s / DeepSeek 1-3s / MiMo 6-14s | 客户端计时（交叉验证证据） |
| 词表 / logits 峰值 | 151936 / [2,8000,vocab]×2B ≈ 4.85GB | 分块 log_softmax 后峰值 300MB |
| 步时 | 本地 ~16min；veRL 实测 50-70min（设计目标 3-5min） | 本地 I/O bound；veRL TQ 瓶颈 |
| 预算 | ¥2.18/卡/h；2 卡 50 步 ¥22-28 | AutoDL 4090（2026-08-30） |
| SFT → GRPO（同管线对比） | dev EM 1.43% → 81.2% | 81.2% 为训练批次内 correct_rate（step 3 峰值），非留出集；仅作同管线前后对比，不作跨论文比较 |

### E.3 常见报错 → 根因 → 修法（12 轮调试提炼）

| 报错/现象 | 根因 | 修法 |
|---|---|---|
| ImportError: cannot import name 'unpack_mixed_mm' | torch 多次安装的新旧文件残留 | 按报错路径删陈旧文件 |
| OSError: Can't load image processor ... preprocessor_config.json | vLLM 0.28 把 Qwen3.5 路由到多模态实现 | 从上游仓库补 preprocessor_config.json |
| RewardLoopWorker FileNotFoundError（driver 正常） | Ray worker 不继承 shell 的 PYTHONPATH | site-packages 写 .pth |
| from_pretrained 卡死数十分钟 | DNS 只解析 IPv6 且无路由 | HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1 |
| 第一步 forward OOM（Tried to allocate 5.35 GiB） | 大词表（151936）logits 峰值 4.85GB | 分块 log_softmax（512 token/块）+ 截断 + 限 token 数 |
| update_weights OOM（峰值 20.67GB、缺口 400MB） | 权重同步在 GPU 上物化全量权重 | 分片取参 + FULL_STATE_DICT + offload_to_cpu 流式 |
| max_num_seqs (256) exceeds available Mamba cache blocks (229) | Mamba 状态块按序列预分配、不可分页 | max_num_seqs=229 |
| rebuild_ipc list_args[6] 越界 | 2048MB bucket 超 CUDA IPC 上限（2GB−1）静默走 shm 路径 | update_weights_bucket_megabytes=512 |
| TQ ACK 超时日志 12 次、pending 卡死 | 启动期 thundering herd | TQ_DATA_UPDATE_RESPONSE_TIMEOUT=120 + 重试补丁 |
| ref FSDP 后常驻 8.46GB | veRL 0.10 ref CPUOffload 在 torch 2.13 失效 | use_kl_loss=false（GRPO 无 KL 配方，关 ref） |
| reward 里 prm 恒 0（静默） | 顶层 override 键被 veRL 0.10 静默忽略 | 对齐 reward.custom_reward_function.reward_kwargs.* |
| load_extern_object 把 module 当文件路径 | 缺 pkg:// 前缀 | path=pkg://verl_search_r1.reward_fn |
| chat template 生成垃圾 token（无异常） | tokenize=True 返回非 dict 子类对象 | tokenize=False + 手动 encode |
| 预生成缓存命中率 0% | 缓存 key（问题原文）≠ 运行时 query（模型改写） | 改运行时缓存 |
| 每步 50-70min | TQ 调度 51K 操作/步、736/min | 批量提交/减子操作/搜索异步化（D.5） |

---

## 结语

工程能力的可迁移内核，本文想传递的只有三件事：

1. **每个坑都要挖到"机制层"**——文件系统语义、进程模型、类型系统、调度
   架构；现象会过时，机制不过时。
2. **成本意识是工程决策的第一约束**——probe 失败成本 0、训练失败成本
   GPU 小时；止损规则写在跑起来之前。
3. **结论用实验买**——"搜索太慢"的直觉被"换后端步时不变"推翻，直觉给
   方向，实验给结论。
