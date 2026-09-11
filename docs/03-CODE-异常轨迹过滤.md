# 代码带读：异常轨迹过滤（anomalous trajectory filtering）

> 代码位置：`C:\new\intern\plan\projects\search-r1\train_grpo_local.py`
> 相关概念：组内 advantage（GRPO）、loss_mask、数据质量层
> 配套问答：07-INTERVIEW_QA.md Q11；教程 02-TUTORIAL.md §2.8.5

---

## 0. 一分钟总览：过滤发生在训练循环的哪个环节

```
一个 step 的流程（train_grpo_local.py main()）：

  take_batch 取 8 题
    → rollout_batch_local 每题采 8 条轨迹（共 64 条）
    → ① is_anomalous 逐条打标记                    ← 本带读的核心
    → ② compute_group_advantages 组内优势重算
    → ③ （可选）PRM-Lite / LATA 叠加
    → ④ build_training_sequences 构建训练序列       ← 异常轨迹在这里被跳过
    → ⑤ grpo_loss 计算损失
```

异常过滤是 ①+②+④ 三处代码合起来的效果，缺一处都不完整。

---

## 1. 检测：`is_anomalous`（第 372-379 行）

```python
def is_anomalous(traj: Trajectory) -> bool:
    if not traj.final_text or not traj.final_text.strip():
        return True
    if traj.search_calls >= MAX_SEARCH_CALLS and not traj.valid_format:
        return True
    if traj.search_calls == 0 and not traj.valid_format:
        return True
    return False
```

**逐行解读**：

```python
if not traj.final_text or not traj.final_text.strip():
    return True
```
第 1 条规则：最终输出为 None 或纯空白。
`traj.final_text` 是轨迹最后一轮 assistant 输出的文本；None/空白意味着
模型"什么都没说出来"（生成中断、截断、崩坏）——没有任何可学习的内容。

```python
if traj.search_calls >= MAX_SEARCH_CALLS and not traj.valid_format:
    return True
```
第 2 条规则：搜索额度用尽（MAX_SEARCH_CALLS=4）但仍没有合法 Answer 行。
`search_calls` 是这条轨迹实际调搜索 API 的次数；搜满 4 次还没答出来 =
"搜了又搜"的死循环模式，环境交互失控，轨迹报废。

```python
if traj.search_calls == 0 and not traj.valid_format:
    return True
```
第 3 条规则：一次都没搜、且格式错。
没搜且没答案 = 搜索环节根本没触发（协议断裂 / API 挂 / 模型卡死），
整条轨迹对"训练搜索行为"这个目标零价值。

```python
return False
```
其余情况都是"正常"轨迹——**包括"搜过但格式坏"的轨迹**（它搜了、
有输出，只是 Answer 行写坏了）。这类是策略行为，不是系统故障，
必须留在组里当反面教材（奖励 −0.1 会拉低组均值，教模型"守格式"）。

**设计原则一句话**：异常 = 系统故障（模型控制不了的）；正常 = 策略行为
（模型能控制、该学的）。判定看"行为有没有发生"，不看"结果好不好"。

---

## 2. 训练循环里的调用点（第 754-761 行）

```python
# Detect anomalous
for t in trajectories:
    t.anomalous = is_anomalous(t)
    if t.anomalous:
        t.advantage = 0.0        # ← 兜底置零（第 2 步会再确认一次）

# Compute group advantages
compute_group_advantages(trajectories, args.group_size)
```

**解读**：
- 标记完成后**立刻把 advantage 置 0**——防止后面任何代码路径漏处理；
- 紧接着调用组优势计算（第 3 节），真正的"排除重算"在那里发生。

---

## 3. 组统计量重算：`compute_group_advantages`（第 386-402 行）

```python
def compute_group_advantages(trajectories: list[Trajectory], group_size: int) -> None:
    n_questions = len(trajectories) // group_size
    for qi in range(n_questions):
        group = trajectories[qi * group_size:(qi + 1) * group_size]
        rewards = [t.reward for t in group if not t.anomalous]   # ★ 关键行
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
```

**逐行解读**：

```python
group = trajectories[qi * group_size:(qi + 1) * group_size]
```
64 条轨迹按题目分组：第 qi 题的 8 条轨迹（rollout 时按题连续存放）。

```python
rewards = [t.reward for t in group if not t.anomalous]   # ★ 关键行
```
★ **排除异常后**再收集奖励——组均值/标准差只由"正常轨迹"决定。
这就是"基线干净"的实现：组均值代表"正常试一次这道题的水平"，
不包含系统故障样本。

```python
if not rewards:
    for t in group:
        t.advantage = 0.0
    continue
```
边界：全组 8 条都是异常 → 组内没有任何正常样本可参照 →
所有优势置 0 → 这组不产生任何梯度（宁可不学也不乱学）。
实测：难题多的批次曾出现 anomalous 50/64 条，这个分支真实触发过。

```python
mean_r = np.mean(rewards)
std_r = np.std(rewards) + 1e-8
```
正常轨迹的均值与标准差；+1e-8 防全组奖励相同时除零。

```python
for t in group:
    if t.anomalous:
        t.advantage = 0.0
    else:
        t.advantage = (t.reward - mean_r) / std_r
```
逐条赋值：异常轨迹 0（不参与比较、不产生信号）；
正常轨迹按 GRPO 公式 (r − μ)/σ 归一化。

**手算对比（为什么"排除重算"重要）**：

```
组内 8 条：4 对(1.0)、3 错(0.0)、1 异常(−0.1)

不排除（错误做法）: mean = 0.4875 → 答对轨迹 A ≈ +1.05   ← 优势虚高 30%
排除后（正确做法）: mean = 0.5714 → 答对轨迹 A ≈ +0.80   ← 诚实
```

---

## 4. 训练序列构建的跳过（第 418-420 行）

```python
for traj in trajectories:
    if traj.anomalous or traj.advantage == 0.0:
        continue
```

**解读**：两个条件任一满足就跳过——
- `traj.anomalous`：异常轨迹（无论如何都不训）；
- `traj.advantage == 0.0`：优势为零的轨迹也不训（零优势 × 任何
  logprob = 零损失，训了也是白算，跳过省显存省时间）。

被跳过的轨迹不会进入 `all_ids` / `loss_mask` / `ref_logprobs` 的拼接，
自然产生**零梯度**——"不参与训练"不是把 mask 置 0，而是**根本不在
训练序列里**（更干净）。

---

## 5. 一次 step 的完整执行流程（64 条轨迹走一遍）

```
① rollout 结束：64 条轨迹，各有 final_text / search_calls / valid_format / reward
② is_anomalous 逐条标记：假设 3 条命中（2 条空白、1 条没搜且格式错）
③ 3 条异常 → advantage 置 0
④ 分组重算：第 2 题的 8 条里有 1 条异常 → 用 7 条算 mean/std
   第 6 题的 8 条里有 2 条异常 → 用 6 条算 mean/std
   其余 6 组全正常 → 用 8 条算
⑤ 正常轨迹拿到 (r−μ)/σ；异常轨迹保持 0
⑥ build_training_sequences：61 条进训练序列，3 条被跳过
⑦ grpo_loss 只对这 61 条算梯度
```

---

## 6. 三个边界情况

| 边界 | 行为 | 理由 |
|---|---|---|
| 全组异常（rewards 为空） | 全组 advantage=0，跳过 | 无参照样本，宁可不学 |
| 全组奖励相同 | std+1e-8 兜底，advantage 全 0 | 分不出好坏就不学 |
| 非异常但 advantage=0 | 也跳过训练序列 | 零优势训了是白算 |

---

## 7. 验证数据与监控

- **机制验证**：10 步验证，160 条轨迹检出 4 条异常（2.5%）——不是全杀
  也不是漏杀，机制行为正确（2026-08-06 记录）；
- **训练日志**：每步打印 `anomalous=3`（本步异常条数）；
- **PyTRIO 版**：`rollout/anomalous_trajectories` + `rollout/anomaly_rate`
  两个指标进 SwanLab，可画趋势图。

---

## 8. 面试追问清单（4 问）

1. **为什么异常轨迹的 advantage 是 0 而不是 −1？** → 0 = 不参与学习
   （中性）；−1 会变成"压死这个输出模式"的强信号，而异常是系统故障，
   不该被当成策略错误来惩罚。
2. **为什么还要从组统计量里排除，直接置 0 不就行了？** → 不排除的话
   异常轨迹的 −0.1 会把组均值拉低、方差撑大，正常轨迹的优势被虚高
   （手算见第 3 节）。
3. **全组异常怎么办？** → 全组置 0、跳过本组，等下一步重新采样
   （每步都重新 rollout，环境恢复后自然好转）。
4. **veRL 管线为什么没有这套过滤？** → 本地管线的搜索是同步串行的，
   环境抖动（API 限流/截断）更频繁；veRL 侧由 reward 的 −0.1 档 +
   response_mask 机制承担类似职责，且调试阶段 probe 已经把环境问题
   挡在训练外。

---

## 9. 代码位置索引

| 代码 | 位置（train_grpo_local.py） |
|---|---|
| `is_anomalous` 三条规则 | 372-379 行 |
| 训练循环打标记 + 置零 | 754-761 行 |
| `compute_group_advantages` 排除重算 | 386-402 行 |
| `build_training_sequences` 跳过 | 418-420 行 |
| anomalous 计数打印 | ~814 行 |
