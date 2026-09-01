"""LLDS (Lazy Likelihood-Displacement Stabilization) 正则化模块。

LLDS 是 ICML 2026 (Deng et al.) 提出的轻量级似然保持正则化方法，用于防止
GRPO 训练中的 LLD（懒惰似然位移）问题。

核心思想：在 GRPO loss 上叠加惩罚项，仅当当前策略的似然低于参考策略时才激活，
且只惩罚导致似然下降的 token。

    L_llds = λ * Σ max(log π_ref(token) - log π_θ(token), 0)

其中 max(·, 0) 实现了单向惩罚 —— 似然上升不惩罚，下降才惩罚。

变体：
    LLDS-R : Response-level gate — 轨迹总似然下降时，对所有 token 施加惩罚
    LLDS-A : Action-level gate (推荐) — 只对参与训练（advantage != 0）的 token 施加惩罚
    LLDS-MA: Mask Answer — 只对搜索/工具调用 token 施加惩罚，保护搜索行为不被压缩

与 PyTRIO 集成思路：
    1. 在 rollout 后调用 store_reference() 保存参考 logprobs
    2. 在 forward_backward 中或之后获取当前模型 logprobs
    3. 调用 compute_regularization() 计算惩罚项
    4. 将惩罚项加入 GRPO loss

参考：
    Deng et al., "On GRPO Collapse in Search-R1: The Lazy Likelihood-Displacement
    Death Spiral", arXiv 2512.04220, ICML 2026.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from protocol import parse_assistant


@dataclass
class LLDSConfig:
    """LLDS 正则化配置。

    Attributes:
        lambda_reg: 正则化强度，论文推荐 0.05-0.1。
        variant: LLDS 变体，可选 "R"（Response-level）、"A"（Action-level）、
                 "MA"（Mask Answer）。
        mask_answer: 是否额外排除 answer token。当 variant="MA" 时自动为 True。
                    设置为 True 时，即使 variant 不是 "MA"，也会排除 answer token。
    """

    lambda_reg: float = 0.05
    variant: str = "A"
    mask_answer: bool = False

    def __post_init__(self) -> None:
        """校验变体参数并在 variant="MA" 时自动启用 mask_answer。"""
        valid_variants = frozenset({"R", "A", "MA"})
        if self.variant not in valid_variants:
            raise ValueError(
                f"无效的 LLDS 变体 '{self.variant}'，可选: {sorted(valid_variants)}"
            )
        if self.variant == "MA":
            self.mask_answer = True
        if self.lambda_reg < 0:
            raise ValueError(f"lambda_reg 必须非负，当前值: {self.lambda_reg}")


@dataclass
class _TokenRecord:
    """单 token 的参考数据。

    按照与 build_datum() 相同的方式对齐 —— 即 full_tokens 顺序，
    其中 observation token 的 ref_logprob 为 0.0 且 is_completion_token 为 False。

    Attributes:
        ref_logprob: 参考策略在该 token 位置的对数概率（来自 rollout 采样）。
        token_id: 该位置的目标 token ID。
        is_completion_token: 是否为 assistant 生成的 token（非 observation）。
        is_tool_token: 对于 completion token，该 token 属于 tool call 还是 answer。
                       observation token 此字段无意义，固定为 False。
    """

    ref_logprob: float
    token_id: int
    is_completion_token: bool = False
    is_tool_token: bool = False


@dataclass
class _TrajectoryRef:
    """一条轨迹的完整参考数据。

    Attributes:
        ref_logprobs: 按 full_tokens 顺序对齐的参考 logprobs。
        advantages: 按 full_tokens 顺序对齐的 advantage 值（同 build_datum）。
        completion_mask: 按 full_tokens 顺序对齐的完成 token 掩码。
        tool_mask: 按 full_tokens 顺序对齐的工具 token 掩码。
        trajectory_advantage: 该轨迹的整体 advantage（用于 LLDS-R gate 决策）。
        total_ref_logprob: 所有 completion token 的参考 logprob 之和（用于 LLDS-R gate）。
    """

    ref_logprobs: list[float]
    advantages: list[float]
    completion_mask: list[bool]
    tool_mask: list[bool]
    trajectory_advantage: float
    total_ref_logprob: float


class LLDSTracker:
    """LLDS 正则化跟踪器。

    负责：
    1. 保存 rollout 阶段采样的参考 logprobs（来自旧策略）。
    2. 判断是否需要对轨迹施加正则化（gate 逻辑）。
    3. 计算 LLDS 惩罚项。

    典型用法::

        config = LLDSConfig(lambda_reg=0.05, variant="A")
        tracker = LLDSTracker(config)

        # 1. rollout 后保存参考数据
        for idx, trajectory in enumerate(trajectories):
            tracker.store_reference(trajectory, trajectory_id=idx)

        # 2. 训练时计算正则化惩罚
        # current_logprobs 需要与参考数据按同一 token 顺序对齐
        penalty = tracker.compute_regularization(
            trajectory_id=idx,
            current_logprobs=[...],  # 当前模型在每个 token 位置的 logprob
        )
        total_loss = grpo_loss + config.lambda_reg * penalty
    """

    def __init__(self, config: LLDSConfig | None = None) -> None:
        """初始化 LLDS 跟踪器。

        Args:
            config: LLDS 配置。如果为 None，使用默认配置（variant="A", lambda_reg=0.05）。
        """
        self.config = config or LLDSConfig()
        self._trajectories: dict[int, _TrajectoryRef] = {}

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def store_reference(self, trajectory: Any, trajectory_id: int) -> int:
        """从 rollout 轨迹中保存参考 logprobs 和相关元数据。

        解析每条轨迹的 assistant turn，按 build_datum() 的 token 对齐方式
        存储参考 logprobs、advantage 掩码和 token 分类（tool/answer）。

        Args:
            trajectory: rollout 产生的 Trajectory 对象。需要包含:
                        - turns: list[AssistantTurn]，每个 turn 有 completion_tokens、
                          logprobs、text
                        - advantage: float，该轨迹的 group-relative advantage
            trajectory_id: 轨迹的唯一标识符，后续 compute_regularization 使用。

        Returns:
            存储的 completion token 数量。

        Raises:
            ValueError: 如果 trajectory 没有 assistant turn 或数据不一致。
        """
        if not trajectory.turns:
            raise ValueError(
                f"轨迹 {trajectory_id} 没有 assistant turn，无法存储参考 logprobs"
            )

        # 按 build_datum() 相同的方式遍历，对齐 reference logprobs 与 full_tokens
        full_ref_logprobs: list[float] = []
        full_advantages: list[float] = []
        full_completion_mask: list[bool] = []
        full_tool_mask: list[bool] = []
        total_ref_logprob = 0.0
        completion_count = 0

        for turn_index, turn in enumerate(trajectory.turns):
            num_completion = len(turn.completion_tokens)
            if num_completion == 0:
                # 空 assistant turn：跳过，无需存储
                continue
            if num_completion != len(turn.logprobs):
                raise ValueError(
                    f"轨迹 {trajectory_id} 第 {turn_index + 1} 个 turn 的 "
                    f"completion token ({num_completion}) 与 logprob "
                    f"({len(turn.logprobs)}) 长度不一致"
                )

            # 解析 turn 类型：tool call 还是 answer
            parsed = parse_assistant(turn.text)
            is_tool = parsed.kind == "tool"

            # 模拟 observation token 的占位（与 build_datum 对齐）
            # 首轮之前是 prompt tokens（非 completion），后续轮次之前是 observation tokens
            # 这里只关心 completion token 的对齐，observation 区域用 0 填充
            if turn_index == 0:
                # 首轮：prompt_tokens 中的 observation 部分
                # 在 build_datum 中，首轮的 delta_observation 就是整个 prompt_tokens
                obs_len = len(turn.prompt_tokens)
            else:
                # 后续轮次：prompt_tokens 中超出已有 tokens 的部分是 observation
                # 简化处理：按 delta = len(prompt_tokens) - sum(prev lengths) 计算
                # 由于我们无法在此处知道精确的 delta（需要累计 full_tokens），
                # 使用 0 作为 observation 占位，只在后续对齐
                obs_len = 0  # observation tokens 由调用方负责对齐

            # observation 区域不参与 LLDS 计算
            full_ref_logprobs.extend([0.0] * obs_len)
            full_advantages.extend([0.0] * obs_len)
            full_completion_mask.extend([False] * obs_len)
            full_tool_mask.extend([False] * obs_len)

            # completion tokens 存储参考 logprobs
            full_ref_logprobs.extend(turn.logprobs)
            full_advantages.extend([trajectory.advantage] * num_completion)
            full_completion_mask.extend([True] * num_completion)
            full_tool_mask.extend([is_tool] * num_completion)

            total_ref_logprob += sum(turn.logprobs)
            completion_count += num_completion

        # 对完整序列统一右移一位（与 build_datum 的右移对齐）：
        # 输入 tokens = full_tokens[:-1], 目标 tokens = full_tokens[1:]
        # 对应地，logprobs/advantages 也右移一位
        if len(full_ref_logprobs) > 1:
            full_ref_logprobs = full_ref_logprobs[1:]
            full_advantages = full_advantages[1:]
            full_completion_mask = full_completion_mask[1:]
            full_tool_mask = full_tool_mask[1:]

        if completion_count == 0:
            raise ValueError(
                f"轨迹 {trajectory_id} 没有 completion token，无法存储参考 logprobs"
            )

        self._trajectories[trajectory_id] = _TrajectoryRef(
            ref_logprobs=full_ref_logprobs,
            advantages=full_advantages,
            completion_mask=full_completion_mask,
            tool_mask=full_tool_mask,
            trajectory_advantage=trajectory.advantage,
            total_ref_logprob=total_ref_logprob,
        )

        return completion_count

    def should_activate(
        self,
        trajectory_id: int,
        current_logprobs: list[float],
    ) -> bool:
        """判断轨迹级别的 LLDS gate 是否应该激活。

        仅对 LLDS-R 变体有意义：比较轨迹总似然是否下降。
        对于 LLDS-A 和 LLDS-MA，gate 是 per-token 的（由 max(0, ref-θ) 实现），
        此方法始终返回 True。

        Args:
            trajectory_id: 轨迹标识符。
            current_logprobs: 当前模型在对应位置的 logprobs，
                              长度必须与参考数据一致。

        Returns:
            对于 LLDS-R：当前似然低于参考似然时返回 True。
            对于其他变体：始终返回 True（per-token gate 在 compute_regularization 中处理）。
        """
        ref = self._trajectories[trajectory_id]

        if self.config.variant == "R":
            # Response-level gate: 只在总似然下降时激活
            total_current = sum(
                cur_lp
                for cur_lp, is_comp in zip(current_logprobs, ref.completion_mask, strict=True)
                if is_comp
            )
            return total_current < ref.total_ref_logprob

        # LLDS-A 和 LLDS-MA：per-token gate，此处总是"激活"
        return True

    def compute_regularization(
        self,
        trajectory_id: int,
        current_logprobs: list[float],
    ) -> float:
        """计算 LLDS 正则化惩罚项。

        公式：
            penalty = Σ max(log π_ref(t) - log π_θ(t), 0)  [仅对适用 token]

        Args:
            trajectory_id: 轨迹标识符，必须已通过 store_reference 注册。
            current_logprobs: 当前模型在对应位置的 logprobs。
                              长度必须与参考数据一致（由 store_reference 保证）。

        Returns:
            标量惩罚值（正数，float）。

        Raises:
            KeyError: 如果 trajectory_id 未注册。
            ValueError: 如果 current_logprobs 长度与参考数据不匹配。
        """
        ref = self._trajectories[trajectory_id]

        if len(current_logprobs) != len(ref.ref_logprobs):
            raise ValueError(
                f"轨迹 {trajectory_id}: current_logprobs 长度 ({len(current_logprobs)}) "
                f"与参考数据长度 ({len(ref.ref_logprobs)}) 不匹配"
            )

        # 对于 LLDS-R，先检查 gate
        if self.config.variant == "R":
            total_current = sum(
                cur_lp
                for cur_lp, is_comp in zip(
                    current_logprobs, ref.completion_mask, strict=True
                )
                if is_comp
            )
            if total_current >= ref.total_ref_logprob:
                # 似然未下降，不施加惩罚
                return 0.0

        penalty = 0.0
        for idx, (cur_lp, ref_lp) in enumerate(
            zip(current_logprobs, ref.ref_logprobs, strict=True)
        ):
            # 基础条件：必须是 completion token（非 observation）
            if not ref.completion_mask[idx]:
                continue

            # 变体筛选
            if self.config.variant == "A":
                # Action-level: 只对 advantage != 0 的 token
                if ref.advantages[idx] == 0.0:
                    continue
            # LLDS-R 不额外筛选，所有 completion token 都参与

            if self.config.mask_answer:
                # Mask Answer: 只对 tool token（非 answer token）
                # 此过滤独立于 variant，可与 variant="A" 组合使用
                if not ref.tool_mask[idx]:
                    continue

            # 单向惩罚：只在当前似然低于参考时才激活
            if cur_lp < ref_lp:
                penalty += ref_lp - cur_lp

        return penalty

    def compute_batch_regularization(
        self,
        trajectory_logprobs: dict[int, list[float]],
    ) -> float:
        """批量计算多条轨迹的 LLDS 正则化惩罚项之和。

        便捷方法，等效于对每条轨迹分别调用 compute_regularization 并求和。

        Args:
            trajectory_logprobs: 轨迹 ID 到当前模型 logprobs 列表的映射。

        Returns:
            所有轨迹的 LLDS 惩罚项之和（正数，float）。
        """
        total = 0.0
        for trajectory_id, current_logprobs in trajectory_logprobs.items():
            total += self.compute_regularization(trajectory_id, current_logprobs)
        return total

    def get_reference_logprobs(self, trajectory_id: int) -> list[float]:
        """获取指定轨迹的参考 logprobs（仅供调试和测试）。

        Args:
            trajectory_id: 轨迹标识符。

        Returns:
            参考 logprobs 列表。
        """
        return list(self._trajectories[trajectory_id].ref_logprobs)

    def get_effective_token_count(self, trajectory_id: int) -> int:
        """获取指定轨迹中实际参与 LLDS 计算的 token 数量。

        根据当前变体配置计算有效的 token 数量（用于统计和调试）。

        Args:
            trajectory_id: 轨迹标识符。

        Returns:
            参与 LLDS 计算的 token 数量。
        """
        ref = self._trajectories[trajectory_id]
        count = 0
        for idx in range(len(ref.ref_logprobs)):
            if not ref.completion_mask[idx]:
                continue
            if self.config.variant == "A" and ref.advantages[idx] == 0.0:
                continue
            if self.config.mask_answer and not ref.tool_mask[idx]:
                continue
            count += 1
        return count

    @property
    def trajectory_count(self) -> int:
        """已存储参考数据的轨迹数量。"""
        return len(self._trajectories)

    def clear(self) -> None:
        """清除所有已存储的参考数据（用于开始新的训练 step）。"""
        self._trajectories.clear()


# ---------------------------------------------------------------------------
# 辅助函数：与 PyTRIO 集成时，从 model output 提取 per-token logprobs
# ---------------------------------------------------------------------------


def extract_completion_logprobs(
    current_logprobs_full: list[float],
    completion_mask: list[bool],
) -> list[float]:
    """从完整序列 logprobs 中提取 completion token 对应的部分。

    在 PyTRIO 集成中，模型为整个 input 序列输出 logprobs，但 LLDS 只需
    completion token 部分。此函数根据 completion mask 提取并返回对应的 logprobs。

    Args:
        current_logprobs_full: 模型为整个序列输出的 logprobs（长度 = 序列长度 - 1，
                               因为输入 tokens 产生对 target tokens 的预测）。
        completion_mask: 与 logprobs 同长度的布尔列表，True 表示 completion token。

    Returns:
        仅包含 completion token 的 logprobs 子列表。

    Raises:
        ValueError: 如果长度不匹配。
    """
    if len(current_logprobs_full) != len(completion_mask):
        raise ValueError(
            f"current_logprobs_full 长度 ({len(current_logprobs_full)}) "
            f"与 completion_mask 长度 ({len(completion_mask)}) 不匹配"
        )
    return [
        lp
        for lp, is_comp in zip(current_logprobs_full, completion_mask, strict=True)
        if is_comp
    ]


# ---------------------------------------------------------------------------
# 自定义 PyTRIO loss_fn 参考实现
# ---------------------------------------------------------------------------


def llds_loss_fn(
    *,
    llds_tracker: LLDSTracker,
    trajectory_id: int,
    base_loss: float,
    current_logprobs: list[float],
    lambda_scale: float | None = None,
) -> float:
    """在 GRPO loss 上叠加 LLDS 正则化的辅助函数。

    用于 PyTRIO 自定义 loss_fn 中，或作为 forward_backward 的 post-processing。

    Args:
        llds_tracker: 已通过 store_reference 填充的 LLDS 跟踪器。
        trajectory_id: 当前轨迹的标识符。
        base_loss: GRPO 基础 loss（importance_sampling 输出）。
        current_logprobs: 当前模型输出的 per-token logprobs，
                          必须与 store_reference 时对齐。
        lambda_scale: 如果为 None，使用 LLDSConfig.lambda_reg。
                      如果提供，覆盖配置中的 lambda_reg（用于动态调度）。

    Returns:
        base_loss + λ * L_llds 的总 loss。

    Raises:
        KeyError: 如果 trajectory_id 未在 llds_tracker 中注册。
    """
    llds = llds_tracker.compute_regularization(trajectory_id, current_logprobs)
    lam = lambda_scale if lambda_scale is not None else llds_tracker.config.lambda_reg
    return base_loss + lam * llds
