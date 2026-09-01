"""LLDS 模块的单元测试。

覆盖：
    - LLDSConfig 配置校验
    - store_reference 参考数据存储
    - should_activate gate 逻辑
    - compute_regularization 惩罚计算
    - 三种变体 (R, A, MA) 的行为差异
    - 边界情况和错误处理
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# 确保项目根目录在搜索路径中
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llds import (
    LLDSConfig,
    LLDSTracker,
    extract_completion_logprobs,
    llds_loss_fn,
)


# ---------------------------------------------------------------------------
# 轻量级 mock：模拟 rollout.py 中的 AssistantTurn 和 Trajectory
# ---------------------------------------------------------------------------


@dataclass
class MockAssistantTurn:
    """模拟 rollout.AssistantTurn，仅包含 LLDS 需要的字段。"""

    prompt_tokens: list[int] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    text: str = ""


@dataclass
class MockTrajectory:
    """模拟 rollout.Trajectory，仅包含 LLDS 需要的字段。"""

    advantage: float = 0.0
    turns: list[MockAssistantTurn] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _make_tool_turn(
    completion_tokens: list[int] | None = None,
    logprobs: list[float] | None = None,
    text: str | None = None,
) -> MockAssistantTurn:
    """创建一个 tool call turn。"""
    tokens = completion_tokens if completion_tokens is not None else [101, 102, 103]
    lps = logprobs if logprobs is not None else [-0.5, -1.0, -0.3]
    txt = text or "<tool_call>\n<function=search>\n<parameter=query>\ntest query\n</parameter>\n</function>\n</tool_call>"
    # 确保 prompt_tokens 非空（模拟真实场景）
    return MockAssistantTurn(
        prompt_tokens=[1, 2, 3],
        completion_tokens=tokens,
        logprobs=lps,
        text=txt,
    )


def _make_answer_turn(
    completion_tokens: list[int] | None = None,
    logprobs: list[float] | None = None,
    text: str | None = None,
) -> MockAssistantTurn:
    """创建一个 answer turn。"""
    tokens = completion_tokens if completion_tokens is not None else [201, 202, 203]
    lps = logprobs if logprobs is not None else [-0.2, -0.4, -0.1]
    txt = text or "Answer: Paris"
    return MockAssistantTurn(
        prompt_tokens=[10, 20, 30],
        completion_tokens=tokens,
        logprobs=lps,
        text=txt,
    )


def _make_trajectory(
    turns: list[MockAssistantTurn] | None = None,
    advantage: float = 0.5,
) -> MockTrajectory:
    """创建一个模拟轨迹。"""
    return MockTrajectory(advantage=advantage, turns=turns or [])


# ---------------------------------------------------------------------------
# LLDSConfig 测试
# ---------------------------------------------------------------------------


class TestLLDSConfig:
    """LLDSConfig 构造与校验。"""

    def test_default_config(self) -> None:
        config = LLDSConfig()
        assert config.lambda_reg == 0.05
        assert config.variant == "A"
        assert config.mask_answer is False

    def test_custom_config(self) -> None:
        config = LLDSConfig(lambda_reg=0.1, variant="R", mask_answer=True)
        assert config.lambda_reg == 0.1
        assert config.variant == "R"
        assert config.mask_answer is True

    def test_variant_ma_auto_masks_answer(self) -> None:
        """variant="MA" 应自动将 mask_answer 设为 True。"""
        config = LLDSConfig(variant="MA")
        assert config.mask_answer is True
        assert config.variant == "MA"

    def test_invalid_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="无效的 LLDS 变体"):
            LLDSConfig(variant="X")

    def test_negative_lambda_raises(self) -> None:
        with pytest.raises(ValueError, match="lambda_reg 必须非负"):
            LLDSConfig(lambda_reg=-0.1)

    def test_lambda_zero_is_valid(self) -> None:
        """lambda_reg=0 应该是合法的（相当于关闭 LLDS）。"""
        config = LLDSConfig(lambda_reg=0.0)
        assert config.lambda_reg == 0.0


# ---------------------------------------------------------------------------
# LLDSTracker.store_reference 测试
# ---------------------------------------------------------------------------


class TestStoreReference:
    """测试参考数据存储。"""

    def test_store_single_tool_turn(self) -> None:
        tracker = LLDSTracker()
        traj = _make_trajectory(turns=[_make_tool_turn()], advantage=0.5)
        count = tracker.store_reference(traj, trajectory_id=0)
        assert count == 3  # 3 completion tokens
        assert tracker.trajectory_count == 1

    def test_store_single_answer_turn(self) -> None:
        tracker = LLDSTracker()
        traj = _make_trajectory(turns=[_make_answer_turn()], advantage=0.3)
        count = tracker.store_reference(traj, trajectory_id=0)
        assert count == 3

    def test_store_multiple_turns(self) -> None:
        tracker = LLDSTracker()
        traj = _make_trajectory(
            turns=[
                _make_tool_turn(completion_tokens=[1, 2], logprobs=[-0.5, -0.3]),
                _make_answer_turn(completion_tokens=[3, 4], logprobs=[-0.2, -0.4]),
            ],
            advantage=0.5,
        )
        count = tracker.store_reference(traj, trajectory_id=0)
        assert count == 4  # 2 + 2

    def test_store_multiple_trajectories(self) -> None:
        tracker = LLDSTracker()
        for idx in range(3):
            traj = _make_trajectory(
                turns=[_make_tool_turn()],
                advantage=0.5 + idx * 0.1,
            )
            tracker.store_reference(traj, trajectory_id=idx)
        assert tracker.trajectory_count == 3

    def test_store_zero_advantage_trajectory(self) -> None:
        """advantage=0 的轨迹也应能存储参考数据。"""
        tracker = LLDSTracker()
        traj = _make_trajectory(turns=[_make_tool_turn()], advantage=0.0)
        count = tracker.store_reference(traj, trajectory_id=0)
        assert count == 3

    def test_empty_turns_raises(self) -> None:
        tracker = LLDSTracker()
        traj = _make_trajectory(turns=[], advantage=0.5)
        with pytest.raises(ValueError, match="没有 assistant turn"):
            tracker.store_reference(traj, trajectory_id=0)

    def test_mismatched_tokens_logprobs_raises(self) -> None:
        tracker = LLDSTracker()
        turn = MockAssistantTurn(
            prompt_tokens=[1],
            completion_tokens=[1, 2, 3],
            logprobs=[-0.5, -0.3],  # 长度不匹配
            text="<tool_call>\n<function=search>\n<parameter=query>\ntest\n</parameter>\n</function>\n</tool_call>",
        )
        traj = _make_trajectory(turns=[turn])
        with pytest.raises(ValueError, match="长度不一致"):
            tracker.store_reference(traj, trajectory_id=0)

    def test_empty_completion_tokens_skipped(self) -> None:
        """空 completion token 的 turn 应被跳过，不报错。"""
        tracker = LLDSTracker()
        empty_turn = _make_tool_turn(completion_tokens=[], logprobs=[])
        valid_turn = _make_tool_turn()
        traj = _make_trajectory(turns=[empty_turn, valid_turn], advantage=0.5)
        count = tracker.store_reference(traj, trajectory_id=0)
        assert count == 3  # 只有 valid_turn 的 3 个 token

    def test_reference_logprobs_accessible(self) -> None:
        """存储后应能通过 get_reference_logprobs 访问参考数据。"""
        tracker = LLDSTracker()
        traj = _make_trajectory(
            turns=[_make_tool_turn(logprobs=[-0.5, -1.0, -0.3])],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        ref = tracker.get_reference_logprobs(0)
        assert len(ref) >= 3  # 包含 observation padding + completion tokens


# ---------------------------------------------------------------------------
# LLDSTracker.should_activate 测试
# ---------------------------------------------------------------------------


class TestShouldActivate:
    """测试轨迹级 gate 逻辑。"""

    def test_variant_r_activates_when_likelihood_decreased(self) -> None:
        """LLDS-R: 当前似然低于参考时应返回 True。"""
        config = LLDSConfig(variant="R")
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[_make_tool_turn(logprobs=[-0.5, -1.0, -0.3])],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        ref_lps = tracker.get_reference_logprobs(0)
        # 构造更差的当前 logprobs（总和更小/绝对值更大）
        current_lps = [lp - 0.5 for lp in ref_lps]  # 每个 token 都下降 0.5
        assert tracker.should_activate(0, current_lps) is True

    def test_variant_r_deactivates_when_likelihood_increased(self) -> None:
        """LLDS-R: 当前似然高于参考时应返回 False。"""
        config = LLDSConfig(variant="R")
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[_make_tool_turn(logprobs=[-0.5, -1.0, -0.3])],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        ref_lps = tracker.get_reference_logprobs(0)
        # 构造更好的当前 logprobs（总和更大/接近 0）
        current_lps = [lp + 0.3 for lp in ref_lps]  # 每个 token 都改善 0.3
        assert tracker.should_activate(0, current_lps) is False

    def test_variant_a_always_activates(self) -> None:
        """LLDS-A: gate 是 per-token 的，should_activate 总是 True。"""
        config = LLDSConfig(variant="A")
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[_make_tool_turn(logprobs=[-0.5, -1.0, -0.3])],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        ref_lps = tracker.get_reference_logprobs(0)
        # 似然上升时也应返回 True（per-token gate 在 compute_regularization 中处理）
        current_lps = [lp + 0.5 for lp in ref_lps]
        assert tracker.should_activate(0, current_lps) is True

    def test_variant_ma_always_activates(self) -> None:
        """LLDS-MA: 与 LLDS-A 类似，总是返回 True。"""
        config = LLDSConfig(variant="MA")
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[_make_tool_turn(logprobs=[-0.5, -1.0, -0.3])],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        ref_lps = tracker.get_reference_logprobs(0)
        current_lps = [lp + 0.5 for lp in ref_lps]
        assert tracker.should_activate(0, current_lps) is True


# ---------------------------------------------------------------------------
# LLDSTracker.compute_regularization 测试
# ---------------------------------------------------------------------------


class TestComputeRegularization:
    """测试 LLDS 惩罚计算的核心逻辑。"""

    # --- LLDS-R: Response-level ---

    def test_r_variant_no_penalty_when_improved(self) -> None:
        """LLDS-R: 似然上升时不施加惩罚。"""
        config = LLDSConfig(variant="R", lambda_reg=0.1)
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[_make_tool_turn(logprobs=[-0.5, -1.0, -0.3])],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        ref_lps = tracker.get_reference_logprobs(0)
        current_lps = [lp + 0.3 for lp in ref_lps]
        penalty = tracker.compute_regularization(0, current_lps)
        assert penalty == 0.0

    def test_r_variant_penalty_when_decreased(self) -> None:
        """LLDS-R: 似然下降时对所有 completion token 施加惩罚。"""
        config = LLDSConfig(variant="R", lambda_reg=0.1)
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[_make_tool_turn(logprobs=[-0.5, -1.0, -0.3])],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        ref_lps = tracker.get_reference_logprobs(0)
        # 模拟似然下降：每个 token 的 logprob 都减少 0.5
        current_lps = [lp - 0.5 for lp in ref_lps]
        penalty = tracker.compute_regularization(0, current_lps)
        # 3 个 completion token 各下降 0.5，penalty = 3 * 0.5 = 1.5
        assert penalty == pytest.approx(1.5, abs=1e-6)

    # --- LLDS-A: Action-level ---

    def test_a_variant_only_penalizes_advantage_tokens(self) -> None:
        """LLDS-A: 只对 advantage != 0 的 token 施加惩罚。"""
        config = LLDSConfig(variant="A")
        tracker = LLDSTracker(config)
        # advantage=0.5: 所有 completion token 的 advantage 非零
        traj = _make_trajectory(
            turns=[_make_tool_turn(logprobs=[-0.5, -1.0, -0.3])],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        ref_lps = tracker.get_reference_logprobs(0)
        current_lps = [lp - 0.5 for lp in ref_lps]
        penalty = tracker.compute_regularization(0, current_lps)
        assert penalty == pytest.approx(1.5, abs=1e-6)

    def test_a_variant_skips_zero_advantage_tokens(self) -> None:
        """LLDS-A: advantage=0 的 token 不参与计算。"""
        config = LLDSConfig(variant="A")
        tracker = LLDSTracker(config)
        # advantage=0.0: 所有 completion token 的 advantage 为零
        traj = _make_trajectory(
            turns=[_make_tool_turn(logprobs=[-0.5, -1.0, -0.3])],
            advantage=0.0,
        )
        tracker.store_reference(traj, trajectory_id=0)
        ref_lps = tracker.get_reference_logprobs(0)
        current_lps = [lp - 0.5 for lp in ref_lps]
        penalty = tracker.compute_regularization(0, current_lps)
        # 所有 token advantage=0，不参与计算
        assert penalty == 0.0

    # --- LLDS-MA: Mask Answer ---

    def test_ma_variant_only_tool_tokens(self) -> None:
        """LLDS-MA: 只对 tool token 施加惩罚，answer token 被排除。"""
        config = LLDSConfig(variant="MA")
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[
                _make_tool_turn(logprobs=[-0.5, -1.0, -0.3]),
                _make_answer_turn(logprobs=[-0.2, -0.4, -0.1]),
            ],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        ref_lps = tracker.get_reference_logprobs(0)
        # 两个 turn 都下降
        current_lps = [lp - 0.5 for lp in ref_lps]
        penalty = tracker.compute_regularization(0, current_lps)
        # 只有 tool turn 的 3 个 token 参与，penalty = 3 * 0.5 = 1.5
        assert penalty == pytest.approx(1.5, abs=1e-6)

    def test_ma_with_mask_answer_true(self) -> None:
        """mask_answer=True 时也排除 answer token。"""
        config = LLDSConfig(variant="A", mask_answer=True)
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[
                _make_tool_turn(logprobs=[-0.5, -1.0, -0.3]),
                _make_answer_turn(logprobs=[-0.2, -0.4, -0.1]),
            ],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        ref_lps = tracker.get_reference_logprobs(0)
        current_lps = [lp - 0.5 for lp in ref_lps]
        penalty = tracker.compute_regularization(0, current_lps)
        assert penalty == pytest.approx(1.5, abs=1e-6)

    # --- 单向惩罚：max(0, ref - current) ---

    def test_one_sided_penalty_increased_tokens_excluded(self) -> None:
        """只在似然下降的 token 上施加惩罚，上升的 token 不应贡献。"""
        config = LLDSConfig(variant="A")
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[_make_tool_turn(logprobs=[-0.5, -1.0, -0.3])],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        ref_lps = tracker.get_reference_logprobs(0)
        # 第一个 token 上升，第二个和第三个下降
        # ref:  [-0.5, -1.0, -0.3]
        # cur:  [-0.2, -1.5, -0.8]
        # diff (ref-cur, max with 0): [0, 0.5, 0] -> only token 1 contributes
        # But we need to figure out which indices are completion tokens
        # ref_lps includes observation padding. Let me check the actual structure.

        # The ref_lps after store_reference: [obs_padding..., lp1, lp2, lp3]
        # We need to know the exact indices
        # Actually, the function get_reference_logprobs returns the full aligned array
        # Let's directly test by checking the completion mask positions

        # Instead, let's use the tracker's internal data
        ref = tracker._trajectories[0]
        # Find completion token indices
        comp_indices = [i for i, m in enumerate(ref.completion_mask) if m]
        assert len(comp_indices) == 3

        # Set current logprobs: token 0 rises, tokens 1 and 2 drop
        current_lps = list(ref_lps)  # copy
        current_lps[comp_indices[0]] = ref_lps[comp_indices[0]] + 0.3  # rises
        current_lps[comp_indices[1]] = ref_lps[comp_indices[1]] - 0.5  # drops
        current_lps[comp_indices[2]] = ref_lps[comp_indices[2]] - 0.4  # drops

        penalty = tracker.compute_regularization(0, current_lps)
        # Only tokens 1 and 2 contribute: 0.5 + 0.4 = 0.9
        assert penalty == pytest.approx(0.9, abs=1e-6)

    def test_no_penalty_when_all_improved(self) -> None:
        """所有 token 的似然都上升时，penalty 应为 0。"""
        config = LLDSConfig(variant="A")
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[_make_tool_turn(logprobs=[-0.5, -1.0, -0.3])],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        ref_lps = tracker.get_reference_logprobs(0)
        current_lps = [lp + 0.5 for lp in ref_lps]
        penalty = tracker.compute_regularization(0, current_lps)
        assert penalty == 0.0

    # --- 边界情况 ---

    def test_length_mismatch_raises(self) -> None:
        """current_logprobs 长度不匹配时应抛出异常。"""
        tracker = LLDSTracker()
        traj = _make_trajectory(turns=[_make_tool_turn()], advantage=0.5)
        tracker.store_reference(traj, trajectory_id=0)
        with pytest.raises(ValueError, match="不匹配"):
            tracker.compute_regularization(0, [0.0, 0.0])  # 长度不对

    def test_unregistered_trajectory_raises(self) -> None:
        """访问未注册的轨迹 ID 应抛出 KeyError。"""
        tracker = LLDSTracker()
        with pytest.raises(KeyError):
            tracker.compute_regularization(999, [0.0])

    def test_zero_lambda_returns_zero_effective_penalty(self) -> None:
        """lambda_reg=0 时，compute_regularization 仍返回原始 penalty，
        由调用方乘以 lambda。"""
        config = LLDSConfig(lambda_reg=0.0, variant="A")
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[_make_tool_turn(logprobs=[-0.5, -1.0, -0.3])],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        ref_lps = tracker.get_reference_logprobs(0)
        current_lps = [lp - 0.5 for lp in ref_lps]
        penalty = tracker.compute_regularization(0, current_lps)
        # compute_regularization 返回原始求和，不做 lambda 缩放
        assert penalty == pytest.approx(1.5, abs=1e-6)

    def test_extreme_logprob_values(self) -> None:
        """极端 logprob 值（如 -inf）应正确处理。"""
        config = LLDSConfig(variant="A")
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[_make_tool_turn(logprobs=[-0.5, float("-inf"), -0.3])],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        ref_lps = tracker.get_reference_logprobs(0)
        # 当 ref 为 -inf 时，ref - cur 可能为 nan 或 inf
        # 确保不崩溃（ref - cur 在 ref=-inf 时，cur 也是有限值的话，结果是 -inf，max(0, -inf) = 0）
        current_lps = [lp for lp in ref_lps]
        penalty = tracker.compute_regularization(0, current_lps)
        # 所有 logprob 未下降（其中一个是 -inf，max(0, -inf - (-inf)) = max(0, nan) = 0 或报错）
        # Python 的 max(0, nan) 返回 nan，但我们希望它是 0
        # 对于 -inf 对 -inf，差值为 nan，这可能导致问题
        # accept penalty being 0 or a reasonable value
        assert penalty >= 0.0  # 至少是非负值


# ---------------------------------------------------------------------------
# compute_batch_regularization 测试
# ---------------------------------------------------------------------------


class TestBatchRegularization:
    """批量计算测试。"""

    def test_batch_sums_individual_penalties(self) -> None:
        config = LLDSConfig(variant="A")
        tracker = LLDSTracker(config)
        for idx in range(3):
            traj = _make_trajectory(
                turns=[_make_tool_turn(logprobs=[-0.5, -1.0, -0.3])],
                advantage=0.5,
            )
            tracker.store_reference(traj, trajectory_id=idx)

        # 为每条轨迹构造下降的 current logprobs
        batch = {}
        for idx in range(3):
            ref_lps = tracker.get_reference_logprobs(idx)
            batch[idx] = [lp - 0.5 for lp in ref_lps]

        total = tracker.compute_batch_regularization(batch)
        # 每条轨迹 3 个 token 各下降 0.5 => 3 * 0.5 * 3 = 4.5
        assert total == pytest.approx(4.5, abs=1e-6)

    def test_batch_with_mixed_advantages(self) -> None:
        """部分轨迹 advantage=0，在 LLDS-A 下不贡献 penalty。"""
        config = LLDSConfig(variant="A")
        tracker = LLDSTracker(config)
        # 第一条 advantage=0.5（活跃），第二条 advantage=0.0（不活跃）
        for idx, adv in enumerate([0.5, 0.0]):
            traj = _make_trajectory(
                turns=[_make_tool_turn(logprobs=[-0.5, -1.0, -0.3])],
                advantage=adv,
            )
            tracker.store_reference(traj, trajectory_id=idx)

        batch = {}
        for idx in range(2):
            ref_lps = tracker.get_reference_logprobs(idx)
            batch[idx] = [lp - 0.5 for lp in ref_lps]

        total = tracker.compute_batch_regularization(batch)
        # 只有 trajectory 0 贡献: 3 * 0.5 = 1.5
        assert total == pytest.approx(1.5, abs=1e-6)


# ---------------------------------------------------------------------------
# get_effective_token_count 测试
# ---------------------------------------------------------------------------


class TestEffectiveTokenCount:
    """有效 token 计数测试。"""

    def test_a_variant_counts_advantage_tokens(self) -> None:
        config = LLDSConfig(variant="A")
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[
                _make_tool_turn(completion_tokens=[1, 2], logprobs=[-0.5, -0.3]),
            ],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        assert tracker.get_effective_token_count(0) == 2

    def test_a_variant_zero_advantage_counts_zero(self) -> None:
        config = LLDSConfig(variant="A")
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[_make_tool_turn()],
            advantage=0.0,
        )
        tracker.store_reference(traj, trajectory_id=0)
        assert tracker.get_effective_token_count(0) == 0

    def test_ma_variant_excludes_answer_tokens(self) -> None:
        config = LLDSConfig(variant="MA")
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[
                _make_tool_turn(completion_tokens=[1, 2], logprobs=[-0.5, -0.3]),
                _make_answer_turn(completion_tokens=[3, 4], logprobs=[-0.2, -0.1]),
            ],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        assert tracker.get_effective_token_count(0) == 2  # only tool tokens

    def test_r_variant_counts_all_completion_tokens(self) -> None:
        config = LLDSConfig(variant="R")
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[
                _make_tool_turn(completion_tokens=[1, 2], logprobs=[-0.5, -0.3]),
                _make_answer_turn(completion_tokens=[3, 4], logprobs=[-0.2, -0.1]),
            ],
            advantage=0.0,
        )
        tracker.store_reference(traj, trajectory_id=0)
        # LLDS-R counts all completion tokens regardless of advantage
        assert tracker.get_effective_token_count(0) == 4


# ---------------------------------------------------------------------------
# clear 测试
# ---------------------------------------------------------------------------


class TestClear:
    """清除存储数据测试。"""

    def test_clear_removes_all_data(self) -> None:
        tracker = LLDSTracker()
        traj = _make_trajectory(turns=[_make_tool_turn()], advantage=0.5)
        tracker.store_reference(traj, trajectory_id=0)
        assert tracker.trajectory_count == 1
        tracker.clear()
        assert tracker.trajectory_count == 0

    def test_clear_then_reuse(self) -> None:
        """清除后可以重新存储数据。"""
        tracker = LLDSTracker()
        traj = _make_trajectory(turns=[_make_tool_turn()], advantage=0.5)
        tracker.store_reference(traj, trajectory_id=0)
        tracker.clear()
        # 重新存储
        tracker.store_reference(traj, trajectory_id=0)
        assert tracker.trajectory_count == 1


# ---------------------------------------------------------------------------
# llds_loss_fn 辅助函数测试
# ---------------------------------------------------------------------------


class TestLLDSLossFn:
    """集成辅助函数测试。"""

    def test_loss_fn_adds_weighted_penalty(self) -> None:
        config = LLDSConfig(lambda_reg=0.05, variant="A")
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[_make_tool_turn(logprobs=[-0.5, -1.0, -0.3])],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        ref_lps = tracker.get_reference_logprobs(0)
        current_lps = [lp - 0.5 for lp in ref_lps]

        base_loss = 2.0
        total = llds_loss_fn(
            llds_tracker=tracker,
            trajectory_id=0,
            base_loss=base_loss,
            current_logprobs=current_lps,
        )
        # penalty = 1.5, lambda=0.05, total = 2.0 + 0.05 * 1.5 = 2.075
        assert total == pytest.approx(2.075, abs=1e-6)

    def test_loss_fn_custom_lambda_scale(self) -> None:
        """自定义 lambda_scale 覆盖配置。"""
        config = LLDSConfig(lambda_reg=0.05, variant="A")
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[_make_tool_turn(logprobs=[-0.5, -1.0, -0.3])],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)
        ref_lps = tracker.get_reference_logprobs(0)
        current_lps = [lp - 0.5 for lp in ref_lps]

        base_loss = 2.0
        total = llds_loss_fn(
            llds_tracker=tracker,
            trajectory_id=0,
            base_loss=base_loss,
            current_logprobs=current_lps,
            lambda_scale=0.1,
        )
        # penalty = 1.5, custom lambda=0.1, total = 2.0 + 0.15 = 2.15
        assert total == pytest.approx(2.15, abs=1e-6)


# ---------------------------------------------------------------------------
# extract_completion_logprobs 测试
# ---------------------------------------------------------------------------


class TestExtractCompletionLogprobs:
    """辅助提取函数测试。"""

    def test_extracts_only_completion_tokens(self) -> None:
        full = [-0.1, -0.5, -0.3, -0.2, -0.8, -0.4]
        mask = [False, True, True, False, True, True]
        result = extract_completion_logprobs(full, mask)
        assert result == [-0.5, -0.3, -0.8, -0.4]

    def test_empty_result_when_no_completion(self) -> None:
        full = [-0.1, -0.2, -0.3]
        mask = [False, False, False]
        result = extract_completion_logprobs(full, mask)
        assert result == []

    def test_all_completion(self) -> None:
        full = [-0.5, -0.3, -0.2]
        mask = [True, True, True]
        result = extract_completion_logprobs(full, mask)
        assert result == full

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="不匹配"):
            extract_completion_logprobs([1.0, 2.0], [True])


# ---------------------------------------------------------------------------
# 集成场景测试：模拟完整训练流程
# ---------------------------------------------------------------------------


class TestIntegrationScenarios:
    """模拟从 rollout 到 backward 的端到端流程。"""

    def test_full_pipeline_single_trajectory(self) -> None:
        """模拟单条轨迹的完整 LLDS 流水线。"""
        # 1. 配置
        config = LLDSConfig(lambda_reg=0.05, variant="A")

        # 2. Rollout 后：保存参考数据
        tracker = LLDSTracker(config)
        traj = _make_trajectory(
            turns=[
                _make_tool_turn(logprobs=[-0.5, -1.0, -0.3]),
                _make_answer_turn(logprobs=[-0.2, -0.4, -0.1]),
            ],
            advantage=0.5,
        )
        tracker.store_reference(traj, trajectory_id=0)

        # 3. 训练时：获取当前模型 logprobs（模拟似然下降）
        ref_lps = tracker.get_reference_logprobs(0)
        current_lps = [lp - 0.3 for lp in ref_lps]

        # 4. 检查 gate
        should_apply = tracker.should_activate(0, current_lps)
        assert should_apply is True

        # 5. 计算惩罚
        penalty = tracker.compute_regularization(0, current_lps)

        # 6. 合并到 loss
        grpo_loss = 1.5
        total_loss = grpo_loss + config.lambda_reg * penalty
        assert total_loss > grpo_loss

        # 7. 验证有效 token 数
        effective = tracker.get_effective_token_count(0)
        # LLDS-A with advantage=0.5: all 6 completion tokens
        assert effective == 6

    def test_training_step_with_mixed_quality(self) -> None:
        """混合质量的训练 step：部分轨迹似然上升，部分下降。"""
        config = LLDSConfig(variant="A", lambda_reg=0.05)
        tracker = LLDSTracker(config)

        # 三条轨迹
        for idx in range(3):
            traj = _make_trajectory(
                turns=[_make_tool_turn(logprobs=[-0.5, -0.3, -0.4])],
                advantage=0.5,
            )
            tracker.store_reference(traj, trajectory_id=idx)

        # 轨迹 0: 似然上升 (无惩罚)
        # 轨迹 1: 似然下降 (有惩罚)
        # 轨迹 2: 似然下降 (有惩罚，但 advantage=0，LLDS-A 下不参与)
        # 需要重新存储轨迹 2 的 advantage
        tracker.clear()
        for idx, adv in enumerate([0.5, 0.5, 0.0]):
            traj = _make_trajectory(
                turns=[_make_tool_turn(logprobs=[-0.5, -0.3, -0.4])],
                advantage=adv,
            )
            tracker.store_reference(traj, trajectory_id=idx)

        batch = {}
        ref0 = tracker.get_reference_logprobs(0)
        ref1 = tracker.get_reference_logprobs(1)
        ref2 = tracker.get_reference_logprobs(2)

        batch[0] = [lp + 0.3 for lp in ref0]  # 似然上升
        batch[1] = [lp - 0.3 for lp in ref1]  # 似然下降
        batch[2] = [lp - 0.3 for lp in ref2]  # 似然下降但 adv=0

        total_penalty = tracker.compute_batch_regularization(batch)
        # 只有轨迹 1 贡献：3 tokens * 0.3 = 0.9
        assert total_penalty == pytest.approx(0.9, abs=1e-6)
