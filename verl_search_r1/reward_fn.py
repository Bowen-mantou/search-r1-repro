"""Custom reward function for Search-R1 + veRL 0.10.

Integrates Search-R1's answer scoring (``reward.score_answer``) and optional
PRM-Lite process reward (``reward_lite.score_with_prm``) as a veRL-compatible
reward function.

veRL 0.10 calling convention (experimental/reward_loop/reward_manager/naive.py):
the raw reward fn is invoked keyword-only with
``data_source`` / ``solution_str`` / ``ground_truth`` / ``extra_info``
and must return either a float or ``{"score": float, **per-metric}``
(extra keys land in ``reward_extra_info``).

Usage in veRL YAML config:
    reward:
      custom_reward_function:
        path: verl_search_r1.reward_fn
        name: search_r1_reward
        reward_kwargs:
          enable_prm_lite: true
          enable_lata: false

Or when launching:
    reward.custom_reward_function.path=pkg://verl_search_r1.reward_fn \
    reward.custom_reward_function.name=search_r1_reward \
    +reward.custom_reward_function.reward_kwargs.enable_prm_lite=true \
    +reward.custom_reward_function.reward_kwargs.enable_lata=false
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

# Ensure the parent 03-search-r1 directory is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from reward import score_answer
from reward_lite import LATAScaler, PRMLiteScorer, apply_lata, score_with_prm


def search_r1_reward(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: list[str] | None = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute Search-R1 reward for a single trajectory.

    Args:
        data_source: dataset name (e.g. "nq", "hotpotqa") — unused for scoring,
            kept for interface parity.
        solution_str: decoded model response text.
        ground_truth: list of accepted answers.
        extra_info: dict merged from dataset ``extra_info`` + agent-loop
            ``tool_extra_fields`` (response_text, search_calls, turns, ...).

    Returns:
        ``{"score": float, ...}`` — extra keys become per-sample metrics in
        ``reward_extra_info``.
    """
    enable_prm_lite = kwargs.get("enable_prm_lite", False)
    enable_lata = kwargs.get("enable_lata", False)

    extra = extra_info or {}
    final_text = solution_str or extra.get("response_text") or ""
    search_calls = extra.get("search_calls", 0)

    gt = ground_truth or []

    # Score the answer
    result = score_answer(final_text, gt)
    reward_val = result.reward

    # Overlay PRM-Lite process reward
    prm_val = 0.0
    if enable_prm_lite and final_text:
        prm_scorer = PRMLiteScorer()
        prm_result = prm_scorer.score(
            _PRMTrajectoryAdapter(
                final_text=final_text,
                question=extra.get("question", ""),
                turns=_extract_turns(extra),
                search_calls=search_calls,
            )
        )
        prm_val = prm_result.process_reward
        reward_val += prm_val

    out: dict[str, Any] = {
        "score": float(reward_val),
        "exact_match": float(result.exact_match),
        "valid_format": float(result.valid_format),
    }
    if enable_prm_lite:
        out["prm_process_reward"] = float(prm_val)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _extract_turns(extra: dict[str, Any]) -> list:
    """Extract turn information from agent-loop extra_fields.

    Turns may arrive as a list of dicts (each normalised to a
    _FakeTurn-compatible object).
    """
    turns = extra.get("turns") or []
    if not turns:
        return []
    return [t if not isinstance(t, dict) else _FakeTurn(t.get("text", "")) for t in turns]


class _PRMTrajectoryAdapter:
    """Minimal trajectory adapter for PRMLiteScorer.

    Provides the duck-typed interface expected by reward_lite.py:
      - ``turns``: list of objects with ``text`` and ``completion_text`` attrs
      - ``final_text``: str
      - ``example.question``: str
      - ``search_calls``: int
      - ``messages``: list of role/content dicts
    """

    def __init__(self, final_text: str, question: str, turns: list, search_calls: int):
        self.final_text = final_text
        self.turns = turns or [_FakeTurn(final_text)]
        self.search_calls = search_calls
        self.messages: list[dict[str, str]] = []
        self.example = _FakeExample(question)
        self.anomalous = False
        self.reward = 0.0


class _FakeTurn:
    def __init__(self, text: str):
        self.text = text
        self.completion_text = text


class _FakeExample:
    def __init__(self, question: str):
        self.question = question
