"""PRM-Lite + LATA 模块的单元测试。

覆盖：
    - 12 条 Penalty 规则（每条至少一个触发 + 一个不触发用例）
    - 10 条 Bonus 规则（每条至少一个触发 + 一个不触发用例）
    - 规则互斥与上限截断
    - 辅助函数（文本分析、特征提取）
    - LATA 对长/短轨迹的 advantage 缩放效果
    - PRM-Lite + LATA 组合
    - 边界情况：空轨迹、单轮轨迹、多轮轨迹
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reward_lite import (
    LATAScaler,
    PRMLiteConfig,
    PRMLiteResult,
    PRMLiteScorer,
    _common_substring_length,
    _extract_answer,
    _extract_queries_from_turns,
    _extract_search_result_blocks,
    _is_multi_hop,
    _is_question_sentence,
    _is_trivially_simple,
    _query_specificity,
    _trajectory_completion_token_count,
    _word_count,
    _word_overlap_ratio,
    apply_lata,
    score_with_prm,
)


# ===========================================================================
# 轻量级 mock：模拟 rollout 及训练流程中的对象
# ===========================================================================


@dataclass
class MockTurn:
    """模拟 AssistantTurn，仅包含 PRM-Lite / LATA 需要的字段。"""

    text: str = ""
    completion_tokens: list[int] = field(default_factory=list)


@dataclass
class MockExample:
    """模拟 Example，提供 question 字段。"""

    question: str = ""


@dataclass
class MockTrajectory:
    """模拟 Trajectory，覆盖 PRM-Lite + LATA 所需全部字段。"""

    turns: list[MockTurn] = field(default_factory=list)
    final_text: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    search_calls: int = 0
    example: MockExample = field(default_factory=MockExample)
    reward: float = 0.0
    question_index: int = 0
    advantage: float = 0.0


# ===========================================================================
# 测试辅助函数
# ===========================================================================


def _tool_text(query: str) -> str:
    """生成一段包含 search tool-call 的 assistant turn 文本。"""
    return (
        "<tool_call>\n"
        "<function=search>\n"
        "<parameter=query>\n"
        f"{query}\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )


def _tool_text_with_prefix(prefix: str, query: str) -> str:
    """生成包含搜索前推理文本的 tool-call 文本。"""
    return prefix + "\n" + _tool_text(query)


def _make_turn(
    text: str = "",
    completion_tokens: list[int] | None = None,
) -> MockTurn:
    return MockTurn(text=text, completion_tokens=completion_tokens if completion_tokens is not None else [101, 102, 103])


def _make_tool_msg(content: str) -> dict[str, Any]:
    return {"role": "tool", "content": content}


def _make_traj(
    *,
    turns: list[MockTurn] | None = None,
    final_text: str = "",
    messages: list[dict[str, Any]] | None = None,
    search_calls: int = 0,
    question: str = "",
    reward: float = 0.0,
    question_index: int = 0,
) -> MockTrajectory:
    return MockTrajectory(
        turns=turns or [],
        final_text=final_text,
        messages=messages or [],
        search_calls=search_calls,
        example=MockExample(question=question),
        reward=reward,
        question_index=question_index,
    )


# ===========================================================================
# 辅助函数测试
# ===========================================================================


class TestWordCount:
    def test_simple(self) -> None:
        assert _word_count("hello world") == 2

    def test_empty(self) -> None:
        assert _word_count("") == 0
        assert _word_count("   ") == 0

    def test_single(self) -> None:
        assert _word_count("hello") == 1


class TestIsQuestionSentence:
    def test_what_question(self) -> None:
        assert _is_question_sentence("what is the capital of France") is True

    def test_who_question(self) -> None:
        assert _is_question_sentence("who discovered penicillin") is True

    def test_question_mark(self) -> None:
        assert _is_question_sentence("Paris is in France?") is True

    def test_keyword_not_question(self) -> None:
        assert _is_question_sentence("Paris capital France population") is False

    def test_empty(self) -> None:
        assert _is_question_sentence("") is False


class TestWordOverlapRatio:
    def test_full_overlap(self) -> None:
        # 去停用词后词集合相同
        ratio = _word_overlap_ratio("Paris is the capital", "capital Paris beautiful")
        assert ratio == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        ratio = _word_overlap_ratio("hello world", "foo bar baz")
        # hello, world vs foo, bar, baz (stopwords removed: all are non-stopwords)
        assert ratio == 0.0

    def test_partial_overlap(self) -> None:
        ratio = _word_overlap_ratio("Paris capital France", "Paris city population")
        # {"paris", "capital", "france"} vs {"paris", "city", "population"}
        # inter = {"paris"}, min_len = 3, ratio = 1/3
        assert ratio == pytest.approx(1.0 / 3.0)

    def test_all_stopwords(self) -> None:
        # 全部是停用词
        ratio = _word_overlap_ratio("the a is", "of in on")
        assert ratio == 0.0

    def test_empty_input(self) -> None:
        assert _word_overlap_ratio("", "hello") == 0.0
        assert _word_overlap_ratio("hello", "") == 0.0


class TestCommonSubstringLength:
    def test_long_match(self) -> None:
        s = "the quick brown fox jumps over the lazy dog " * 3  # ~132 chars
        length = _common_substring_length(s, s, min_len=10)
        assert length >= 100  # capped at len(shorter) but max 100

    def test_exact_match(self) -> None:
        length = _common_substring_length("hello world abcdef", "world abcdef xyz", min_len=5)
        assert length == 12  # "world abcdef"

    def test_no_match(self) -> None:
        length = _common_substring_length("aaaa", "bbbb", min_len=3)
        assert length == 0

    def test_min_len_filter(self) -> None:
        # 最长公共子串长度 < min_len，返回 0
        length = _common_substring_length("hello world", "world hello", min_len=50)
        assert length == 0

    def test_case_insensitive(self) -> None:
        length = _common_substring_length("Hello World", "hello world today", min_len=5)
        assert length == 11  # "hello world"


class TestQuerySpecificity:
    def test_high_specificity(self) -> None:
        # 多个大写开头 + 长文本
        score = _query_specificity("Albert Einstein Nobel Prize Physics 1921")
        words = 6  # len_score = min(6/10, 1) = 0.6, cap_score = 5/6 ≈ 0.833
        assert score > 0.5

    def test_low_specificity(self) -> None:
        score = _query_specificity("hello")
        # 1 word, lowercase: len_score = 0.1, cap_score = 0
        assert score == pytest.approx(0.05)

    def test_empty(self) -> None:
        assert _query_specificity("") == 0.0

    def test_increasing(self) -> None:
        """更具体的 query 应得分更高。"""
        s1 = _query_specificity("hello")
        s2 = _query_specificity("Paris France Eiffel Tower")
        assert s2 > s1


class TestIsTriviallySimple:
    def test_arithmetic(self) -> None:
        assert _is_trivially_simple("what is 2+2") is True

    def test_greeting(self) -> None:
        assert _is_trivially_simple("hello!") is True

    def test_short_query(self) -> None:
        assert _is_trivially_simple("hi") is True

    def test_complex_question(self) -> None:
        assert _is_trivially_simple("What was the impact of the French Revolution on European politics") is False

    def test_empty(self) -> None:
        assert _is_trivially_simple("") is True  # _word_count("") == 0 <= 2


class TestIsMultiHop:
    def test_comparison(self) -> None:
        assert _is_multi_hop("compare iPhone 15 and Samsung S24") is True

    def test_compound_question(self) -> None:
        assert _is_multi_hop("When was Einstein born and what did he discover") is True

    def test_very_long_question(self) -> None:
        assert _is_multi_hop("x " * 20) is True  # > 15 words

    def test_simple_fact(self) -> None:
        assert _is_multi_hop("capital of France") is False

    def test_temporal(self) -> None:
        assert _is_multi_hop("who was president before Obama") is True


class TestExtractAnswer:
    def test_valid_answer(self) -> None:
        assert _extract_answer("Answer: Paris") == "Paris"

    def test_multiline_with_answer(self) -> None:
        text = "Some reasoning\nAnswer: 42\nMore text"
        assert _extract_answer(text) == "42"

    def test_no_answer(self) -> None:
        assert _extract_answer("Just some text") is None

    def test_multiple_answers(self) -> None:
        text = "Answer: Paris\nAnswer: Lyon"
        assert _extract_answer(text) is None

    def test_empty_answer(self) -> None:
        assert _extract_answer("Answer:   ") is None

    def test_case_insensitive(self) -> None:
        assert _extract_answer("answer: Paris") == "Paris"


class TestExtractQueriesFromTurns:
    def test_single_query(self) -> None:
        turns = [_make_turn(text=_tool_text("Paris capital"))]
        assert _extract_queries_from_turns(turns) == ["Paris capital"]

    def test_multiple_queries(self) -> None:
        turns = [
            _make_turn(text=_tool_text("first query")),
            _make_turn(text=_tool_text("second query")),
        ]
        assert _extract_queries_from_turns(turns) == ["first query", "second query"]

    def test_no_tool_call(self) -> None:
        turns = [_make_turn(text="Just thinking without any tool call")]
        assert _extract_queries_from_turns(turns) == []

    def test_empty_turns(self) -> None:
        assert _extract_queries_from_turns([]) == []


class TestExtractSearchResultBlocks:
    def test_single_block(self) -> None:
        msgs = [_make_tool_msg("Result A content")]
        blocks = _extract_search_result_blocks(msgs)
        assert blocks == ["Result A content"]

    def test_multiple_blocks_in_one_message(self) -> None:
        msgs = [_make_tool_msg("Block 1\n\nBlock 2\n\nBlock 3")]
        blocks = _extract_search_result_blocks(msgs)
        assert blocks == ["Block 1", "Block 2", "Block 3"]

    def test_multiple_messages(self) -> None:
        msgs = [
            _make_tool_msg("Result 1"),
            _make_tool_msg("Result 2"),
        ]
        blocks = _extract_search_result_blocks(msgs)
        assert blocks == ["Result 1", "Result 2"]

    def test_no_tool_messages(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        blocks = _extract_search_result_blocks(msgs)
        assert blocks == []

    def test_empty_content_skipped(self) -> None:
        msgs = [_make_tool_msg(""), _make_tool_msg("valid")]
        blocks = _extract_search_result_blocks(msgs)
        assert blocks == ["valid"]


class TestTrajectoryCompletionTokenCount:
    def test_single_turn(self) -> None:
        traj = _make_traj(turns=[_make_turn(completion_tokens=[1, 2, 3, 4])])
        assert _trajectory_completion_token_count(traj) == 4

    def test_multiple_turns(self) -> None:
        traj = _make_traj(
            turns=[
                _make_turn(completion_tokens=[1, 2]),
                _make_turn(completion_tokens=[3, 4, 5]),
            ]
        )
        assert _trajectory_completion_token_count(traj) == 5

    def test_empty_turns(self) -> None:
        traj = _make_traj()
        assert _trajectory_completion_token_count(traj) == 0

    def test_empty_tokens(self) -> None:
        traj = _make_traj(turns=[_make_turn(completion_tokens=[])])
        assert _trajectory_completion_token_count(traj) == 0


# ===========================================================================
# PRMLiteConfig 测试
# ===========================================================================


class TestPRMLiteConfig:
    def test_default_config(self) -> None:
        cfg = PRMLiteConfig()
        assert cfg.max_assistant_turns == 6
        assert cfg.long_query_threshold == 200
        assert cfg.max_penalty == -0.2
        assert cfg.max_bonus == 0.2

    def test_all_rules_enabled_by_default(self) -> None:
        cfg = PRMLiteConfig()
        for attr in dir(cfg):
            if attr.startswith(("p", "b")) and not attr.startswith("p1"):  # skip partial
                pass
        assert cfg.p1_empty_query is True
        assert cfg.b10_gradual_focus is True

    def test_disable_rule(self) -> None:
        cfg = PRMLiteConfig(p1_empty_query=False, b1_keyword_query=False)
        assert cfg.p1_empty_query is False
        assert cfg.b1_keyword_query is False
        assert cfg.p2_long_query is True  # 其他规则不受影响

    def test_custom_thresholds(self) -> None:
        cfg = PRMLiteConfig(
            max_assistant_turns=3,
            long_query_threshold=100,
            concise_answer_min=10,
            concise_answer_max=50,
        )
        assert cfg.max_assistant_turns == 3
        assert cfg.long_query_threshold == 100
        assert cfg.concise_answer_min == 10
        assert cfg.concise_answer_max == 50


# ===========================================================================
# Penalty 规则测试 —— 每条至少一个触发 + 一个不触发
# ===========================================================================


class TestPenalties:
    """12 条 penalty 规则的单元测试。"""

    # -- P1: empty / whitespace-only query -----------------------------------

    def test_p1_trigger_empty_query(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p1_empty_query(["normal", "  "]) == pytest.approx(-0.05)

    def test_p1_trigger_empty_string(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p1_empty_query(["", "q"]) == pytest.approx(-0.05)

    def test_p1_not_trigger_all_valid(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p1_empty_query(["q1", "q2"]) == 0.0

    def test_p1_not_trigger_no_queries(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p1_empty_query([]) == 0.0

    # -- P2: query longer than threshold ------------------------------------

    def test_p2_trigger_too_long(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p2_long_query(["a" * 201]) == pytest.approx(-0.03)

    def test_p2_not_trigger_normal(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p2_long_query(["normal query"]) == 0.0

    def test_p2_boundary_at_threshold(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p2_long_query(["a" * 200]) == 0.0  # exactly threshold

    # -- P3: consecutive identical queries ----------------------------------

    def test_p3_trigger_exact_duplicate(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p3_duplicate_query(["Paris", "Paris"]) == pytest.approx(-0.05)

    def test_p3_trigger_case_insensitive(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p3_duplicate_query(["PARIS", "paris"]) == pytest.approx(-0.05)

    def test_p3_not_trigger_different(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p3_duplicate_query(["Paris", "Lyon"]) == 0.0

    def test_p3_not_trigger_non_consecutive(self) -> None:
        """只有 consecutive 的重复才触发。"""
        scorer = PRMLiteScorer()
        assert scorer._check_p3_duplicate_query(["Paris", "Lyon", "Paris"]) == 0.0

    def test_p3_not_trigger_single_query(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p3_duplicate_query(["only one"]) == 0.0

    # -- P4: no valid Answer: line -------------------------------------------

    def test_p4_trigger_no_answer(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p4_invalid_answer(None) == pytest.approx(-0.1)

    def test_p4_not_trigger_has_answer(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p4_invalid_answer("Paris") == 0.0

    # -- P5: excessive assistant turns ---------------------------------------

    def test_p5_trigger_too_many_turns(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p5_excessive_turns(7) == pytest.approx(-0.05)

    def test_p5_not_trigger_within_limit(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p5_excessive_turns(6) == 0.0

    def test_p5_not_trigger_zero_turns(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p5_excessive_turns(0) == 0.0

    # -- P6: extracted answer too short --------------------------------------

    def test_p6_trigger_short_answer(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p6_short_answer("ab") == pytest.approx(-0.03)

    def test_p6_trigger_exactly_4_chars(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p6_short_answer("abcd") == pytest.approx(-0.03)

    def test_p6_not_trigger_long_enough(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p6_short_answer("Paris") == 0.0

    def test_p6_not_trigger_null_answer(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p6_short_answer(None) == 0.0

    # -- P7: query is a full question sentence -------------------------------

    def test_p7_trigger_what(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p7_question_query(["what is the capital of France"]) == pytest.approx(-0.02)

    def test_p7_trigger_question_mark(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p7_question_query(["Is France in Europe?"]) == pytest.approx(-0.02)

    def test_p7_not_trigger_keywords(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p7_question_query(["Paris capital France population"]) == 0.0

    def test_p7_not_trigger_empty(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p7_question_query([]) == 0.0

    # -- P8: search done but answer ignores results --------------------------

    def test_p8_trigger_no_overlap(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p8_no_reference(
            has_search=True,
            answer="xyz abc def",
            search_content="quick brown fox jumps lazy dog",
        ) == pytest.approx(-0.05)

    def test_p8_not_trigger_has_overlap(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p8_no_reference(
            has_search=True,
            answer="brown fox",
            search_content="the quick brown fox jumps",
        ) == 0.0

    def test_p8_not_trigger_no_search(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p8_no_reference(
            has_search=False,
            answer="xyz",
            search_content="",
        ) == 0.0

    def test_p8_not_trigger_no_answer(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p8_no_reference(
            has_search=True,
            answer=None,
            search_content="some content",
        ) == 0.0

    # -- P9: multiple results but only first used ----------------------------

    def test_p9_trigger_only_first_used(self) -> None:
        scorer = PRMLiteScorer()
        # block0 和 answer 有高重叠，block1 与 answer 基本无重叠
        blocks = ["aaa bbb ccc ddd eee", "xxx yyy zzz www vvv"]
        answer = "aaa bbb ccc ddd eee"
        assert scorer._check_p9_ignore_results(blocks, answer) == pytest.approx(-0.03)

    def test_p9_not_trigger_uses_both(self) -> None:
        scorer = PRMLiteScorer()
        blocks = ["aaa bbb ccc ddd", "xxx yyy zzz www"]
        # 与两个 block 都有重叠
        answer = "aaa xxx result"
        assert scorer._check_p9_ignore_results(blocks, answer) == 0.0

    def test_p9_not_trigger_single_block(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p9_ignore_results(["only one block"], "answer") == 0.0

    # -- P10: verbatim copy from search results -------------------------------

    def test_p10_trigger_long_copy(self) -> None:
        scorer = PRMLiteScorer()
        long_text = "A" * 60
        assert scorer._check_p10_copy_paste(long_text, long_text) == pytest.approx(-0.03)

    def test_p10_not_trigger_no_copy(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p10_copy_paste("original answer", "search content here") == 0.0

    def test_p10_not_trigger_empty(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p10_copy_paste("", "") == 0.0

    # -- P11: query unrelated to question ------------------------------------

    def test_p11_trigger_irrelevant(self) -> None:
        scorer = PRMLiteScorer()
        # 完全无重叠
        assert scorer._check_p11_irrelevant_query(
            queries=["xyz abc"],
            question="Paris capital",
        ) == pytest.approx(-0.05)

    def test_p11_not_trigger_overlap(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p11_irrelevant_query(
            queries=["Paris capital population"],
            question="What is the capital of Paris",
        ) == 0.0

    def test_p11_not_trigger_no_queries(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p11_irrelevant_query([], "question") == 0.0

    # -- P12: search on trivially simple fact ---------------------------------

    def test_p12_trigger_simple_question_with_search(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p12_unnecessary_search(
            search_calls=1,
            question="what is 2+2",
        ) == pytest.approx(-0.02)

    def test_p12_not_trigger_no_search(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p12_unnecessary_search(search_calls=0, question="hi") == 0.0

    def test_p12_not_trigger_complex_question(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_p12_unnecessary_search(
            search_calls=1,
            question="Explain the theory of general relativity in detail",
        ) == 0.0


# ===========================================================================
# Bonus 规则测试 —— 每条至少一个触发 + 一个不触发
# ===========================================================================


class TestBonuses:
    """10 条 bonus 规则的单元测试。"""

    # -- B1: concise keyword query (<=5 words, not question) ------------------

    def test_b1_trigger_keyword(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b1_keyword_query(["Paris capital population 2024"]) == pytest.approx(0.03)

    def test_b1_not_trigger_question(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b1_keyword_query(["what is the capital of France"]) == 0.0

    def test_b1_not_trigger_too_many_words(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b1_keyword_query(["word1 word2 word3 word4 word5 word6"]) == 0.0

    def test_b1_not_trigger_no_queries(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b1_keyword_query([]) == 0.0

    # -- B2: synthesises >=2 result blocks -----------------------------------

    def test_b2_trigger_two_blocks_used(self) -> None:
        scorer = PRMLiteScorer()
        blocks = ["Paris is the capital of France", "France has a population of 67 million"]
        answer = "capital Paris France population million"
        assert scorer._check_b2_multi_reference(blocks, answer) == pytest.approx(0.05)

    def test_b2_not_trigger_only_one_block_used(self) -> None:
        scorer = PRMLiteScorer()
        blocks = ["Paris is the capital of France", "Galaxy far far away unrelated..."]
        answer = "Paris is the capital"
        assert scorer._check_b2_multi_reference(blocks, answer) == 0.0

    def test_b2_not_trigger_single_block(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b2_multi_reference(["only one"], "answer text") == 0.0

    # -- B3: think/integration text around searches ---------------------------

    def test_b3_trigger_thinking_before_search(self) -> None:
        scorer = PRMLiteScorer()
        prefix = "Let me think about this question carefully step by step to analyze"
        traj = _make_traj(
            turns=[_make_turn(text=_tool_text_with_prefix(prefix, "Paris"))],
            final_text="Answer: Paris",
        )
        assert scorer._check_b3_think_integration(traj) == pytest.approx(0.02)

    def test_b3_trigger_non_tool_reasoning_turn(self) -> None:
        scorer = PRMLiteScorer()
        traj = _make_traj(
            turns=[_make_turn(text="Based on the search results, I can conclude that the answer is Paris. This is supported by multiple sources.")],
            final_text="Answer: Paris",
        )
        assert scorer._check_b3_think_integration(traj) == pytest.approx(0.02)

    def test_b3_not_trigger_just_tool_call(self) -> None:
        scorer = PRMLiteScorer()
        traj = _make_traj(
            turns=[_make_turn(text=_tool_text("Paris"))],
            final_text="Answer: Paris",
        )
        assert scorer._check_b3_think_integration(traj) == 0.0

    # -- B4: first query overlaps question strongly ---------------------------

    def test_b4_trigger_high_first_overlap(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b4_first_search_hit(
            queries=["Paris capital population"],
            question="What is the capital of Paris and its population",
        ) == pytest.approx(0.05)

    def test_b4_not_trigger_low_overlap(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b4_first_search_hit(
            queries=["abc xyz"],
            question="Paris capital",
        ) == 0.0

    def test_b4_not_trigger_no_queries(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b4_first_search_hit([], "question") == 0.0

    # -- B5: answer length in [20, 100] --------------------------------------

    def test_b5_trigger_concise(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b5_concise_answer("A" * 50) == pytest.approx(0.03)

    def test_b5_trigger_boundary_min(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b5_concise_answer("A" * 20) == pytest.approx(0.03)

    def test_b5_trigger_boundary_max(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b5_concise_answer("A" * 100) == pytest.approx(0.03)

    def test_b5_not_trigger_too_short(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b5_concise_answer("short") == 0.0

    def test_b5_not_trigger_too_long(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b5_concise_answer("A" * 120) == 0.0

    def test_b5_not_trigger_null_answer(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b5_concise_answer(None) == 0.0

    # -- B6: exactly one Answer: line ----------------------------------------

    def test_b6_trigger_single_answer(self) -> None:
        scorer = PRMLiteScorer()
        traj = _make_traj(final_text="Some reasoning\nAnswer: Paris\nMore notes")
        # answer = _extract_answer("Some reasoning\nAnswer: Paris\nMore notes") = "Paris"
        assert scorer._check_b6_correct_format("Paris", traj) == pytest.approx(0.03)

    def test_b6_not_trigger_no_answer(self) -> None:
        scorer = PRMLiteScorer()
        traj = _make_traj(final_text="No answer line here")
        assert scorer._check_b6_correct_format(None, traj) == 0.0

    def test_b6_not_trigger_multiple_answers(self) -> None:
        scorer = PRMLiteScorer()
        traj = _make_traj(final_text="Answer: Paris\nAnswer: Lyon")
        # _extract_answer returns None (len(matches) == 2)
        # But we test with a non-None answer passed directly
        assert scorer._check_b6_correct_format("Paris", traj) == 0.0

    # -- B7: search rounds match question complexity --------------------------

    def test_b7_trigger_simple_question_1_search(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b7_reasonable_rounds(
            search_calls=1,
            question="capital of France",
        ) == pytest.approx(0.03)

    def test_b7_trigger_multi_hop_3_searches(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b7_reasonable_rounds(
            search_calls=3,
            question="compare iPhone vs Samsung and which has better camera resolution 2024",
        ) == pytest.approx(0.03)

    def test_b7_not_trigger_zero_searches(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b7_reasonable_rounds(search_calls=0, question="test") == 0.0

    def test_b7_not_trigger_simple_too_many_searches(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b7_reasonable_rounds(
            search_calls=5,
            question="capital of France",
        ) == 0.0

    # -- B8: answer uses own words, no long copy ------------------------------

    def test_b8_trigger_original_answer(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b8_own_words(
            answer="paris city the capital of france",
            search_content="France is a country in Europe",
        ) == pytest.approx(0.03)

    def test_b8_not_trigger_copy_detected(self) -> None:
        scorer = PRMLiteScorer()
        long_text = "A" * 40
        assert scorer._check_b8_own_words(long_text, long_text) == 0.0

    # -- B9: diverse keywords across consecutive queries ----------------------

    def test_b9_trigger_diverse(self) -> None:
        scorer = PRMLiteScorer()
        # 低重叠
        assert scorer._check_b9_diverse_keywords(
            ["Paris capital", "population demography census"],
        ) == pytest.approx(0.04)

    def test_b9_not_trigger_similar(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b9_diverse_keywords(
            ["Paris capital France", "Paris France capital"],
        ) == 0.0

    def test_b9_not_trigger_single_query(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b9_diverse_keywords(["only one"]) == 0.0

    # -- B10: gradual focus (increasing specificity) --------------------------

    def test_b10_trigger_increasing_specificity(self) -> None:
        scorer = PRMLiteScorer()
        # low -> medium -> high specificity
        queries = [
            "earthquake",
            "Japan Earthquake magnitude",
            "Tohoku Earthquake Richter Scale 2011",
        ]
        assert scorer._check_b10_gradual_focus(queries) == pytest.approx(0.03)

    def test_b10_not_trigger_decreasing(self) -> None:
        scorer = PRMLiteScorer()
        queries = [
            "Tohoku Earthquake Richter Scale 2011 details",
            "Japan earthquake",
            "quake",
        ]
        assert scorer._check_b10_gradual_focus(queries) == 0.0

    def test_b10_not_trigger_single_query(self) -> None:
        scorer = PRMLiteScorer()
        assert scorer._check_b10_gradual_focus(["only one"]) == 0.0


# ===========================================================================
# 规则互斥与上限截断测试
# ===========================================================================


class TestMutualExclusion:
    """P4 vs B6 互斥、P10 抑制 B8、以及 caps 截断。"""

    def _make_scoring_traj(
        self,
        final_text: str,
        turns: list[MockTurn] | None = None,
        messages: list[dict] | None = None,
        search_calls: int = 0,
        question: str = "",
    ) -> MockTrajectory:
        return _make_traj(
            turns=turns or [],
            final_text=final_text,
            messages=messages or [],
            search_calls=search_calls,
            question=question,
        )

    def test_p4_wins_over_b6(self) -> None:
        """P4 (-0.1) 的绝对值大于 B6 (0.03)，P4 胜出。"""
        scorer = PRMLiteScorer()
        # final_text 没有 Answer: 行 → P4 fires (-0.1)
        traj = self._make_scoring_traj(final_text="No answer line", question="test")
        result = scorer.score(traj)
        assert "p4" in result.penalties
        assert "b6" not in result.bonuses
        assert result.all_rules.get("b6", 0.0) == 0.0

    def test_b6_wins_over_p4_when_p4_disabled(self) -> None:
        """当 P4 被禁用时，B6 正常触发。"""
        cfg = PRMLiteConfig(p4_invalid_answer=False)
        scorer = PRMLiteScorer(cfg)
        traj = self._make_scoring_traj(final_text="Answer: Paris", question="test")
        result = scorer.score(traj)
        assert "p4" not in result.all_rules  # disabled rule not evaluated
        assert result.all_rules["b6"] == pytest.approx(0.03)

    def test_p10_suppresses_b8(self) -> None:
        """P10 触发时 B8 不应出现在结果中。"""
        scorer = PRMLiteScorer()
        long_snippet = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars"
        # P10 fires (verbatim copy from search)
        traj = _make_traj(
            final_text="Answer: " + long_snippet,
            messages=[_make_tool_msg(long_snippet)],
            turns=[_make_turn(text=_tool_text("Eiffel Tower"))],
            search_calls=1,
            question="Eiffel Tower",
        )
        result = scorer.score(traj)
        if "p10" in result.penalties:
            assert "b8" not in result.bonuses

    def test_penalty_capped(self) -> None:
        """总 penalty 不应超过 max_penalty (-0.2)。"""
        scorer = PRMLiteScorer()
        # 触发多条惩罚规则：
        # P4: no Answer: -> -0.1
        # P5: 7 turns -> -0.05
        # P7: question query -> -0.02
        # P12: simple question with search -> -0.02
        turns = [
            _make_turn(text=_tool_text("what is 2+2")) for _ in range(7)
        ]
        traj = _make_traj(
            turns=turns,
            final_text="No answer",
            messages=[_make_tool_msg("some text")],
            search_calls=1,
            question="hello",
        )
        result = scorer.score(traj)
        # P4 fires (-0.1) + potentially P5 (-0.05) + P12 (-0.02) + ...
        assert result.total_penalty >= -0.2  # 被 cap

    def test_bonus_capped(self) -> None:
        """总 bonus 不应超过 max_bonus (0.2)。"""
        cfg = PRMLiteConfig()
        scorer = PRMLiteScorer(cfg)
        # 构造尽可能多 bonus 触发的轨迹
        traj = _make_traj(
            turns=[
                _make_turn(text=_tool_text("Paris capital Eiffel population")),
                _make_turn(text=_tool_text("Tourism visitors Paris 2024")),
            ],
            final_text="Answer: Paris is the capital city of France with rich cultural heritage",
            messages=[
                _make_tool_msg("Paris is the capital of France\n\nFrance population is 67 million"),
            ],
            search_calls=2,
            question="What is the capital of France and how many people live there",
        )
        result = scorer.score(traj)
        assert result.total_bonus <= 0.2


# ===========================================================================
# scorer.score() 集成测试
# ===========================================================================


class TestScorerScore:
    """通过 scorer.score() 测试完整的特征提取 + 规则评估流水线。"""

    def test_happy_path_multi_turn(self) -> None:
        """正常的多轮搜索轨迹应触发多条 bonus，少量 penalty。"""
        scorer = PRMLiteScorer()
        traj = _make_traj(
            turns=[
                _make_turn(
                    text=_tool_text_with_prefix(
                        "Let me first understand the question about France",
                        "Paris capital France",
                    ),
                    completion_tokens=[1, 2, 3],
                ),
                _make_turn(
                    text=_tool_text_with_prefix(
                        "Now let me look up the population",
                        "France population 2024 census",
                    ),
                    completion_tokens=[4, 5],
                ),
            ],
            final_text="Answer: Paris is the capital of France with a population of 67 million",
            messages=[
                _make_tool_msg(
                    "Paris is the capital of France and largest city\n\n"
                    "France has approximately 67 million people as of 2024 census"
                ),
            ],
            search_calls=2,
            question="What is the capital of France and its population",
        )
        result = scorer.score(traj)
        assert "p4" not in result.penalties  # 有有效 Answer
        assert "b4" in result.bonuses  # 第一次查询与问题重叠
        assert "b6" in result.bonuses  # 正确格式
        assert result.total_bonus > 0

    def test_all_rules_populated(self) -> None:
        """all_rules 字典应包含所有启用的规则，未触发的值为 0.0。"""
        scorer = PRMLiteScorer()
        traj = _make_traj(
            turns=[_make_turn(text=_tool_text("Paris"))],
            final_text="Answer: Paris",
            messages=[_make_tool_msg("Paris is the capital of France")],
            search_calls=1,
            question="capital of France",
        )
        result = scorer.score(traj)
        for rule_id in ["p1", "p2", "p3", "p4", "b1", "b2", "b6"]:
            assert rule_id in result.all_rules, f"{rule_id} missing from all_rules"

    def test_disabled_rule_not_evaluated(self) -> None:
        """禁用规则不应出现在 all_rules 中（也不应被触发）。"""
        cfg = PRMLiteConfig(p7_question_query=False, b1_keyword_query=False)
        scorer = PRMLiteScorer(cfg)
        traj = _make_traj(
            turns=[_make_turn(text=_tool_text("what is Paris"))],
            final_text="Answer: Paris",
            question="capital of France",
        )
        result = scorer.score(traj)
        # 禁用规则不在 all_rules 中
        assert "p7" not in result.all_rules
        assert "b1" not in result.all_rules

    def test_default_scorer_reusable(self) -> None:
        """scorer.score() 多次调用各自独立，无状态污染。"""
        scorer = PRMLiteScorer()
        # 第一次：不好的轨迹
        bad = _make_traj(
            final_text="No answer",
            question="test",
        )
        r1 = scorer.score(bad)
        assert r1.total_penalty < 0

        # 第二次：好的轨迹
        good = _make_traj(
            turns=[_make_turn(text=_tool_text("Paris"))],
            final_text="Answer: Paris",
            messages=[_make_tool_msg("Paris capital France")],
            search_calls=1,
            question="France capital",
        )
        r2 = scorer.score(good)
        assert r2.total_bonus >= 0  # 不应被上一次的 penalty 污染


# ===========================================================================
# LATA 测试：长/短轨迹的 advantage 缩放
# ===========================================================================


class TestLATA:
    """LATA: Length-Adaptive Temporal Advantage 测试。"""

    def test_single_group_mean_centered(self) -> None:
        """同一组内 advantages 总和应为 0（去均值后）。"""
        trajectories = [
            _make_traj(
                reward=1.0,
                question_index=0,
                turns=[_make_turn(completion_tokens=[1])],
            ),
            _make_traj(
                reward=0.5,
                question_index=0,
                turns=[_make_turn(completion_tokens=[1])],
            ),
            _make_traj(
                reward=0.0,
                question_index=0,
                turns=[_make_turn(completion_tokens=[1])],
            ),
        ]
        LATAScaler.compute_advantages(trajectories)
        total_adv = sum(t.advantage for t in trajectories)
        # 每条轨迹的 L=1, 所以 advantage = (reward - mean) / sqrt(1) = reward - mean
        assert total_adv == pytest.approx(0.0, abs=1e-12)

    def test_long_trajectory_scaled_down(self) -> None:
        """长轨迹（更多 token）的 advantage 应被 sqrt(L) 缩小。"""
        short = _make_traj(
            reward=2.0,
            question_index=0,
            turns=[_make_turn(completion_tokens=[1])],  # L = 1
        )
        long = _make_traj(
            reward=2.0,
            question_index=0,
            turns=[_make_turn(completion_tokens=list(range(100)))],  # L = 100
        )
        trajectories = [short, long]
        LATAScaler.compute_advantages(trajectories)
        # 两者 reward 相同 → 去均值后 raw_adv 相同
        # short: raw_adv / sqrt(1), long: raw_adv / sqrt(100) = raw_adv / 10
        assert abs(short.advantage) == pytest.approx(abs(long.advantage) * 10.0, abs=1e-6)

    def test_zero_token_trajectory_unscaled(self) -> None:
        """L=0 时 advantage 不缩放（退化为 raw advantage）。"""
        t1 = _make_traj(
            reward=1.0,
            question_index=0,
            turns=[_make_turn(completion_tokens=[])],  # L = 0
        )
        t2 = _make_traj(
            reward=0.0,
            question_index=0,
            turns=[_make_turn(completion_tokens=[])],  # L = 0
        )
        trajectories = [t1, t2]
        LATAScaler.compute_advantages(trajectories)
        mean = (1.0 + 0.0) / 2
        assert t1.advantage == pytest.approx(1.0 - mean)
        assert t2.advantage == pytest.approx(0.0 - mean)

    def test_multiple_groups(self) -> None:
        """不同 question_index 的组应独立计算均值。"""
        g1 = [
            _make_traj(reward=3.0, question_index=0, turns=[_make_turn(completion_tokens=[1])]),
            _make_traj(reward=1.0, question_index=0, turns=[_make_turn(completion_tokens=[1])]),
        ]
        g2 = [
            _make_traj(reward=0.0, question_index=1, turns=[_make_turn(completion_tokens=[1])]),
        ]
        trajectories = g1 + g2
        LATAScaler.compute_advantages(trajectories)
        # g1 mean = 2.0; g1[0] adv = 1.0, g1[1] adv = -1.0
        # g2 mean = 0.0; g2[0] adv = 0.0
        assert g1[0].advantage == pytest.approx(1.0)
        assert g1[1].advantage == pytest.approx(-1.0)
        assert g2[0].advantage == pytest.approx(0.0)

    def test_short_vs_long_advantage_comparison(self) -> None:
        """相同 reward deviation 时，长轨迹 advantage 绝对值更小。"""
        short = _make_traj(
            reward=5.0,
            question_index=0,
            turns=[_make_turn(completion_tokens=[1])],  # L=1
        )
        long = _make_traj(
            reward=5.0,
            question_index=0,
            turns=[_make_turn(completion_tokens=list(range(25)))],  # L=25
        )
        baseline = _make_traj(
            reward=1.0,
            question_index=0,
            turns=[_make_turn(completion_tokens=[1])],
        )
        trajectories = [short, long, baseline]
        LATAScaler.compute_advantages(trajectories)
        # short L=1: adv = (5 - mean) / 1
        # long L=25: adv = (5 - mean) / 5
        # long 的 advantage 绝对值更小
        assert abs(short.advantage) > abs(long.advantage)


# ===========================================================================
# PRM-Lite + LATA 组合测试
# ===========================================================================


class TestPRMLiteAndLATACombined:
    """端到端组合：先评分 PRM-Lite，再应用 LATA 归一化。"""

    def test_combined_pipeline(self) -> None:
        """模拟真实训练流程：base reward + PRM-Lite -> LATA。"""
        # 构造一组轨迹（同一问题）
        good_traj = _make_traj(
            turns=[
                _make_turn(
                    text=_tool_text("Paris capital France"),
                    completion_tokens=[1, 2, 3, 4, 5],
                ),
                _make_turn(
                    text="Based on search results, Paris is the capital of France",
                    completion_tokens=[6, 7, 8, 9, 10],
                ),
            ],
            final_text="Answer: Paris",
            messages=[_make_tool_msg("Paris is the capital of France\n\nFrance population 67M")],
            search_calls=1,
            question="capital of France",
            reward=1.5,  # base answer reward
            question_index=0,
        )

        bad_traj = _make_traj(
            turns=[
                _make_turn(
                    text=_tool_text(""),
                    completion_tokens=[1, 2],
                ),
            ],
            final_text="No answer",
            messages=[],
            search_calls=1,
            question="capital of France",
            reward=0.0,
            question_index=0,
        )

        trajectories = [good_traj, bad_traj]

        # Step 1: 加上 PRM-Lite process reward
        for t in trajectories:
            pr = score_with_prm(t)
            t.reward += pr

        # good_traj 的 reward 应该比原始更高（bonus > penalty）
        assert good_traj.reward > 1.5
        # bad_traj 的 reward 应该比原始更低（penalty 主导）
        assert bad_traj.reward < 0.0

        # Step 2: 应用 LATA
        apply_lata(trajectories)

        # 验证 advantages 已设置
        for t in trajectories:
            assert hasattr(t, "advantage")
            assert isinstance(t.advantage, float)

        # 同组 advantage：good_traj 应高于 mean（正 advantage），
        # bad_traj 应低于 mean（负 advantage）
        # 注意：因为 sqrt(L) 缩放，不同长度的轨迹 advantage 总和不一定为零
        assert good_traj.advantage > 0  # reward 高于均值
        assert bad_traj.advantage < 0   # reward 低于均值


# ===========================================================================
# 便捷函数测试
# ===========================================================================


class TestConvenienceFunctions:
    def test_score_with_prm_returns_float(self) -> None:
        traj = _make_traj(
            turns=[_make_turn(text=_tool_text("Paris"))],
            final_text="Answer: Paris",
            messages=[_make_tool_msg("Paris capital")],
            search_calls=1,
            question="France capital",
        )
        reward = score_with_prm(traj)
        assert isinstance(reward, float)
        assert -0.2 <= reward <= 0.2

    def test_score_with_prm_custom_config(self) -> None:
        cfg = PRMLiteConfig(p1_empty_query=False, p2_long_query=False)
        traj = _make_traj(
            turns=[_make_turn(text=_tool_text("Paris"))],
            final_text="Answer: Paris",
            question="test",
        )
        reward = score_with_prm(traj, config=cfg)
        assert isinstance(reward, float)

    def test_apply_lata_sets_advantage(self) -> None:
        trajectories = [
            _make_traj(reward=1.0, question_index=0, turns=[_make_turn(completion_tokens=[1, 2])]),
        ]
        apply_lata(trajectories)
        assert trajectories[0].advantage == pytest.approx(0.0)  # single in group


# ===========================================================================
# 边界情况测试
# ===========================================================================


class TestEdgeCases:
    """空轨迹、单轮、多轮等极端输入。"""

    # -- 空轨迹 ---------------------------------------------------------------

    def test_empty_trajectory_no_turns(self) -> None:
        """没有任何 assistant turns 的轨迹。"""
        scorer = PRMLiteScorer()
        traj = _make_traj(
            final_text="Answer: Paris",
            question="test",
        )
        result = scorer.score(traj)
        # B6 fires: exactly one Answer: line
        assert result.total_bonus == pytest.approx(0.03)
        assert result.all_rules["b6"] == pytest.approx(0.03)
        # 检查 all_rules 可访问
        assert isinstance(result.all_rules, dict)

    def test_empty_trajectory_no_final_text(self) -> None:
        """final_text 为空，没有 Answer: 行。"""
        scorer = PRMLiteScorer()
        traj = _make_traj(question="test")
        result = scorer.score(traj)
        assert "p4" in result.penalties  # no Answer: → P4

    def test_empty_trajectory_no_messages(self) -> None:
        """messages 为空列表。"""
        scorer = PRMLiteScorer()
        traj = _make_traj(
            turns=[_make_turn(text=_tool_text("Paris"))],
            final_text="Answer: Paris",
            question="France capital",
        )
        result = scorer.score(traj)
        assert result is not None  # 不崩溃即可

    def test_empty_lata_group(self) -> None:
        """LATA: 空轨迹列表不应崩溃。"""
        # 空列表应正常通过（没有组需要处理）
        LATAScaler.compute_advantages([])

    # -- 单轮轨迹 -------------------------------------------------------------

    def test_single_turn_basic(self) -> None:
        """单轮搜索 + 回答的轨迹。"""
        scorer = PRMLiteScorer()
        traj = _make_traj(
            turns=[_make_turn(text=_tool_text("Paris population 2024"))],
            final_text="Answer: Paris has a population of 2.1 million in 2024",
            messages=[_make_tool_msg("Paris population 2024: approximately 2.1 million")],
            search_calls=1,
            question="What is the population of Paris in 2024",
        )
        result = scorer.score(traj)
        assert "p4" not in result.penalties
        assert result.total_bonus > 0

    def test_single_turn_no_search(self) -> None:
        """单轮，但没有搜索（直接回答）。"""
        scorer = PRMLiteScorer()
        traj = _make_traj(
            turns=[_make_turn(text="Answer: Paris")],
            final_text="Answer: Paris",
            search_calls=0,
            question="capital of France",
        )
        result = scorer.score(traj)
        # B6 应触发（正确格式），P4 不触发
        assert "p4" not in result.penalties
        assert result.all_rules["b6"] == pytest.approx(0.03)

    def test_single_turn_lata(self) -> None:
        """单条轨迹 LATA 应正常工作。"""
        traj = _make_traj(
            reward=1.0,
            question_index=0,
            turns=[_make_turn(completion_tokens=[1, 2, 3, 4])],
        )
        LATAScaler.compute_advantages([traj])
        assert traj.advantage > -1e-6  # 单条时平均即自身，advantage ≈ 0
        assert traj.advantage == pytest.approx(0.0, abs=1e-10)

    # -- 多轮轨迹 -------------------------------------------------------------

    def test_multi_turn_many_searches(self) -> None:
        """多轮搜索轨迹：触发多条规则。"""
        scorer = PRMLiteScorer()
        turns = [
            _make_turn(text=_tool_text("Paris")),
            _make_turn(text=_tool_text("Paris")),  # P3: duplicate
            _make_turn(text=_tool_text("what is the Eiffel Tower")),  # P7: question
            _make_turn(text=_tool_text("a" * 201)),  # P2: long query
        ]
        traj = _make_traj(
            turns=turns,
            final_text="Answer: Paris",
            messages=[_make_tool_msg("Paris is the capital of France")],
            search_calls=4,
            question="capital of France",
        )
        result = scorer.score(traj)
        assert result.total_penalty < 0  # 多条 penalty 触发
        assert result.total_bonus >= 0

    def test_multi_turn_lata_scaling(self) -> None:
        """多轮轨迹累加 token 数后 LATA 缩放。"""
        t1 = _make_traj(
            reward=3.0,
            question_index=0,
            turns=[
                _make_turn(completion_tokens=list(range(10))),   # 10 tokens
                _make_turn(completion_tokens=list(range(10))),   # 10 tokens
            ],  # total L = 20
        )
        t2 = _make_traj(
            reward=1.0,
            question_index=0,
            turns=[_make_turn(completion_tokens=[1])],  # L = 1
        )
        trajectories = [t1, t2]
        LATAScaler.compute_advantages(trajectories)
        mean = 2.0
        # t1: raw = 1.0, L=20, adv = 1.0 / sqrt(20) ≈ 0.2236
        # t2: raw = -1.0, L=1, adv = -1.0 / 1 = -1.0
        assert t1.advantage == pytest.approx(1.0 / math.sqrt(20), abs=1e-4)
        assert t2.advantage == pytest.approx(-1.0, abs=1e-6)

    def test_many_turns_exceeding_limit(self) -> None:
        """轮数超过 P5 上限。"""
        scorer = PRMLiteScorer()
        turns = [_make_turn(text=_tool_text(f"query {i}")) for i in range(10)]
        traj = _make_traj(
            turns=turns,
            final_text="Answer: Paris",
            messages=[_make_tool_msg("Paris capital")],
            search_calls=10,
            question="capital of France",
        )
        result = scorer.score(traj)
        assert "p5" in result.penalties

    def test_question_with_special_chars(self) -> None:
        """问题包含特殊字符不应导致崩溃。"""
        scorer = PRMLiteScorer()
        traj = _make_traj(
            turns=[_make_turn(text=_tool_text("test"))],
            final_text="Answer: 42",
            question="What's the answer to life, the universe & everything?",
        )
        result = scorer.score(traj)
        assert result is not None

    def test_answer_with_unicode(self) -> None:
        """非 ASCII 答案文本。"""
        scorer = PRMLiteScorer()
        traj = _make_traj(
            turns=[_make_turn(text=_tool_text("东京 人口"))],
            final_text="Answer: 东京の人口は約1400万人です",
            messages=[_make_tool_msg("東京都の人口推計 令和6年")],
            search_calls=1,
            question="东京人口",
        )
        result = scorer.score(traj)
        assert result is not None


# ===========================================================================
# scorere_weight_bounds 测试：process_reward 应始终在 [-0.2, +0.2]
# ===========================================================================


class TestRewardBounds:
    """验证 PRM-Lite process_reward 始终在配置的 caps 范围内。"""

    def test_within_bounds_normal(self) -> None:
        scorer = PRMLiteScorer()
        traj = _make_traj(
            turns=[_make_turn(text=_tool_text("test"))],
            final_text="Answer: test",
            question="question",
        )
        result = scorer.score(traj)
        assert -0.2 <= result.process_reward <= 0.2

    def test_within_bounds_extreme_penalties(self) -> None:
        """即使触发所有 penalty，process_reward 也不低于 -0.2。"""
        scorer = PRMLiteScorer()
        # 构造一条尽量触发多 penalty 的轨迹
        turns = [
            _make_turn(text=_tool_text("")),  # P1
            _make_turn(text=_tool_text("what is hello world today " * 20)),  # P2 + P7
            _make_turn(text=_tool_text("what is hello world today " * 20)),  # P3 duplicate of P2
        ] + [
            _make_turn(text=_tool_text("what is the Eiffel Tower history")) for _ in range(5)  # P5
        ]
        traj = _make_traj(
            turns=turns,
            final_text="No answer",  # P4
            messages=[_make_tool_msg("unrelated content xyz abc")],  # P8 + maybe P11
            search_calls=len(turns),
            question="hello",  # P12
        )
        result = scorer.score(traj)
        assert result.process_reward >= -0.2

    def test_within_bounds_extreme_bonuses(self) -> None:
        """即使触发所有 bonus，process_reward 也不超过 +0.2。"""
        scorer = PRMLiteScorer()
        traj = _make_traj(
            turns=[
                _make_turn(
                    text=_tool_text_with_prefix(
                        "Let me analyze the question about the history and details of France",
                        "France history revolution Napoleon",
                    ),
                    completion_tokens=[1, 2, 3],
                ),
                _make_turn(
                    text=_tool_text_with_prefix(
                        "Now checking different aspects",
                        "Paris tourism landmarks culture",
                    ),
                    completion_tokens=[4, 5],
                ),
            ],
            final_text="Answer: Paris is the capital of France with a rich cultural heritage",
            messages=[
                _make_tool_msg(
                    "Paris France capital history culture\n\n"
                    "France tourism landmarks Eiffel Tower Louvre\n\n"
                    "Napoleon Bonaparte French Revolution history"
                ),
            ],
            search_calls=2,
            question="What is the history and culture of Paris France including tourism landmarks",
        )
        result = scorer.score(traj)
        assert result.process_reward <= 0.2
