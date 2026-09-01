"""PRM-Lite process reward model and LATA advantage normalization.

PRM-Lite provides 12 penalty rules and 10 bonus rules as lightweight heuristics
for evaluating search-augmented reasoning trajectories.  LATA (Length-Adaptive
Temporal Advantage) normalizes group-relative advantages by sqrt(L) where L is
the total number of completion tokens, preventing systematic underestimation of
long reasoning chains.

Typical integration with train.py::

    from reward_lite import score_with_prm, apply_lata

    # After rollout computes base reward:
    for t in trajectories:
        t.reward = t.reward + score_with_prm(t)

    # Replace or supplement standard group advantages:
    apply_lata(trajectories)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

# Re-use the tool-call pattern from protocol.py instead of duplicating.
from protocol import TOOL_CALL_PATTERN


# ============================================================================
# Answer extraction (mirrors reward.score_answer / reward.extract_answer)
# ============================================================================

_ANSWER_PATTERN = re.compile(
    r"^\s*Answer:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE
)


def _extract_answer(text: str) -> str | None:
    """Return the single non-empty Answer: line, or None."""
    matches = _ANSWER_PATTERN.findall(text)
    if len(matches) != 1:
        return None
    answer = matches[0].strip()
    return answer or None


def _has_valid_answer(text: str) -> bool:
    return _extract_answer(text) is not None


# ============================================================================
# Shared text-analysis helpers
# ============================================================================

# Words that signal a natural-language question when they appear first.
_QUESTION_STARTS: frozenset[str] = frozenset(
    {
        "what", "who", "where", "when", "why", "how", "which", "whom",
        "whose", "is", "are", "was", "were", "do", "does", "did",
        "can", "could", "would", "should", "shall", "will",
        "have", "has", "had", "am", "may", "might", "must",
    }
)

# Very high-frequency English words excluded from word-overlap ratios.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
        "at", "to", "for", "and", "or", "but", "with", "by", "from",
        "it", "its", "be", "been", "being", "not", "no", "that", "this",
        "these", "those", "as", "has", "had", "have", "do", "does", "did",
    }
)


def _word_count(text: str) -> int:
    return len(text.strip().split())


def _is_question_sentence(query: str) -> bool:
    """True when *query* reads like a full English question, not keywords."""
    text = query.strip().lower()
    if not text:
        return False
    parts = text.split()
    if not parts:
        return False
    return parts[0] in _QUESTION_STARTS or "?" in text


def _word_overlap_ratio(text1: str, text2: str) -> float:
    """Jaccard-like word overlap (stopword-filtered, normalised by shorter set)."""
    w1 = set(text1.lower().split()) - _STOPWORDS
    w2 = set(text2.lower().split()) - _STOPWORDS
    if not w1 or not w2:
        return 0.0
    inter = w1 & w2
    return len(inter) / min(len(w1), len(w2))


def _common_substring_length(text1: str, text2: str,
                             min_len: int = 10) -> int:
    """Longest contiguous substring shared by *text1* and *text2* (>= *min_len*)."""
    t1, t2 = text1.lower(), text2.lower()
    # Search from the shorter string for efficiency.
    if len(t1) > len(t2):
        t1, t2 = t2, t1
    max_possible = min(len(t1), 100)
    for length in range(max_possible, min_len - 1, -1):
        seen: set[str] = {
            t1[i : i + length] for i in range(len(t1) - length + 1)
        }
        for j in range(len(t2) - length + 1):
            if t2[j : j + length] in seen:
                return length
    return 0


def _query_specificity(query: str) -> float:
    """Heuristic specificity score (0-1); higher means more focused."""
    words = query.strip().split()
    if not words:
        return 0.0
    length_score = min(len(words) / 10.0, 1.0)
    cap_score = sum(1 for w in words if w and w[0].isupper()) / len(words)
    return 0.5 * length_score + 0.5 * cap_score


# ============================================================================
# Trajectory-derived feature extractors
# ============================================================================

def _extract_queries_from_turns(turns: list) -> list[str]:
    """Return ordered list of search queries parsed from assistant turns."""
    queries: list[str] = []
    for turn in turns:
        text = getattr(turn, "text", None) or getattr(turn, "completion_text", "")
        match = TOOL_CALL_PATTERN.search(text)
        if match:
            q = match.group(1).strip()
            if q:
                queries.append(q)
    return queries


def _extract_search_result_blocks(messages: list) -> list[str]:
    """Split tool-message content into per-result blocks.

    Individual results are joined by ``\\n\\n`` (see :func:`rollout.fit_tool_content`).
    """
    blocks: list[str] = []
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if content:
                blocks.extend(p for p in content.split("\n\n") if p.strip())
    return blocks


def _extract_search_content(messages: list) -> str:
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if content:
                parts.append(content)
    return "\n".join(parts)


def _trajectory_completion_token_count(trajectory: Any) -> int:
    """Sum of ``len(turn.completion_tokens)`` across all turns."""
    total = 0
    for turn in getattr(trajectory, "turns", []):
        tokens = getattr(turn, "completion_tokens", [])
        total += len(tokens)
    return total


# ============================================================================
# "Simple fact" detection for P12
# ============================================================================

_SIMPLE_FACT_PATTERNS: list[re.Pattern] = [
    # Arithmetic
    re.compile(r"^what\s+is\s+\d+\s*[\+\-\*\\/]\s*\d+", re.IGNORECASE),
    # Chitchat
    re.compile(r"^(what\s+is\s+your\s+name|how\s+are\s+you)", re.IGNORECASE),
    re.compile(r"^what\s+(day|time)\s+is\s+it", re.IGNORECASE),
    re.compile(r"^(say\s+hello|repeat\s+after\s+me)", re.IGNORECASE),
    # Trivially short
    re.compile(r"^(hi|hello|hey|ok|thanks|bye|yes|no)[\s!.]*$", re.IGNORECASE),
]


def _is_trivially_simple(question: str) -> bool:
    text = question.strip().lower()
    for pat in _SIMPLE_FACT_PATTERNS:
        if pat.search(text):
            return True
    if _word_count(text) <= 2:
        return True
    return False


# ============================================================================
# Multi-hop detection for B7
# ============================================================================

_MULTI_HOP_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\band\b.*\b(what|who|which|when|where|how)\b", re.IGNORECASE
    ),
    re.compile(r"\b(compare|difference|versus|vs\.?)\b", re.IGNORECASE),
    re.compile(r"\b(both|two|multiple|several)\b", re.IGNORECASE),
    re.compile(
        r"\b(before|after|following|preceding|earlier|later)\b", re.IGNORECASE
    ),
    re.compile(r"\b(first|then|next|finally)\b", re.IGNORECASE),
    re.compile(
        r"\b(author|founder|creator|inventor|discoverer|president|ceo)\b"
        r".*\b(born|from|in|of|country|nation)\b",
        re.IGNORECASE,
    ),
    re.compile(r",\s*.*\b(and|or)\b", re.IGNORECASE),
]


def _is_multi_hop(question: str) -> bool:
    for pat in _MULTI_HOP_PATTERNS:
        if pat.search(question):
            return True
    if _word_count(question) > 15:
        return True
    return False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PRMLiteConfig:
    """Per-rule switches and thresholds for PRM-Lite.

    Set any ``p*`` / ``b*`` field to ``False`` to disable that rule.
    """

    # -- Penalty switches ----------------------------------------------------
    p1_empty_query: bool = True          # empty / whitespace-only search query
    p2_long_query: bool = True           # query > *long_query_threshold* chars
    p3_duplicate_query: bool = True      # consecutive identical queries
    p4_invalid_answer: bool = True       # no valid ``Answer:`` line
    p5_excessive_turns: bool = True      # turns > *max_assistant_turns*
    p6_short_answer: bool = True         # extracted answer < *short_answer_threshold*
    p7_question_query: bool = True       # query is full question, not keywords
    p8_no_reference: bool = True         # search done but answer ignores results
    p9_ignore_results: bool = True       # multiple results but only first used
    p10_copy_paste: bool = True          # verbatim copy from search results
    p11_irrelevant_query: bool = True    # query unrelated to question
    p12_unnecessary_search: bool = True  # search on trivially simple facts

    # -- Bonus switches ------------------------------------------------------
    b1_keyword_query: bool = True        # concise keyword query (<=5 words)
    b2_multi_reference: bool = True      # synthesises >=2 result blocks
    b3_think_integration: bool = True    # think/integration text around searches
    b4_first_search_hit: bool = True     # first query overlaps question strongly
    b5_concise_answer: bool = True       # answer length in [20, 100] chars
    b6_correct_format: bool = True       # exactly one non-empty ``Answer:`` line
    b7_reasonable_rounds: bool = True    # search calls match question complexity
    b8_own_words: bool = True            # answer summarises, not copies
    b9_diverse_keywords: bool = True     # queries use different keyword strategies
    b10_gradual_focus: bool = True       # queries narrow from broad to focused

    # -- Global caps ---------------------------------------------------------
    max_penalty: float = -0.2
    max_bonus: float = 0.2

    # -- Thresholds ----------------------------------------------------------
    max_assistant_turns: int = 6
    long_query_threshold: int = 200
    short_answer_threshold: int = 5
    concise_answer_min: int = 20
    concise_answer_max: int = 100
    copy_paste_min_overlap: int = 50
    irrelevant_overlap_threshold: float = 0.1
    max_keyword_query_words: int = 5


# ============================================================================
# Scoring result
# ============================================================================

@dataclass
class PRMLiteResult:
    """Container returned by :meth:`PRMLiteScorer.score`."""

    penalties: dict[str, float]   # rule_id -> negative value  (fired rules only)
    bonuses: dict[str, float]     # rule_id -> positive value  (fired rules only)
    total_penalty: float          # sum of penalties, capped
    total_bonus: float            # sum of bonuses,   capped
    process_reward: float         # total_penalty + total_bonus
    all_rules: dict[str, float]   # every evaluated rule (zero when not fired)


# ============================================================================
# PRM-Lite Scorer
# ============================================================================

class PRMLiteScorer:
    """Apply the 12 penalty and 10 bonus heuristics to one trajectory.

    Typical usage::

        scorer = PRMLiteScorer()
        result = scorer.score(trajectory)
        combined = trajectory.reward + result.process_reward
    """

    def __init__(self, config: PRMLiteConfig | None = None) -> None:
        self.config = config or PRMLiteConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, trajectory: Any) -> PRMLiteResult:
        """Evaluate all enabled rules against *trajectory*.

        *trajectory* is duck-typed and must expose at minimum:
          - ``turns``: list of objects with a ``text`` attribute
          - ``final_text``: str
          - ``messages``: list of dicts (``role`` / ``content``)
          - ``search_calls``: int
          - ``example.question``: str
        """
        cf = self.config

        # Feature extraction ------------------------------------------------
        queries = _extract_queries_from_turns(getattr(trajectory, "turns", []))
        answer = _extract_answer(getattr(trajectory, "final_text", ""))
        question = getattr(
            getattr(trajectory, "example", None), "question", ""
        )
        messages = getattr(trajectory, "messages", [])
        search_blocks = _extract_search_result_blocks(messages)
        search_content = _extract_search_content(messages)
        has_search = len(search_blocks) > 0
        turn_count = len(getattr(trajectory, "turns", []))
        search_calls = getattr(trajectory, "search_calls", 0)

        # Rule evaluation ---------------------------------------------------
        all_rules: dict[str, float] = {}
        penalties: dict[str, float] = {}
        bonuses: dict[str, float] = {}

        # Penalties -------------------------------------------------------
        for rule_id, enabled, fn in [
            ("p1", cf.p1_empty_query, lambda: self._check_p1_empty_query(queries)),
            ("p2", cf.p2_long_query, lambda: self._check_p2_long_query(queries)),
            ("p3", cf.p3_duplicate_query, lambda: self._check_p3_duplicate_query(queries)),
            ("p4", cf.p4_invalid_answer, lambda: self._check_p4_invalid_answer(answer)),
            ("p5", cf.p5_excessive_turns, lambda: self._check_p5_excessive_turns(turn_count)),
            ("p6", cf.p6_short_answer, lambda: self._check_p6_short_answer(answer)),
            ("p7", cf.p7_question_query, lambda: self._check_p7_question_query(queries)),
            ("p8", cf.p8_no_reference, lambda: self._check_p8_no_reference(has_search, answer, search_content)),
            ("p9", cf.p9_ignore_results, lambda: self._check_p9_ignore_results(search_blocks, answer)),
            ("p10", cf.p10_copy_paste, lambda: self._check_p10_copy_paste(answer, search_content)),
            ("p11", cf.p11_irrelevant_query, lambda: self._check_p11_irrelevant_query(queries, question)),
            ("p12", cf.p12_unnecessary_search, lambda: self._check_p12_unnecessary_search(search_calls, question)),
        ]:
            if enabled:
                val = fn()
                all_rules[rule_id] = val
                if val != 0.0:
                    penalties[rule_id] = val

        # Bonuses ---------------------------------------------------------
        for rule_id, enabled, fn in [
            ("b1", cf.b1_keyword_query, lambda: self._check_b1_keyword_query(queries)),
            ("b2", cf.b2_multi_reference, lambda: self._check_b2_multi_reference(search_blocks, answer)),
            ("b3", cf.b3_think_integration, lambda: self._check_b3_think_integration(trajectory)),
            ("b4", cf.b4_first_search_hit, lambda: self._check_b4_first_search_hit(queries, question)),
            ("b5", cf.b5_concise_answer, lambda: self._check_b5_concise_answer(answer)),
            ("b6", cf.b6_correct_format, lambda: self._check_b6_correct_format(answer, trajectory)),
            ("b7", cf.b7_reasonable_rounds, lambda: self._check_b7_reasonable_rounds(search_calls, question)),
            ("b8", cf.b8_own_words, lambda: self._check_b8_own_words(answer, search_content)),
            ("b9", cf.b9_diverse_keywords, lambda: self._check_b9_diverse_keywords(queries)),
            ("b10", cf.b10_gradual_focus, lambda: self._check_b10_gradual_focus(queries)),
        ]:
            if enabled:
                val = fn()
                all_rules[rule_id] = val
                if val != 0.0:
                    bonuses[rule_id] = val

        # -- Mutual-exclusion adjustments ------------------------------------
        # P4 (invalid answer) and B6 (correct format) are mutually exclusive.
        if "p4" in penalties and "b6" in bonuses:
            if abs(penalties["p4"]) >= abs(bonuses["b6"]):
                del bonuses["b6"]
                all_rules["b6"] = 0.0
            else:
                del penalties["p4"]
                all_rules["p4"] = 0.0

        # P10 (copy-paste) and B8 (own words) are opposing.
        if "p10" in penalties:
            bonuses.pop("b8", None)
            all_rules["b8"] = 0.0

        # -- Caps ------------------------------------------------------------
        total_penalty = (
            max(sum(penalties.values()), cf.max_penalty)
            if penalties
            else 0.0
        )
        total_bonus = (
            min(sum(bonuses.values()), cf.max_bonus) if bonuses else 0.0
        )

        return PRMLiteResult(
            penalties=penalties,
            bonuses=bonuses,
            total_penalty=total_penalty,
            total_bonus=total_bonus,
            process_reward=total_penalty + total_bonus,
            all_rules=all_rules,
        )

    # ==================================================================
    # Penalty rule methods  (return 0.0 or negative float)
    # ==================================================================

    def _check_p1_empty_query(self, queries: list[str]) -> float:
        """P1: search query is empty or whitespace-only."""
        if not queries:
            return 0.0
        for q in queries:
            if not q.strip():
                return -0.05
        return 0.0

    def _check_p2_long_query(self, queries: list[str]) -> float:
        """P2: search query longer than threshold characters."""
        for q in queries:
            if len(q) > self.config.long_query_threshold:
                return -0.03
        return 0.0

    def _check_p3_duplicate_query(self, queries: list[str]) -> float:
        """P3: same query appears consecutively (>=2 times)."""
        if len(queries) < 2:
            return 0.0
        for i in range(len(queries) - 1):
            if queries[i].strip().lower() == queries[i + 1].strip().lower():
                return -0.05
        return 0.0

    def _check_p4_invalid_answer(self, answer: str | None) -> float:
        """P4: no valid ``Answer:`` line in final text."""
        if answer is None:
            return -0.1
        return 0.0

    def _check_p5_excessive_turns(self, turn_count: int) -> float:
        """P5: assistant turns exceed configured maximum."""
        if turn_count > self.config.max_assistant_turns:
            return -0.05
        return 0.0

    def _check_p6_short_answer(self, answer: str | None) -> float:
        """P6: extracted answer shorter than threshold."""
        if answer is not None and len(answer.strip()) < self.config.short_answer_threshold:
            return -0.03
        return 0.0

    def _check_p7_question_query(self, queries: list[str]) -> float:
        """P7: query reads like a complete question sentence."""
        for q in queries:
            if _is_question_sentence(q):
                return -0.02
        return 0.0

    def _check_p8_no_reference(
        self,
        has_search: bool,
        answer: str | None,
        search_content: str,
    ) -> float:
        """P8: search was performed but answer has no content overlap with results."""
        if not has_search or not answer or not search_content.strip():
            return 0.0
        if _word_overlap_ratio(answer, search_content) < 0.05:
            return -0.05
        return 0.0

    def _check_p9_ignore_results(
        self,
        search_blocks: list[str],
        answer: str | None,
    ) -> float:
        """P9: multiple search results but answer only references the first."""
        if len(search_blocks) < 2 or not answer:
            return 0.0
        first = search_blocks[0]
        others = " ".join(search_blocks[1:])
        overlap_first = _word_overlap_ratio(answer, first)
        overlap_others = _word_overlap_ratio(answer, others)
        if overlap_first > 0.1 and overlap_others < 0.02:
            return -0.03
        return 0.0

    def _check_p10_copy_paste(
        self,
        answer: str | None,
        search_content: str,
    ) -> float:
        """P10: answer contains a long verbatim substring from search results."""
        if not answer or not search_content:
            return 0.0
        longest = _common_substring_length(
            answer, search_content,
            min_len=self.config.copy_paste_min_overlap,
        )
        if longest >= self.config.copy_paste_min_overlap:
            return -0.03
        return 0.0

    def _check_p11_irrelevant_query(
        self,
        queries: list[str],
        question: str,
    ) -> float:
        """P11: query has essentially no word overlap with the question."""
        if not queries or not question:
            return 0.0
        for q in queries:
            if _word_overlap_ratio(q, question) < self.config.irrelevant_overlap_threshold:
                return -0.05
        return 0.0

    def _check_p12_unnecessary_search(
        self, search_calls: int, question: str
    ) -> float:
        """P12: search was called on a trivially simple fact."""
        if search_calls > 0 and _is_trivially_simple(question):
            return -0.02
        return 0.0

    # ==================================================================
    # Bonus rule methods  (return 0.0 or positive float)
    # ==================================================================

    def _check_b1_keyword_query(self, queries: list[str]) -> float:
        """B1: at least one query is <=5 words and not a question sentence."""
        if not queries:
            return 0.0
        for q in queries:
            if (
                _word_count(q) <= self.config.max_keyword_query_words
                and not _is_question_sentence(q)
            ):
                return 0.03
        return 0.0

    def _check_b2_multi_reference(
        self,
        search_blocks: list[str],
        answer: str | None,
    ) -> float:
        """B2: answer references at least two distinct search result blocks."""
        if len(search_blocks) < 2 or not answer:
            return 0.0
        refs = sum(
            1 for b in search_blocks if _word_overlap_ratio(answer, b) > 0.05
        )
        if refs >= 2:
            return 0.05
        return 0.0

    def _check_b3_think_integration(self, trajectory: Any) -> float:
        """B3: at least one turn contains substantial reasoning text around searches."""
        turns = getattr(trajectory, "turns", [])
        for turn in turns:
            text = getattr(turn, "text", None) or getattr(turn, "completion_text", "")
            match = TOOL_CALL_PATTERN.search(text)
            if match:
                prefix = text[: match.start()].strip()
                if len(prefix) >= 20:
                    return 0.02
            else:
                if len(text.strip()) >= 20 and not _has_valid_answer(text):
                    return 0.02
        return 0.0

    def _check_b4_first_search_hit(
        self, queries: list[str], question: str
    ) -> float:
        """B4: first search query shares substantial words with the question."""
        if not queries or not question:
            return 0.0
        if _word_overlap_ratio(queries[0], question) > 0.3:
            return 0.05
        return 0.0

    def _check_b5_concise_answer(self, answer: str | None) -> float:
        """B5: answer length within [20, 100] characters."""
        if answer is None:
            return 0.0
        length = len(answer.strip())
        if self.config.concise_answer_min <= length <= self.config.concise_answer_max:
            return 0.03
        return 0.0

    def _check_b6_correct_format(
        self, answer: str | None, trajectory: Any
    ) -> float:
        """B6: exactly one non-empty ``Answer:`` line."""
        if answer is None:
            return 0.0
        text = getattr(trajectory, "final_text", "")
        matches = _ANSWER_PATTERN.findall(text)
        if len(matches) == 1 and matches[0].strip():
            return 0.03
        return 0.0

    def _check_b7_reasonable_rounds(
        self, search_calls: int, question: str
    ) -> float:
        """B7: search round count matches question complexity.

        Simple question: 1-2 rounds; multi-hop: 2-4 rounds.
        """
        if search_calls == 0:
            return 0.0
        multi_hop = _is_multi_hop(question)
        if multi_hop and 2 <= search_calls <= 4:
            return 0.03
        if not multi_hop and 1 <= search_calls <= 2:
            return 0.03
        return 0.0

    def _check_b8_own_words(
        self, answer: str | None, search_content: str
    ) -> float:
        """B8: answer uses own words; no long copy from search results."""
        if not answer or not search_content:
            return 0.0
        longest = _common_substring_length(answer, search_content, min_len=30)
        if longest < 30:
            return 0.03
        return 0.0

    def _check_b9_diverse_keywords(self, queries: list[str]) -> float:
        """B9: consecutive queries use notably different keywords (overlap < 0.5)."""
        if len(queries) < 2:
            return 0.0
        for i in range(len(queries) - 1):
            if _word_overlap_ratio(queries[i], queries[i + 1]) < 0.5:
                return 0.04
        return 0.0

    def _check_b10_gradual_focus(self, queries: list[str]) -> float:
        """B10: query specificity monotonically increases across turns."""
        if len(queries) < 2:
            return 0.0
        scores = [_query_specificity(q) for q in queries]
        increases = sum(
            1 for i in range(len(scores) - 1) if scores[i + 1] > scores[i]
        )
        if increases >= len(scores) // 2:
            return 0.03
        return 0.0


# ============================================================================
# LATA: Length-Adaptive Temporal Advantage
# ============================================================================

class LATAScaler:
    """Length-Adaptive Advantage normalisation.

    Formula::

        advantage_lata = (reward - group_mean) / sqrt(L)

    where *L* is the total number of completion tokens across all assistant
    turns in the trajectory.  Dividing by sqrt(L) protects long reasoning
    chains from systematic underestimation while leaving short trajectories
    mostly unchanged.
    """

    @staticmethod
    def compute_advantages(trajectories: list) -> None:
        """Compute LATA advantages in-place on every trajectory.

        Trajectories are grouped by ``question_index`` (same question = same
        group).  Each trajectory's ``advantage`` field is overwritten with the
        LATA-normalised value.

        Raises:
            ValueError: if any group has zero trajectories.
        """
        groups: dict[int, list] = {}
        for t in trajectories:
            qi = getattr(t, "question_index", 0)
            groups.setdefault(qi, []).append(t)

        for group in groups.values():
            if not group:
                raise ValueError("Empty trajectory group encountered")
            mean_reward = sum(
                getattr(t, "reward", 0.0) for t in group
            ) / len(group)

            for t in group:
                raw_adv = getattr(t, "reward", 0.0) - mean_reward
                L = _trajectory_completion_token_count(t)
                if L > 0:
                    t.advantage = raw_adv / math.sqrt(L)
                else:
                    t.advantage = raw_adv


# ============================================================================
# Convenience wrappers for train.py
# ============================================================================

_DEFAULT_SCORER: PRMLiteScorer = PRMLiteScorer()


def score_with_prm(
    trajectory: Any,
    config: PRMLiteConfig | None = None,
) -> float:
    """Score one trajectory with PRM-Lite and return the process reward.

    Intended to be called after the base answer reward has already been
    computed::

        from reward_lite import score_with_prm

        trajectory.reward = score_answer(...).reward
        trajectory.reward += score_with_prm(trajectory)

    Args:
        trajectory: A :class:`Trajectory` (or duck-typed equivalent).
        config: Optional :class:`PRMLiteConfig` override.

    Returns:
        Process reward in ``[-0.2, +0.2]``.
    """
    if config is not None:
        scorer = PRMLiteScorer(config)
        result = scorer.score(trajectory)
    else:
        result = _DEFAULT_SCORER.score(trajectory)
    return result.process_reward


def apply_lata(trajectories: list) -> None:
    """Apply LATA advantage normalisation to *trajectories* in-place.

    Can be used either as a drop-in replacement for
    :func:`rollout.assign_group_advantages`, or called after it to
    re-normalise::

        from reward_lite import apply_lata

        apply_lata(trajectories)          # standalone
        # or:
        assign_group_advantages(trajectories)
        apply_lata(trajectories)          # post-normalise

    Args:
        trajectories: List of :class:`Trajectory` objects.
    """
    LATAScaler.compute_advantages(trajectories)
