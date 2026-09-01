"""Search-R1 Agent Loop for veRL.

Implements a multi-turn search-augmented agent loop compatible with
veRL's ``AgentLoopBase`` interface.  Each turn the LLM can either:
  1. output a search query (parsed via <tool_call> tags)
  2. output a final answer (parsed via ``Answer:`` line)

The agent loop handles search execution, formats results, and feeds them
back into the conversation.  Token-in-Token-out (TITO) protocol is followed
to ensure veRL's PPO/GRPO trainer can compute correct token-level advantages.

Verified against veRL AgentLoopBase API (trainer_config, server_manager,
tokenizer, processor, **kwargs).

Integration:
    actor_rollout_ref.rollout.agent.default_agent_loop=search_r1_agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path=path/to/agent_config.yaml
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# ── veRL imports (available inside a veRL environment) ──
try:
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopOutput,
        register,
    )
    from verl.utils.profiler import simple_timer
except ImportError:
    # Allow this module to be imported for inspection outside veRL.
    AgentLoopBase = object  # type: ignore[assignment,misc]
    AgentLoopOutput = None  # type: ignore[assignment,misc]
    register = lambda name: lambda cls: cls  # type: ignore[no-untyped-def]  # noqa: E731
    simple_timer = lambda name, metrics: __import__("contextlib").nullcontext()  # type: ignore[no-untyped-def]  # noqa: E731

# ── Search-R1 shared modules ──
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
import sys

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from protocol import (
    SEARCH_TOOL,
    SYSTEM_PROMPT,
    parse_assistant,
)
from search import (
    SearchClient,
    SearchResult,
    create_search_client,
    format_item,
)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

MAX_SEARCH_CALLS = 2        # 从4降到2，减少搜索延迟影响
MAX_ASSISTANT_TURNS = 4      # 从6降到4，配合搜索轮次减少
MAX_ASSISTANT_TOKENS = 1024
MAX_TOOL_RESPONSE_CHARS = 2048


@dataclass
class SearchR1Config:
    """Agent-loop-specific configuration (populated from YAML)."""

    search_backend: str = "deepseek"
    search_model: str = "deepseek-v4-flash"
    search_timeout: float = 60.0
    max_search_calls: int = MAX_SEARCH_CALLS
    max_assistant_turns: int = MAX_ASSISTANT_TURNS
    max_assistant_tokens: int = MAX_ASSISTANT_TOKENS


# ═══════════════════════════════════════════════════════════════════════════
# Agent loop
# ═══════════════════════════════════════════════════════════════════════════


@register("search_r1_agent")
class SearchR1AgentLoop(AgentLoopBase):
    """Multi-turn search agent loop for Search-R1 GRPO training.

    Protocol (same as ``train_grpo_local.py``):
        - System prompt defines the search tool and answer format.
        - Assistant outputs parsed for ``<tool_call>`` or ``Answer:``.
        - Search results wrapped in ``<tool_response>`` as user message.
        - Terminates on answer or exceeding turn/search limits.

    Class-level singletons:
        _shared_client / _shared_cache: 所有 sample 共享同一个搜索客户端和
        缓存，避免 64 个实例各自创建连接 + 重复查询 + 无速率限制。
    """

    # 类级别单例：所有 agent loop 实例共享搜索客户端和缓存
    _shared_client: SearchClient | None = None
    _shared_cache: dict[str, Any] = {}
    _shared_config_key: str = ""
    _cache_hits: int = 0
    _cache_misses: int = 0

    def __init__(
        self,
        trainer_config: Any,
        server_manager: Any,
        tokenizer: Any,
        processor: Any,
        dataset_cls: Any = None,
        data_config: Any = None,
        hf_model_type: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialise with veRL's injected dependencies.

        Args:
            trainer_config: ``DictConfigWrap`` wrapping the full veRL config tree.
            server_manager: ``AsyncLLMServerManager`` for LLM inference.
            tokenizer: HuggingFace ``AutoTokenizer``.
            processor: HuggingFace ``AutoProcessor`` (unused — text-only agent).
            dataset_cls/data_config: veRL main 0.10 makes these required args on
                ``AgentLoopBase.__init__`` — accepted and forwarded.
            **kwargs: agent-loop fields from agent_config.yaml (search_backend, …).
        """
        if AgentLoopBase is object:
            # Inspection-only fallback (veRL not installed): stash what we use.
            super().__init__()
            self.config = trainer_config
            self.rollout_config = getattr(
                getattr(trainer_config, "actor_rollout_ref", None), "rollout", None
            )
            self.tokenizer = tokenizer
            self.server_manager = server_manager
        else:
            super().__init__(
                trainer_config,
                server_manager,
                tokenizer,
                processor,
                dataset_cls,
                data_config,
                hf_model_type=hf_model_type,
                **kwargs,
            )
        # Older veRL set self.loop in the base class; main 0.10 may not.
        if not hasattr(self, "loop"):
            self.loop = asyncio.get_event_loop()

        # Extract agent-loop config. veRL main (0.10+) registers agent loops from
        # `agent_loop_config_path` (a YAML list of {name, _target_, ...}); the entry's
        # extra fields arrive here as __init__ kwargs via hydra.utils.instantiate.
        # Fall back to the config-tree node (probe path) and then defaults.
        agent_node = getattr(self.config.actor_rollout_ref.rollout, "agent", None) or {}
        sr1 = getattr(agent_node, "search_r1", None) or {}

        def _cfg(key: str, default: Any) -> Any:
            value = kwargs.get(key)
            if value is not None:
                return value
            value = getattr(sr1, key, None)
            return default if value is None else value

        self.search_config = SearchR1Config(
            search_backend=_cfg("search_backend", "deepseek"),
            search_model=_cfg("search_model", "deepseek-v4-flash"),
            search_timeout=float(_cfg("search_timeout", 60.0)),
            max_search_calls=int(_cfg("max_search_calls", MAX_SEARCH_CALLS)),
            max_assistant_turns=int(_cfg("max_assistant_turns", MAX_ASSISTANT_TURNS)),
            max_assistant_tokens=int(_cfg("max_assistant_tokens", MAX_ASSISTANT_TOKENS)),
        )

        # 搜索客户端和缓存使用类级别单例，所有 sample 共享。
        # 避免 64 个实例各自创建连接 + 缓存不共享 + 无速率限制。
        config_key = (
            f"{self.search_config.search_backend}:"
            f"{self.search_config.search_model}:"
            f"{self.search_config.search_timeout}"
        )
        if SearchR1AgentLoop._shared_client is None or SearchR1AgentLoop._shared_config_key != config_key:
            env_path = _SCRIPT_DIR / ".env"
            SearchR1AgentLoop._shared_client = create_search_client(
                self.search_config.search_backend,
                str(env_path) if env_path.exists() else None,
                model=self.search_config.search_model,
                timeout=self.search_config.search_timeout,
            )
            SearchR1AgentLoop._shared_cache = {}
            SearchR1AgentLoop._shared_config_key = config_key
            # 运行时缓存：同 step 内相似搜索 query 自然命中，无需预生成
        self._search_client = SearchR1AgentLoop._shared_client
        self._search_cache = SearchR1AgentLoop._shared_cache

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        sampling_params: dict[str, Any],
        priority: int = 0,
        **kwargs: Any,
    ) -> "AgentLoopOutput":  # type: ignore[valid-type]
        """Execute the Search-R1 agent loop for a single prompt.

        Args:
            sampling_params: LLM sampling parameters (temperature, top_p, …).
            priority: veRL main 0.10 passes a per-sample priority; unused here.
            **kwargs: Must contain ``raw_prompt`` (list of role/content messages).

        Returns:
            AgentLoopOutput with prompt_ids, response_ids, and response_mask.
        """
        metrics: dict[str, float] = {}
        messages = list(kwargs["raw_prompt"])

        # Prepend system prompt with tool definition
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        # Tokenize prompt via chat template (with tool definitions)
        # veRL base class does NOT provide apply_chat_template — use tokenizer directly.
        prompt_ids = await self.loop.run_in_executor(
            None,
            self._apply_chat_template,
            messages,
        )
        # Normalise: some tokenizers return nested lists or tensors.
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        prompt_ids = [int(t) for t in prompt_ids]

        all_response_ids: list[int] = []
        all_response_mask: list[int] = []
        search_calls = 0
        final_answer_text: str | None = None
        # Engine limits: the multi-turn context grows each turn (tool tokens become
        # part of the next call's prompt), so it must stay inside both vLLM's
        # max_model_len and the response budget — otherwise requests fail mid-rollout.
        response_length = int(getattr(self.rollout_config, "response_length", 4096) or 4096)
        max_model_len = int(getattr(self.rollout_config, "max_model_len", 6144) or 6144)

        # ── Multi-turn loop ──
        for _turn_idx in range(self.search_config.max_assistant_turns):
            current_prompt = prompt_ids + all_response_ids
            budget = min(
                max_model_len - len(current_prompt) - 16,
                response_length - len(all_response_ids) - 16,
            )
            if budget < 64:
                break  # 上下文已满，无法再生成有效回合
            max_tokens = min(self.search_config.max_assistant_tokens, budget)

            # Generate one assistant turn
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=current_prompt,
                    sampling_params={
                        **sampling_params,
                        "max_tokens": max_tokens,
                    },
                )

            completion_ids = list(output.token_ids)
            completion_text = self._decode_safe(completion_ids)
            parsed = parse_assistant(completion_text)

            # LLM-generated tokens → mask = 1 (trained)
            all_response_ids.extend(completion_ids)
            all_response_mask.extend([1] * len(completion_ids))

            if parsed.kind == "tool" and parsed.query:
                if search_calls >= self.search_config.max_search_calls:
                    break
                search_calls += 1

                # Execute search (sync → async via thread)，带缓存
                with simple_timer("search_execute", metrics):
                    cache_key = parsed.query.strip().lower()
                    if cache_key in self._search_cache:
                        result = self._search_cache[cache_key]
                        SearchR1AgentLoop._cache_hits += 1
                    else:
                        SearchR1AgentLoop._cache_misses += 1
                        result = await asyncio.to_thread(
                            self._search_client.search, parsed.query
                        )
                        self._search_cache[cache_key] = result
                        # 每100次 miss 打印一次命中率
                        total = SearchR1AgentLoop._cache_hits + SearchR1AgentLoop._cache_misses
                        if total % 100 == 0:
                            hit_rate = SearchR1AgentLoop._cache_hits / total * 100
                            logger.info(
                                "Search cache: %d hits / %d misses (%.1f%% hit rate), cache_size=%d",
                                SearchR1AgentLoop._cache_hits,
                                SearchR1AgentLoop._cache_misses,
                                hit_rate,
                                len(self._search_cache),
                            )

                tool_text = self._format_search_result(result)
                tool_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.encode(tool_text, add_special_tokens=False),
                )
                if hasattr(tool_ids, "tolist"):
                    tool_ids = tool_ids.tolist()
                tool_ids = [int(t) for t in tool_ids]

                # Tool response tokens → mask = 0 (not trained)
                all_response_ids.extend(tool_ids)
                all_response_mask.extend([0] * len(tool_ids))

            elif parsed.kind == "answer":
                final_answer_text = parsed.content or completion_text
                break  # valid final answer
            else:
                break  # invalid output (counts as anomalous in reward)

            # Tool tokens just added; stop if the response budget is exhausted
            # (the final answer turn must still fit inside response_length).
            if len(all_response_ids) >= response_length - 64:
                break

        # Build output. veRL main 0.10: AgentLoopOutput is a pydantic model —
        # `metrics` takes a dict (coerced to AgentLoopMetrics), and everything the
        # reward function needs beyond the dataset columns travels in
        # `extra_fields` (the reward worker exposes it as
        # non_tensor_batch["tool_extra_fields"]).
        extra_fields: dict[str, Any] = {
            "response_text": final_answer_text or "",
            "search_calls": search_calls,
        }
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=all_response_ids,
            response_mask=all_response_mask,
            response_logprobs=None,
            multi_modal_data={},
            num_turns=search_calls + 1,
            metrics=metrics,
            extra_fields=extra_fields,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> list[int]:
        """Render + tokenize the chat with tool definitions, returning ``list[int]``.

        Uses ``tokenize=False`` (always returns a ``str`` — stable across
        transformers 4.x/5.x) and encodes manually, the same pattern as veRL's
        own vLLM rollout.  The ``tokenize=True`` path on transformers 5.10
        returns a dict-like object that is not a ``dict`` subclass, so the
        previous ``isinstance(out, dict)`` normalisation silently iterated its
        key names — we avoid that path entirely.  ``enable_thinking`` is a
        Qwen3-family extension; fall back to the plain template when the
        installed tokenizer does not accept it.
        """
        kwargs: dict[str, Any] = {
            "tools": [SEARCH_TOOL],
            "add_generation_prompt": True,
            "tokenize": False,
        }
        try:
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(text, (list, tuple)):  # defensive: single sample
            text = text[0]
        enc = self.tokenizer(text, add_special_tokens=False)
        ids = getattr(enc, "input_ids", None)
        if ids is None and isinstance(enc, dict):
            ids = enc.get("input_ids")
        if ids is None:
            raise ValueError(
                f"tokenizer encode returned no input_ids: {type(enc).__name__}"
            )
        if hasattr(ids, "tolist"):  # tensor / ndarray
            ids = ids.tolist()
        if ids and isinstance(ids[0], (list, tuple)):  # batched single sample
            ids = ids[0]
        return [int(t) for t in ids]

    def _decode_safe(self, token_ids: list[int]) -> str:
        """Decode tokens to text, stripping special/stop tokens."""
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        eos = getattr(self.tokenizer, "eos_token", None)
        if eos and text.endswith(eos):
            text = text[: -len(eos)]
        return text

    @staticmethod
    def _format_search_result(result: SearchResult) -> str:
        """Format search results into a tool response string."""
        if not result.ok:
            return (
                "<tool_response>"
                f"Search error: {result.error or 'unknown'}"
                "</tool_response>"
            )
        if not result.items:
            return "<tool_response>Search returned no results.</tool_response>"
        blocks = "\n\n".join(
            format_item(item, i) for i, item in enumerate(result.items, 1)
        )
        return f"<tool_response>{blocks[:MAX_TOOL_RESPONSE_CHARS]}</tool_response>"
