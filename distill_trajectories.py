#!/usr/bin/env python3
r"""用 DeepSeek Chat API 生成带搜索轨迹的蒸馏数据（E2-A 教师蒸馏）。

从 datasets/train.jsonl 随机抽取 100 条 NQ + 100 条 HotpotQA 问题，
调用 DeepSeek Chat API 让模型模拟搜索 Agent 生成完整的多轮搜索-回答轨迹，
输出 JSONL 用于 Search-R1 SFT 冷启动。

用法:
    uv run python distill_trajectories.py
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

# ===========================================================================
# 配置
# ===========================================================================

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    print("ERROR: DEEPSEEK_API_KEY not found in .env", file=sys.stderr)
    sys.exit(1)

MODEL = "deepseek-chat"                     # DeepSeek Chat API 模型名
TRAIN_FILE = HERE / "datasets" / "train.jsonl"
OUTPUT_FILE = HERE / "distilled_trajectories.jsonl"

N_SAMPLES_NQ = 200
N_SAMPLES_HOTPOTQA = 200
RANDOM_SEED = 42

MAX_RETRIES = 5                             # 手动重试次数（on top of openai 默认 2 次）
REQUEST_TIMEOUT = 120.0                     # 请求超时（秒）
MAX_TOKENS = 4096                           # 单次生成最大 token 数
TEMPERATURE = 0.7                           # 生成温度
REQUEST_INTERVAL = 0.5                      # 请求间隔（秒）

# ===========================================================================
# Prompt 模板
# ===========================================================================

SYSTEM_PROMPT = """\
You are a question-answering agent with access to a web search tool.

## Search Tool
Function: search(query: str) — search the web. Query must be concise English.

## Assistant Turn Protocol
In each turn you MUST output your reasoning first, then EITHER:
  (A) One tool call using this EXACT format:
      <tool_call><function=search><parameter=query>YOUR QUERY</parameter></function></tool_call>
  (B) A final answer on a line starting with "Answer: ":
      Answer: YOUR CONCISE ANSWER

IMPORTANT: One action per turn — a tool call or an answer, never both.
You may search 1–4 times across separate turns.

## Your Task
Generate a COMPLETE multi-turn search trajectory for the question below.
For every tool call you make, you must also SIMULATE realistic search results
(using your own knowledge). Include source titles, text snippets, and URLs.

Structure the trajectory with these EXACT section markers:

  ---BEGIN ASSISTANT---
  (your reasoning + either a tool call or the final answer)
  ---BEGIN TOOL---
  (simulated search results — with source names, snippets, URLs)
  ---END TOOL---

Repeat the ---BEGIN ASSISTANT--- / ---BEGIN TOOL--- / ---END TOOL--- pattern
as needed. The final ---BEGIN ASSISTANT--- must contain the answer.

DO NOT use "---BEGIN" or "---END" anywhere inside a section's content.

## Rules
1. Think step-by-step: decide what info you need before each search.
2. One tool call per assistant turn. Use concise English queries.
3. Simulate detailed, realistic search results with source names and snippets.
4. Simple questions: 1–2 searches. Multi-hop / comparative: 3–4 searches.
5. When evidence is sufficient, give the answer starting with "Answer: ".
6. Do NOT include both a tool call and an answer in the same turn.

## Example 1 — simple factual question
Question: What year was the Eiffel Tower built?

---BEGIN ASSISTANT---
This is a straightforward factual question. I need to find the construction year.
<tool_call><function=search><parameter=query>Eiffel Tower construction year</parameter></function></tool_call>
---BEGIN TOOL---
[1] Title: Eiffel Tower - Wikipedia
    Content: The Eiffel Tower was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair in Paris. It was completed on March 31, 1889.
    URL: https://en.wikipedia.org/wiki/Eiffel_Tower
[2] Title: Eiffel Tower History & Facts — History.com
    Content: Construction of the Eiffel Tower began in 1887 and was completed in 1889. Designed by Gustave Eiffel, it stands 330 meters tall.
    URL: https://www.history.com/topics/landmarks/eiffel-tower
---END TOOL---
---BEGIN ASSISTANT---
The search results consistently show the Eiffel Tower was completed in 1889.
Answer: 1889

## Example 2 — multi-hop / comparative question
Question: Which magazine was started first, Arthur's Magazine or First for Women?

---BEGIN ASSISTANT---
This is a comparison question. I need the founding dates of both magazines.
<tool_call><function=search><parameter=query>Arthur's Magazine founding date year</parameter></function></tool_call>
---BEGIN TOOL---
[1] Title: Arthur's Magazine - Wikipedia
    Content: Arthur's Magazine was an American literary periodical published in Philadelphia from 1844 to 1846. Founded by Timothy Shay Arthur.
    URL: https://en.wikipedia.org/wiki/Arthur%27s_Magazine
---END TOOL---
---BEGIN ASSISTANT---
Arthur's Magazine was founded in 1844. Now I need First for Women's founding date.
<tool_call><function=search><parameter=query>First for Women magazine founded year</parameter></function></tool_call>
---BEGIN TOOL---
[1] Title: First for Women (magazine) - Wikipedia
    Content: First for Women is a women's magazine published by A360 Media. It was first published in 1989. It covers health, nutrition, beauty, and lifestyle.
    URL: https://en.wikipedia.org/wiki/First_for_Women
---END TOOL---
---BEGIN ASSISTANT---
Arthur's Magazine (1844) was founded earlier than First for Women (1989).
Answer: Arthur's Magazine"""

USER_TEMPLATE = """Question: {question}"""

# ===========================================================================
# 正则 — 轨迹解析
# ===========================================================================

SECTION_SPLITTER = re.compile(
    r"---(?:BEGIN (ASSISTANT|TOOL)---|END TOOL---)"
)

ANSWER_RE = re.compile(r"\bAnswer:\s*(.*?)$", re.MULTILINE)

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=search>\s*<parameter=query>\s*"
    r"(.*?)"
    r"\s*</parameter>\s*</function>\s*</tool_call>",
    re.DOTALL,
)


def parse_trajectory(
    raw_text: str, question: str, system_prompt: str
) -> tuple[list[dict[str, str]], str] | None:
    """Parse model output into (messages, trajectory_text) using section markers.

    Returns None if parsing produces fewer than 3 messages (system + user + answer).
    """
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    parts = SECTION_SPLITTER.split(text)
    # parts 格局: [preamble, group_val, text, group_val, text, ...]
    # group_val 为 "ASSISTANT" / "TOOL" / None（END TOOL 时未参与匹配）

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    traj_parts: list[str] = []

    # 跳到第一个有效 marker
    idx = 0
    while idx < len(parts) and parts[idx] not in ("ASSISTANT", "TOOL"):
        idx += 1

    current_role: str | None = None
    current_content: list[str] = []

    def _flush() -> None:
        nonlocal current_role, current_content
        if current_role and current_content:
            content = "\n".join(current_content).strip()
            if content:
                messages.append({"role": current_role, "content": content})
                traj_parts.append(content)
        current_role = None
        current_content = []

    while idx < len(parts) - 1:
        role_label = parts[idx]
        content_text = parts[idx + 1]

        if role_label in ("ASSISTANT", "TOOL"):
            _flush()
            current_role = "assistant" if role_label == "ASSISTANT" else "tool"
            current_content = [content_text]

        idx += 2

    _flush()

    if len(messages) < 3:
        return None

    # ── 后处理：合并同角色连续消息 ──
    merged: list[dict[str, str]] = [messages[0], messages[1]]
    for msg in messages[2:]:
        if merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(msg)

    # ── 后处理：插入缺失的 tool 消息 ──
    fixed: list[dict[str, str]] = [merged[0], merged[1]]
    for msg in merged[2:]:
        if (
            msg["role"] == "assistant"
            and fixed[-1]["role"] == "assistant"
            and "<tool_call>" in fixed[-1]["content"]
            and "Answer:" not in fixed[-1]["content"]
        ):
            fixed.append(
                {"role": "tool", "content": "[Search results for the above query]"}
            )
        fixed.append(msg)
    messages = fixed

    trajectory_text = "\n\n".join(traj_parts)
    return messages, trajectory_text


def fallback_parse(
    raw_text: str, question: str, system_prompt: str
) -> tuple[list[dict[str, str]], str] | None:
    """Fallback parser when structured section markers are absent.

    Scans for <tool_call ... /> and Answer: patterns in the raw text.
    """
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    tool_matches = list(TOOL_CALL_RE.finditer(text))

    if not tool_matches:
        # 无 tool_call：整段当作直接回答
        ans_m = ANSWER_RE.search(text)
        if ans_m:
            messages.append(
                {"role": "assistant", "content": f"Answer: {ans_m.group(1).strip()}"}
            )
        else:
            messages.append({"role": "assistant", "content": text.strip()})
        traj_text = messages[-1]["content"]
        return messages, traj_text

    last_end = 0
    for match in tool_matches:
        query = match.group(1).strip()
        before = text[last_end : match.start()].strip()
        tc_full = match.group(0)

        if before:
            ans_in_before = ANSWER_RE.search(before)
            if ans_in_before:
                think = before[: ans_in_before.start()].strip()
                if think:
                    messages.append({"role": "assistant", "content": think})
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Answer: {ans_in_before.group(1).strip()}",
                    }
                )
            else:
                messages.append(
                    {"role": "assistant", "content": f"{before}\n\n{tc_full}".strip()}
                )
        else:
            messages.append({"role": "assistant", "content": tc_full})

        # 插入占位 tool 结果
        messages.append(
            {"role": "tool", "content": f'[Search results for: "{query}"]'}
        )
        last_end = match.end()

    remaining = text[last_end:].strip()
    if remaining:
        ans_in_rem = ANSWER_RE.search(remaining)
        if ans_in_rem:
            think = remaining[: ans_in_rem.start()].strip()
            if think:
                messages.append({"role": "assistant", "content": think})
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Answer: {ans_in_rem.group(1).strip()}",
                }
            )
        else:
            messages.append({"role": "assistant", "content": remaining})

    if len(messages) < 3:
        return None

    traj_parts = [
        m["content"] for m in messages if m["role"] in ("assistant", "tool")
    ]
    trajectory_text = "\n\n".join(traj_parts)
    return messages, trajectory_text


def validate_trajectory(messages: list[dict[str, str]]) -> list[str]:
    """Check a parsed trajectory for common issues. Returns list of warnings."""
    warnings: list[str] = []
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]

    if not assistant_msgs:
        return ["No assistant messages found"]

    # 最后的 assistant 消息应有 Answer:
    last = assistant_msgs[-1]["content"]
    if not ANSWER_RE.search(last):
        warnings.append("No 'Answer:' found in the final assistant turn")

    # tool_call 数量
    n_tc = sum(
        len(TOOL_CALL_RE.findall(m["content"])) for m in assistant_msgs
    )
    if n_tc == 0:
        warnings.append("No tool calls found in trajectory")
    elif n_tc > 6:
        warnings.append(f"High number of tool calls: {n_tc}")

    return warnings


# ===========================================================================
# 数据加载
# ===========================================================================

def load_and_sample(
    train_file: Path, n_nq: int, n_hotpotqa: int, seed: int
) -> list[dict[str, Any]]:
    """Load train.jsonl and randomly sample the requested counts.

    Results are interleaved (NQ, HotpotQA, NQ, HotpotQA, ...).
    """
    nq_all: list[dict[str, Any]] = []
    hotpotqa_all: list[dict[str, Any]] = []

    with train_file.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            src = row["data_source"]
            if src == "nq":
                nq_all.append(row)
            elif src == "hotpotqa":
                hotpotqa_all.append(row)

    rng = random.Random(seed)
    nq_sampled = rng.sample(nq_all, min(n_nq, len(nq_all)))
    hotpotqa_sampled = rng.sample(hotpotqa_all, min(n_hotpotqa, len(hotpotqa_all)))

    # 交错排列
    result: list[dict[str, Any]] = []
    for nq_item, hq_item in zip(nq_sampled, hotpotqa_sampled):
        result.append(nq_item)
        result.append(hq_item)
    # 补齐不相等部分
    if len(nq_sampled) > len(hotpotqa_sampled):
        result.extend(nq_sampled[len(hotpotqa_sampled):])
    if len(hotpotqa_sampled) > len(nq_sampled):
        result.extend(hotpotqa_sampled[len(nq_sampled):])

    return result


def load_completed_ids(output_file: Path) -> set[str]:
    """Collect already-processed question IDs (checkpoint / resume support)."""
    if not output_file.exists():
        return set()
    completed: set[str] = set()
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                completed.add(str(row["id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


# ===========================================================================
# API 客户端
# ===========================================================================

def _call_api(
    client: OpenAI,
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Invoke chat completion with a manual retry loop on top of the SDK's built-in retries."""
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = resp.choices[0].message.content
            return content if content else ""
        except RateLimitError as exc:
            wait = 2.0 * (2**attempt)
            print(
                f"    限流 (429), 等待 {wait:.0f}s "
                f"(重试 {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(wait)
            last_err = exc
        except APITimeoutError as exc:
            print(
                f"    请求超时, 重试中 (尝试 {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(1.0 * (2**attempt))
            last_err = exc
        except APIError as exc:
            # 5xx or other server error
            if getattr(exc, "status_code", 500) >= 500:
                wait = 2.0 * (2**attempt)
                print(
                    f"    服务器错误, 等待 {wait:.0f}s "
                    f"(重试 {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(wait)
                last_err = exc
                continue
            # Non-retryable (e.g. 400 Bad Request)
            raise RuntimeError(f"API 错误: {exc}") from exc
        except Exception as exc:
            last_err = exc
            if attempt < MAX_RETRIES:
                time.sleep(1.0 * (2**attempt))
                continue
            raise

    raise RuntimeError(f"超过最大重试次数: {last_err}")


# ===========================================================================
# 主流程
# ===========================================================================

def main() -> None:
    """Entry point."""
    print("=" * 60)
    print("Search-R1 教师蒸馏数据生成 (E2-A)")
    print("=" * 60)

    # ── 1. 加载并抽样 ──
    print(f"\n[1/4] 加载训练数据: {TRAIN_FILE}")
    questions = load_and_sample(
        TRAIN_FILE, N_SAMPLES_NQ, N_SAMPLES_HOTPOTQA, RANDOM_SEED
    )
    n_nq = sum(1 for q in questions if q["data_source"] == "nq")
    n_hq = sum(1 for q in questions if q["data_source"] == "hotpotqa")
    print(f"  抽样: {len(questions)} 条 (NQ={n_nq}, HotpotQA={n_hq})")

    # ── 2. 断点续传 ──
    print(f"\n[2/4] 检查断点: {OUTPUT_FILE}")
    completed_ids = load_completed_ids(OUTPUT_FILE)
    pending = [q for q in questions if str(q["id"]) not in completed_ids]
    print(f"  已完成: {len(completed_ids)} 条")
    print(f"  待处理: {len(pending)} 条")

    if not pending:
        print("\n所有问题已处理完毕, 退出。")
        return

    # ── 3. 初始化 API 客户端 ──
    print(f"\n[3/4] 初始化 DeepSeek Chat API  (model={MODEL})")
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://api.deepseek.com",
        timeout=REQUEST_TIMEOUT,
        max_retries=1,  # SDK 内置重试 1 次，外层还有手动循环
    )

    # ── 4. 生成轨迹 ──
    print(f"\n[4/4] 开始生成 {len(pending)} 条轨迹\n")

    mode = "a" if OUTPUT_FILE.exists() else "w"
    out_f = OUTPUT_FILE.open(mode, encoding="utf-8")

    success = 0
    fail = 0
    t_start = time.time()

    try:
        for idx, row in enumerate(pending, start=1):
            qid = str(row["id"])
            question = row["question"]
            answers = row["answers"]
            data_source = row["data_source"]

            q_preview = (
                question[:100] + "..." if len(question) > 100 else question
            )
            print(
                f"[{idx:03d}/{len(pending):03d}] "
                f"[{data_source.upper():8s}] {q_preview}"
            )

            try:
                api_msgs: list[dict[str, str]] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_TEMPLATE.format(question=question),
                    },
                ]

                raw = _call_api(
                    client,
                    api_msgs,
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )

                # ── 解析 ──
                result = parse_trajectory(raw, question, SYSTEM_PROMPT)
                used_fallback = False
                if result is None:
                    print("    [WARN] 主解析失败, 尝试 fallback 解析")
                    result = fallback_parse(raw, question, SYSTEM_PROMPT)
                    used_fallback = True

                if result is None:
                    print(f'    [FAIL] 无法解析轨迹 —— 跳过 "{qid}"')
                    # 把原始输出保存到文件便于调试
                    _save_raw_on_failure(OUTPUT_FILE, qid, raw)
                    fail += 1
                    continue

                messages, traj_text = result

                # ── 验证 ──
                for w in validate_trajectory(messages):
                    print(f"    [WARN] {w}")

                # ── 统计 ──
                n_asst = sum(1 for m in messages if m["role"] == "assistant")
                n_tool_msgs = sum(1 for m in messages if m["role"] == "tool")
                n_tc = sum(
                    len(TOOL_CALL_RE.findall(m["content"]))
                    for m in messages
                    if m["role"] == "assistant"
                )

                tag = "FALLBACK" if used_fallback else "OK"
                print(
                    f"    [{tag}] {n_asst}A + {n_tool_msgs}T, "
                    f"{n_tc} searches"
                )

                # ── 写入 ──
                record = {
                    "id": qid,
                    "question": question,
                    "answers": answers,
                    "data_source": data_source,
                    "messages": messages,
                    "trajectory_text": traj_text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                success += 1

            except Exception as exc:
                print(f"    [FAIL] {exc}")
                fail += 1

            # 请求间隔，降低限流风险
            if idx < len(pending):
                time.sleep(REQUEST_INTERVAL)

    finally:
        out_f.close()

    elapsed = time.time() - t_start
    total_saved = success + len(completed_ids)

    print(f"\n{'=' * 60}")
    print(f"完成!  耗时 {elapsed:.1f}s")
    print(f"  本次成功: {success}")
    print(f"  本次失败: {fail}")
    print(f"  历史累计: {len(completed_ids)}")
    print(f"  合计保存: {total_saved}")
    print(f"  输出文件: {OUTPUT_FILE}")
    print(f"{'=' * 60}")


def _save_raw_on_failure(output_file: Path, qid: str, raw_text: str) -> None:
    """Append the raw (unparseable) model output to a sibling debug file."""
    debug_path = output_file.with_suffix(".failures.jsonl")
    record = {"id": qid, "raw_output": raw_text}
    with debug_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
