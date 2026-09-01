#!/usr/bin/env python3
r"""将蒸馏轨迹中的 DeepSeek 原生工具调用标签转换为 Search-R1 简化标签。

转换规则：
  <tool_call><function=search><parameter=query>QUERY</parameter></function></tool_call>
    → <search>QUERY</search>

  tool role 消息的 content → <information>content</information>

  Answer: XXX → <answer>XXX</answer>

用法:
  uv run python fix_distill_tags.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

# ===========================================================================
# 路径配置
# ===========================================================================

HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "distilled_trajectories.jsonl"
OUTPUT_FILE = HERE / "distilled_trajectories_fixed.jsonl"

# ===========================================================================
# 正则
# ===========================================================================

# 匹配所有变体的 tool_call 标签:
#   标准:     <tool_call><function=search><parameter=query>Q</parameter></function></tool_call>
#   缺 query:  <tool_call><function=search><parameter>Q</parameter></function></tool_call>
#   错关:     <tool_call><function=search><parameter>Q</parameter></function></search></tool_call>
#   缺 tool_:  <tool_call><function=search><parameter>Q</parameter></function></call>
#   query 内:  <tool_call><function=search><parameter>query=Q</parameter></function></tool_call>
#   query 后置: <tool_call><function=search><parameter>query</parameter>Q</function></tool_call>
# 采用 tag-stripping 策略: 匹配完整的 tool_call 块, 然后剥离所有 XML 标签提取查询文本。
# 处理 DeepSeek 生成的各种工具调用闭合标签变体:
#   </tool_call>  </call>  </talk>  </tool>  </tool call>  </search>
# 注意: </search> 仅在 <tool_call> 块内匹配, 不会匹配我们新生成的 <search> 标签,
#       因为匹配必须以 <tool_call> 开头。
TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>"
    r"(.*?)"
    r"</(?:tool_?\s*)?(?:call|talk|tool|search)>",
    re.DOTALL,
)

# 用于剥离 XML 标签的正则
XML_TAG_RE = re.compile(r"<[^>]+>")

ANSWER_RE = re.compile(r"Answer:\s*(.*?)$", re.MULTILINE | re.IGNORECASE)

# ===========================================================================
# 核心转换逻辑
# ===========================================================================


def _clean_query(raw_inner: str) -> str:
    """从 tool_call 块的内部内容中提取纯净的查询文本。

    策略: 剥离所有 XML 标签, 然后去掉可能残留的 "query=" 或 "query" 前缀。
    """
    # 剥离所有 XML 标签
    stripped = XML_TAG_RE.sub("", raw_inner).strip()
    # 去掉可能的前缀 "query=" 或 孤立的 "query"
    if stripped.startswith("query="):
        stripped = stripped[len("query="):].strip()
    elif stripped == "query":
        stripped = ""
    elif stripped.startswith("query ") or stripped.startswith("query\n"):
        stripped = stripped[len("query"):].strip()
    return stripped


def convert_content_tool_calls(text: str) -> str:
    """将文本中所有 tool_call 变体标签批量替换为 <search> 格式。"""
    def _replace(m: re.Match) -> str:
        inner = m.group(1)
        query = _clean_query(inner)
        return f"<search>{query}</search>"
    return TOOL_CALL_BLOCK_RE.sub(_replace, text)


def convert_assistant_content(text: str, is_final: bool) -> str:
    """转换 assistant 消息内容：替换 tool_call 标签，可选替换 Answer 为 <answer>。

    Args:
        text: assistant 消息的 content 文本。
        is_final: 是否是最后一条 assistant 消息（只有最后一条才替换 Answer）。

    Returns:
        转换后的文本。
    """
    converted = convert_content_tool_calls(text)

    if is_final:
        # 替换 Answer: XXX → <answer>XXX</answer>
        # 保留 Answer: 前面的推理文本
        converted = ANSWER_RE.sub(r"<answer>\1</answer>", converted)

    return converted


def convert_tool_content(text: str) -> str:
    """用 <information> 标签包裹搜索结果。"""
    return f"<information>{text}</information>"


def fix_trajectory(record: dict) -> tuple[dict, int]:
    """转换单条轨迹，返回 (修正后的 record, 替换次数)。

    Args:
        record: 原始轨迹 dict，包含 id/question/answers/data_source/messages/trajectory_text。

    Returns:
        (fixed_record, replacement_count)
    """
    messages = record.get("messages", [])
    if not messages:
        return record, 0

    replacement_count = 0
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    is_final_tracker = {}  # 标记最后一条 assistant 消息的 id()

    if assistant_msgs:
        final_assistant = assistant_msgs[-1]
        is_final_tracker[id(final_assistant)] = True

    fixed_messages = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            is_final = bool(is_final_tracker.get(id(msg), False))
            new_content = convert_assistant_content(content, is_final=is_final)

            # 统计替换次数
            tc_before = len(TOOL_CALL_BLOCK_RE.findall(content))
            tc_after = len(re.findall(r"<search>", new_content))
            replacement_count += max(tc_before, tc_after)

            if is_final:
                answer_before = bool(ANSWER_RE.search(content))
                answer_after = bool(re.search(r"<answer>", new_content))
                if answer_before or answer_after:
                    replacement_count += 1

            fixed_messages.append({"role": role, "content": new_content})

        elif role == "tool":
            new_content = convert_tool_content(content)
            replacement_count += 1
            fixed_messages.append({"role": role, "content": new_content})

        elif role == "system":
            # 系统提示中包含示例 tool_call 标签，也一并转换
            new_content = content
            tc_count = len(TOOL_CALL_BLOCK_RE.findall(content))
            if tc_count > 0:
                new_content = convert_content_tool_calls(content)
                replacement_count += tc_count
            fixed_messages.append({"role": role, "content": new_content})

        else:
            fixed_messages.append(dict(msg))

    # ── 转换 trajectory_text ──
    traj_text = record.get("trajectory_text", "")
    new_traj_text = convert_content_tool_calls(traj_text)

    # 替换 Answer: → <answer>
    new_traj_text = ANSWER_RE.sub(r"<answer>\1</answer>", new_traj_text)

    fixed_record = {
        "id": record.get("id"),
        "question": record.get("question"),
        "answers": record.get("answers"),
        "data_source": record.get("data_source"),
        "messages": fixed_messages,
        "trajectory_text": new_traj_text,
    }
    return fixed_record, replacement_count


# ===========================================================================
# 验证
# ===========================================================================


def validate_record(record: dict, idx: int) -> list[str]:
    """验证一条转换后的轨迹是否符合预期格式。返回问题列表。"""
    issues: list[str] = []
    messages = record.get("messages", [])
    traj_text = record.get("trajectory_text", "")

    # 检查所有 messages 中
    all_texts = [m.get("content", "") for m in messages] + [traj_text]
    combined = "\n".join(all_texts)

    # 必须有 <search> 或 <answer> 或 <information> 标签
    has_search = "<search>" in combined
    has_info = "<information>" in combined
    has_answer = "<answer>" in combined

    if not has_search:
        issues.append(f"记录 {idx}: 缺少 <search> 标签")
    if not has_info:
        issues.append(f"记录 {idx}: 缺少 <information> 标签")
    if not has_answer:
        # 有些轨迹可能是纯 tool_call 但没有明确答案行
        pass

    # 不应再有旧的 tool_call 标签
    has_old_tool_call = bool(TOOL_CALL_BLOCK_RE.search(combined))
    has_old_function = "<function=search>" in combined
    has_old_param = "<parameter=query>" in combined

    if has_old_tool_call:
        issues.append(f"记录 {idx}: 仍然包含 <tool_call> 标签")
    if has_old_function:
        issues.append(f"记录 {idx}: 仍然包含 <function=search> 标签")
    if has_old_param:
        issues.append(f"记录 {idx}: 仍然包含 <parameter=query> 标签")

    # 检查 tool role 的消息是否被正确包裹
    for i, msg in enumerate(messages):
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if not content.startswith("<information>") or not content.endswith(
                "</information>"
            ):
                issues.append(
                    f"记录 {idx}, message {i}: tool 消息未被 <information> 包裹"
                )

    return issues


# ===========================================================================
# 主流程
# ===========================================================================


def main() -> None:
    """读取、转换、验证并写入蒸馏轨迹数据。"""
    print("=" * 60)
    print("Search-R1 蒸馏数据标签格式修复")
    print("=" * 60)

    # ── 1. 读取 ──
    print(f"\n[1/4] 读取输入: {INPUT_FILE}")
    records = []
    skipped = 0
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [跳过] 第 {line_no} 行 JSON 解析失败: {exc}")
                skipped += 1

    total_input = len(records) + skipped
    print(f"  读取 {total_input} 条, 其中有效 {len(records)} 条, 跳过 {skipped} 条")

    # ── 2. 转换 ──
    print(f"\n[2/4] 转换标签格式...")
    total_replacements = 0
    fixed_records = []
    for record in records:
        fixed, count = fix_trajectory(record)
        fixed_records.append(fixed)
        total_replacements += count

    print(f"  转换完成: {len(fixed_records)} 条")
    print(f"  标签替换: {total_replacements} 处")

    # ── 3. 验证 ──
    print(f"\n[3/4] 验证转换结果...")

    # 随机抽查 5 条
    rng = random.Random(42)
    sample_indices = sorted(rng.sample(range(len(fixed_records)), min(5, len(fixed_records))))

    all_issues: list[str] = []
    for idx in sample_indices:
        issues = validate_record(fixed_records[idx], idx)
        all_issues.extend(issues)

    # 打印抽查结果
    print(f"\n  随机抽查 5 条 (索引: {sample_indices}):")
    for idx in sample_indices:
        rec = fixed_records[idx]
        msgs = rec.get("messages", [])
        n_asst = sum(1 for m in msgs if m.get("role") == "assistant")
        n_tool = sum(1 for m in msgs if m.get("role") == "tool")

        # 检查标签
        all_content = "\n".join(m.get("content", "") for m in msgs)
        has_search = "<search>" in all_content
        has_info = "<information>" in all_content
        has_answer = "<answer>" in all_content
        has_old = bool(TOOL_CALL_BLOCK_RE.search(all_content))

        status = "OK" if (has_search and has_info and not has_old) else "ISSUES"
        print(
            f"    [{idx:03d}] id={rec['id']} | "
            f"{n_asst}A + {n_tool}T | "
            f"search={'Y' if has_search else 'N'} "
            f"info={'Y' if has_info else 'N'} "
            f"answer={'Y' if has_answer else 'N'} "
            f"old_tags={'Y' if has_old else 'N'} | {status}"
        )

    if all_issues:
        print(f"\n  发现 {len(all_issues)} 个问题:")
        for issue in all_issues[:10]:
            print(f"    - {issue}")
        if len(all_issues) > 10:
            print(f"    ... 以及另外 {len(all_issues) - 10} 个问题")
    else:
        print(f"\n  抽查全部通过, 未发现问题")

    # ── 4. 写入 ──
    print(f"\n[4/4] 写入输出: {OUTPUT_FILE}")
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for rec in fixed_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  写入 {len(fixed_records)} 条轨迹")

    # ── 汇总统计 ──
    print(f"\n{'=' * 60}")
    print(f"转换统计")
    print(f"{'=' * 60}")
    print(f"  输入文件: {INPUT_FILE}")
    print(f"  输出文件: {OUTPUT_FILE}")
    print(f"  输入记录数: {total_input}")
    print(f"  输出记录数: {len(fixed_records)}")
    print(f"  标签替换数: {total_replacements} 处")
    print(f"  抽查通过: {'Y' if not all_issues else 'N'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
