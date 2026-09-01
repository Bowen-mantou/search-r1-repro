"""JSONL → parquet for veRL agent-loop training (E6-B).

veRL agent loops receive each row's `prompt` (list of role/content messages)
via data.return_raw_chat=True, and the custom reward function reads
`reward_model.ground_truth` + `question`/`data_source` columns.

Usage (on server):
    python verl_search_r1/prepare_data.py datasets/train.jsonl datasets/train.parquet
    python verl_search_r1/prepare_data.py datasets/test.jsonl  datasets/test.parquet
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def convert(src: Path, dst: Path) -> None:
    rows = []
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append({
                "id": rec.get("id", ""),
                "question": rec["question"],
                "answers": rec["answers"],
                "data_source": rec.get("data_source", "nq"),
                # veRL agent loop receives this as kwargs["raw_prompt"]
                "prompt": [{"role": "user", "content": rec["question"]}],
                # veRL custom reward function reads ground_truth from here
                "reward_model": {"ground_truth": rec["answers"], "style": "rule"},
            })
    df = pd.DataFrame(rows)
    df.to_parquet(dst, index=False)
    print(f"  {src.name}: {len(df)} rows -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()
    convert(args.src, args.dst)
