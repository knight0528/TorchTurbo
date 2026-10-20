#!/usr/bin/env python3
"""Convert DAPO math prompt/label rows to ms-swift messages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_row(row: dict) -> dict:
    prompt = row["prompt"]
    if isinstance(prompt, list):
        content = "\n".join(
            item["content"] if isinstance(item, dict) else str(item)
            for item in prompt
        )
    else:
        content = str(prompt)
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": str(row["label"])},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.input_path.open(encoding="utf-8") as source, args.output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            target.write(json.dumps(convert_row(json.loads(line))) + "\n")


if __name__ == "__main__":
    main()
