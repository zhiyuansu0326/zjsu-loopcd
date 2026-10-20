#!/usr/bin/env python3
"""Derive the paper's zero-shot prompts from exactly the existing ARC test rows."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    source = ROOT / "data/arc_easy_8shot.jsonl"
    records = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(records) == 2376 and len({row["id"] for row in records}) == 2376
    for row in records:
        row["context"] = f"Question: {row['question']}\nAnswer:"
        row["fewshot_ids"] = []
    dest = ROOT / "data/arc_easy_0shot.jsonl"
    dest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    metadata = {
        "source": "data/arc_easy_8shot.jsonl",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output": "data/arc_easy_0shot.jsonl",
        "output_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "n": len(records), "shots": 0,
        "template": "Question: {question}\\nAnswer:",
        "choices_and_gold_unchanged": True,
        "paper": "Table A2: Huginn, Parcae, Looped-Qwen3 use zero-shot ARC-Easy.",
    }
    (ROOT / "data/protocol_0shot.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
