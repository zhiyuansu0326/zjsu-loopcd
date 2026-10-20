#!/usr/bin/env python3
"""Prepare the complete ARC-Easy test split using pinned lm-eval conventions.

Uses only the Python standard library. Fetches selected JSONL members from AI2's
official ZIP with HTTP byte ranges, without downloading its large search corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import random
import sys
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS_COMMIT = "d6de81643928d653435c431bae19945d41d32520"
HARNESS_REPO = "https://github.com/EleutherAI/lm-evaluation-harness"
ARCHIVE_URL = "https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip"
ARCHIVE_SIZE = 680841265
ARCHIVE_ETAG = '"1127a48008065e274e98c503e844e064-41"'
ARCHIVE_VERSION = "nRMm.3j2cEmnJuI_ZfN4Bdm9QPoKdUWX"
MEMBER_PREFIX = "ARC-V1-Feb2018-2/"
RAW_HASHES = {
    "ARC-Easy/ARC-Easy-Train.jsonl": "c62e5482d8c01f2d5a7f24239ac9db3b12f9e47267c4e3f088c6ffe41d5545ef",
    "ARC-Easy/ARC-Easy-Dev.jsonl": "f4364b808a68df6bf500ce1070fb0105d40cacd2391c210deb00ff83ba75d1f4",
    "ARC-Easy/ARC-Easy-Test.jsonl": "ce32a0774b4beb1977d2c6b16dda5dd2b24fdc9c7f506882c452678676bd09be",
    "README.txt": "4f9f190ab3398a1278dc4940335beaa2c9e9a69a15ffd8e894a69d2d7697f81a",
}
SOURCE_HASHES = {
    "LICENSE.md": "a806e42547620dffbc2ec0333322c8b3523b4af11e059be14ef81aa5b1ae021f",
    "lm_eval/api/model.py": "b8b1db336e168b53d9ca808fe837d5bf8943d9dca76a14cf747e4f3a22d50545",
    "lm_eval/api/samplers.py": "b9541ec0ea1144d2003301ae6584f86abfab2237aebeafb0f7dd2a3adaf82c42",
    "lm_eval/api/task.py": "4c25a4f1949488483b833ce4fcf91cfd3e066084a9ea00079c6879ae3ba95657",
    "lm_eval/api/utils.py": "027862cf6efb657861b78a8827ad33a88ad3a1284f9086cc00b2826c74423dfa",
    "lm_eval/config/task.py": "c9ec7ce71a5ee6c84cd85436eec47a8fdf9842f2bd7011cc336c83ecb4cd958a",
    "lm_eval/defaults.py": "135d3c74587354d890aba98fbcb95bde6808b74ebd236fb404c1c7c4b101d1ec",
    "lm_eval/evaluator.py": "581e7767820aaa8503e6b23682360a6dbcd0c0f8e5b50b53e3f282c64b5bc986",
    "lm_eval/models/huggingface.py": "56d215b8f713d7579f9c7b58a9b8771b09c88e0e4fb8d1b6dd9068033b614b80",
    "lm_eval/tasks/arc/arc_easy.yaml": "96da1d9efe1df88659481cedc13985cb347dc068554a2eae561303fa39bfab3f",
}
LOADER_URL = (
    "https://huggingface.co/datasets/allenai/ai2_arc/resolve/"
    "4d4ed1b0a54f45df33cd8eb45c157cc00cdaf35e/ai2_arc.py"
)
LOADER_HASH = "c0e2af39a9692c9f3163df11b8d97f807f918a538f04a957d9a91dbf2940ae18"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verified_file(path: Path, expected: str) -> bool:
    if not path.exists():
        return False
    actual = sha256(path.read_bytes())
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {path}: {actual} != {expected}")
    return True


def save_verified(path: Path, data: bytes, expected: str) -> None:
    if sha256(data) != expected:
        raise ValueError(f"Downloaded bytes failed SHA256 verification: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()


class RemoteZip(io.RawIOBase):
    """Small seekable HTTP reader for zipfile's central-directory/member reads."""

    def __init__(self) -> None:
        self.position = 0
        self.transferred = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.position, 2: ARCHIVE_SIZE}[whence]
        self.position = base + offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, size: int = -1) -> bytes:
        size = min(size if size >= 0 else ARCHIVE_SIZE, ARCHIVE_SIZE - self.position)
        if size <= 0:
            return b""
        if size > 5_000_000:
            raise ValueError("Unexpected large ZIP read; refusing corpus download")
        end = self.position + size - 1
        request = urllib.request.Request(
            ARCHIVE_URL,
            headers={"Range": f"bytes={self.position}-{end}", "If-Match": ARCHIVE_ETAG},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            expected = f"bytes {self.position}-{end}/{ARCHIVE_SIZE}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Server did not return the requested byte range")
            data = response.read()
        if len(data) != size:
            raise ValueError("Incomplete byte range")
        self.position += size
        self.transferred += size
        return data


def acquire() -> tuple[list[dict], int]:
    records = []
    for relative, expected in SOURCE_HASHES.items():
        path = ROOT / "sources/lm_eval" / relative
        url = f"https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/{HARNESS_COMMIT}/{relative}"
        if not verified_file(path, expected):
            save_verified(path, fetch(url), expected)
        records.append({"path": str(path.relative_to(ROOT)), "url": url, "sha256": expected})
    loader = ROOT / "data/ai2_arc_loader.py"
    if not verified_file(loader, LOADER_HASH):
        save_verified(loader, fetch(LOADER_URL), LOADER_HASH)
    records.append({"path": str(loader.relative_to(ROOT)), "url": LOADER_URL, "sha256": LOADER_HASH})
    missing = [
        relative for relative, expected in RAW_HASHES.items()
        if not verified_file(ROOT / "data/raw" / relative, expected)
    ]
    transferred = 0
    if missing:
        reader = RemoteZip()
        with zipfile.ZipFile(reader) as archive:
            for relative in missing:
                save_verified(
                    ROOT / "data/raw" / relative,
                    archive.read(MEMBER_PREFIX + relative),
                    RAW_HASHES[relative],
                )
        transferred = reader.transferred
    for relative, expected in RAW_HASHES.items():
        records.append({
            "path": "data/raw/" + relative,
            "url": ARCHIVE_URL,
            "zip_member": MEMBER_PREFIX + relative,
            "sha256": expected,
        })
    return records, transferred


def read_split(name: str) -> list[dict]:
    path = ROOT / f"data/raw/ARC-Easy/ARC-Easy-{name}.jsonl"
    documents = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        choices = raw["question"]["choices"]
        labels = [choice["label"] for choice in choices]
        documents.append({
            "id": raw["id"],
            "question": raw["question"]["stem"],
            "choices": [choice["text"] for choice in choices],
            "labels": labels,
            "gold": labels.index(raw["answerKey"]),
        })
    return documents


def prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def build_rows(train: list[dict], test: list[dict], seed: int) -> list[dict]:
    """Mirror ContextSampler.sample and ConfigurableTask.fewshot_context."""
    rng = random.Random(seed)
    rows = []
    for document in test:
        examples = rng.sample(train, 8)
        context = "".join(
            prompt(item["question"]) + " " + item["choices"][item["gold"]] + "\n\n"
            for item in examples
        ) + prompt(document["question"])
        rows.append({
            **document,
            "context": context,
            "fewshot_ids": [item["id"] for item in examples],
        })
    return rows


def validate(rows: list[dict], train: list[dict], test: list[dict], seed: int) -> dict:
    # Execute the pinned upstream sampler independently to check every selection.
    path = ROOT / "sources/lm_eval/lm_eval/api/samplers.py"
    spec = importlib.util.spec_from_file_location("pinned_arc_sampler", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sampler = module.ContextSampler(train, rnd=seed)
    train_ids = {item["id"] for item in train}
    test_ids = {item["id"] for item in test}
    assert len(train) == len(train_ids) == 2251
    assert len(test) == len(test_ids) == len(rows) == 2376
    assert train_ids.isdisjoint(test_ids)
    for row in rows:
        assert row["fewshot_ids"] == [item["id"] for item in sampler.sample(8)]
        assert len(set(row["fewshot_ids"])) == 8
        assert set(row["fewshot_ids"]) <= train_ids
        assert len(row["choices"]) == len(row["labels"])
        assert all(isinstance(choice, str) and len(choice) > 0 for choice in row["choices"])
        assert 0 <= row["gold"] < len(row["choices"])
        assert row["context"].endswith(prompt(row["question"]))
    # Changing test labels must not change examples or contexts.
    changed = [{**item, "gold": (item["gold"] + 1) % len(item["choices"])} for item in test]
    alternate = build_rows(train, changed, seed)
    assert all(a["context"] == b["context"] for a, b in zip(rows, alternate))
    assert all(a["fewshot_ids"] == b["fewshot_ids"] for a, b in zip(rows, alternate))
    return {
        "upstream_sampler_all_2376_selections_match": True,
        "train_test_ids_disjoint": True,
        "test_label_invariance": True,
        "count_choices": {str(k): sum(len(row["choices"]) == k for row in rows) for k in (3, 4, 5)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    records, transferred = acquire()
    train, dev, test = read_split("Train"), read_split("Dev"), read_split("Test")
    assert len(dev) == 570
    rows = build_rows(train, test, args.seed)
    checks = validate(rows, train, test, args.seed)
    output = ROOT / "data/arc_easy_8shot.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    protocol = {
        "task": "arc_easy",
        "dataset": "allenai/ai2_arc",
        "subset": "ARC-Easy",
        "split": "test",
        "counts": {"train": len(train), "validation": len(dev), "test": len(test)},
        "num_fewshot": 8,
        "fewshot_source": "train",
        "fewshot_seed": args.seed,
        "fewshot_sampling": "random.Random(seed).sample(train, 8) for each test document in original order; RNG advances between documents",
        "seed_status": "Paper does not specify the seed; 1234 is the pinned harness default and an explicit reproduction assumption",
        "paper_url": "https://arxiv.org/html/2610.02185v1",
        "paper_settings": "Table A2: Ouro ARC-Easy 8-shot. Table A4: logit guidance fixed 0.5, adaptive cap 1.0. Table 2: 84.09/84.39/84.26 percent baseline/fixed/adaptive.",
        "harness_repository": HARNESS_REPO,
        "harness_commit": HARNESS_COMMIT,
        "harness_version_status": "Paper does not pin a version; this commit is an explicit reproduction assumption",
        "apply_chat_template": False,
        "system_instruction": None,
        "prompt_template": "Question: {question}\nAnswer:",
        "fewshot_separator": "\n\n",
        "fewshot_answer": "one leading space followed by full correct answer text; no A/B/C/D label",
        "continuation": "one leading space followed by full candidate text",
        "token_boundary": "Move trailing whitespace from context to continuation; tokenize context+continuation and context with identical tokenizer special-token settings; continuation_ids=whole_ids[len(context_ids):]. Score only continuation tokens at causal positions shifted by one.",
        "tokenizer_special_tokens": "Use tokenizer defaults consistently; tokenizer identity/revision and BOS policy must be recorded by the inference runner.",
        "acc": "argmax over sum of natural log probabilities of all continuation tokens",
        "acc_norm": "argmax over continuation log-likelihood sum / len(original_choice_text), using Python Unicode character count, excluding the added leading space; NOT token count",
        "ties": "First maximum in original choice order, matching numpy.argmax",
        "paper_metric_ambiguity": "Paper says likelihood-based accuracy without specifying acc versus acc_norm; report both and do not select a metric after seeing the result.",
        "selection_policy": "All 2376 test rows retained, in original order; no tuning or selection based on test labels",
        "archive": {
            "url": ARCHIVE_URL,
            "bytes": ARCHIVE_SIZE,
            "etag": ARCHIVE_ETAG,
            "observed_s3_version_id": ARCHIVE_VERSION,
            "retrieval": "HTTP Range requests for selected ZIP members, guarded by If-Match",
            "integrity": "ZIP CRC and hard-coded SHA256 verified for every extracted member; full archive SHA256 not measured because the unrelated corpus was not fetched",
        },
        "sources": records,
        "output": {"path": str(output.relative_to(ROOT)), "sha256": sha256(output.read_bytes()), "rows": len(rows)},
        "validation": checks,
    }
    (ROOT / "data/protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output), "rows": len(rows),
        "sha256": protocol["output"]["sha256"],
        "archive_bytes_transferred_this_run": transferred, "validation": checks,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
