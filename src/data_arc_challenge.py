"""Pinned ARC-Challenge data, using the official ARC archive and lm-eval format."""

from __future__ import annotations

import fcntl
import importlib.util
import json
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

from scripts import prepare_arc as arc


ROOT = Path(__file__).resolve().parents[1]
TASK = "arc_challenge"
HARNESS_COMMIT = arc.HARNESS_COMMIT
COUNTS = {"Train": 1119, "Dev": 299, "Test": 1172}
RAW_HASHES = {
    "Train": "e02837c53ee86f6d73c259f91c77725c5062ef4531130b7f473a9d9b01118a77",
    "Dev": "55224364abd2477f66a9a57c5cbef109fb71d83019aa5e37aec0e7ab757746ca",
    "Test": "9fa8ffb3e3a1f88cb302d53dd95aae81e39209ad9bcf974376b319d8fd35717c",
}
ZIP_CRC32 = {"Train": "d1a0022d", "Dev": "08ff48b0", "Test": "bc2fd48c"}
SOURCE_HASHES = {
    **arc.SOURCE_HASHES,
    "lm_eval/tasks/arc/arc_challenge.yaml":
        "682621c26ae24b1065a2d732460a55a8ef2b9f2544574431666c9309a9efd7bd",
}


def _raw_path(split: str) -> Path:
    return ROOT / f"data/raw/ARC-Challenge/ARC-Challenge-{split}.jsonl"


def acquire() -> list[dict]:
    """Fetch only missing pinned sources and the three small official ZIP members."""
    records = []
    for relative, expected in SOURCE_HASHES.items():
        path = ROOT / "sources/lm_eval" / relative
        url = (
            "https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/"
            f"{HARNESS_COMMIT}/{relative}"
        )
        if not arc.verified_file(path, expected):
            arc.save_verified(path, arc.fetch(url), expected)
        records.append({
            "path": str(path.relative_to(ROOT)), "url": url, "sha256": expected,
            "bytes": path.stat().st_size,
        })
    loader = ROOT / "data/ai2_arc_loader.py"
    if not arc.verified_file(loader, arc.LOADER_HASH):
        arc.save_verified(loader, arc.fetch(arc.LOADER_URL), arc.LOADER_HASH)
    records.append({
        "path": str(loader.relative_to(ROOT)), "url": arc.LOADER_URL,
        "sha256": arc.LOADER_HASH, "bytes": loader.stat().st_size,
    })
    missing = [
        split for split, expected in RAW_HASHES.items()
        if not arc.verified_file(_raw_path(split), expected)
    ]
    if missing:
        reader = arc.RemoteZip()
        with zipfile.ZipFile(reader) as archive:
            for split in missing:
                member = f"{arc.MEMBER_PREFIX}ARC-Challenge/ARC-Challenge-{split}.jsonl"
                info = archive.getinfo(member)
                if format(info.CRC, "08x") != ZIP_CRC32[split]:
                    raise ValueError(f"Unexpected official ZIP CRC for {member}")
                arc.save_verified(_raw_path(split), archive.read(member), RAW_HASHES[split])
    for split, expected in RAW_HASHES.items():
        path = _raw_path(split)
        records.append({
            "path": str(path.relative_to(ROOT)), "url": arc.ARCHIVE_URL,
            "zip_member": f"{arc.MEMBER_PREFIX}ARC-Challenge/ARC-Challenge-{split}.jsonl",
            "zip_crc32": ZIP_CRC32[split], "sha256": expected,
            "bytes": path.stat().st_size,
        })
    return records


def read_split(split: str) -> list[dict]:
    if split not in COUNTS:
        raise ValueError(f"Unknown ARC-Challenge split: {split}")
    arc.verified_file(_raw_path(split), RAW_HASHES[split])
    documents = []
    for line in _raw_path(split).read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        choices = raw["question"]["choices"]
        labels = [choice["label"] for choice in choices]
        documents.append({
            "id": raw["id"], "question": raw["question"]["stem"],
            "choices": [choice["text"] for choice in choices], "labels": labels,
            "gold": labels.index(raw["answerKey"]),
        })
    if len(documents) != COUNTS[split]:
        raise ValueError(f"Unexpected {split} count: {len(documents)}")
    return documents


def build_rows(train: list[dict], test: list[dict], shots: int, seed: int) -> list[dict]:
    if shots not in (0, 25):
        raise ValueError("This reproduction supports ARC-Challenge 0 or 25 shots.")
    rng = random.Random(seed)
    rows = []
    for index, document in enumerate(test):
        examples = rng.sample(train, shots)
        context = "".join(
            arc.prompt(item["question"]) + " " + item["choices"][item["gold"]] + "\n\n"
            for item in examples
        ) + arc.prompt(document["question"])
        rows.append({
            **document, "context": context,
            "fewshot_ids": [item["id"] for item in examples],
            "task": TASK, "split": "test", "source_index": index,
        })
    return rows


def _upstream_sampler(train: list[dict], seed: int):
    path = ROOT / "sources/lm_eval/lm_eval/api/samplers.py"
    arc.verified_file(path, SOURCE_HASHES["lm_eval/api/samplers.py"])
    name = "_pinned_arc_challenge_sampler"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.ContextSampler(train, rnd=seed)


def audit_rows(
    rows: list[dict], train: list[dict], test: list[dict], shots: int, seed: int,
) -> dict:
    """Compare every prepared field and full prompt against raw docs and upstream RNG."""
    if shots not in (0, 25):
        raise ValueError("Unsupported few-shot count")
    train_ids = {item["id"] for item in train}
    test_ids = {item["id"] for item in test}
    if len(train_ids) != len(train) or len(test_ids) != len(test):
        raise ValueError("Duplicate raw document IDs")
    if not train_ids.isdisjoint(test_ids):
        raise ValueError("Training and test IDs overlap")
    if len(rows) != len(test):
        raise ValueError("Prepared test count differs from raw test count")
    sampler = _upstream_sampler(train, seed)
    for index, (row, document) in enumerate(zip(rows, test)):
        for key in ("id", "question", "choices", "labels", "gold"):
            if row.get(key) != document[key]:
                raise ValueError(f"Prepared row {index} differs from raw field {key}")
        if (
            row.get("task") != TASK or row.get("split") != "test"
            or row.get("source_index") != index
        ):
            raise ValueError(f"Prepared row {index} has incorrect source metadata")
        if (
            len(row["choices"]) != len(row["labels"])
            or not all(isinstance(choice, str) and choice for choice in row["choices"])
            or len(set(row["labels"])) != len(row["labels"])
            or not 0 <= row["gold"] < len(row["choices"])
        ):
            raise ValueError(f"Invalid choices/labels in row {index}")
        examples = sampler.sample(shots)
        expected_ids = [item["id"] for item in examples]
        if row.get("fewshot_ids") != expected_ids or len(set(expected_ids)) != shots:
            raise ValueError(f"Few-shot selection differs from pinned harness at row {index}")
        # Construct independently of build_rows, including complete answer strings.
        expected_context = "\n\n".join(
            f"Question: {item['question']}\nAnswer: {item['choices'][item['gold']]}"
            for item in examples
        )
        if expected_context:
            expected_context += "\n\n"
        expected_context += f"Question: {document['question']}\nAnswer:"
        if row.get("context") != expected_context:
            raise ValueError(f"Prepared prompt differs from raw/harness format at row {index}")
    changed = [
        {**item, "gold": (item["gold"] + 1) % len(item["choices"])} for item in test
    ]
    alternate = build_rows(train, changed, shots, seed)
    if any(
        left["context"] != right["context"] or left["fewshot_ids"] != right["fewshot_ids"]
        for left, right in zip(rows, alternate)
    ):
        raise ValueError("Test labels influenced the input prompts")
    return {
        "rows_checked": len(rows), "all_raw_fields_match": True,
        "upstream_sampler_all_selections_match": True, "full_contexts_match": True,
        "train_test_ids_disjoint": True, "test_label_invariance": True,
        "count_choices": dict(sorted(Counter(str(len(row["choices"])) for row in rows).items())),
        "candidate_count": sum(len(row["choices"]) for row in rows),
    }


def audit_arc_challenge(shots: int, seed: int = 1234) -> dict:
    """Verify already prepared local data; never download or modify files."""
    path = ROOT / f"data/tasks/{TASK}/{shots}shot.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return audit_rows(rows, read_split("Train"), read_split("Test"), shots, seed)


def audit_prepared(shots: int, seed: int = 1234) -> dict:
    """Shared dataset-audit API; read-only alias for the full ARC-Challenge audit."""
    return audit_arc_challenge(shots, seed)


def _write_unchanged_or_new(path: Path, data: bytes) -> None:
    if path.exists() and path.read_bytes() == data:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def prepare_arc_challenge(shots: int, seed: int = 1234) -> Path:
    """Download/verify sources, prepare full 0/25-shot test data, and return its path.

    Contexts are never truncated here. A shared protocol records both variants.
    Repeated calls preserve identical output and protocol bytes.
    """
    if shots not in (0, 25) or isinstance(shots, bool):
        raise ValueError("This reproduction supports ARC-Challenge 0 or 25 shots.")
    output_dir = ROOT / f"data/tasks/{TASK}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = acquire()
        train, dev, test = (read_split(split) for split in ("Train", "Dev", "Test"))
        rows = build_rows(train, test, shots, seed)
        checks = audit_rows(rows, train, test, shots, seed)
        output = output_dir / f"{shots}shot.jsonl"
        serialized = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ).encode("utf-8")
        protocol_path = output_dir / "protocol.json"
        old_protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
        if old_protocol and old_protocol.get("fewshot_seed") != seed:
            raise ValueError("Existing ARC-Challenge variants use a different seed.")
        if output.exists() and output.read_bytes() != serialized:
            raise ValueError(f"Existing prepared data does not match pinned raw/protocol: {output}")
        _write_unchanged_or_new(output, serialized)
        variants = {}
        for count in (0, 25):
            path = output_dir / f"{count}shot.jsonl"
            if path.exists():
                audit = checks if count == shots else audit_arc_challenge(count, seed)
                variants[str(count)] = {
                    "path": str(path.relative_to(ROOT)), "shots": count,
                    "sha256": arc.sha256(path.read_bytes()), "bytes": path.stat().st_size,
                    "rows": len(test), "audit": audit,
                }
        protocol = {
            "schema_version": 1, "task": TASK, "dataset": "allenai/ai2_arc",
            "subset": "ARC-Challenge", "split": "test",
            "counts": {"train": len(train), "validation": len(dev), "test": len(test)},
            "fewshot_source": "train", "fewshot_seed": seed,
            "fewshot_sampling": (
                "One random.Random(seed), advancing sample(train, shots) for each "
                "test document in official order; pinned ContextSampler verified for every row."
            ),
            "seed_status": "Paper does not specify seed; 1234 is the pinned harness default.",
            "paper_settings": "Table A2: Huginn 0-shot; Ouro, Parcae, Looped-Qwen 25-shot.",
            "paper_url": "https://arxiv.org/html/2610.02185v1",
            "harness_repository": arc.HARNESS_REPO, "harness_commit": HARNESS_COMMIT,
            "archive": {
                "url": arc.ARCHIVE_URL, "size_bytes": arc.ARCHIVE_SIZE,
                "etag": arc.ARCHIVE_ETAG, "observed_s3_version_id": arc.ARCHIVE_VERSION,
                "verification": (
                    "Selected HTTP ranges guarded by If-Match ETag; ZIP CRC32 and "
                    "pinned SHA256 of every extracted member. Full archive SHA256 was not computed."
                ),
            },
            "prompt": {
                "doc_to_text": "Question: {question}\nAnswer:",
                "target_delimiter": " ", "fewshot_delimiter": "\n\n",
                "fewshot_target": "Full correct answer text",
                "chat_template": False, "context_truncation_during_preparation": False,
            },
            "scoring": {
                "acc": "Argmax of sum of natural log probabilities of every continuation token.",
                "acc_norm": (
                    "Argmax of the same sum divided by Python len(original choice text), "
                    "excluding the prepended target delimiter."
                ),
                "ties": "First choice wins (numpy.argmax convention).",
                "tokenization": (
                    "Pinned harness causal pair encoding; no chat template; family-specific "
                    "BOS and any inference-time left context truncation are recorded by evaluator."
                ),
            },
            "sources": records,
            "implementation": {
                "path": "src/data_arc_challenge.py",
                "sha256": arc.sha256(Path(__file__).read_bytes()),
                "range_downloader": "scripts/prepare_arc.py",
                "range_downloader_sha256": arc.sha256(Path(arc.__file__).read_bytes()),
            },
            "outputs": variants,
        }
        _write_unchanged_or_new(
            protocol_path,
            (json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        return output
