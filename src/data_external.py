"""Pinned, standard-library-only SciQ/PIQA preparation for likelihood evaluation.

Public API: prepare_external(task, shots=0, seed=1234) -> protocol dictionary.
The returned dictionary is also written to data/tasks/<task>/protocol.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HARNESS_COMMIT = "d6de81643928d653435c431bae19945d41d32520"
SCIQ_CARD_REVISION = "2c94ad3e1aafab77146f384e23536f97a4849815"
COUNTS = {"sciq": 1000, "piqa": 1838}
SPLITS = {"sciq": "test", "piqa": "validation"}
PRIMARY_METRICS = {"sciq": "acc", "piqa": "acc_norm"}

# Archives/site files do not advertise immutable release revisions. Their exact
# bytes are pinned by SHA256 instead of treating a floating URL as a revision.
DOWNLOADS = {
    "sciq": [
        ("SciQ.zip", "https://ai2-public-datasets.s3.amazonaws.com/sciq/SciQ.zip",
         "7f3312f6ac6b09970b32942d106a8c44ec0dad46a0369f17d635aff8e348a87c"),
        ("README.official.md",
         f"https://huggingface.co/datasets/allenai/sciq/resolve/{SCIQ_CARD_REVISION}/README.md",
         "f16f71b220a0e205672f6f0c8afb40e37f7a158541f759593745fe52162f8ad8"),
    ],
    "piqa": [
        ("physicaliqa-train-dev.zip",
         "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip",
         "54d32a04f59a7e354396f321723c8d7ec35cc6b08506563d8d1ffcc15ce98ddd"),
        ("valid.jsonl", "https://yonatanbisk.com/piqa/data/valid.jsonl",
         "93503cc97c679e459b065c3d13e848282e44b2a25213985bed3e5d458abef72d"),
        ("valid-labels.lst", "https://yonatanbisk.com/piqa/data/valid-labels.lst",
         "b4192dc3a2a0363d9d60ccf79800cbbe2f32ebb17726efdde6970e0b8131bceb"),
        ("official_data_page.html", "https://yonatanbisk.com/piqa/data/",
         "618bda94355e50ed52a1d3d7abaa6e2c523ed4c4062ea3fce0eacaa9beb7a294"),
    ],
}
MEMBERS = {
    "sciq": [
        ("SciQ.zip", "SciQ dataset-2 3/test.json", "test.json",
         "7340983c427dc2ffd3dccf1adb758b6cdc9455bd22bf3db0ddf6ffcae0452006"),
        ("SciQ.zip", "SciQ dataset-2 3/readme.txt", "readme.txt",
         "67c00d15da00e2ba4045930f81b051bbfab38c956dad0bb3d0c697a47efb69ec"),
        ("SciQ.zip", "SciQ dataset-2 3/license.txt", "license.txt",
         "6f2f301537b1f94b025af00467a9c3412bc249a40b4075a6d31b1c21ad7359e3"),
    ],
    "piqa": [
        ("physicaliqa-train-dev.zip", "physicaliqa-train-dev/dev.jsonl", "dev.jsonl",
         "a73f46cb5d72b2cc399522a79f27728a429d9b89d9a236243fcfda8c02588d54"),
        ("physicaliqa-train-dev.zip", "physicaliqa-train-dev/dev-labels.lst", "dev-labels.lst",
         "b4192dc3a2a0363d9d60ccf79800cbbe2f32ebb17726efdde6970e0b8131bceb"),
    ],
}
HARNESS_FILES = {
    "sciq": {
        "sciq.yaml": "d0409dc9466fed64e908c45111489a73c4a004d6af9a968dc5d85228823696de",
        "README.md": "5c25c17f729fa4f589a818aa2a8811feac96a852be5d028e8661a8ce5470e346",
    },
    "piqa": {
        "piqa.yaml": "e874f7956ccac325b888d4a5dc600bad867b79fba2bd358fe8761ab8263270fb",
        "README.md": "788cef081bb9fff2afe8a30984ce17fffa7c0401674d5abd4f558306e713825a",
    },
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _record(path, **metadata):
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": sha256_file(path), **metadata}


def _write_verified(path, data, expected):
    _require(hashlib.sha256(data).hexdigest() == expected, f"Downloaded/extracted checksum mismatch: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _download_verified(path, url, expected):
    if path.exists():
        _require(sha256_file(path) == expected, f"Existing source checksum mismatch: {path}")
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "LoopCD-Reproduction/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            _write_verified(path, response.read(), expected)
    return _record(path, url=url, pinning="sha256")


def _ensure_assets(task):
    raw_dir = ROOT / "data/raw" / task
    raw_files = []
    for filename, url, expected in DOWNLOADS[task]:
        raw_files.append(_download_verified(raw_dir / filename, url, expected))
    for archive, member, filename, expected in MEMBERS[task]:
        path = raw_dir / filename
        if path.exists():
            _require(sha256_file(path) == expected, f"Existing extracted source checksum mismatch: {path}")
        else:
            with zipfile.ZipFile(raw_dir / archive) as handle:
                _write_verified(path, handle.read(member), expected)
        raw_files.append(_record(path, archive=str((raw_dir / archive).relative_to(ROOT)), member=member))
    sources = []
    for filename, expected in HARNESS_FILES[task].items():
        relative = f"lm_eval/tasks/{task}/{filename}"
        path = ROOT / "sources/lm_eval" / relative
        url = f"https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/{HARNESS_COMMIT}/{relative}"
        sources.append({**_download_verified(path, url, expected), "revision": HARNESS_COMMIT})
    license_path = ROOT / "sources/lm_eval/LICENSE.md"
    _require(sha256_file(license_path) == "a806e42547620dffbc2ec0333322c8b3523b4af11e059be14ef81aa5b1ae021f",
             "Pinned lm-eval license checksum mismatch")
    sources.append(_record(license_path, revision=HARNESS_COMMIT, license="MIT",
                           url=f"https://github.com/EleutherAI/lm-evaluation-harness/blob/{HARNESS_COMMIT}/LICENSE.md"))
    return raw_files, sources


def _string_fields(raw, fields):
    for field in fields:
        _require(isinstance(raw.get(field), str), f"Expected string field: {field}")


def normalize_sciq(raw, index):
    """Exactly match the pinned harness, including empty support's leading LF."""
    _string_fields(raw, ("question", "support", "distractor1", "distractor2",
                         "distractor3", "correct_answer"))
    return {
        "id": f"sciq:test:{index:06d}", "task": "sciq", "split": "test", "source_index": index,
        "question": raw["question"],
        "choices": [raw["distractor1"], raw["distractor2"], raw["distractor3"], raw["correct_answer"]],
        "labels": ["A", "B", "C", "D"], "gold": 3,
        "context": raw["support"].lstrip() + "\nQuestion: " + raw["question"] + "\nAnswer:",
        "fewshot_ids": [], "support": raw["support"],
    }


def normalize_piqa(raw, label, index):
    """The official archive's dev split is the public labeled validation split."""
    _string_fields(raw, ("id", "goal", "sol1", "sol2"))
    _require(bool(raw["id"]), "PIQA source ID is empty")
    _require(type(label) is int and label in (0, 1), "PIQA label must be integer 0 or 1")
    return {
        "id": "piqa:validation:" + raw["id"], "task": "piqa", "split": "validation",
        "source_index": index, "source_id": raw["id"], "question": raw["goal"],
        "choices": [raw["sol1"], raw["sol2"]], "labels": ["A", "B"], "gold": label,
        "context": "Question: " + raw["goal"] + "\nAnswer:", "fewshot_ids": [],
    }


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _load_raw(task):
    raw_dir = ROOT / "data/raw" / task
    if task == "sciq":
        return json.loads((raw_dir / "test.json").read_text()), None
    documents = _jsonl(raw_dir / "dev.jsonl")
    label_text = (raw_dir / "dev-labels.lst").read_text().splitlines()
    _require(len(documents) == len(label_text), "PIQA document/label line counts differ")
    _require(all(label in ("0", "1") for label in label_text), "PIQA labels must be 0/1 lines")
    # Independently published author-site files confirm text/order and label alignment.
    author_documents = _jsonl(raw_dir / "valid.jsonl")
    _require(len(documents) == len(author_documents), "PIQA author-site/archive counts differ")
    _require(all(all(doc[field] == author[field] for field in ("goal", "sol1", "sol2"))
                 for doc, author in zip(documents, author_documents)),
             "PIQA author-site/archive content or order differs")
    _require((raw_dir / "dev-labels.lst").read_bytes() == (raw_dir / "valid-labels.lst").read_bytes(),
             "PIQA author-site/archive labels differ")
    return documents, [int(label) for label in label_text]


def audit_external(task, rows):
    """Audit every prepared row directly against canonical source fields."""
    _require(task in COUNTS, f"Unsupported external task: {task}")
    raw, labels = _load_raw(task)
    _require(len(raw) == len(rows) == COUNTS[task], f"Wrong complete split size for {task}")
    _require(len({row["id"] for row in rows}) == COUNTS[task], "Duplicate prepared IDs")
    for index, (doc, row) in enumerate(zip(raw, rows)):
        _require(row["source_index"] == index, "Source order changed")
        _require(row["task"] == task and row["split"] == SPLITS[task], "Wrong task/split")
        _require(row["fewshot_ids"] == [], "Unexpected demonstration")
        _require(isinstance(row["choices"], list) and all(isinstance(x, str) and x for x in row["choices"]),
                 "Empty/nontext choice")
        _require(isinstance(row["labels"], list) and all(isinstance(x, str) for x in row["labels"]),
                 "Nontext choice label")
        _require(type(row["gold"]) is int and 0 <= row["gold"] < len(row["choices"]), "Invalid target index")
        if task == "sciq":
            _require(row["id"] == f"sciq:test:{index:06d}" and row["question"] == doc["question"]
                     and row["support"] == doc["support"], "SciQ source identity/content differs")
            _require(row["choices"] == [doc[k] for k in ("distractor1", "distractor2", "distractor3", "correct_answer")]
                     and row["labels"] == ["A", "B", "C", "D"] and row["gold"] == 3,
                     "SciQ choice order/target differs")
            context = f"{doc['support'].lstrip()}\nQuestion: {doc['question']}\nAnswer:"
        else:
            _require(row["id"] == "piqa:validation:" + doc["id"] and row["source_id"] == doc["id"]
                     and row["question"] == doc["goal"], "PIQA source identity/content differs")
            _require(row["choices"] == [doc["sol1"], doc["sol2"]] and row["labels"] == ["A", "B"]
                     and row["gold"] == labels[index], "PIQA choice order/target differs")
            context = f"Question: {doc['goal']}\nAnswer:"
        _require(row["context"] == context, "Prepared context differs from canonical template")
    return {
        "passed": True, "documents": len(rows), "unique_ids": len({r["id"] for r in rows}),
        "candidates": sum(len(row["choices"]) for row in rows),
        "all_source_fields_order_choices_labels_and_contexts_match": True,
        "support_empty_count": sum(not doc["support"] for doc in raw) if task == "sciq" else None,
        "piqa_author_site_matches_archive": True if task == "piqa" else None,
    }


def audit_prepared(task, shots=0, seed=1234):
    """Read-only audit entry point for the generic result analyzer; no downloads."""
    _require(task in COUNTS, f"Unsupported external task: {task}")
    _require(type(shots) is int and shots == 0, "SciQ/PIQA protocol supports zero-shot only")
    raw_dir = ROOT / "data/raw" / task
    verified = {}
    for filename, _, expected in DOWNLOADS[task]:
        path = raw_dir / filename
        _require(sha256_file(path) == expected, f"Canonical downloaded source checksum mismatch: {path}")
        verified[str(path.relative_to(ROOT))] = expected
    for _, _, filename, expected in MEMBERS[task]:
        path = raw_dir / filename
        _require(sha256_file(path) == expected, f"Canonical extracted source checksum mismatch: {path}")
        verified[str(path.relative_to(ROOT))] = expected
    for filename, expected in HARNESS_FILES[task].items():
        path = ROOT / f"sources/lm_eval/lm_eval/tasks/{task}/{filename}"
        _require(sha256_file(path) == expected, f"Pinned harness checksum mismatch: {path}")
        verified[str(path.relative_to(ROOT))] = expected
    protocol_path = ROOT / f"data/tasks/{task}/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    output_path = ROOT / f"data/tasks/{task}/0shot.jsonl"
    _require(protocol["task"] == task and protocol["split"] == SPLITS[task]
             and protocol["shots"] == shots and protocol["seed"] == seed
             and protocol["primary_metric"] == PRIMARY_METRICS[task]
             and protocol["reported_metrics"] == ["acc", "acc_norm"]
             and protocol["harness_commit"] == HARNESS_COMMIT, "Prepared protocol mismatch")
    _require(protocol["output"]["path"] == str(output_path.relative_to(ROOT))
             and protocol["output"]["sha256"] == sha256_file(output_path)
             and protocol["output"]["rows"] == COUNTS[task], "Prepared output hash/count mismatch")
    for record in protocol["raw_files"] + protocol["sources"]:
        _require(sha256_file(ROOT / record["path"]) == record["sha256"], "Protocol source checksum mismatch")
    audit = audit_external(task, _jsonl(output_path))
    _require(protocol["count"] == audit["documents"] and protocol["candidates"] == audit["candidates"],
             "Protocol audit counts mismatch")
    return {**audit, "canonical_raw_and_frozen_harness_hashes_verified": True,
            "raw_and_source_sha256": verified, "output_sha256": sha256_file(output_path),
            "protocol_sha256": sha256_file(protocol_path)}


def prepare_external(task, shots=0, seed=1234):
    """Prepare the complete frozen split and return the written protocol dict.

    Requires no GPU, datasets library, tokenizer, or model. Cached source bytes
    are always checked; a corrupted cache is rejected instead of overwritten.
    """
    _require(task in COUNTS, f"Unsupported external task: {task}")
    _require(type(shots) is int and shots == 0, "SciQ/PIQA protocol supports zero-shot only")
    _require(type(seed) is int, "Seed must be an integer")
    raw_files, sources = _ensure_assets(task)
    raw, labels = _load_raw(task)
    rows = [normalize_sciq(doc, index) if task == "sciq" else normalize_piqa(doc, labels[index], index)
            for index, doc in enumerate(raw)]
    audit = audit_external(task, rows)
    output_dir = ROOT / "data/tasks" / task
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "0shot.jsonl"
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    _write_verified(output_path, serialized, hashlib.sha256(serialized).hexdigest())
    license_info = (
        {"name": "Creative Commons Attribution-NonCommercial 3.0 Unported",
         "spdx": "CC-BY-NC-3.0", "source": "data/raw/sciq/license.txt",
         "url": "https://creativecommons.org/licenses/by-nc/3.0/"}
        if task == "sciq" else
        {"name": "Academic Free License v3.0", "spdx": "AFL-3.0",
         "source": "data/raw/piqa/official_data_page.html", "url": "https://yonatanbisk.com/piqa/data/"}
    )
    protocol = {
        "schema_version": 1, "task": task, "split": SPLITS[task], "count": len(rows),
        "candidates": audit["candidates"], "shots": shots, "seed": seed,
        "seed_effect": "None for zero-shot data preparation; retained for experiment provenance.",
        "primary_metric": PRIMARY_METRICS[task], "reported_metrics": ["acc", "acc_norm"],
        "primary_metric_status": "Predeclared reproduction convention; the paper does not unambiguously specify this normalization. Both metrics are retained.",
        "harness_repository": "https://github.com/EleutherAI/lm-evaluation-harness",
        "harness_commit": HARNESS_COMMIT,
        "harness_dataset_path": "allenai/sciq" if task == "sciq" else "baber/piqa",
        "data_source": "Official AI2 release archive" if task == "sciq"
                       else "Official AI2 release archive, cross-checked against author-hosted data",
        "source_revision": {"kind": "sha256", "archive_sha256": DOWNLOADS[task][0][2]},
        "official_sciq_card_revision": SCIQ_CARD_REVISION if task == "sciq" else None,
        "prompt_template": "{{support.lstrip()}}\nQuestion: {{question}}\nAnswer:" if task == "sciq"
                           else "Question: {{goal}}\nAnswer:",
        "choice_fields": ["distractor1", "distractor2", "distractor3", "correct_answer"] if task == "sciq"
                         else ["sol1", "sol2"],
        "target_rule": "Fixed index 3, matching original field order" if task == "sciq"
                       else "Aligned dev-labels.lst integer 0/1",
        "support_policy": "Include support.lstrip(); retain all 1000 test rows, including empty support."
                          if task == "sciq" else None,
        "split_note": "Official labeled test split." if task == "sciq"
                      else "Public labeled validation (archive dev). No private/unreleased test labels used or inferred.",
        "selection_policy": "Complete official split in source order; no filtering, answer shuffling, or label-based selection.",
        "continuation": "one leading space followed by complete candidate answer text",
        "apply_chat_template": False, "system_instruction": None, "truncation": False,
        "token_boundary": "Move context trailing whitespace to continuation; encode context and context+continuation identically; score all continuation tokens only.",
        "tokenizer_special_tokens": "Chosen and recorded by model adapter; not added during data preparation.",
        "acc": "argmax of summed conditional natural-log token probabilities",
        "acc_norm": "argmax of summed conditional natural-log token probabilities / Python len(original choice text); Unicode characters, not tokens, excluding added leading space",
        "ties": "first maximum in original candidate order",
        "license": license_info, "raw_files": raw_files, "sources": sources,
        "output": {"path": str(output_path.relative_to(ROOT)), "sha256": sha256_file(output_path), "rows": len(rows)},
        "validation": audit,
    }
    protocol_path = output_dir / "protocol.json"
    data = (json.dumps(protocol, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _write_verified(protocol_path, data, hashlib.sha256(data).hexdigest())
    return protocol
