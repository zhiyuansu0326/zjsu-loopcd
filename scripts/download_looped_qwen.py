#!/usr/bin/env python3
"""Download the pinned official dense Qwen3-4B used by the LoopCD wrapper."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import Path
import time

import requests


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/Qwen3-4B"
MANIFEST = ROOT / "models/looped_qwen_manifest.json"
REPOSITORY = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
FILES = (
    "LICENSE", "README.md", "config.json", "generation_config.json",
    "merges.txt", "vocab.json", "tokenizer.json", "tokenizer_config.json",
    "model.safetensors.index.json", "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors",
)


def verify(path: Path, entry: dict) -> dict:
    if path.stat().st_size != entry["size"]:
        raise ValueError(f"Size mismatch: {path}")
    digest = hashlib.sha256()
    blob = hashlib.sha1(f"blob {entry['size']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            blob.update(chunk)
    expected = entry.get("lfs", {}).get("sha256")
    if expected:
        if digest.hexdigest() != expected:
            raise ValueError(f"Official LFS SHA256 mismatch: {path}")
    elif blob.hexdigest() != entry["blobId"]:
        raise ValueError(f"Official Git blob SHA1 mismatch: {path}")
    return {
        "filename": entry["rfilename"], "bytes": entry["size"],
        "sha256": digest.hexdigest(), "official_lfs_sha256": expected,
        "official_git_blob_sha1": entry["blobId"],
        "source_url": f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{entry['rfilename']}",
    }


def download(entry: dict) -> dict:
    filename = entry["rfilename"]
    destination = MODEL_DIR / filename
    if destination.exists():
        result = verify(destination, entry)
        print(f"Verified {filename}", flush=True)
        return result
    partial = destination.with_suffix(destination.suffix + ".part")
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{filename}"
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == entry["size"]:
            break
        if offset > entry["size"]:
            raise ValueError(f"Overlong partial file: {partial}")
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(
                f"{url}?download=true&transfer={time.time_ns()}",
                headers=headers, stream=True, timeout=(60, 180),
            ) as response:
                response.raise_for_status()
                append = offset and response.status_code == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Unexpected Content-Range")
                with partial.open("ab" if append else "wb") as stream:
                    last_report = time.monotonic()
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                        if time.monotonic() - last_report >= 20:
                            print(f"{filename}: {stream.tell():,}/{entry['size']:,} bytes", flush=True)
                            last_report = time.monotonic()
            if partial.stat().st_size != entry["size"]:
                raise requests.RequestException("Incomplete transfer")
            break
        except requests.RequestException as error:
            if attempt == 5:
                raise
            print(f"Retry {filename}: {type(error).__name__}", flush=True)
            time.sleep(min(2 * (attempt + 1), 10))
    result = verify(partial, entry)
    partial.replace(destination)
    print(f"Downloaded and verified {filename}: {entry['size']:,} bytes", flush=True)
    return result


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    metadata = response.json()
    if metadata["sha"] != REVISION:
        raise ValueError("Pinned revision mismatch")
    available = {entry["rfilename"]: entry for entry in metadata["siblings"]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        # Retrieve small configuration/tokenizer files before launching weight shards.
        records = list(pool.map(download, [available[name] for name in FILES[:9]]))
        records += list(pool.map(download, [available[name] for name in FILES[9:]]))
    manifest = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "official_metadata_url": url,
        "complete": True,
        "stored_dtype": "bfloat16",
        "model_variant": "Original Qwen3-4B release; not the distinct Qwen3-4B-Base checkpoint",
        "runtime_dtype": "Selected by adapter constructor; recorded separately by each evaluation run",
        "files": records,
        "total_bytes": sum(entry["bytes"] for entry in records),
        "wrapper": {
            "source": "LoopCD arXiv:2610.02185v1 Table A1, Appendix A.2",
            "prelude_zero_based": [0, 14],
            "window_zero_based": [15, 18],
            "tail_zero_based": [19, 35],
            "loops": 8,
            "update": "h_next=(1-1/8)*h + (1/8)*g(h)",
            "reference": "first completed damped substep, then all 17 tail layers, final norm and shared lm_head",
            "anchor_weight": 0,
            "use_cache": False,
            "fewshot": 0,
            "fixed_strength": 0.5,
            "adaptive_cap": 1.0,
            "assumptions": [
                "LoopCD does not identify an exact checkpoint revision or release a complete wrapper.",
                "Use Qwen/Qwen3-4B, matching the stated dense Qwen3-4B name; the exact authors' checkpoint choice is not proven.",
                "Use only the damped Euler rule explicitly given by LoopCD; no anchored blend or additional normalization inside recurrence.",
                "The upstream Training-Free repository default K=6, beta=0.5 is not the LoopCD K=8 recipe.",
                "All likelihood positions use complete causal sequences without KV-cache reuse; decode-time cache choices do not apply.",
            ],
        },
    }
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(MANIFEST)
    print(f"Complete: {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
