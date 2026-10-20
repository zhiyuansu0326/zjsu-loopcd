#!/usr/bin/env python3
"""Fetch official, pinned Parcae weights, tokenizer, and unmodified source."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "69284c13746e849104f738d6d1a347b1f457df76"
TOKENIZER_REPO = "SandyResearch/parcae-tokenizer"
TOKENIZER_REVISION = "6247b5d0592876b73660f34c2dd16c9db9e9045c"
MODELS = {
    "370m": ("SandyResearch/parcae-370m", "439284464ee4999bd1f762da7d044613a4828efe", "Parcae-370M"),
    "1.3b": ("SandyResearch/parcae-1.3b", "517f8e6add927e3d73e15beec13a5b500537b9cc", "Parcae-1.3B"),
}


def get_json(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def verify(path, size, sha256=None, git_blob=None):
    if path.stat().st_size != size:
        raise ValueError(f"Size mismatch: {path}")
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            sha.update(chunk)
            git.update(chunk)
    if sha256 and sha.hexdigest() != sha256:
        raise ValueError(f"LFS SHA256 mismatch: {path}")
    if not sha256 and git_blob and git.hexdigest() != git_blob:
        raise ValueError(f"Git blob SHA1 mismatch: {path}")
    return sha.hexdigest()


def fetch(url, path, size, sha256=None, git_blob=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        part = path.with_name(path.name + ".part")
        for attempt in range(6):
            offset = part.stat().st_size if part.exists() else 0
            if offset == size:
                break
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                transfer_url = url + ("&" if "?" in url else "?") + f"transfer={time.time_ns()}"
                with requests.get(transfer_url, headers=headers, stream=True, timeout=(60, 180)) as response:
                    response.raise_for_status()
                    mode = "ab" if offset and response.status_code == 206 else "wb"
                    if mode == "ab" and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Unexpected range response")
                    with part.open(mode) as stream:
                        for chunk in response.iter_content(8 * 1024**2):
                            stream.write(chunk)
                if part.stat().st_size != size:
                    raise requests.RequestException("Incomplete transfer")
                break
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(2 * (attempt + 1))
        digest = verify(part, size, sha256, git_blob)
        part.replace(path)
        print(f"Downloaded {path.relative_to(ROOT)}: {size:,} bytes", flush=True)
    else:
        digest = verify(path, size, sha256, git_blob)
        print(f"Verified {path.relative_to(ROOT)}: {size:,} bytes", flush=True)
    return {
        "path": str(path.relative_to(ROOT)), "url": url, "bytes": size, "sha256": digest,
        "official_lfs_sha256": sha256, "official_git_blob_sha1": git_blob,
    }


def stable_json(path, data):
    text = json.dumps(data, indent=2) + "\n"
    if path.exists() and json.loads(path.read_text()) == data:
        return
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def download_source():
    api = get_json(
        f"https://api.github.com/repos/sandyresearch/parcae/git/trees/{SOURCE_REVISION}?recursive=1"
    )
    extra = {"README.md", "LICENSE", "pyproject.toml", "receval/models/parcae.py",
             "receval/tasks/lm_eval.py", "receval/settings.py", "scripts/eval.py"}
    files = [f for f in api["tree"] if f["type"] == "blob" and
             (f["path"].startswith("parcae_lm/") or f["path"] in extra)]
    source_dir = ROOT / "sources/parcae"

    def one(item):
        url = f"https://raw.githubusercontent.com/sandyresearch/parcae/{SOURCE_REVISION}/{item['path']}"
        result = fetch(url, source_dir / item["path"], item["size"], git_blob=item["sha"])
        return {"path": item["path"], "url": url, "bytes": result["bytes"],
                "sha256": result["sha256"], "git_blob_sha1": item["sha"]}

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(one, files))
    stable_json(source_dir / "manifest.json", {
        "source": "https://github.com/sandyresearch/parcae",
        "revision": SOURCE_REVISION, "files": records,
    })


def download_model(key):
    repo, revision, folder = MODELS[key]
    api = get_json(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true")
    assert api["sha"] == revision
    available = {f["rfilename"]: f for f in api["siblings"]}
    token_api = get_json(
        f"https://huggingface.co/api/models/{TOKENIZER_REPO}/revision/{TOKENIZER_REVISION}?blobs=true"
    )
    assert token_api["sha"] == TOKENIZER_REVISION
    token = next(f for f in token_api["siblings"] if f["rfilename"] == "tokenizer.json")
    directory = ROOT / "models" / folder
    records = []
    for filename in ("README.md", "config.json", "pytorch_model.bin", "tokenizer.json"):
        metadata = token if filename == "tokenizer.json" else available[filename]
        origin, commit = (TOKENIZER_REPO, TOKENIZER_REVISION) if filename == "tokenizer.json" else (repo, revision)
        records.append(fetch(
            f"https://huggingface.co/{origin}/resolve/{commit}/{filename}",
            directory / filename, metadata["size"],
            sha256=metadata.get("lfs", {}).get("sha256"), git_blob=metadata["blobId"],
        ))
    manifest = {
        "repository": repo, "revision": revision, "complete": True,
        "tokenizer_repository": TOKENIZER_REPO, "tokenizer_revision": TOKENIZER_REVISION,
        "source_repository": "https://github.com/sandyresearch/parcae",
        "source_revision": SOURCE_REVISION,
        "total_bytes": sum(r["bytes"] for r in records), "files": records,
    }
    stable_json(ROOT / "models" / f"parcae_{key.replace('.', '_')}_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", *MODELS), default="all")
    args = parser.parse_args()
    download_source()
    selected = list(MODELS) if args.model == "all" else [args.model]
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(download_model, selected))


if __name__ == "__main__":
    main()
