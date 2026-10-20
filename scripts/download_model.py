#!/usr/bin/env python3
"""Download the official Ouro base checkpoint without duplicating a hub cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote

import requests


REPO_ID = "ByteDance/Ouro-1.4B"
REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Ouro-1.4B"
MANIFEST_PATH = ROOT / "models" / "model_manifest.json"
SMALL_FILES = (
    "config.json",
    "configuration_ouro.py",
    "modeling_ouro.py",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "README.md",
)


def hashes(path: Path) -> tuple[str, str]:
    """Return SHA256 and the Git blob SHA1 used by the official HF API."""
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1()
    git_blob.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            git_blob.update(chunk)
    return sha256.hexdigest(), git_blob.hexdigest()


def verify(path: Path, metadata: dict) -> dict:
    size = path.stat().st_size
    expected_size = metadata["size"]
    if size != expected_size:
        raise ValueError(f"{path.name}: expected {expected_size} bytes, found {size}")
    sha256, git_blob = hashes(path)
    lfs_sha256 = metadata.get("lfs", {}).get("sha256")
    if lfs_sha256:
        if sha256 != lfs_sha256:
            raise ValueError(f"{path.name}: LFS SHA256 mismatch")
    elif git_blob != metadata["blobId"]:
        raise ValueError(f"{path.name}: Git blob SHA1 mismatch")
    return {
        "path": str(path.relative_to(ROOT)),
        "filename": metadata["rfilename"],
        "bytes": size,
        "sha256": sha256,
        "official_git_blob_sha1": metadata["blobId"],
        "official_lfs_sha256": lfs_sha256,
    }


def download(session: requests.Session, metadata: dict) -> dict:
    filename = metadata["rfilename"]
    destination = MODEL_DIR / filename
    url = (
        f"https://huggingface.co/{REPO_ID}/resolve/"
        f"{REVISION}/{quote(filename, safe='/')}?download=true"
    )
    if destination.exists():
        result = verify(destination, metadata)
        result["source_url"] = url
        print(f"Verified existing {filename}: {result['bytes']:,} bytes", flush=True)
        return result

    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 7):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == metadata["size"]:
            break
        if offset > metadata["size"]:
            raise ValueError(f"{partial}: partial download exceeds expected size")
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            # A unique query avoids an expired signed CDN redirect cached upstream.
            transfer_url = f"{url}&transfer_attempt={time.time_ns()}"
            with session.get(
                transfer_url, headers=headers, stream=True, timeout=(60, 180)
            ) as response:
                response.raise_for_status()
                if offset and response.status_code == 206:
                    if not response.headers.get("Content-Range", "").startswith(
                        f"bytes {offset}-"
                    ):
                        raise ValueError(f"{filename}: unexpected Content-Range")
                    mode = "ab"
                else:
                    offset = 0
                    mode = "wb"
                downloaded = offset
                last_report = time.monotonic()
                with partial.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        stream.write(chunk)
                        downloaded += len(chunk)
                        if time.monotonic() - last_report >= 20:
                            print(
                                f"{filename}: {downloaded:,}/{metadata['size']:,} bytes",
                                flush=True,
                            )
                            last_report = time.monotonic()
            if partial.stat().st_size != metadata["size"]:
                raise requests.RequestException(f"{filename}: incomplete transfer")
            break
        except requests.RequestException as exc:
            if attempt == 6:
                raise
            print(f"{filename}: transfer attempt {attempt} failed: {exc}", flush=True)
            time.sleep(min(attempt * 2, 10))
    result = verify(partial, metadata)
    partial.replace(destination)
    result["path"] = str(destination.relative_to(ROOT))
    result["source_url"] = url
    print(f"Downloaded and verified {filename}: {result['bytes']:,} bytes", flush=True)
    return result


def write_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(MANIFEST_PATH)


def same_snapshot(previous: dict, current: dict) -> bool:
    """Ignore download timestamps when the verified model snapshot is unchanged."""
    keys = (
        "repository", "source", "revision", "model_variant", "official_metadata_url",
        "complete", "integrity_checks", "total_bytes",
    )
    return all(previous.get(key) == current.get(key) for key in keys) and (
        sorted(previous.get("files", []), key=lambda item: item["filename"])
        == sorted(current["files"], key=lambda item: item["filename"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--small-files-only",
        action="store_true",
        help="Fetch configuration, source, and tokenizer without model weights.",
    )
    args = parser.parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = "ouro-single-model-reproduction/1.0"
    api_url = f"https://huggingface.co/api/models/{REPO_ID}/revision/{REVISION}?blobs=true"
    response = session.get(api_url, timeout=60)
    response.raise_for_status()
    api = response.json()
    if api["sha"] != REVISION:
        raise ValueError("Official API did not resolve to the pinned revision")
    available = {item["rfilename"]: item for item in api["siblings"]}
    previous = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    previous_complete = previous.get("complete") is True
    wanted = list(SMALL_FILES)
    if not args.small_files_only or previous_complete:
        wanted.append("model.safetensors")
    manifest = {
        "repository": REPO_ID,
        "source": f"https://huggingface.co/{REPO_ID}",
        "revision": REVISION,
        "model_variant": "base (not Thinking)",
        "official_metadata_url": api_url,
        "download_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "integrity_checks": "LFS SHA256 for weights; Git blob SHA1 for small files; SHA256 recorded for every file.",
        "files": [],
    }
    for filename in wanted:
        if previous_complete and args.small_files_only:
            # A complete snapshot is retained only after verifying every file.
            # Do not fetch missing weights in --small-files-only mode.
            if not (MODEL_DIR / filename).exists():
                raise FileNotFoundError(
                    f"Complete manifest refers to missing {filename}; "
                    "rerun without --small-files-only to repair the download. "
                    "The existing manifest has been left unchanged."
                )
        manifest["files"].append(download(session, available[filename]))
        manifest["total_bytes"] = sum(item["bytes"] for item in manifest["files"])
        # Keep an existing complete manifest intact until all checks succeed.
        # New or interrupted downloads still record their incremental progress.
        if not previous_complete:
            write_manifest(manifest)
    manifest["complete"] = previous_complete or not args.small_files_only
    if previous_complete and same_snapshot(previous, manifest):
        print(f"Reusing verified complete manifest unchanged: {MANIFEST_PATH}", flush=True)
        return
    if previous_complete and args.small_files_only:
        raise ValueError(
            "Existing complete manifest differs from the verified pinned snapshot; "
            "rerun without --small-files-only to update it. "
            "The existing manifest has been left unchanged."
        )
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest)
    print(f"Manifest saved: {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
