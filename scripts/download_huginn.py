#!/usr/bin/env python3
"""Download the immutable official Huginn release and verify every shard."""

import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
TARGET = ROOT / "models/Huginn-0125"
SOURCE = ROOT / "sources/huginn"


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            sha.update(block)
    return sha.hexdigest()


def download(entry):
    name = entry["rfilename"]
    destination = TARGET / name
    expected = entry.get("lfs", {}).get("sha256")
    if destination.exists() and destination.stat().st_size == entry["size"]:
        actual = digest(destination)
        if expected is None or actual == expected:
            print(json.dumps({"file": name, "status": "verified-existing"}), flush=True)
            return {"path": name, "bytes": entry["size"], "sha256": actual}
    temporary = destination.with_name(destination.name + ".partial")
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
    for attempt in range(4):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=120) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with temporary.open(mode) as handle:
                    while block := response.read(8 * 1024**2):
                        handle.write(block)
            if temporary.stat().st_size != entry["size"]:
                raise ValueError(f"Incomplete download: {name}")
            actual = digest(temporary)
            if expected is not None and actual != expected:
                raise ValueError(f"SHA256 mismatch: {name}")
            temporary.replace(destination)
            print(json.dumps({"file": name, "status": "verified", "bytes": entry["size"]}), flush=True)
            return {"path": name, "bytes": entry["size"], "sha256": actual}
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    metadata_url = f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true"
    metadata = json.load(urllib.request.urlopen(metadata_url))
    assert metadata["sha"] == REVISION
    (SOURCE / "hf_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    entries = [
        row for row in metadata["siblings"]
        if row["rfilename"].endswith((".py", ".json", ".safetensors")) or row["rfilename"] == "README.md"
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(download, entries))
    for name in ("raven_config_minimal.py", "raven_modeling_minimal.py", "README.md"):
        (SOURCE / name).write_bytes((TARGET / name).read_bytes())
    manifest = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "source_url": f"https://huggingface.co/{REPOSITORY}/tree/{REVISION}",
        "files": sorted(files, key=lambda row: row["path"]),
        "source_code_unmodified": True,
        "architecture": {"prelude_layers": 2, "recurrent_layers": 4, "coda_layers": 2,
                         "hidden_size": 5280, "context_length": 4096, "vocabulary": 65536},
    }
    (ROOT / "models/huginn_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "complete", "revision": REVISION}), flush=True)


if __name__ == "__main__":
    main()
