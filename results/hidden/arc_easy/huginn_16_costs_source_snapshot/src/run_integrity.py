"""Read-only checks for resuming a scored ARC run.

Call ``validate_resume_manifest`` before writing a run manifest, then load prior
records with ``load_completed_scores``. Add ``native_source_hashes`` to the
current manifest before the comparison. These checks intentionally reject old
manifests that do not record enough information to prove compatibility.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODES = ("baseline", "loopcd_fixed", "loopcd_adaptive", "reference")
REQUIRED_MANIFEST_KEYS = frozenset({
    "name", "configuration", "documents", "candidates", "data_sha256",
    "model_manifest_sha256", "code_hashes", "native_source_hashes",
    "dtype", "float32_matmul_precision", "batch_size", "token_budget", "seed",
    "add_bos", "chat_template", "max_input_tokens", "truncation",
    "torch", "transformers", "adapter_metadata",
})
# Moving the same run to another host/job is safe when all numerical settings,
# versions, source hashes, and the GPU model (when recorded) remain identical.
NON_SEMANTIC_MANIFEST_KEYS = frozenset({"host", "slurm_job_id"})


def sha256_file(path: str | Path) -> str:
    """Hash large files without reading the entire file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_source_hashes(
    family: str, model_path: str | Path, *, root: str | Path = ROOT,
) -> dict[str, str]:
    """Hash the native Python source used by an adapter, without loading models.

    Parcae records the complete downloaded Python source tree (a conservative
    superset of executed modules). Ouro/Huginn record top-level checkpoint
    Python files. Looped-Qwen records the installed transformers Qwen3 package.
    Common library internals are represented by package versions in the run
    manifest; this function does not claim to hash every transitive dependency.
    """
    root = Path(root).resolve()
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = root / model_path
    if family == "parcae":
        directory = root / "sources/parcae"
        paths = sorted(directory.rglob("*.py"))
        prefix = None
    elif family in {"ouro", "huginn"}:
        directory = model_path.resolve()
        paths = sorted(directory.glob("*.py"))
        prefix = None
    elif family == "looped_qwen":
        spec = importlib.util.find_spec("transformers.models.qwen3")
        if spec is None or not spec.submodule_search_locations:
            raise FileNotFoundError("Installed transformers.models.qwen3 source not found")
        locations = list(spec.submodule_search_locations)
        if len(locations) != 1:
            raise ValueError("Ambiguous transformers.models.qwen3 source locations")
        directory = Path(locations[0]).resolve()
        paths = sorted(directory.rglob("*.py"))
        prefix = "transformers/models/qwen3"
    else:
        raise ValueError(f"Unsupported model family: {family}")
    paths = [path for path in paths if path.is_file()]
    if not paths:
        raise FileNotFoundError(f"No native Python sources found for {family}: {directory}")
    result = {}
    for path in paths:
        if prefix:
            name = f"{prefix}/{path.relative_to(directory).as_posix()}"
        else:
            try:
                name = path.absolute().relative_to(root).as_posix()
            except ValueError:
                name = str(path.absolute())
        result[name] = sha256_file(path)
    return result


def _require_manifest_fields(manifest: dict, label: str) -> None:
    if not isinstance(manifest, dict):
        raise ValueError(f"{label} manifest must be a JSON object")
    missing = REQUIRED_MANIFEST_KEYS - manifest.keys()
    if missing:
        raise ValueError(f"{label} manifest lacks required fields: {', '.join(sorted(missing))}")
    if not isinstance(manifest["native_source_hashes"], dict) or not manifest["native_source_hashes"]:
        raise ValueError(f"{label} manifest must record nonempty native_source_hashes")


def compare_resume_manifests(previous: dict, current: dict) -> None:
    """Reject changed/missing settings, including future newly recorded fields.

    Only host and job identifiers are ignored. In particular seed, package
    versions, adapter metadata, source hashes, and any recorded GPU model are
    compared. No previous manifest is rewritten or upgraded by this function.
    """
    _require_manifest_fields(previous, "Previous")
    _require_manifest_fields(current, "Current")
    old = {key: value for key, value in previous.items() if key not in NON_SEMANTIC_MANIFEST_KEYS}
    new = {key: value for key, value in current.items() if key not in NON_SEMANTIC_MANIFEST_KEYS}
    changed = [
        key for key in sorted(old.keys() | new.keys())
        if key not in old or key not in new
        or json.dumps(old[key], sort_keys=True, allow_nan=False)
        != json.dumps(new[key], sort_keys=True, allow_nan=False)
    ]
    if changed:
        raise ValueError(f"Resume configuration changed: {', '.join(changed)}")


def validate_resume_manifest(
    manifest_path: str | Path, score_path: str | Path, current_manifest: dict,
) -> dict | None:
    """Preflight resume state *before* a caller writes any new provenance files."""
    manifest_path, score_path = Path(manifest_path), Path(score_path)
    if score_path.exists() and score_path.stat().st_size and not manifest_path.is_file():
        raise ValueError("Nonempty candidate scores exist without a previous run manifest; refusing reuse")
    _require_manifest_fields(current_manifest, "Current")
    if not manifest_path.exists():
        return None
    try:
        previous = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read previous run manifest: {manifest_path}") from exc
    compare_resume_manifests(previous, current_manifest)
    return previous


def _is_integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_completed_scores(
    score_path: str | Path, candidates: Iterable, *, modes=DEFAULT_MODES,
) -> dict[tuple[int, int], dict]:
    """Read and validate saved scores against the current encoded candidates.

    Candidate objects must expose question_index, choice_index, targets, and
    input_ids, as src.loopcd.Candidate does. Malformed/truncated JSONL is rejected
    with its line number; no records are silently deleted or repaired.
    """
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("Score modes must be nonempty and unique")
    expected = {}
    for candidate in candidates:
        key = (candidate.question_index, candidate.choice_index)
        if key in expected:
            raise ValueError(f"Duplicate expected candidate: {key}")
        expected[key] = candidate
    score_path = Path(score_path)
    if not score_path.exists():
        return {}
    completed = {}
    required_modes = set(modes)
    with score_path.open("r") as stream:
        for line_number, line in enumerate(stream, 1):
            label = f"{score_path.name}:{line_number}"
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{label}: malformed or truncated candidate JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{label}: candidate record must be an object")
            question, choice = row.get("question_index"), row.get("choice_index")
            if not _is_integer(question) or not _is_integer(choice):
                raise ValueError(f"{label}: candidate indices must be integers")
            key = (question, choice)
            if key not in expected:
                raise ValueError(f"{label}: saved candidate outside current candidate set: {key}")
            if key in completed:
                raise ValueError(f"{label}: duplicate saved candidate: {key}")
            candidate = expected[key]
            targets = row.get("target_ids")
            if (
                not isinstance(targets, list)
                or not all(_is_integer(token) and token >= 0 for token in targets)
                or targets != list(candidate.targets)
            ):
                raise ValueError(f"{label}: target_ids changed for candidate {key}")
            length = row.get("input_length")
            if not _is_integer(length) or length != len(candidate.input_ids):
                raise ValueError(f"{label}: input_length changed for candidate {key}")
            scores, token_scores = row.get("scores"), row.get("token_logprobs")
            if not isinstance(scores, dict) or set(scores) != required_modes:
                raise ValueError(f"{label}: missing or unexpected score modes")
            if not isinstance(token_scores, dict) or set(token_scores) != required_modes:
                raise ValueError(f"{label}: missing or unexpected token_logprobs modes")
            for mode in modes:
                values = token_scores[mode]
                if not isinstance(values, list) or len(values) != len(targets):
                    raise ValueError(f"{label}: token_logprobs length mismatch for {mode}")
                if not _finite_number(scores[mode]) or not all(_finite_number(v) for v in values):
                    raise ValueError(f"{label}: non-finite or invalid score for {mode}")
                if not math.isclose(scores[mode], sum(values), rel_tol=1e-12, abs_tol=1e-9):
                    raise ValueError(f"{label}: saved score does not equal token sum for {mode}")
            completed[key] = row
    return completed
