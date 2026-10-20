#!/usr/bin/env python3
"""Audit all four tasks and seal their reports after the GPU jobs finish."""
import argparse
from datetime import datetime, timezone
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.run_integrity import sha256_file

BASE = ROOT / "results/multitask"

PENDING = (
    "Final verification of the 28 task/model configurations is pending or incomplete. "
    "See [completion.json](results/multitask/completion.json) for the current status. "
    "Original scores and execution source snapshots are preserved."
)
FINISHED = (
    "All **28 task/model configurations are complete and independently audited**: "
    "7 configurations across ARC-Easy, ARC-Challenge, SciQ, and PIQA. "
    "See the [four-task report](results/multitask/REPORT.md), "
    "[JSON](results/multitask/comparison.json), and [CSV](results/multitask/comparison.csv). "
    "Original scores and execution source snapshots are preserved."
)


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    status = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "complete": False, "state": "running",
    }
    history = BASE / "finalization_history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for filename in ("completion.json", "artifact_manifest.json"):
        previous = BASE / filename
        if previous.exists():
            history.mkdir(parents=True, exist_ok=True)
            shutil.move(str(previous), str(history / filename))
    atomic_json(BASE / "completion.json", status)
    readme_path = ROOT / "README.md"
    if readme_path.exists():
        readme_path.write_text(readme_path.read_text().replace(FINISHED, PENDING))
    try:
        jobs_path = BASE / "jobs.json"
        if jobs_path.exists() and status["slurm_job_id"]:
            jobs = json.loads(jobs_path.read_text())
            accounting = subprocess.run(
                ["sacct", "-X", "-j", ",".join(row["job_id"] for row in jobs),
                 "--format=JobIDRaw,Partition,Account,State,Elapsed,ExitCode",
                 "--noheader", "--parsable2"],
                capture_output=True, text=True, check=True,
            )
            (BASE / "slurm_accounting_final.txt").write_text(accounting.stdout)
        with (BASE / "tests_final.log").open("w") as log:
            subprocess.run(["make", "test"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(
            [sys.executable, "-B", "scripts/analyze.py", "--task", "all"],
            cwd=ROOT, check=True,
        )
        result = json.loads((BASE / "comparison.json").read_text())
        if not result["complete"] or len(result["tasks"]) != 4:
            raise ValueError("All four tasks must pass their full audits before finalization.")
        status["configurations"] = sum(len(task["models"]) for task in result["tasks"].values())
        if status["configurations"] != 28:
            raise ValueError("Expected 28 complete task/model configurations.")
        status["question_evaluations"] = sum(
            model["n"] for task in result["tasks"].values() for model in task["models"].values()
        )
        status["scored_candidates"] = sum(
            model["audit"]["unique_candidates"]
            for task in result["tasks"].values() for model in task["models"].values()
        )
        status["answer_token_logprobs_all_four_policies"] = 4 * sum(
            model["audit"]["answer_token_count_per_policy"]
            for task in result["tasks"].values() for model in task["models"].values()
        )
        readme_path = ROOT / "README.md"
        readme = readme_path.read_text()
        if PENDING in readme:
            readme_path.write_text(readme.replace(PENDING, FINISHED))
        files = set()
        for name in ("AGENTS.md", "README.md", "Makefile", ".gitignore",
                     "requirements.txt", "requirements.freeze.txt",
                     "results/README.md", "results/cleanup_20261020.json"):
            files.add(ROOT / name)
        files.update(ROOT.glob("*.pdf"))
        files.update((ROOT / "models").glob("*manifest.json"))
        for directory in ("src", "scripts", "configs", "tests", "docs", "data", "sources",
                          "results/arc_easy", "results/arc_all_models", "results/multitask"):
            files.update((ROOT / directory).rglob("*"))
        inventory = {}
        for path in sorted(files):
            if (not path.is_file() or "__pycache__" in path.parts
                    or path.suffix in {".pyc", ".tmp", ".out", ".err", ".log", ".zip"}
                    or path in (BASE / "artifact_manifest.json", BASE / "completion.json")):
                continue
            inventory[str(path.relative_to(ROOT))] = {
                "bytes": path.stat().st_size, "sha256": sha256_file(path),
            }
        artifact = {
            "complete": True, **{k: status[k] for k in (
                "configurations", "question_evaluations", "scored_candidates",
                "answer_token_logprobs_all_four_policies",
            )},
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Four-task LoopCD-Logits evaluation; 21 new configurations and seven reused ARC-Easy runs.",
            "weight_integrity": "Pinned model manifests included; large weights are not rehashed here.",
            "inventory_note": "Logs, caches, downloaded ZIP archives, environment and model weights are excluded.",
            "files": inventory,
        }
        atomic_json(BASE / "artifact_manifest.json", artifact)
        status.update(
            complete=True, state="complete", artifact_manifest_sha256=sha256_file(BASE / "artifact_manifest.json"),
            report="results/multitask/REPORT.md",
        )
    except Exception as exc:
        status["state"] = "failed"
        artifact_path = BASE / "artifact_manifest.json"
        if artifact_path.exists():
            history.mkdir(parents=True, exist_ok=True)
            shutil.move(str(artifact_path), str(history / "failed_attempt_artifact.json"))
        if readme_path.exists():
            readme_path.write_text(readme_path.read_text().replace(FINISHED, PENDING))
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(BASE / "completion.json", status)
        print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
