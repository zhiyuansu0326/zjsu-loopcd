#!/usr/bin/env python3
"""Prepare every task through one task registry; no GPU or model loading."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.registry import load_registry


def main():
    models, tasks = load_registry()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", *tasks["tasks"]], default="all")
    args = parser.parse_args()
    selected = tasks["tasks"] if args.task == "all" else [args.task]
    for task in selected:
        if task == "arc_easy":
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts/prepare_arc.py"),
                            "--seed", str(models["seed"])], check=True, cwd=ROOT)
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts/prepare_arc_zero.py")],
                           check=True, cwd=ROOT)
            directory = ROOT / "data/tasks/arc_easy"
            directory.mkdir(parents=True, exist_ok=True)
            for shots, source in ((0, "protocol_0shot.json"), (8, "protocol.json")):
                shutil.copy2(ROOT / "data" / source, directory / f"protocol_{shots}shot.json")
        elif task == "arc_challenge":
            from src.data_arc_challenge import prepare_arc_challenge
            for shots in sorted(set(tasks["tasks"][task]["shots_by_family"].values())):
                print(prepare_arc_challenge(shots, seed=models["seed"]))
        else:
            from src.data_external import prepare_external
            print(json.dumps(prepare_external(task, shots=0, seed=models["seed"]), indent=2))


if __name__ == "__main__":
    main()
