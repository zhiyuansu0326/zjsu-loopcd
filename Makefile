.DEFAULT_GOAL := help
SHELL := /bin/bash
PYTHON ?= $(CURDIR)/venv/bin/python
MODEL ?= ouro_26
TASK ?= all
OUTPUT ?= results/repeats/$(TASK)/$(MODEL)
AUDIT_OUTPUT ?= results/maintenance/latest_audit.json
MATMUL_PRECISION ?=
REDUCED_OUTPUT ?= $(if $(filter all,$(TASK)),results/reduced_depth,results/reduced_depth/$(TASK))
HIDDEN_OUTPUT ?= results/hidden

export PYTHONDONTWRITEBYTECODE := 1

.PHONY: help check-env test audit report finalize prepare-data download-models submit submit-reduced report-reduced submit-hidden report-hidden audit-hidden

help:
	@printf '%s\n' \
	  'make check-env        Verify the retained Python environment and dependencies' \
	  'make test             Run CPU unit tests; no full checkpoint loading' \
	  'make audit TASK=all   Audit saved runs without rewriting their results' \
	  'make report           Rebuild comparison.json and REPORT.md from saved runs' \
	  'make finalize         Test/audit all 28 configurations and write final artifact hashes' \
	  'make prepare-data TASK=all  Prepare/verify task data using the shared registry' \
	  'make download-models  Fetch/verify the six pinned model checkpoints' \
	  'make submit TASK=sciq MODEL=ouro_26 OUTPUT=results/repeats/sciq/ouro_26' \
	  '                      Submit one full GPU evaluation with its validated precision' \
	  '                      Override precision explicitly with MATMUL_PRECISION=highest' \
	  'make submit-reduced TASK=arc_challenge REDUCED_OUTPUT=results/repeats/huginn_half_depth' \
	  '                      Evaluate/time Huginn with half-depth guidance' \
	  'make report-reduced TASK=all  Audit/rebuild all four half-depth tasks and equal-weight means' \
	  'make submit-hidden TASK=sciq MODEL=huginn_16 HIDDEN_OUTPUT=results/repeats/hidden' \
	  '                      Evaluate/time fixed Hidden guidance with the shared evaluator' \
	  'make report-hidden    Audit/report Huginn R16/R32 across the four tasks'

check-env:
	"$(PYTHON)" -B -c 'import sys, torch, transformers; print("Python:", sys.version.split()[0]); print("Environment:", sys.prefix); print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda); print("Transformers:", transformers.__version__); print("CUDA visible in this process:", torch.cuda.is_available())'
	"$(PYTHON)" -B -m pip check

test:
	HF_HOME="$(CURDIR)/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$(PYTHON)" -B -m unittest discover -s tests -v

audit:
	@mkdir -p "$(dir $(AUDIT_OUTPUT))"
	"$(PYTHON)" -B scripts/analyze.py --task "$(TASK)" --partial > "$(AUDIT_OUTPUT)"
	"$(PYTHON)" -B -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["complete"], r.get("incomplete",r); print("Audit passed;", sys.argv[1])' "$(AUDIT_OUTPUT)"

report:
	"$(PYTHON)" -B scripts/analyze.py --task "$(TASK)"

finalize:
	"$(PYTHON)" -B scripts/finalize_results.py

prepare-data:
	"$(PYTHON)" -B scripts/prepare_tasks.py --task "$(TASK)"

download-models:
	"$(PYTHON)" -B scripts/download_model.py
	"$(PYTHON)" -B scripts/download_ouro26.py
	"$(PYTHON)" -B scripts/download_huginn.py
	"$(PYTHON)" -B scripts/download_parcae.py
	"$(PYTHON)" -B scripts/download_looped_qwen.py

submit:
	"$(PYTHON)" -B -c 'import sys; from src.registry import resolve_run; r=resolve_run(sys.argv[1],sys.argv[2]); assert r["data_path"].is_file(), "Run make prepare-data first"' "$(TASK)" "$(MODEL)"
	@mkdir -p results/multitask/logs
	@set -e; precision="$(MATMUL_PRECISION)"; \
	  if [[ -z "$$precision" ]]; then \
	    precision="$$( "$(PYTHON)" -B -c 'import json,sys; from src.registry import resolve_run; from pathlib import Path; r=resolve_run(sys.argv[1],sys.argv[2]); p=Path("results/multitask/precision_decisions.json"); overrides=json.loads(p.read_text())["overrides"] if p.exists() else {}; print(overrides.get("/".join(sys.argv[1:]), {}).get("selected",r["model"]["matmul_precision"]))' "$(TASK)" "$(MODEL)" )"; \
	  fi; \
	  [[ "$$precision" == high || "$$precision" == highest ]] || { echo "MATMUL_PRECISION must be high or highest" >&2; exit 2; }; \
	  sbatch --partition=scavenger-h200 --account=scavenger-h200 --qos=normal scripts/run_suite.slurm --task "$(TASK)" --model "$(MODEL)" --output "$(OUTPUT)" --matmul-precision "$$precision"

submit-reduced:
	"$(PYTHON)" -B -c 'import sys; from src.registry import resolve_run; r=resolve_run(sys.argv[1],"huginn_16","huginn_half_depth"); assert r["data_path"].is_file(), "Run make prepare-data first"' "$(TASK)"
	@mkdir -p results/reduced_depth/logs
	sbatch --partition=scavenger-h200 --account=scavenger-h200 --qos=normal scripts/run_reduced_depth.slurm "$(TASK)" "$(REDUCED_OUTPUT)"

report-reduced:
	"$(PYTHON)" -B scripts/analyze_reduced_depth.py --task "$(TASK)" --output "$(REDUCED_OUTPUT)"

submit-hidden:
	"$(PYTHON)" -B -c 'import sys; from src.registry import resolve_run; r=resolve_run(sys.argv[1],sys.argv[2],guidance_space="hidden"); assert r["data_path"].is_file(), "Run make prepare-data first"' "$(TASK)" "$(MODEL)"
	@mkdir -p results/hidden/logs
	sbatch --partition=scavenger-h200 --account=scavenger-h200 --qos=normal scripts/run_hidden.slurm "$(TASK)" "$(MODEL)" "$(HIDDEN_OUTPUT)"

report-hidden:
	"$(PYTHON)" -B scripts/analyze_hidden.py --output "$(HIDDEN_OUTPUT)"

audit-hidden:
	"$(PYTHON)" -B scripts/analyze_hidden.py --output "$(HIDDEN_OUTPUT)" --partial
