#!/bin/bash
#SBATCH --job-name=loopcd-final-audit
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=results/multitask/logs/final-audit-%j.out
#SBATCH --error=results/multitask/logs/final-audit-%j.err

set -euo pipefail
if [[ "${SLURM_JOB_PARTITION:-}" != "scavenger-h200" ]]; then
    echo "This experiment is authorized only on scavenger-h200; refusing to run." >&2
    exit 2
fi
cd "${LOOPCD_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$PWD/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
venv/bin/python -B scripts/finalize_results.py
