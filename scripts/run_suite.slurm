#!/bin/bash
#SBATCH --job-name=loopcd-multitask
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/multitask/logs/slurm-%j.out
#SBATCH --error=results/multitask/logs/slurm-%j.err

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
export OMP_NUM_THREADS=6
venv/bin/python -B scripts/evaluate.py "$@"
