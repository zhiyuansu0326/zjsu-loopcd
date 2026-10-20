#!/bin/bash
#SBATCH --job-name=loopcd-huginn-hidden
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/hidden/logs/slurm-%j.out
#SBATCH --error=results/hidden/logs/slurm-%j.err

set -euo pipefail
if [[ "${SLURM_JOB_PARTITION:-}" != "scavenger-h200" || "${SLURM_JOB_ACCOUNT:-}" != "scavenger-h200" ]]; then
    echo "Hidden evaluation requires partition and account scavenger-h200." >&2
    exit 2
fi
cd "${LOOPCD_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$PWD/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=6

task="${1:-arc_challenge}"
model="${2:-huginn_16}"
output="${3:-results/hidden}"
venv/bin/python -B scripts/evaluate.py --task "$task" --model "$model" \
    --guidance-space hidden --matmul-precision highest --output "$output/$task/$model"
venv/bin/python -B scripts/benchmark_hidden_huginn.py --task "$task" --model "$model" \
    --output "$output/$task/${model}_costs.json"
