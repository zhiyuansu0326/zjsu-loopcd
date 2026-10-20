#!/bin/bash
#SBATCH --job-name=loopcd-huginn-half
#SBATCH --partition=scavenger-h200
#SBATCH --account=scavenger-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/reduced_depth/logs/slurm-%j.out
#SBATCH --error=results/reduced_depth/logs/slurm-%j.err

set -euo pipefail
if [[ "${SLURM_JOB_PARTITION:-}" != "scavenger-h200" || "${SLURM_JOB_ACCOUNT:-}" != "scavenger-h200" ]]; then
    echo "This experiment requires partition and account scavenger-h200." >&2
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
output="${2:-results/reduced_depth/$task}"
venv/bin/python -B scripts/evaluate.py --task "$task" --model huginn_16 --experiment huginn_half_depth \
    --output "$output/huginn_16_ref5"
venv/bin/python -B scripts/benchmark_huginn.py --task "$task" \
    --output "$output/huginn_costs.json"
if [[ "${LOOPCD_DEFER_ANALYSIS:-0}" != "1" ]]; then
    venv/bin/python -B scripts/analyze_reduced_depth.py --task "$task" --output "$output"
fi
