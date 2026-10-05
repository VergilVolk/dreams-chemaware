#!/bin/bash
#SBATCH --job-name=grand_consensus
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/grand_consensus_%j.out
#SBATCH --error=data/validation/grand_consensus_%j.err

# Post-hoc development screen over every fixed consensus arm. This is useful
# for mechanism selection only and is never labelled external confirmation.
set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

BENCHMARK="${GNPS_BENCHMARK:-data/validation/gnps_gold_silver_10ppm_benchmark_v1}"
BUNDLE="${SCORE_BUNDLE:-data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz}"
OUT="data/validation/grand_fusion_fixed_consensus_${SLURM_JOB_ID}/report.json"

required=(
  "$BUNDLE" "$BENCHMARK/panel_identity_disjoint.npz"
  "$BENCHMARK/panel_formula_disjoint.npz"
  tasks/analyze_grand_fusion_fixed_consensus.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "Missing consensus input: $path" >&2; exit 5; }
done

python -m py_compile tasks/analyze_grand_fusion_fixed_consensus.py
python -u tasks/analyze_grand_fusion_fixed_consensus.py \
  --score-bundle "$BUNDLE" \
  --identity-panel "$BENCHMARK/panel_identity_disjoint.npz" \
  --formula-panel "$BENCHMARK/panel_formula_disjoint.npz" \
  --output "$OUT"
echo "Fixed-consensus diagnostic complete: $OUT"

