#!/bin/bash
# Assemble the four-module entity evidence workflow without fitting a fusion head.
# GNPS is development/consumed; this job never accesses Enveda-180.
#SBATCH --job-name=entity_evidence
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/entity_evidence_%j.out
#SBATCH --error=data/validation/entity_evidence_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

BENCHMARK="${GNPS_BENCHMARK:-data/validation/gnps_gold_silver_10ppm_benchmark_v1}"
SCORE_BUNDLE="${GNPS_SCORE_BUNDLE:-data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz}"
CHEMAWARE_TRANSFER_RUN="${CHEMAWARE_TRANSFER_RUN:-data/validation/gnps_noise_chemaware_v2_transfer_run_2358497}"
CHEMAWARE_CACHE="$CHEMAWARE_TRANSFER_RUN/noise_v1_chemaware_v2_cache"
OUT="data/validation/unified_entity_evidence_ledger_run_${SLURM_JOB_ID}"

required=(
  "$BENCHMARK/panel_identity_disjoint.npz"
  "$BENCHMARK/panel_formula_disjoint.npz"
  "$SCORE_BUNDLE"
  "$CHEMAWARE_CACHE/action_ledger_identity_disjoint.npz"
  "$CHEMAWARE_CACHE/action_ledger_formula_disjoint.npz"
  tasks/build_unified_entity_evidence_ledger.py
  tasks/test_unified_entity_evidence_ledger.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "UNIFIED_ENTITY_LEDGER_MISSING_INPUT: $path" >&2; exit 5; }
done
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 4; }

python -m py_compile tasks/build_unified_entity_evidence_ledger.py
python -u tasks/test_unified_entity_evidence_ledger.py

if [[ -n "${ENTITY_MANIFEST:-}" ]]; then
  [[ -f "$ENTITY_MANIFEST" ]] || { echo "Missing ENTITY_MANIFEST: $ENTITY_MANIFEST" >&2; exit 6; }
fi
if [[ -n "${BIOAWARE_EVENTS:-}" ]]; then
  [[ -f "$BIOAWARE_EVENTS" ]] || { echo "Missing BIOAWARE_EVENTS: $BIOAWARE_EVENTS" >&2; exit 7; }
fi

run_builder() {
  python -u tasks/build_unified_entity_evidence_ledger.py \
    --benchmark "$BENCHMARK" \
    --score-bundle "$SCORE_BUNDLE" \
    --chemaware-action-cache "$CHEMAWARE_CACHE" \
    --dataset-id gnps_gold_silver_10ppm \
    --dataset-role development_consumed \
    --output "$OUT" \
    "$@"
}

if [[ -n "${ENTITY_MANIFEST:-}" && -n "${BIOAWARE_EVENTS:-}" ]]; then
  run_builder --entity-manifest "$ENTITY_MANIFEST" --bioaware-events "$BIOAWARE_EVENTS"
elif [[ -n "${ENTITY_MANIFEST:-}" ]]; then
  run_builder --entity-manifest "$ENTITY_MANIFEST"
elif [[ -n "${BIOAWARE_EVENTS:-}" ]]; then
  run_builder --bioaware-events "$BIOAWARE_EVENTS"
else
  run_builder
fi

echo "Unified entity evidence ledger complete: $OUT"
