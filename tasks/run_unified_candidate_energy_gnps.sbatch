#!/bin/bash
# Build and cross-fit one continuous unified candidate-energy model on the
# consumed GNPS development panels.  This job does not access Enveda-180.
# It reuses the frozen Noise/WSE/P2b bundle and the already completed frozen
# ChemAware V2 action ledgers; only the sealed ChemAware stage-1 encoder may
# need label-blind encoding.
#SBATCH --job-name=unified_energy
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=18:00:00
#SBATCH --output=data/validation/unified_energy_%j.out
#SBATCH --error=data/validation/unified_energy_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

BENCHMARK="${GNPS_BENCHMARK:-data/validation/gnps_gold_silver_10ppm_benchmark_v1}"
IDENTITY_PANEL="${GNPS_IDENTITY_PANEL:-data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz}"
FORMULA_PANEL="${GNPS_FORMULA_PANEL:-$BENCHMARK/panel_formula_disjoint.npz}"
BASE_BUNDLE="${GNPS_SCORE_BUNDLE:-data/validation/noise_gnps_article_benchmark_run_2349091/method_scores.npz}"
STAGE1="${CHEMAWARE_STAGE1_CHECKPOINT:-data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt}"
ARCH="${DREAMS_ARCHITECTURE_CHECKPOINT:-dreams/models/pretrained/ssl_model_server.pt}"
V2_RUN="${CHEMAWARE_V2_TRANSFER_RUN:-data/validation/gnps_noise_chemaware_v2_transfer_run_2358497/noise_v1_chemaware_v2_cache}"
OUT="data/validation/unified_candidate_energy_run_${SLURM_JOB_ID}"
SCRATCH_ROOT="${SLURM_TMPDIR:-/tmp}/unified_candidate_energy_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "UNIFIED_ENERGY_OUTPUT_COLLISION: $OUT" >&2; exit 3; }
mkdir -p "$OUT" "$SCRATCH_ROOT"
cleanup() { rm -rf -- "$SCRATCH_ROOT"; }
trap cleanup EXIT

required=(
  "$BENCHMARK/manifest.csv.gz"
  "$IDENTITY_PANEL"
  "$FORMULA_PANEL"
  "$BASE_BUNDLE"
  "$V2_RUN/action_ledger_identity_disjoint.npz"
  "$V2_RUN/action_ledger_formula_disjoint.npz"
  tasks/build_chemaware_v2_sparse_action_evidence.py
  tasks/build_grand_unified_evidence_bundle.py
  tasks/extend_gnps_article_score_bundle.py
  tasks/train_unified_candidate_energy_crossfit.py
  tasks/test_unified_candidate_energy.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "UNIFIED_ENERGY_MISSING_INPUT: $path" >&2; exit 4; }
done

python -m py_compile \
  tasks/build_chemaware_v2_sparse_action_evidence.py \
  tasks/build_grand_unified_evidence_bundle.py \
  tasks/extend_gnps_article_score_bundle.py \
  tasks/train_unified_candidate_energy_crossfit.py
python -u tasks/test_unified_candidate_energy.py

# A previously encoded, checkpoint-matched cache can be supplied explicitly.
# Otherwise the frozen stage-1 checkpoint is encoded once without labels.
if [[ -n "${GNPS_CHEMAWARE_STAGE1_EMBEDDINGS:-}" ]]; then
  STAGE1_EMB="$GNPS_CHEMAWARE_STAGE1_EMBEDDINGS"
  [[ -f "$STAGE1_EMB" ]] || { echo "UNIFIED_ENERGY_MISSING_STAGE1_CACHE: $STAGE1_EMB" >&2; exit 5; }
else
  [[ -f "$STAGE1" ]] || { echo "UNIFIED_ENERGY_MISSING_INPUT: $STAGE1" >&2; exit 4; }
  [[ -f "$ARCH" ]] || { echo "UNIFIED_ENERGY_MISSING_INPUT: $ARCH" >&2; exit 4; }
  [[ -f "$BENCHMARK/spectra.mgf" ]] || { echo "UNIFIED_ENERGY_MISSING_INPUT: $BENCHMARK/spectra.mgf" >&2; exit 4; }
  STAGE1_SLIM="$SCRATCH_ROOT/chemaware_stage1_slim.pt"
  STAGE1_EMB="$SCRATCH_ROOT/chemaware_stage1_embeddings.npz"
  python -u tasks/prepare_official_embedding_checkpoint.py \
    --source "$STAGE1" --output "$STAGE1_SLIM"
  python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
    --benchmark "$BENCHMARK" \
    --checkpoint "$STAGE1_SLIM" \
    --architecture-checkpoint "$ARCH" \
    --output "$STAGE1_EMB" \
    --panel-file "identity_disjoint=$IDENTITY_PANEL" \
    --panel-file "formula_disjoint=$FORMULA_PANEL" \
    --batch-size 64 \
    --label chemaware-stage1
fi
[[ -f "${STAGE1_EMB%.npz}.json" ]] || {
  echo "UNIFIED_ENERGY_MISSING_STAGE1_CACHE_REPORT: ${STAGE1_EMB%.npz}.json" >&2
  exit 5
}

python -u tasks/extend_gnps_article_score_bundle.py \
  --benchmark "$BENCHMARK" \
  --base-bundle "$BASE_BUNDLE" \
  --identity-panel "$IDENTITY_PANEL" \
  --formula-panel "$FORMULA_PANEL" \
  --embedding-method "chemaware_stage1_encoder=$STAGE1_EMB" \
  --output "$OUT/extended_method_scores.npz"

for panel in identity_disjoint formula_disjoint; do
  if [[ "$panel" == "identity_disjoint" ]]; then
    PANEL_PATH="$IDENTITY_PANEL"
  else
    PANEL_PATH="$FORMULA_PANEL"
  fi
  python -u tasks/build_chemaware_v2_sparse_action_evidence.py \
    --panel "$PANEL_PATH" \
    --action-ledger "$V2_RUN/action_ledger_${panel}.npz" \
    --output "$OUT/chemaware_v2_action_${panel}.npz"

  python -u tasks/build_grand_unified_evidence_bundle.py \
    --panel "$panel" \
    --panel-npz "$PANEL_PATH" \
    --base-scores "$OUT/extended_method_scores.npz" \
    --module noise_v1 \
    --module chemaware_stage1_encoder \
    --module weighted_spectral_entropy \
    --module p2b_noise_v1_frozen \
    --module chemaware_v2_action \
    --extra "chemaware_v2_action=$OUT/chemaware_v2_action_${panel}.npz" \
    --out "$OUT/evidence_${panel}.npz"
done

# Hyperparameters and formula-hash folds are fixed in the program.  The
# formula panel is predicted only; it never contributes an optimization label.
python -u tasks/train_unified_candidate_energy_crossfit.py \
  --identity-bundle "$OUT/evidence_identity_disjoint.npz" \
  --formula-bundle "$OUT/evidence_formula_disjoint.npz" \
  --output "$OUT/crossfit" \
  --run-ablations \
  --device cuda

cp "${STAGE1_EMB%.npz}.json" "$OUT/chemaware_stage1_encoding.json"
echo "UNIFIED_CANDIDATE_ENERGY_GNPS_COMPLETE: $OUT"
