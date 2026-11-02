#!/bin/bash
# Frozen auxiliary TransExION comparison on the consumed GNPS development axes.
# Submit with: sbatch tasks/run_transexion_gnps_same_axis_1gpu.sbatch
#
# Scientific boundary: the released checkpoint was trained with GNPS-derived
# spectra and predicts pairwise structural similarity.  This run is therefore
# an auxiliary, source-contaminated development comparison, never an
# independent identity-SOTA claim.
#SBATCH --job-name=tx_gnps_axis
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/transexion_gnps_%j.out
#SBATCH --error=data/validation/transexion_gnps_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

GNPS="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
IDENTITY="data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz"
FORMULA="$GNPS/panel_formula_disjoint.npz"
MGF="$GNPS/spectra.mgf"
REPO="data/external/sota_identity_methods_20261102/TransExION"
CHECKPOINT="$REPO/model/GNPS_MassBank.ms.model"
RUN16="data/validation/gnps_run16_mega_evaluation_inputs_20261102"
OUT="${TRANSEXION_GNPS_OUT:-data/validation/transexion_gnps_same_axis_run_${SLURM_JOB_ID}}"

required=(
  "$MGF" "$IDENTITY" "$FORMULA" "$CHECKPOINT"
  "$REPO/lrp/model.py" "$REPO/lrp/data.py" "$REPO/spectrum/io.py"
  "$RUN16/scores_identity_disjoint.npz" "$RUN16/scores_formula_disjoint.npz"
  tasks/transexion_same_axis_adapter.py
  tasks/public_sota_embedding_adapter.py
  tasks/evaluate_mega_identity_benchmark.py
  tasks/test_transexion_same_axis_adapter.py
)
missing=()
for path in "${required[@]}"; do
  [[ -s "$path" ]] || missing+=("$path")
done
if (( ${#missing[@]} > 0 )); then
  printf 'TRANSEXION_GNPS_MISSING: %s\n' "${missing[@]}" >&2
  exit 4
fi

python -m py_compile \
  tasks/transexion_same_axis_adapter.py \
  tasks/public_sota_embedding_adapter.py \
  tasks/evaluate_mega_identity_benchmark.py
python -u tasks/test_transexion_same_axis_adapter.py

mkdir -p "$OUT/identity" "$OUT/formula"

if [[ ! -s "$OUT/identity/score_report.json" ]]; then
  python -u tasks/transexion_same_axis_adapter.py score \
    --mgf "$MGF" --panel "$IDENTITY" \
    --repo "$REPO" --checkpoint "$CHECKPOINT" \
    --device cuda --batch-size 16 --progress-every 10000 \
    --method-name "TransExION (GNPS-trained; auxiliary)" \
    --output "$OUT/identity/score_cache.npz" \
    --report "$OUT/identity/score_report.json"
fi

if [[ ! -s "$OUT/formula/score_report.json" ]]; then
  python -u tasks/transexion_same_axis_adapter.py score \
    --mgf "$MGF" --panel "$FORMULA" \
    --repo "$REPO" --checkpoint "$CHECKPOINT" \
    --device cuda --batch-size 16 --progress-every 10000 \
    --method-name "TransExION (GNPS-trained; auxiliary)" \
    --output "$OUT/formula/score_cache.npz" \
    --report "$OUT/formula/score_report.json"
fi

if [[ ! -s "$OUT/identity/merged_scores.npz" ]]; then
  python -u tasks/public_sota_embedding_adapter.py merge \
    --score-cache "$RUN16/scores_identity_disjoint.npz" \
    --score-cache "$OUT/identity/score_cache.npz" \
    --output "$OUT/identity/merged_scores.npz"
fi
if [[ ! -s "$OUT/formula/merged_scores.npz" ]]; then
  python -u tasks/public_sota_embedding_adapter.py merge \
    --score-cache "$RUN16/scores_formula_disjoint.npz" \
    --score-cache "$OUT/formula/score_cache.npz" \
    --output "$OUT/formula/merged_scores.npz"
fi

if [[ ! -s "$OUT/identity/evaluation/report.json" ]]; then
  python -u tasks/evaluate_mega_identity_benchmark.py \
    --panel "$IDENTITY" --scores "$OUT/identity/merged_scores.npz" \
    --output-dir "$OUT/identity/evaluation" \
    --reference-method full23_v132 \
    --bootstrap-resamples 10000 --bootstrap-seed 20261102 \
    --evaluation-role development --truth-status consumed
fi
if [[ ! -s "$OUT/formula/evaluation/report.json" ]]; then
  python -u tasks/evaluate_mega_identity_benchmark.py \
    --panel "$FORMULA" --scores "$OUT/formula/merged_scores.npz" \
    --output-dir "$OUT/formula/evaluation" \
    --reference-method full23_v132 \
    --bootstrap-resamples 10000 --bootstrap-seed 20261102 \
    --evaluation-role development --truth-status consumed
fi

echo "TRANSEXION_GNPS_SAME_AXIS_AUXILIARY_COMPLETE: $OUT"
