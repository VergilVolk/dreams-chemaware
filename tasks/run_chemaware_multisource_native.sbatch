#!/bin/bash
#SBATCH --job-name=chem_multi_src
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/chem_multi_src_%j.out
#SBATCH --error=data/validation/chem_multi_src_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
BASE="data/validation/chemaware_high_coverage_native/run_2340524"
PHASEA_RUN="data/validation/chemaware_online_role_budget_native/run_2345481"
PHASEA="$PHASEA_RUN/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt"
PHASEA_POOL="$PHASEA_RUN/phasea_triplets/train_pool.npz"
PHASEA_VAL="$PHASEA_RUN/phasea_triplets/val_pool.npz"
MANIFEST="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
ICEBERG_GRAPH="data/validation/chemaware_full_manifest_iceberg_graph_v1"
ICEBERG_TEACHER="data/validation/chemaware_full_manifest_iceberg_teacher_700_v1"
OUT="data/validation/chemaware_multisource_native/run_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT/source_ledgers"

required_files=(
  "$OFFICIAL" "$ARCHITECTURE" "$DATA" "$PHASEA" "$PHASEA_POOL" "$PHASEA_VAL"
  "$MANIFEST" "$ICEBERG_GRAPH/graph.npz" "$ICEBERG_GRAPH/source_query_index.npy"
  "$ICEBERG_GRAPH/report.json" "$ICEBERG_TEACHER/selected_queries.npy"
  "$ICEBERG_TEACHER/query_ptr.npy" "$ICEBERG_TEACHER/scores_and_ranks.npz"
  "$ICEBERG_TEACHER/report.json"
  "$BASE/evidence/train_triplet_evidence.npz"
  "$BASE/evidence/selection_triplet_evidence.npz"
  "$BASE/evidence/confirmation_triplet_evidence.npz"
  tasks/export_chemaware_iceberg_source_ledger.py
  tasks/convert_chemaware_sirius_source_ledger.py
  tasks/build_chemaware_multisource_native_triplets.py
  tasks/test_chemaware_multisource_native_triplets.py
  tasks/encode_chemaware_checkpoint_manifest_rows.py
  tasks/train_chemaware_dreams_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/select_chemaware_residual_checkpoint.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing ChemAware multisource file: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_chemaware_multisource_native_triplets.py
python -u tasks/export_chemaware_iceberg_source_ledger.py \
  --manifest "$MANIFEST" \
  --graph-dir "$ICEBERG_GRAPH" \
  --teacher-dir "$ICEBERG_TEACHER" \
  --output "$OUT/source_ledgers/iceberg"

source_args=(--source-ledger "$OUT/source_ledgers/iceberg")
# SIRIUS is optional.  When a completed score directory is supplied, it becomes
# a second independent source; the builder still does not average source values.
if [[ -n "${SIRIUS_SCORE_DIR:-}" ]]; then
  [[ -f "$SIRIUS_SCORE_DIR/candidate_scores.tsv" && -f "$SIRIUS_SCORE_DIR/report.json" ]] || {
    echo "Invalid SIRIUS_SCORE_DIR: $SIRIUS_SCORE_DIR" >&2
    exit 2
  }
  python -u tasks/convert_chemaware_sirius_source_ledger.py \
    --scores "$SIRIUS_SCORE_DIR/candidate_scores.tsv" \
    --score-report "$SIRIUS_SCORE_DIR/report.json" \
    --output "$OUT/source_ledgers/sirius"
  source_args+=(--source-ledger "$OUT/source_ledgers/sirius")
fi

# Extract current geometry once from the protected +2.1266 pp checkpoint.  This
# is not Phase-A reproduction and does not update weights.
srun --export=ALL --preserve-env python -u tasks/encode_chemaware_checkpoint_manifest_rows.py \
  --checkpoint "$PHASEA" \
  --evidence-dir "$BASE/evidence" \
  --manifest "$MANIFEST" \
  --data "$DATA" \
  --official-checkpoint "$OFFICIAL" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --output "$OUT/phasea_embeddings" \
  --batch-size 64 --n-highest-peaks 100 --device cuda

python -u tasks/build_chemaware_multisource_native_triplets.py \
  --phasea-pool "$PHASEA_POOL" \
  --validation-pool "$PHASEA_VAL" \
  "${source_args[@]}" \
  --manifest "$MANIFEST" \
  --embedding-rows "$OUT/phasea_embeddings/rows.npy" \
  --phasea-embeddings "$OUT/phasea_embeddings/embeddings_f32.npy" \
  --data "$DATA" \
  --output "$OUT/triplets" \
  --margin 0.1 --maximum-source-events-per-query 2 --minimum-supporting-sources 1

read -r SOURCE_EVENTS SOURCE_QUERIES CORRECTED_WINNERS < <(python -c \
  "import json; r=json.load(open('$OUT/triplets/report.json')); print(r['source_events_added'],r['source_event_queries'],r['source_corrected_phasea_current_winners'])")
if [[ "$SOURCE_EVENTS" -lt 100 || "$SOURCE_QUERIES" -lt 80 || "$CORRECTED_WINNERS" -lt 30 ]]; then
  echo "CHEMAWARE_MULTISOURCE_COVERAGE_STOP: events=$SOURCE_EVENTS queries=$SOURCE_QUERIES corrected_winners=$CORRECTED_WINNERS" >&2
  exit 4
fi

# Direct continuation from Phase A.  The complete 5,956-event Phase-A pool is
# the exact prefix; only source-proven native identity triplets are appended.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
  --data "$DATA" \
  --train-pool "$OUT/triplets/train_pool.npz" \
  --val-pool "$OUT/triplets/val_pool.npz" \
  --official-checkpoint "$PHASEA" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --output "$OUT/training" \
  --seed 3407 --lr 2e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
  --max-epochs 3 --max-steps 1000 --checkpoint-mode fixed_steps \
  --save-every-n-steps 250 --n-highest-peaks 100 --device cuda

checkpoint_args=(
  --checkpoint "official=$OFFICIAL"
  --checkpoint "phaseA_2pp=$PHASEA"
)
mapfile -t trained_checkpoints < <(
  find "$OUT/training" -maxdepth 1 -type f -name 'step-*.ckpt' | sort
)
[[ ${#trained_checkpoints[@]} -eq 4 ]] || {
  echo "Expected four multisource checkpoints, found ${#trained_checkpoints[@]}" >&2
  exit 5
}
for checkpoint in "${trained_checkpoints[@]}"; do
  checkpoint_args+=(--checkpoint "multisource_$(basename "$checkpoint" .ckpt)=$checkpoint")
done

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/selection_triplet_evidence.npz" \
  "${checkpoint_args[@]}" \
  --paired-reference phaseA_2pp --formula-role 2 \
  --output "$OUT/role2_checkpoint_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

python -u tasks/select_chemaware_residual_checkpoint.py \
  --evaluation "$OUT/role2_checkpoint_evaluation.json" \
  --base-name phaseA_2pp --require-positive-formula-ci \
  --output "$OUT/checkpoint_selection.json"
SELECTED="$(tr -d '\r\n' < "$OUT/checkpoint_selection.checkpoint.txt")"
SELECTED_STEP="$(tr -d '\r\n' < "$OUT/checkpoint_selection.step.txt")"

if [[ "$SELECTED_STEP" == "0" ]]; then
  echo "CHEMAWARE_MULTISOURCE_ROLE2_STOP: retain protected Phase A; role 3 remains unopened"
  echo "COMPLETE: multisource role-2 audit -> $OUT"
  exit 0
fi

SELECTED_REAL="$(realpath "$SELECTED")"
srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/confirmation_triplet_evidence.npz" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "phaseA_2pp=$PHASEA" \
  --checkpoint "multisource_selected=$SELECTED_REAL" \
  --paired-reference phaseA_2pp --formula-role 3 \
  --output "$OUT/role3_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

mkdir -p "$OUT/protected_candidate"
ln "$SELECTED_REAL" "$OUT/protected_candidate/chemaware_multisource_selected.ckpt"
cp "$OUT/triplets/report.json" "$OUT/protected_candidate/"
cp "$OUT/triplets/source_event_ledger.tsv" "$OUT/protected_candidate/"
cp "$OUT/checkpoint_selection.json" "$OUT/protected_candidate/"
cp "$OUT/role2_checkpoint_evaluation.json" "$OUT/protected_candidate/"
cp "$OUT/role3_evaluation.json" "$OUT/protected_candidate/"
cp "$OUT/training/training_report.json" "$OUT/protected_candidate/"
(
  cd "$OUT/protected_candidate"
  sha256sum -- * > SHA256SUMS
)
echo "COMPLETE: protected ChemAware multisource candidate -> $OUT/protected_candidate"
