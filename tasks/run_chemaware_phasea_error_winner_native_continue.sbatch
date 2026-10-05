#!/bin/bash
# Continue the protected +2.1266 pp Phase-A checkpoint with the unchanged
# DreaMS native triplet objective.  Chemistry is used only to select the exact
# false molecule that still wins under the frozen Phase-A geometry.
#SBATCH --job-name=chem_errwin
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=data/validation/chem_errwin_%j.out
#SBATCH --error=data/validation/chem_errwin_%j.err

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
TRAIN_EVIDENCE="$BASE/evidence/train_triplet_evidence.npz"
SELECTION_EVIDENCE="$BASE/evidence/selection_triplet_evidence.npz"
CONFIRMATION_EVIDENCE="$BASE/evidence/confirmation_triplet_evidence.npz"
PHASEA_RUN="data/validation/chemaware_online_role_budget_native/run_2345481"
PHASEA="$PHASEA_RUN/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt"
PHASEA_POOL="$PHASEA_RUN/phasea_triplets/train_pool.npz"
PHASEA_VAL="$PHASEA_RUN/phasea_triplets/val_pool.npz"
MANIFEST="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
GRAPH_LEDGER="data/validation/chemaware_fragment_graph_native/run_2346306/qualified_source"
MASSBANK_LEDGER="data/validation/chemaware_massbank_candidate_source_all_qualified_v1_20260928"
PHASEA_CACHE="data/validation/chemaware_fragment_graph_native/run_2346306/phasea_embeddings"
OUT="data/validation/chemaware_phasea_error_winner_native/run_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"
required_files=(
  "$OFFICIAL" "$ARCHITECTURE" "$DATA" "$MANIFEST"
  "$PHASEA" "$PHASEA_POOL" "$PHASEA_VAL"
  "$TRAIN_EVIDENCE" "$SELECTION_EVIDENCE" "$CONFIRMATION_EVIDENCE"
  "$GRAPH_LEDGER/report.json" "$GRAPH_LEDGER/candidate_scores.tsv"
  "$MASSBANK_LEDGER/report.json" "$MASSBANK_LEDGER/candidate_scores.tsv"
  "$PHASEA_CACHE/report.json" "$PHASEA_CACHE/rows.npy"
  "$PHASEA_CACHE/embeddings_f32.npy"
  tasks/audit_chemaware_native_resume_checkpoint.py
  tasks/build_chemaware_phasea_error_winner_native_triplets.py
  tasks/train_chemaware_dreams_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/select_chemaware_residual_checkpoint.py
  tasks/audit_chemaware_role3_confirmation.py
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "CHEMAWARE_ERROR_WINNER_MISSING_INPUT: $path" >&2; exit 2; }
done

python -m py_compile \
  tasks/audit_chemaware_native_resume_checkpoint.py \
  tasks/build_chemaware_phasea_error_winner_native_triplets.py \
  tasks/train_chemaware_dreams_native.py

# This is checked before triplet construction and before any GPU optimization.
# If the protected checkpoint has lost native Adam moments, the job stops.
python -u tasks/audit_chemaware_native_resume_checkpoint.py \
  --checkpoint "$PHASEA" --expected-lr 5e-6 --expected-weight-decay 0 \
  --minimum-step 2000 --output "$OUT/phasea_adam_resume_audit.json"

python -u tasks/build_chemaware_phasea_error_winner_native_triplets.py \
  --phasea-pool "$PHASEA_POOL" --validation-pool "$PHASEA_VAL" \
  --legacy-evidence "$TRAIN_EVIDENCE" \
  --source-ledger "$GRAPH_LEDGER" --source-ledger "$MASSBANK_LEDGER" \
  --manifest "$MANIFEST" --embedding-rows "$PHASEA_CACHE/rows.npy" \
  --phasea-embeddings "$PHASEA_CACHE/embeddings_f32.npy" \
  --embedding-report "$PHASEA_CACHE/report.json" \
  --geometry-checkpoint "$PHASEA" --data "$DATA" \
  --output "$OUT/error_winner_triplets" --margin 0.1 \
  --maximum-events-per-formula 8 --maximum-events-per-identity 3 \
  --minimum-new-error-winners 20

read -r PHASEA_EVENTS NEW_EVENTS TOTAL_EVENTS <<< "$(python - "$OUT/error_winner_triplets/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(r["phasea_events_preserved"], r["new_error_winner_events"],
      r["phasea_events_preserved"] + r["new_error_winner_events"])
PY
)"
[[ "$NEW_EVENTS" -ge 20 ]] || { echo "CHEMAWARE_ERROR_WINNER_COVERAGE_STOP: $NEW_EVENTS" >&2; exit 4; }
STEPS_PER_EPOCH=$((TOTAL_EVENTS / 4))
SAVE_EVERY=$((STEPS_PER_EPOCH / 4))
[[ "$STEPS_PER_EPOCH" -ge 4 && "$SAVE_EVERY" -ge 1 ]] || {
  echo "CHEMAWARE_ERROR_WINNER_INVALID_STEP_BUDGET" >&2; exit 5;
}
echo "CHEMAWARE_ERROR_WINNER_POOL: phasea=$PHASEA_EVENTS new=$NEW_EVENTS total=$TOTAL_EVENTS steps=$STEPS_PER_EPOCH"

# Protect the small scientific artifacts before the expensive part.  The
# original Phase-A checkpoint and pools remain immutable at their source.
mkdir -p "$OUT/protected_corpus"
ln "$OUT/error_winner_triplets/train_pool.npz" "$OUT/protected_corpus/train_pool.npz"
ln "$OUT/error_winner_triplets/val_pool.npz" "$OUT/protected_corpus/val_pool.npz"
cp "$OUT/error_winner_triplets/report.json" "$OUT/protected_corpus/"
cp "$OUT/error_winner_triplets/error_winner_ledger.tsv" "$OUT/protected_corpus/"
cp "$OUT/phasea_adam_resume_audit.json" "$OUT/protected_corpus/"
sha256sum \
  "$PHASEA" "$PHASEA_POOL" "$PHASEA_VAL" "$MANIFEST" \
  "$TRAIN_EVIDENCE" "$GRAPH_LEDGER/report.json" \
  "$GRAPH_LEDGER/candidate_scores.tsv" "$MASSBANK_LEDGER/report.json" \
  "$MASSBANK_LEDGER/candidate_scores.tsv" \
  "$OUT/protected_corpus/train_pool.npz" \
  "$OUT/protected_corpus/error_winner_ledger.tsv" \
  > "$OUT/protected_corpus/SHA256SUMS.txt"

# Exactly one shuffled pass over the Phase-A-prefix pool.  Adam moments and
# weights both resume from the same protected checkpoint; native loss and
# unit event sampling are unchanged.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
  --data "$DATA" --train-pool "$OUT/protected_corpus/train_pool.npz" \
  --val-pool "$OUT/protected_corpus/val_pool.npz" \
  --official-checkpoint "$PHASEA" --restore-adam-from "$PHASEA" \
  --architecture-checkpoint "$ARCHITECTURE" --output "$OUT/training" \
  --seed 3407 --lr 5e-6 --weight-decay 0 --triplet-loss-margin 0.1 \
  --batch-size 4 --num-workers 0 --max-epochs 1 --max-steps "$STEPS_PER_EPOCH" \
  --checkpoint-mode fixed_steps --save-every-n-steps "$SAVE_EVERY" \
  --n-highest-peaks 100 --device cuda

checkpoint_args=(--checkpoint "official=$OFFICIAL" --checkpoint "phaseA_2pp=$PHASEA")
mapfile -t trained_checkpoints < <(find "$OUT/training" -maxdepth 1 -type f -name 'step-*.ckpt' | sort)
[[ ${#trained_checkpoints[@]} -eq 4 ]] || {
  echo "Expected four fixed-step checkpoints, found ${#trained_checkpoints[@]}" >&2; exit 6;
}
for checkpoint in "${trained_checkpoints[@]}"; do
  checkpoint_args+=(--checkpoint "error_winner_$(basename "$checkpoint" .ckpt)=$checkpoint")
done

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --manifest "$MANIFEST" --policy "$SELECTION_EVIDENCE" --data "$DATA" \
  --architecture-checkpoint "$ARCHITECTURE" --official-checkpoint "$OFFICIAL" \
  "${checkpoint_args[@]}" --role-contract-dir "$BASE/evidence" \
  --paired-reference phaseA_2pp --formula-role 2 \
  --output "$OUT/role2_checkpoint_evaluation.json" \
  --batch-size 64 --bootstrap-draws 10000 --device cuda

python -u tasks/select_chemaware_residual_checkpoint.py \
  --evaluation "$OUT/role2_checkpoint_evaluation.json" \
  --base-name phaseA_2pp --require-positive-formula-ci \
  --output "$OUT/checkpoint_selection.json"
SELECTED="$(tr -d '\r\n' < "$OUT/checkpoint_selection.checkpoint.txt")"
SELECTED_STEP="$(tr -d '\r\n' < "$OUT/checkpoint_selection.step.txt")"
if [[ "$SELECTED_STEP" == "0" ]]; then
  echo "CHEMAWARE_ERROR_WINNER_ROLE2_STOP: protected Phase-A retained; role 3 unopened"
  exit 0
fi

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --manifest "$MANIFEST" --policy "$CONFIRMATION_EVIDENCE" --data "$DATA" \
  --architecture-checkpoint "$ARCHITECTURE" --official-checkpoint "$OFFICIAL" \
  --checkpoint "official=$OFFICIAL" --checkpoint "phaseA_2pp=$PHASEA" \
  --checkpoint "error_winner_selected=$(realpath "$SELECTED")" \
  --role-contract-dir "$BASE/evidence" --paired-reference phaseA_2pp \
  --formula-role 3 --output "$OUT/role3_evaluation.json" \
  --batch-size 64 --bootstrap-draws 10000 --device cuda

python -u tasks/audit_chemaware_role3_confirmation.py \
  --evaluation "$OUT/role3_evaluation.json" --base-name phaseA_2pp \
  --protected-baseline-name phaseA_2pp --candidate-name error_winner_selected \
  --output "$OUT/role3_confirmation_decision.json"

sha256sum "$SELECTED" "$OUT/role2_checkpoint_evaluation.json" \
  "$OUT/role3_evaluation.json" "$OUT/role3_confirmation_decision.json" \
  > "$OUT/final_SHA256SUMS.txt"
echo "COMPLETE: Phase-A exact-error-winner native continuation -> $OUT"
