#!/bin/bash
#SBATCH --job-name=chem_sirius_tri
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/chem_sirius_tri_%j.out
#SBATCH --error=data/validation/chem_sirius_tri_%j.err

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

if [[ -z "${SIRIUS_RUN:-}" ]]; then
  SCORE_REPORT="$({ find data/validation/chemaware_sirius_source \
    -mindepth 3 -maxdepth 3 -path '*/qualified_source/report.json' \
    -printf '%T@ %p\n' 2>/dev/null || true; } | sort -n | tail -1 | cut -d' ' -f2-)"
  [[ -n "$SCORE_REPORT" ]] || {
    echo "No completed SIRIUS source run found. Submit tasks/run_chemaware_sirius_source.sbatch first." >&2
    exit 2
  }
  SIRIUS_RUN="${SCORE_REPORT%/qualified_source/report.json}"
fi
SIRIUS_SOURCE="$SIRIUS_RUN/qualified_source"
STATIC_SOURCE="$SIRIUS_RUN/qualified_static_source"
RULE_CORPUS="$SIRIUS_RUN/layered_rule_corpus"
OUT="data/validation/chemaware_sirius_native/run_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

required_files=(
  "$OFFICIAL" "$ARCHITECTURE" "$DATA" "$PHASEA" "$PHASEA_POOL" "$PHASEA_VAL"
  "$SIRIUS_SOURCE/candidate_scores.tsv" "$SIRIUS_SOURCE/report.json"
  "$SIRIUS_SOURCE/qualification_report.json"
  "$STATIC_SOURCE/candidate_scores.tsv" "$STATIC_SOURCE/report.json"
  "$STATIC_SOURCE/qualification_report.json"
  "$RULE_CORPUS/rule_corpus.json" "$RULE_CORPUS/report.json" "$MANIFEST"
  "$SIRIUS_RUN/source_panel/candidate_ledger.tsv"
  "$SIRIUS_RUN/source_panel/query_registry.tsv"
  "$SIRIUS_RUN/source_panel/candidate_structures.tsv"
  "$SIRIUS_RUN/source_panel/report.json"
  data/e1/e1_train_triplet_pool_10ppm.npz
  "$BASE/evidence/train_triplet_evidence.npz"
  "$BASE/evidence/selection_triplet_evidence.npz"
  "$BASE/evidence/confirmation_triplet_evidence.npz"
  tasks/encode_chemaware_checkpoint_manifest_rows.py
  tasks/build_chemaware_multisource_native_triplets.py
  tasks/test_chemaware_multisource_native_triplets.py
  tasks/build_chemaware_layered_10k_native_pool.py
  tasks/train_chemaware_dreams_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/select_chemaware_residual_checkpoint.py
  tasks/audit_chemaware_role3_confirmation.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing ChemAware SIRIUS-native file: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_chemaware_multisource_native_triplets.py
cp "$SIRIUS_SOURCE/report.json" "$OUT/sirius_qualified_source_report.json"
cp "$STATIC_SOURCE/report.json" "$OUT/static_qualified_source_report.json"

SIRIUS_AUTHORIZED="$(python -c "import json; print(int(json.load(open('$SIRIUS_SOURCE/report.json'))['formal_triplet_mining_authorized']))")"
STATIC_AUTHORIZED="$(python -c "import json; print(int(json.load(open('$STATIC_SOURCE/report.json'))['formal_triplet_mining_authorized']))")"
source_args=()
if [[ "$SIRIUS_AUTHORIZED" == "1" ]]; then
  source_args+=(--source-ledger "$SIRIUS_SOURCE")
fi
if [[ "$STATIC_AUTHORIZED" == "1" ]]; then
  source_args+=(--source-ledger "$STATIC_SOURCE")
fi
if [[ ${#source_args[@]} -eq 0 ]]; then
  echo "CHEMAWARE_NATIVE_STOP: no source ledger passed matched-control qualification"
  exit 0
fi

# This is geometry extraction from the already protected Phase-A checkpoint,
# not a Phase-A reproduction or a second training stage.
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
  --output "$OUT/source_triplets" \
  --margin 0.1 --maximum-source-events-per-query 4 \
  --minimum-supporting-sources 1 \
  --geometry-label "protected Phase-A +2.1266 pp checkpoint"

read -r SOURCE_EVENTS SOURCE_QUERIES < <(python -c \
  "import json; r=json.load(open('$OUT/source_triplets/report.json')); print(r['source_events_added'], r['source_event_queries'])")
if [[ "$SOURCE_EVENTS" -lt 1000 || "$SOURCE_QUERIES" -lt 500 ]]; then
  echo "CHEMAWARE_SIRIUS_CORPUS_STOP: qualified source coverage is below the frozen gate (events=$SOURCE_EVENTS queries=$SOURCE_QUERIES)"
  echo "No replay padding or repeated references will be used to disguise insufficient chemical coverage."
  exit 0
fi

python -u tasks/build_chemaware_layered_10k_native_pool.py \
  --phasea-pool "$PHASEA_POOL" \
  --source-pool "$OUT/source_triplets/train_pool.npz" \
  --source-report "$OUT/source_triplets/report.json" \
  --source-event-ledger "$OUT/source_triplets/source_event_ledger.tsv" \
  --validation-pool "$PHASEA_VAL" \
  --dreams-replay-pool data/e1/e1_train_triplet_pool_10ppm.npz \
  --data "$DATA" \
  --rule-corpus "$RULE_CORPUS" \
  --output "$OUT/layered_10k_triplets" \
  --target-events 12000 --minimum-total-events 10000 \
  --minimum-chemical-events 1000 --minimum-chemical-queries 500 \
  --seed 3407

# The corpus is a reusable scientific asset independent of whether the next
# embedding checkpoint passes role 2.  Protect it before any optimization.
mkdir -p "$OUT/protected_corpus"
ln "$OUT/layered_10k_triplets/train_pool.npz" \
  "$OUT/protected_corpus/train_pool.npz"
ln "$OUT/layered_10k_triplets/val_pool.npz" \
  "$OUT/protected_corpus/val_pool.npz"
cp "$OUT/layered_10k_triplets/report.json" "$OUT/protected_corpus/"
cp "$OUT/layered_10k_triplets/corpus_event_manifest.tsv" "$OUT/protected_corpus/"
cp "$OUT/layered_10k_triplets/chemical_source_event_ledger.tsv" \
  "$OUT/protected_corpus/"
cp "$SIRIUS_SOURCE/report.json" "$OUT/protected_corpus/sirius_source_report.json"
cp "$SIRIUS_SOURCE/qualification_report.json" \
  "$OUT/protected_corpus/sirius_qualification_report.json"
cp "$STATIC_SOURCE/report.json" "$OUT/protected_corpus/static_source_report.json"
cp "$STATIC_SOURCE/qualification_report.json" \
  "$OUT/protected_corpus/static_qualification_report.json"
ln "$SIRIUS_SOURCE/candidate_scores.tsv" \
  "$OUT/protected_corpus/sirius_candidate_scores.tsv"
ln "$STATIC_SOURCE/candidate_scores.tsv" \
  "$OUT/protected_corpus/static_candidate_scores.tsv"
mkdir -p "$OUT/protected_corpus/source_inputs"
ln "$SIRIUS_RUN/source_panel/candidate_ledger.tsv" \
  "$OUT/protected_corpus/source_inputs/candidate_ledger.tsv"
ln "$SIRIUS_RUN/source_panel/query_registry.tsv" \
  "$OUT/protected_corpus/source_inputs/query_registry.tsv"
ln "$SIRIUS_RUN/source_panel/candidate_structures.tsv" \
  "$OUT/protected_corpus/source_inputs/candidate_structures.tsv"
cp "$SIRIUS_RUN/source_panel/report.json" \
  "$OUT/protected_corpus/source_inputs/source_panel_report.json"
cp -R "$RULE_CORPUS" "$OUT/protected_corpus/layered_rule_corpus"
(
  cd "$OUT/protected_corpus"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

# Direct continuation of the +2.1266 pp embedding.  The complete successful
# Phase-A pool remains present; qualified chemical sources contribute only new
# native triplets.  Their raw values never enter the loss.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
  --data "$DATA" \
  --train-pool "$OUT/layered_10k_triplets/train_pool.npz" \
  --val-pool "$OUT/layered_10k_triplets/val_pool.npz" \
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
  echo "Expected four SIRIUS-native checkpoints, found ${#trained_checkpoints[@]}" >&2
  exit 5
}
for checkpoint in "${trained_checkpoints[@]}"; do
  checkpoint_args+=(--checkpoint "sirius_$(basename "$checkpoint" .ckpt)=$checkpoint")
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
  echo "CHEMAWARE_SIRIUS_ROLE2_STOP: retain protected Phase A; role 3 remains unopened"
  echo "COMPLETE: SIRIUS-native role-2 audit -> $OUT"
  exit 0
fi

SELECTED_REAL="$(realpath "$SELECTED")"
srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/confirmation_triplet_evidence.npz" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "phaseA_2pp=$PHASEA" \
  --checkpoint "sirius_selected=$SELECTED_REAL" \
  --paired-reference phaseA_2pp --formula-role 3 \
  --output "$OUT/role3_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

python -u tasks/audit_chemaware_role3_confirmation.py \
  --evaluation "$OUT/role3_evaluation.json" \
  --base-name phaseA_2pp --candidate-name sirius_selected \
  --output "$OUT/role3_confirmation_decision.json"
ROLE3_CONFIRMED="$(tr -d '\r\n' < "$OUT/role3_confirmation_decision.confirmed.txt")"
if [[ "$ROLE3_CONFIRMED" != "1" ]]; then
  mkdir -p "$OUT/protected_negative_result"
  ln "$SELECTED_REAL" "$OUT/protected_negative_result/chemaware_role2_selected.ckpt"
  cp "$OUT/source_triplets/report.json" "$OUT/protected_negative_result/"
  cp "$OUT/layered_10k_triplets/report.json" "$OUT/protected_negative_result/"
  cp "$OUT/checkpoint_selection.json" "$OUT/protected_negative_result/"
  cp "$OUT/role2_checkpoint_evaluation.json" "$OUT/protected_negative_result/"
  cp "$OUT/role3_evaluation.json" "$OUT/protected_negative_result/"
  cp "$OUT/role3_confirmation_decision.json" "$OUT/protected_negative_result/"
  (
    cd "$OUT/protected_negative_result"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
  )
  echo "CHEMAWARE_ROLE3_STOP: selected role-2 checkpoint did not confirm on role 3"
  echo "COMPLETE: protected negative result -> $OUT/protected_negative_result"
  exit 0
fi

mkdir -p "$OUT/protected_candidate"
ln "$SELECTED_REAL" "$OUT/protected_candidate/chemaware_sirius_selected.ckpt"
cp "$OUT/source_triplets/report.json" "$OUT/protected_candidate/source_triplet_report.json"
cp "$OUT/layered_10k_triplets/report.json" "$OUT/protected_candidate/layered_10k_report.json"
cp "$OUT/layered_10k_triplets/chemical_source_event_ledger.tsv" "$OUT/protected_candidate/"
cp "$OUT/layered_10k_triplets/corpus_event_manifest.tsv" "$OUT/protected_candidate/"
ln "$SIRIUS_SOURCE/candidate_scores.tsv" \
  "$OUT/protected_candidate/sirius_candidate_scores.tsv"
ln "$STATIC_SOURCE/candidate_scores.tsv" \
  "$OUT/protected_candidate/static_candidate_scores.tsv"
cp -R "$RULE_CORPUS" "$OUT/protected_candidate/layered_rule_corpus"
cp "$OUT/checkpoint_selection.json" "$OUT/protected_candidate/"
cp "$OUT/role2_checkpoint_evaluation.json" "$OUT/protected_candidate/"
cp "$OUT/role3_evaluation.json" "$OUT/protected_candidate/"
cp "$OUT/role3_confirmation_decision.json" "$OUT/protected_candidate/"
cp "$OUT/training/training_report.json" "$OUT/protected_candidate/"
(
  cd "$OUT/protected_candidate"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
echo "COMPLETE: protected ChemAware SIRIUS-native candidate -> $OUT/protected_candidate"
