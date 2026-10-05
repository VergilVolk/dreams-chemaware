#!/bin/bash
#SBATCH --job-name=chem_fraggraph
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_fraggraph_%j.out
#SBATCH --error=data/validation/chem_fraggraph_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
BASE="data/validation/chemaware_high_coverage_native/run_2340524"
PHASEA_RUN="data/validation/chemaware_online_role_budget_native/run_2345481"
PHASEA="$PHASEA_RUN/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt"
PHASEA_POOL="$PHASEA_RUN/phasea_triplets/train_pool.npz"
PHASEA_VAL="$PHASEA_RUN/phasea_triplets/val_pool.npz"
MANIFEST="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
TRAIN_EVIDENCE="$BASE/evidence/train_triplet_evidence.npz"
MS_PRED_SOURCE="data/external/ms-pred-src"
EVIDENCE_CONTRACT="dreams/models/chem_aware/fragment_graph_evidence_v1.json"
RUN_ROOT="data/validation/chemaware_fragment_graph_native"
REUSE_RUN="${1:-}"
if [[ "$REUSE_RUN" == "latest-qualified" ]]; then
  REUSE_RUN=""
  while IFS= read -r candidate; do
    if [[ -f "$candidate/qualified_source/report.json" \
          && -f "$candidate/phasea_embeddings/rows.npy" \
          && -f "$candidate/phasea_embeddings/embeddings_f32.npy" ]]; then
      AUTHORIZED_CANDIDATE="$(python -c "import json; print(int(json.load(open('$candidate/qualified_source/report.json')).get('formal_triplet_mining_authorized', False)))")"
      if [[ "$AUTHORIZED_CANDIDATE" == "1" ]]; then
        REUSE_RUN="$candidate"
        break
      fi
    fi
  done < <(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'run_*' -print | sort -Vr)
  [[ -n "$REUSE_RUN" ]] || {
    echo "No prior qualified fragment-graph source with Phase-A embeddings was found." >&2
    exit 2
  }
fi
if [[ -n "$REUSE_RUN" ]]; then
  REUSE_RUN="$(realpath "$REUSE_RUN")"
  RUN_ROOT_REAL="$(realpath "$RUN_ROOT")"
  case "$REUSE_RUN" in
    "$RUN_ROOT_REAL"/run_*) ;;
    *) echo "Refusing fragment-graph reuse outside $RUN_ROOT_REAL: $REUSE_RUN" >&2; exit 2 ;;
  esac
  OUT="$RUN_ROOT/run_${SLURM_JOB_ID}_resume_$(basename "$REUSE_RUN")"
else
  OUT="$RUN_ROOT/run_${SLURM_JOB_ID}"
fi
SHARDS=8

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT/source_shards" "$OUT/logs"

required_files=(
  "$OFFICIAL" "$ARCHITECTURE" "$DATA" "$PHASEA" "$PHASEA_POOL" "$PHASEA_VAL"
  "$MANIFEST" "$TRAIN_EVIDENCE" "$EVIDENCE_CONTRACT"
  data/e1/e1_train_triplet_pool_10ppm.npz
  "$BASE/evidence/selection_triplet_evidence.npz"
  "$BASE/evidence/confirmation_triplet_evidence.npz"
  tasks/chemaware_fragment_graph_core.py
  tasks/build_chemaware_fragment_graph_candidate_source_ledger.py
  tasks/merge_chemaware_fragment_graph_source_ledgers.py
  tasks/qualify_chemaware_candidate_source_ledger.py
  tasks/build_chemaware_dynamic_reference_native_triplets.py
  tasks/build_chemaware_layered_10k_native_pool.py
  tasks/encode_chemaware_checkpoint_manifest_rows.py
  tasks/train_chemaware_dreams_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/select_chemaware_residual_checkpoint.py
  tasks/audit_chemaware_role3_confirmation.py
)
if [[ -z "$REUSE_RUN" ]]; then
  required_files+=(
    "$MS_PRED_SOURCE/src/ms_pred/magma/fragmentation.py"
    "$MS_PRED_SOURCE/src/ms_pred/common/chem_utils.py"
  )
else
  required_files+=(
    "$REUSE_RUN/qualified_source/report.json"
    "$REUSE_RUN/qualified_source/candidate_scores.tsv"
    "$REUSE_RUN/qualified_source/qualification_report.json"
    "$REUSE_RUN/unqualified_source/connected_explanations.tsv"
    "$REUSE_RUN/phasea_embeddings/rows.npy"
    "$REUSE_RUN/phasea_embeddings/embeddings_f32.npy"
  )
fi
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing ChemAware fragment-graph input: $required_file" >&2
    exit 2
  }
done

if [[ -z "$REUSE_RUN" ]]; then
  # Build the full 4,032-query source once.  Eight CPU shards only accelerate
  # the identical deterministic graph algorithm; they do not create variants.
  export OMP_NUM_THREADS=1
  pids=()
  for shard in $(seq 0 $((SHARDS - 1))); do
    python -u tasks/build_chemaware_fragment_graph_candidate_source_ledger.py \
      --evidence "$TRAIN_EVIDENCE" \
      --manifest "$MANIFEST" \
      --data "$DATA" \
      --ms-pred-source-root "$MS_PRED_SOURCE" \
      --evidence-contract "$EVIDENCE_CONTRACT" \
      --output "$OUT/source_shards/shard_${shard}" \
      --shard-count "$SHARDS" --shard-index "$shard" \
      > "$OUT/logs/source_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  source_failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || source_failed=1
  done
  if [[ "$source_failed" -ne 0 ]]; then
    echo "CHEMAWARE_FRAGMENT_GRAPH_SOURCE_FAILED" >&2
    tail -n 80 "$OUT"/logs/source_shard_*.log >&2 || true
    exit 4
  fi

  merge_args=()
  for shard in $(seq 0 $((SHARDS - 1))); do
    merge_args+=(--shard "$OUT/source_shards/shard_${shard}")
  done
  python -u tasks/merge_chemaware_fragment_graph_source_ledgers.py \
    "${merge_args[@]}" \
    --output "$OUT/unqualified_source"

  # This is the sole source gate: formula-disjoint confirmation against the
  # frozen fragment-assignment, topology and mass-shift controls.
  python -u tasks/qualify_chemaware_candidate_source_ledger.py \
    --ledger "$OUT/unqualified_source" \
    --manifest "$MANIFEST" \
    --output "$OUT/qualified_source" \
    --bootstrap-draws 10000 --minimum-applicable-queries 100

  AUTHORIZED="$(python -c "import json; print(int(json.load(open('$OUT/qualified_source/report.json'))['formal_triplet_mining_authorized']))")"
  if [[ "$AUTHORIZED" != "1" ]]; then
    echo "CHEMAWARE_FRAGMENT_GRAPH_SOURCE_STOP: full source did not pass its one frozen gate"
    echo "No GPU training was started. Evidence is retained at $OUT."
    exit 0
  fi

  # Extract the geometry of the protected +2.1266 pp model once.  There is no
  # Phase-A reproduction and no weight update in this step.
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
  srun --export=ALL --preserve-env python -u tasks/encode_chemaware_checkpoint_manifest_rows.py \
    --checkpoint "$PHASEA" \
    --evidence-dir "$BASE/evidence" \
    --manifest "$MANIFEST" \
    --data "$DATA" \
    --official-checkpoint "$OFFICIAL" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$OUT/phasea_embeddings" \
    --batch-size 64 --n-highest-peaks 100 --device cuda
  SOURCE_LEDGER="$OUT/qualified_source"
  UNQUALIFIED_SOURCE="$OUT/unqualified_source"
  EMBEDDING_DIR="$OUT/phasea_embeddings"
else
  SOURCE_LEDGER="$REUSE_RUN/qualified_source"
  UNQUALIFIED_SOURCE="$REUSE_RUN/unqualified_source"
  EMBEDDING_DIR="$REUSE_RUN/phasea_embeddings"
  printf '%s\n' "$REUSE_RUN" > "$OUT/reused_qualified_source.txt"
  echo "REUSE: qualified fragment graph and Phase-A embeddings from $REUSE_RUN"
fi

# The qualified chemistry selects identities only.  The existing mature
# DreaMS reference expander chooses native positive/negative spectra with a
# live hinge under the protected Phase-A embedding.
python -u tasks/build_chemaware_dynamic_reference_native_triplets.py \
  --phasea-pool "$PHASEA_POOL" \
  --validation-pool "$PHASEA_VAL" \
  --source-ledger "$SOURCE_LEDGER" \
  --manifest "$MANIFEST" \
  --embedding-rows "$EMBEDDING_DIR/rows.npy" \
  --phasea-embeddings "$EMBEDDING_DIR/embeddings_f32.npy" \
  --embedding-report "$EMBEDDING_DIR/report.json" \
  --geometry-checkpoint "$PHASEA" \
  --data "$DATA" \
  --output "$OUT/chemical_triplets" \
  --margin 0.1 \
  --maximum-candidates-per-query 6 \
  --maximum-positive-references-per-relation 1 \
  --maximum-negative-references-per-event 1 \
  --maximum-events-per-formula 8 \
  --maximum-events-per-identity 3 \
  --minimum-supporting-sources 1 \
  --matched-control-policy pairwise_dominance \
  --geometry-label "protected Phase-A +2.1266 pp checkpoint"

CHEM_COVERAGE="$(python -c \
  "import json; r=json.load(open('$OUT/chemical_triplets/report.json')); v=(r['dynamic_chemical_events_added'],r['dynamic_chemical_queries'],r['dynamic_chemical_formulas']); assert all(type(x) is int and x >= 0 for x in v); print(*v)")"
read -r CHEM_EVENTS CHEM_QUERIES CHEM_FORMULAS <<< "$CHEM_COVERAGE"
if [[ ! "$CHEM_EVENTS" =~ ^[0-9]+$ || ! "$CHEM_QUERIES" =~ ^[0-9]+$ || ! "$CHEM_FORMULAS" =~ ^[0-9]+$ ]]; then
  echo "CHEMAWARE_FRAGMENT_GRAPH_COVERAGE_PARSE_FAIL: $CHEM_COVERAGE" >&2
  exit 4
fi
if [[ "$CHEM_EVENTS" -lt 1000 || "$CHEM_QUERIES" -lt 500 ]]; then
  echo "CHEMAWARE_FRAGMENT_GRAPH_EVENT_COVERAGE_STOP: events=$CHEM_EVENTS queries=$CHEM_QUERIES formulas=$CHEM_FORMULAS" >&2
  echo "The missing chemical event coverage will not be disguised with repeated events." >&2
  exit 5
fi
echo "CHEMAWARE_FRAGMENT_GRAPH_COVERAGE: events=$CHEM_EVENTS queries=$CHEM_QUERIES formulas=$CHEM_FORMULAS"

PHASEA_EVENTS="$(python -c "import numpy as np; x=np.load('$PHASEA_POOL', allow_pickle=False); print(len(x['anchor_idx']))")"
TARGET_EVENTS=$((PHASEA_EVENTS + CHEM_EVENTS))
if [[ "$TARGET_EVENTS" -lt 12000 ]]; then
  TARGET_EVENTS=12000
fi

# Keep the successful 5,956-event Phase-A prefix and all new chemical events;
# unchanged official DreaMS replay fills coverage to at least 12,000 events.
# If high-quality chemistry itself exceeds that budget, no event is discarded.
python -u tasks/build_chemaware_layered_10k_native_pool.py \
  --phasea-pool "$PHASEA_POOL" \
  --source-pool "$OUT/chemical_triplets/train_pool.npz" \
  --source-report "$OUT/chemical_triplets/report.json" \
  --source-event-ledger "$OUT/chemical_triplets/dynamic_source_event_ledger.tsv" \
  --validation-pool "$PHASEA_VAL" \
  --dreams-replay-pool data/e1/e1_train_triplet_pool_10ppm.npz \
  --data "$DATA" \
  --output "$OUT/layered_12k_triplets" \
  --target-events "$TARGET_EVENTS" --minimum-total-events 10000 \
  --minimum-chemical-events 1000 --minimum-chemical-queries 500 \
  --seed 3407

# Protect the reusable chemical supervision before optimization.
mkdir -p "$OUT/protected_corpus"
ln "$OUT/layered_12k_triplets/train_pool.npz" "$OUT/protected_corpus/train_pool.npz"
ln "$OUT/layered_12k_triplets/val_pool.npz" "$OUT/protected_corpus/val_pool.npz"
cp "$OUT/layered_12k_triplets/report.json" "$OUT/protected_corpus/layered_12k_report.json"
cp "$OUT/layered_12k_triplets/corpus_event_manifest.tsv" "$OUT/protected_corpus/"
cp "$OUT/layered_12k_triplets/chemical_source_event_ledger.tsv" "$OUT/protected_corpus/"
ln "$SOURCE_LEDGER/candidate_scores.tsv" "$OUT/protected_corpus/fragment_graph_candidate_scores.tsv"
cp "$SOURCE_LEDGER/report.json" "$OUT/protected_corpus/fragment_graph_source_report.json"
cp "$SOURCE_LEDGER/qualification_report.json" "$OUT/protected_corpus/"
ln "$UNQUALIFIED_SOURCE/connected_explanations.tsv" "$OUT/protected_corpus/connected_explanations.tsv"
cp "$EVIDENCE_CONTRACT" "$OUT/protected_corpus/"
(
  cd "$OUT/protected_corpus"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

# No new loss, sampler, optimizer, teacher or adapter.  This is the mature
# DreaMS native triplet trainer, initialized directly from the protected model.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
  --data "$DATA" \
  --train-pool "$OUT/layered_12k_triplets/train_pool.npz" \
  --val-pool "$OUT/layered_12k_triplets/val_pool.npz" \
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
  echo "Expected four fixed-step checkpoints, found ${#trained_checkpoints[@]}" >&2
  exit 6
}
for checkpoint in "${trained_checkpoints[@]}"; do
  checkpoint_args+=(--checkpoint "fragment_graph_$(basename "$checkpoint" .ckpt)=$checkpoint")
done

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/selection_triplet_evidence.npz" \
  --role-contract-dir "$BASE/evidence" \
  "${checkpoint_args[@]}" \
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
  echo "CHEMAWARE_FRAGMENT_GRAPH_ROLE2_STOP: retain protected Phase A; role 3 remains unopened"
  echo "COMPLETE: fragment-graph role-2 result -> $OUT"
  exit 0
fi

SELECTED_REAL="$(realpath "$SELECTED")"
srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/confirmation_triplet_evidence.npz" \
  --role-contract-dir "$BASE/evidence" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "phaseA_2pp=$PHASEA" \
  --checkpoint "fragment_graph_selected=$SELECTED_REAL" \
  --paired-reference phaseA_2pp --formula-role 3 \
  --output "$OUT/role3_evaluation.json" \
  --batch-size 64 --bootstrap-draws 10000 --device cuda

python -u tasks/audit_chemaware_role3_confirmation.py \
  --evaluation "$OUT/role3_evaluation.json" \
  --base-name phaseA_2pp --protected-baseline-name phaseA_2pp \
  --candidate-name fragment_graph_selected \
  --output "$OUT/role3_confirmation_decision.json"
ROLE3_CONFIRMED="$(tr -d '\r\n' < "$OUT/role3_confirmation_decision.confirmed.txt")"
if [[ "$ROLE3_CONFIRMED" != "1" ]]; then
  mkdir -p "$OUT/protected_negative_result"
  ln "$SELECTED_REAL" "$OUT/protected_negative_result/chemaware_role2_selected.ckpt"
  cp "$OUT/checkpoint_selection.json" "$OUT/protected_negative_result/"
  cp "$OUT/role2_checkpoint_evaluation.json" "$OUT/protected_negative_result/"
  cp "$OUT/role3_evaluation.json" "$OUT/protected_negative_result/"
  cp "$OUT/role3_confirmation_decision.json" "$OUT/protected_negative_result/"
  cp "$OUT/training/report.json" "$OUT/protected_negative_result/training_report.json"
  (
    cd "$OUT/protected_negative_result"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
  )
  echo "CHEMAWARE_FRAGMENT_GRAPH_ROLE3_STOP: role-2 gain did not confirm on role 3"
  echo "COMPLETE: protected negative result -> $OUT/protected_negative_result"
  exit 0
fi

mkdir -p "$OUT/protected_candidate"
ln "$SELECTED_REAL" "$OUT/protected_candidate/chemaware_fragment_graph_selected.ckpt"
cp "$OUT/layered_12k_triplets/report.json" "$OUT/protected_candidate/corpus_report.json"
cp "$OUT/layered_12k_triplets/chemical_source_event_ledger.tsv" "$OUT/protected_candidate/"
cp "$OUT/checkpoint_selection.json" "$OUT/protected_candidate/"
cp "$OUT/role2_checkpoint_evaluation.json" "$OUT/protected_candidate/"
cp "$OUT/role3_evaluation.json" "$OUT/protected_candidate/"
cp "$OUT/role3_confirmation_decision.json" "$OUT/protected_candidate/"
cp "$OUT/training/report.json" "$OUT/protected_candidate/training_report.json"
(
  cd "$OUT/protected_candidate"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
echo "COMPLETE: protected ChemAware fragment-graph candidate -> $OUT/protected_candidate"
