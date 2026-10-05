#!/bin/bash
# Continue the exploratory fragment-graph step-750 seed with *new*,
# non-conflicting qualified MassBank and fragment-graph candidate relations.
# This is one GPU and reuses only already-qualified source ledgers; it does not
# regenerate fragment graphs, requalify sources, or modify native DreaMS code.
#SBATCH --job-name=chem_union
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=data/validation/chem_union_%j.out
#SBATCH --error=data/validation/chem_union_%j.err

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

# The step-750 checkpoint is an exploratory +0.2532 pp role-2 signal.  It is
# a continuation initialization, not a claim that supersedes protected Phase A.
SEED_RUN="data/validation/chemaware_fragment_graph_native/run_2346408_resume_run_2346306"
START="$SEED_RUN/training/step-000750.ckpt"
GRAPH_LEDGER="data/validation/chemaware_fragment_graph_native/run_2346306/qualified_source"
MASSBANK_LEDGER="data/validation/chemaware_massbank_candidate_source_all_qualified_v1_20260928"
OUT="data/validation/chemaware_fragment_massbank_union_continue/run_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

required_files=(
  "$OFFICIAL" "$ARCHITECTURE" "$DATA" "$MANIFEST" "$PHASEA" "$PHASEA_POOL" "$PHASEA_VAL"
  "$START"
  "$GRAPH_LEDGER/report.json" "$GRAPH_LEDGER/candidate_scores.tsv"
  "$MASSBANK_LEDGER/report.json" "$MASSBANK_LEDGER/candidate_scores.tsv"
  data/e1/e1_train_triplet_pool_10ppm.npz
  "$BASE/evidence/selection_triplet_evidence.npz" "$BASE/evidence/confirmation_triplet_evidence.npz"
  tasks/encode_chemaware_checkpoint_manifest_rows.py
  tasks/build_chemaware_dynamic_reference_native_triplets.py
  tasks/build_chemaware_layered_10k_native_pool.py
  tasks/train_chemaware_dreams_native.py tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/select_chemaware_residual_checkpoint.py tasks/audit_chemaware_role3_confirmation.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || { echo "Missing ChemAware union input: $required_file" >&2; exit 2; }
done

# Preserve the useful exploratory seed and the original 1,330-event corpus.
mkdir -p "$OUT/protected_seed"
ln "$START" "$OUT/protected_seed/fragment_graph_step750.ckpt"
ln "$SEED_RUN/protected_corpus/train_pool.npz" "$OUT/protected_seed/fragment_graph_1330_train_pool.npz"
if [[ -f "$SEED_RUN/role2_checkpoint_evaluation.json" ]]; then
  cp "$SEED_RUN/role2_checkpoint_evaluation.json" "$OUT/protected_seed/fragment_graph_role2.json"
fi
sha256sum \
  "$OFFICIAL" "$PHASEA" "$START" "$MANIFEST" \
  "$BASE/evidence/selection_triplet_evidence.npz" \
  "$BASE/evidence/confirmation_triplet_evidence.npz" \
  "$GRAPH_LEDGER/report.json" "$GRAPH_LEDGER/candidate_scores.tsv" \
  "$MASSBANK_LEDGER/report.json" "$MASSBANK_LEDGER/candidate_scores.tsv" \
  > "$OUT/input_sha256sums.txt"

# Encode the current continuation seed once.  Candidate relations below are
# selected against its actual residual geometry, not a stale Phase-A cache.
srun --export=ALL --preserve-env python -u tasks/encode_chemaware_checkpoint_manifest_rows.py \
  --checkpoint "$START" --evidence-dir "$BASE/evidence" --manifest "$MANIFEST" \
  --data "$DATA" --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE" \
  --output "$OUT/seed_embeddings" --batch-size 64 --n-highest-peaks 100 --device cuda

# A pair is admissible only when its observed true-vs-false delta dominates
# that relation's matched null controls.  Source-family qualification alone is
# insufficient for this formal continuation.
python -u tasks/build_chemaware_dynamic_reference_native_triplets.py \
  --phasea-pool "$PHASEA_POOL" --validation-pool "$PHASEA_VAL" \
  --source-ledger "$GRAPH_LEDGER" --source-ledger "$MASSBANK_LEDGER" \
  --manifest "$MANIFEST" --embedding-rows "$OUT/seed_embeddings/rows.npy" \
  --phasea-embeddings "$OUT/seed_embeddings/embeddings_f32.npy" --data "$DATA" \
  --embedding-report "$OUT/seed_embeddings/report.json" --geometry-checkpoint "$START" \
  --output "$OUT/union_triplets" --margin 0.1 \
  --maximum-candidates-per-query 6 --maximum-positive-references-per-relation 1 \
  --maximum-negative-references-per-event 1 --maximum-events-per-formula 8 \
  --maximum-events-per-identity 3 --minimum-supporting-sources 1 \
  --matched-control-policy pairwise_dominance \
  --geometry-label "fragment-graph step-750 exploratory continuation seed"

CHEM_COVERAGE="$(python -c \
  "import json; r=json.load(open('$OUT/union_triplets/report.json')); v=(r['dynamic_chemical_events_added'],r['dynamic_chemical_queries'],r['dynamic_chemical_formulas'],r['source_correctable_current_winners']); assert all(type(x) is int and x >= 0 for x in v); print(*v)")"
read -r CHEM_EVENTS CHEM_QUERIES CHEM_FORMULAS CORRECTABLE <<< "$CHEM_COVERAGE"
if [[ ! "$CHEM_EVENTS" =~ ^[0-9]+$ || ! "$CHEM_QUERIES" =~ ^[0-9]+$ || ! "$CHEM_FORMULAS" =~ ^[0-9]+$ || ! "$CORRECTABLE" =~ ^[0-9]+$ ]]; then
  echo "CHEMAWARE_UNION_COVERAGE_PARSE_FAIL: $CHEM_COVERAGE" >&2
  exit 4
fi
if [[ "$CHEM_EVENTS" -lt 1000 || "$CHEM_QUERIES" -lt 500 ]]; then
  echo "CHEMAWARE_UNION_EVENT_COVERAGE_STOP: events=$CHEM_EVENTS queries=$CHEM_QUERIES formulas=$CHEM_FORMULAS correctable_winners=$CORRECTABLE" >&2
  exit 5
fi
echo "CHEMAWARE_UNION_COVERAGE: events=$CHEM_EVENTS queries=$CHEM_QUERIES formulas=$CHEM_FORMULAS correctable_winners=$CORRECTABLE"

PHASEA_EVENTS="$(python -c "import numpy as np; x=np.load('$PHASEA_POOL', allow_pickle=False); print(len(x['anchor_idx']))")"
TARGET_EVENTS=$((PHASEA_EVENTS + CHEM_EVENTS))
if [[ "$TARGET_EVENTS" -lt 12000 ]]; then TARGET_EVENTS=12000; fi
python -u tasks/build_chemaware_layered_10k_native_pool.py \
  --phasea-pool "$PHASEA_POOL" --source-pool "$OUT/union_triplets/train_pool.npz" \
  --source-report "$OUT/union_triplets/report.json" \
  --source-event-ledger "$OUT/union_triplets/dynamic_source_event_ledger.tsv" \
  --validation-pool "$PHASEA_VAL" --dreams-replay-pool data/e1/e1_train_triplet_pool_10ppm.npz \
  --data "$DATA" --output "$OUT/layered_12k_triplets" --target-events "$TARGET_EVENTS" \
  --minimum-total-events 10000 --minimum-chemical-events 1000 --minimum-chemical-queries 500 --seed 3407

mkdir -p "$OUT/candidate_corpus"
ln "$OUT/layered_12k_triplets/train_pool.npz" "$OUT/candidate_corpus/train_pool.npz"
ln "$OUT/layered_12k_triplets/val_pool.npz" "$OUT/candidate_corpus/val_pool.npz"
cp "$OUT/layered_12k_triplets/report.json" "$OUT/candidate_corpus/"
cp "$OUT/union_triplets/report.json" "$OUT/candidate_corpus/union_triplets_report.json"
cp "$OUT/union_triplets/dynamic_source_event_ledger.tsv" "$OUT/candidate_corpus/"
cp "$OUT/union_triplets/rejected_relation_ledger.tsv" "$OUT/candidate_corpus/"
ln "$GRAPH_LEDGER/candidate_scores.tsv" "$OUT/candidate_corpus/fragment_graph_candidate_scores.tsv"
ln "$MASSBANK_LEDGER/candidate_scores.tsv" "$OUT/candidate_corpus/massbank_candidate_scores.tsv"
find "$OUT/candidate_corpus" -type f -printf '%P\n' | sort > "$OUT/candidate_corpus/MANIFEST.txt"

# Exact mature native DreaMS training; only the candidate relation corpus and
# the explicit initialization checkpoint differ from the preceding run.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
  --data "$DATA" --train-pool "$OUT/layered_12k_triplets/train_pool.npz" \
  --val-pool "$OUT/layered_12k_triplets/val_pool.npz" \
  --official-checkpoint "$START" --architecture-checkpoint "$ARCHITECTURE" \
  --restore-adam-from "$START" \
  --output "$OUT/training" --seed 3407 --lr 2e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 --max-epochs 3 --max-steps 1000 \
  --checkpoint-mode fixed_steps --save-every-n-steps 250 --n-highest-peaks 100 --device cuda

checkpoint_args=(--checkpoint "official=$OFFICIAL" --checkpoint "phaseA_2pp=$PHASEA" --checkpoint "fragment_graph_step750=$START")
mapfile -t trained_checkpoints < <(find "$OUT/training" -maxdepth 1 -type f -name 'step-*.ckpt' | sort)
[[ ${#trained_checkpoints[@]} -eq 4 ]] || { echo "Expected four fixed-step checkpoints, found ${#trained_checkpoints[@]}" >&2; exit 6; }
for checkpoint in "${trained_checkpoints[@]}"; do
  checkpoint_args+=(--checkpoint "union_$(basename "$checkpoint" .ckpt)=$checkpoint")
done

# Selection is relative to the exploratory step-750 seed, while Phase A is
# still reported as a protected fallback.  Role 3 remains unopened on failure.
srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/selection_triplet_evidence.npz" "${checkpoint_args[@]}" \
  --role-contract-dir "$BASE/evidence" \
  --paired-reference fragment_graph_step750 --formula-role 2 \
  --output "$OUT/role2_checkpoint_evaluation.json" --batch-size 64 --bootstrap-draws 10000 --device cuda
python -u tasks/select_chemaware_residual_checkpoint.py \
  --evaluation "$OUT/role2_checkpoint_evaluation.json" --base-name fragment_graph_step750 \
  --protected-baseline-name phaseA_2pp \
  --exclude-name phaseA_2pp \
  --require-positive-formula-ci --output "$OUT/checkpoint_selection.json"
SELECTED="$(tr -d '\r\n' < "$OUT/checkpoint_selection.checkpoint.txt")"
SELECTED_STEP="$(tr -d '\r\n' < "$OUT/checkpoint_selection.step.txt")"
if [[ "$SELECTED_STEP" == "0" ]]; then
  echo "CHEMAWARE_UNION_ROLE2_STOP: retain protected Phase A; role 3 remains unopened"
  exit 0
fi

SELECTED_REAL="$(realpath "$SELECTED")"
srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/confirmation_triplet_evidence.npz" \
  --role-contract-dir "$BASE/evidence" \
  --checkpoint "official=$OFFICIAL" --checkpoint "phaseA_2pp=$PHASEA" \
  --checkpoint "fragment_graph_step750=$START" --checkpoint "union_selected=$SELECTED_REAL" \
  --paired-reference fragment_graph_step750 --formula-role 3 \
  --output "$OUT/role3_evaluation.json" --batch-size 64 --bootstrap-draws 10000 --device cuda
python -u tasks/audit_chemaware_role3_confirmation.py \
  --evaluation "$OUT/role3_evaluation.json" --base-name fragment_graph_step750 \
  --protected-baseline-name phaseA_2pp --candidate-name union_selected \
  --output "$OUT/role3_confirmation_decision.json"
echo "COMPLETE: ChemAware fragment-graph/MassBank union continuation -> $OUT"
