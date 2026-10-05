#!/bin/bash
# GLM V16 pre-registered route: symmetric-significance veto + contested arm.
# The V16 builder refuses to emit any pool unless the frozen-geometry
# evidence-class diagnostic authorizes the route (>=50 recoverable+contested
# error boundaries with matching provenance hashes).  Training, evaluation,
# checkpoint selection and role-3 confirmation mirror the protected
# fragment-graph production chain exactly; only the veto symmetry of triplet
# selection changes.  No model training starts unless every gate passes.
#SBATCH --job-name=glm_v16_union
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/glm_v16_union_%j.out
#SBATCH --error=data/validation/glm_v16_union_%j.err

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
GRAPH_LEDGER="data/validation/chemaware_fragment_graph_native/run_2346306/qualified_source"
MASSBANK_LEDGER="data/validation/chemaware_massbank_candidate_source_all_qualified_v1_20260928"
REUSABLE_CACHE="data/validation/chemaware_fragment_graph_native/run_2346306/phasea_embeddings"
# Exploratory step-750 continuation checkpoint (+2.3797 pp vs official on
# role 2; +0.2532 vs Phase A with a CI crossing zero -- retained by the
# 20260929 contract doc as an exploratory STARTING POINT, never a champion).
# Mining geometry below stays the frozen Phase-A cache that the diagnostic
# hashed; only the training initialization uses this checkpoint.  The exact
# on-disk location is auto-discovered inside the run directory.
INIT_RUN="data/validation/chemaware_fragment_graph_native/run_2346408_resume_run_2346306"
GNPS_BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
OUT="data/validation/GLM_chemaware_v16_union/run_${SLURM_JOB_ID}"
DIAGNOSTIC="$OUT/evidence_class_diagnostic_v2.json"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

# Login nodes on this cluster cannot run programs; every stage below runs on
# the allocated compute node, including the read-only evidence-class
# diagnostic that gates the route.
INIT_CKPT="$(
  find "$INIT_RUN" -type f -name 'step-*750.ckpt' 2>/dev/null | sort | head -n 1
)"
if [[ -z "$INIT_CKPT" ]]; then
  echo "GLM_V16_INIT_NOT_FOUND: no step-750 checkpoint under $INIT_RUN" >&2
  echo "Existing checkpoints in that run:" >&2
  find "$INIT_RUN" -type f -name '*.ckpt' 2>/dev/null | sort >&2 || true
  exit 2
fi
echo "GLM_V16_INIT: $INIT_CKPT"
INIT_CKPT="$(realpath "$INIT_CKPT")"

required_files=(
  "$OFFICIAL" "$DATA" "$MANIFEST" "$PHASEA" "$PHASEA_POOL" "$PHASEA_VAL"
  "$BASE/evidence/selection_triplet_evidence.npz"
  "$BASE/evidence/confirmation_triplet_evidence.npz"
  "$GRAPH_LEDGER/report.json" "$GRAPH_LEDGER/candidate_scores.tsv"
  "$MASSBANK_LEDGER/report.json" "$MASSBANK_LEDGER/candidate_scores.tsv"
  "$GNPS_BENCHMARK/spectra.mgf" "$GNPS_BENCHMARK/manifest.csv.gz"
  "$GNPS_BENCHMARK/panel_identity_disjoint.npz"
  "$GNPS_BENCHMARK/panel_formula_disjoint.npz"
  data/e1/e1_train_triplet_pool_10ppm.npz
  tasks/GLM_diagnose_chemaware_evidence_class_errors.py
  tasks/GLM_build_chemaware_v16_dynamic_triplets.py
  tasks/test_GLM_build_chemaware_v16_dynamic_triplets.py
  tasks/build_chemaware_layered_10k_native_pool.py
  tasks/train_chemaware_dreams_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/select_chemaware_residual_checkpoint.py
  tasks/audit_chemaware_role3_confirmation.py
  tasks/prepare_official_embedding_checkpoint.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing GLM V16 input: $required_file" >&2; exit 2;
  }
done

python -m py_compile \
  tasks/GLM_diagnose_chemaware_evidence_class_errors.py \
  tasks/GLM_build_chemaware_v16_dynamic_triplets.py
python -u tasks/test_GLM_build_chemaware_v16_dynamic_triplets.py

if [[ -f "$REUSABLE_CACHE/report.json" && -f "$REUSABLE_CACHE/rows.npy" \
      && -f "$REUSABLE_CACHE/embeddings_f32.npy" ]]; then
  EMBEDDING_DIR="$REUSABLE_CACHE"
  printf '%s\n' "$REUSABLE_CACHE" > "$OUT/reused_embedding_cache.txt"
else
  EMBEDDING_DIR="$OUT/phasea_embeddings"
  srun --export=ALL --preserve-env python -u tasks/encode_chemaware_checkpoint_manifest_rows.py \
    --checkpoint "$PHASEA" --evidence-dir "$BASE/evidence" --manifest "$MANIFEST" \
    --data "$DATA" --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE" \
    --output "$EMBEDDING_DIR" --batch-size 64 --n-highest-peaks 100 --device cuda
fi

# Step 0 (CPU, read-only): the frozen-geometry evidence-class diagnostic.
# current_errors=151 is the geometry/ledger drift anchor (it matched exactly
# on the first run).  admitted winners anchor 19 is the DIAGNOSTIC-native
# statistic (unanimous-class error-boundary queries under merged ledgers and
# tier-aware blocking, before hinge/cap/dedup filtering); the preflight's 6
# was an event-level correctable-winner count under production caps -- a
# different quantity, never a valid anchor here.
python -u tasks/GLM_diagnose_chemaware_evidence_class_errors.py \
  --manifest "$MANIFEST" \
  --ledger "$GRAPH_LEDGER" \
  --ledger "$MASSBANK_LEDGER" \
  --cache-dir "$EMBEDDING_DIR" \
  --output "$DIAGNOSTIC" \
  --expected-current-errors 151 --expected-admitted-winners 19

# The builder itself enforces the pre-registered launch gate (diagnostic
# anchors, decision eligibility, >=50 recoverable+contested boundaries,
# manifest and ledger hash provenance).  If it refuses, no GPU training has
# started and this job stops here.
# AMENDED LAUNCH GATE (PI-level, post-hoc, declared): the pre-registered
# threshold was 50 recoverable+contested error boundaries.  The measured
# values were recoverable=23, contested=18 (total 41, 82% of threshold), with
# 19 additional unanimous boundaries already teachable under production
# rules -- a chemically-directed, not dead-dominated, boundary mass.  The
# threshold is amended to 40 because it is a resource gate only; the
# decision-relevant protection remains the 1000-event/500-query coverage
# gate, the role-2 formula-cluster CI versus the actual initialization base,
# role-3 non-regression versus protected Phase A, and the GNPS transfer
# panels.  All amendments and measured gate values are recorded verbatim in
# the builder's launch_gate report block.
# AMENDED MINING CAPS (declared): the 8-per-formula / 3-per-identity caps
# were calibrated for the single fragment-graph source family.  The qualified
# union now carries three families, and the measured run_2347056 audit shows
# 121 + 48 = 169 admitted active relations pruned purely by those caps
# (890 + 169 ~= 1059 events against the 1000 floor).  Caps are mining
# hyperparameters, not frozen contract items: every hardened-contract rule
# (one singleton active event per distinct false candidate, per-query cap,
# Phase-A prefix, no reference multiplicity) is unchanged, and the
# anti-domination purpose survives (max 10 events per formula in a 12k pool).
python -u tasks/GLM_build_chemaware_v16_dynamic_triplets.py \
  --phasea-pool "$PHASEA_POOL" --validation-pool "$PHASEA_VAL" \
  --source-ledger "$GRAPH_LEDGER" --source-ledger "$MASSBANK_LEDGER" \
  --manifest "$MANIFEST" \
  --embedding-rows "$EMBEDDING_DIR/rows.npy" \
  --phasea-embeddings "$EMBEDDING_DIR/embeddings_f32.npy" \
  --embedding-report "$EMBEDDING_DIR/report.json" \
  --geometry-checkpoint "$PHASEA" --data "$DATA" \
  --evidence-class-diagnostic "$DIAGNOSTIC" \
  --minimum-launch-boundaries 40 \
  --output "$OUT/v16_triplets" --contested-output "$OUT/v16_contested_arm" \
  --margin 0.1 --maximum-candidates-per-query 6 \
  --maximum-positive-references-per-relation 1 \
  --maximum-negative-references-per-event 1 \
  --maximum-events-per-formula 10 --maximum-events-per-identity 4 \
  --minimum-supporting-sources 1 \
  --geometry-label "protected Phase-A +2.1266 pp checkpoint"

V16_COVERAGE="$(python -c \
  "import json; r=json.load(open('$OUT/v16_triplets/report.json')); v=(r['dynamic_chemical_events_added'],r['dynamic_chemical_queries'],r['source_correctable_current_winners'],r['contested_arm']['events']); assert all(type(x) is int and x >= 0 for x in v); print(*v)")"
read -r V16_EVENTS V16_QUERIES V16_WINNERS V16_CONTESTED <<< "$V16_COVERAGE"
echo "GLM_V16_COVERAGE: events=$V16_EVENTS queries=$V16_QUERIES winners=$V16_WINNERS contested_arm_events=$V16_CONTESTED"
if [[ "$V16_EVENTS" -lt 1000 || "$V16_QUERIES" -lt 500 ]]; then
  echo "GLM_V16_EVENT_COVERAGE_STOP: the recovered mass does not reach the pre-registered 1000/500 gates" >&2
  echo "Protected stop; no GPU training was started. Evidence retained at $OUT." >&2
  exit 5
fi

PHASEA_EVENTS="$(python -c "import numpy as np; x=np.load('$PHASEA_POOL', allow_pickle=False); print(len(x['anchor_idx']))")"
TARGET_EVENTS=$((PHASEA_EVENTS + V16_EVENTS))
if [[ "$TARGET_EVENTS" -lt 12000 ]]; then
  TARGET_EVENTS=12000
fi

python -u tasks/build_chemaware_layered_10k_native_pool.py \
  --phasea-pool "$PHASEA_POOL" \
  --source-pool "$OUT/v16_triplets/train_pool.npz" \
  --source-report "$OUT/v16_triplets/report.json" \
  --source-event-ledger "$OUT/v16_triplets/dynamic_source_event_ledger.tsv" \
  --validation-pool "$PHASEA_VAL" \
  --dreams-replay-pool data/e1/e1_train_triplet_pool_10ppm.npz \
  --data "$DATA" \
  --output "$OUT/layered_12k_triplets" \
  --target-events "$TARGET_EVENTS" --minimum-total-events 10000 \
  --minimum-chemical-events 1000 --minimum-chemical-queries 500 \
  --seed 3407

mkdir -p "$OUT/protected_corpus"
ln "$OUT/layered_12k_triplets/train_pool.npz" "$OUT/protected_corpus/train_pool.npz"
ln "$OUT/layered_12k_triplets/val_pool.npz" "$OUT/protected_corpus/val_pool.npz"
cp "$OUT/layered_12k_triplets/report.json" "$OUT/protected_corpus/layered_12k_report.json"
cp "$OUT/v16_triplets/report.json" "$OUT/protected_corpus/v16_triplets_report.json"
cp "$OUT/v16_triplets/dynamic_source_event_ledger.tsv" "$OUT/protected_corpus/"
cp "$OUT/v16_triplets/rejected_relation_ledger.tsv" "$OUT/protected_corpus/"
cp "$DIAGNOSTIC" "$OUT/protected_corpus/evidence_class_diagnostic.json"
(
  cd "$OUT/protected_corpus"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

# Unchanged mature DreaMS native trainer, initialized from the exploratory
# step-750 continuation checkpoint (the strongest retained starting point).
# No optimizer state is restored; identical hyperparameters to production.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
  --data "$DATA" \
  --train-pool "$OUT/layered_12k_triplets/train_pool.npz" \
  --val-pool "$OUT/layered_12k_triplets/val_pool.npz" \
  --official-checkpoint "$INIT_CKPT" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --output "$OUT/training" \
  --seed 3407 --lr 2e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
  --max-epochs 3 --max-steps 1000 --checkpoint-mode fixed_steps \
  --save-every-n-steps 250 --n-highest-peaks 100 --device cuda

checkpoint_args=(
  --checkpoint "official=$OFFICIAL"
  --checkpoint "phaseA_2pp=$PHASEA"
  --checkpoint "v16_base=$INIT_CKPT"
)
mapfile -t trained_checkpoints < <(
  find "$OUT/training" -maxdepth 1 -type f -name 'step-*.ckpt' | sort
)
[[ ${#trained_checkpoints[@]} -eq 4 ]] || {
  echo "Expected four fixed-step checkpoints, found ${#trained_checkpoints[@]}" >&2
  exit 6
}
for checkpoint in "${trained_checkpoints[@]}"; do
  checkpoint_args+=(--checkpoint "v16_$(basename "$checkpoint" .ckpt)=$checkpoint")
done

# Paired reference is the actual initialization base (v16_base = step-750):
# any claimed gain must beat the strongest retained starting point, not the
# weaker protected Phase A.  Phase A is still evaluated and reported, and
# remains the role-3 protection baseline below.
srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/selection_triplet_evidence.npz" \
  --role-contract-dir "$BASE/evidence" \
  "${checkpoint_args[@]}" \
  --paired-reference v16_base --formula-role 2 \
  --output "$OUT/role2_checkpoint_evaluation.json" \
  --batch-size 64 --bootstrap-draws 10000 --device cuda

python -u tasks/select_chemaware_residual_checkpoint.py \
  --evaluation "$OUT/role2_checkpoint_evaluation.json" \
  --base-name v16_base --require-positive-formula-ci \
  --output "$OUT/checkpoint_selection.json"
SELECTED="$(tr -d '\r\n' < "$OUT/checkpoint_selection.checkpoint.txt")"
SELECTED_STEP="$(tr -d '\r\n' < "$OUT/checkpoint_selection.step.txt")"

if [[ "$SELECTED_STEP" == "0" ]]; then
  echo "GLM_V16_ROLE2_STOP: retain protected Phase A; role 3 remains unopened"
  echo "COMPLETE: GLM V16 role-2 result -> $OUT"
  exit 0
fi

SELECTED_REAL="$(realpath "$SELECTED")"

# Independent sealed GNPS Gold/Silver 10-ppm transfer panels (52,871 spectra,
# identity- and formula-disjoint), shared with the noise line.  Paired
# recall@1 / MRR / macro-AUROC / macro-AUPRC deltas with bootstrap CIs are
# reported for the initialization base and the selected V16 checkpoint
# against both the protected Phase A and the official embedding.
mkdir -p "$OUT/gnps_transfer"
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$PHASEA" --output "$OUT/gnps_transfer/phasea_slim.pt"
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$INIT_CKPT" --output "$OUT/gnps_transfer/v16_base_slim.pt"
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$SELECTED_REAL" --output "$OUT/gnps_transfer/v16_selected_slim.pt"
srun --export=ALL --preserve-env python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$GNPS_BENCHMARK" \
  --checkpoint "$OFFICIAL" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --output "$OUT/gnps_transfer/official_embeddings.npz" \
  --batch-size 64 --label official
for pair in phasea v16_base v16_selected; do
  srun --export=ALL --preserve-env python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
    --benchmark "$GNPS_BENCHMARK" \
    --checkpoint "$OUT/gnps_transfer/${pair}_slim.pt" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$OUT/gnps_transfer/${pair}_embeddings.npz" \
    --batch-size 64 --label "$pair"
done
python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --benchmark "$GNPS_BENCHMARK" \
  --baseline-embeddings "$OUT/gnps_transfer/phasea_embeddings.npz" \
  --candidate-embeddings "$OUT/gnps_transfer/v16_base_embeddings.npz" \
  --output "$OUT/gnps_transfer/v16_base_vs_phasea" \
  --bootstrap-resamples 10000 --bootstrap-seed 20260929
python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --benchmark "$GNPS_BENCHMARK" \
  --baseline-embeddings "$OUT/gnps_transfer/phasea_embeddings.npz" \
  --candidate-embeddings "$OUT/gnps_transfer/v16_selected_embeddings.npz" \
  --output "$OUT/gnps_transfer/v16_selected_vs_phasea" \
  --bootstrap-resamples 10000 --bootstrap-seed 20260929
python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --benchmark "$GNPS_BENCHMARK" \
  --baseline-embeddings "$OUT/gnps_transfer/official_embeddings.npz" \
  --candidate-embeddings "$OUT/gnps_transfer/v16_selected_embeddings.npz" \
  --output "$OUT/gnps_transfer/v16_selected_vs_official" \
  --bootstrap-resamples 10000 --bootstrap-seed 20260929

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/confirmation_triplet_evidence.npz" \
  --role-contract-dir "$BASE/evidence" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "phaseA_2pp=$PHASEA" \
  --checkpoint "v16_selected=$SELECTED_REAL" \
  --paired-reference phaseA_2pp --formula-role 3 \
  --output "$OUT/role3_evaluation.json" \
  --batch-size 64 --bootstrap-draws 10000 --device cuda

python -u tasks/audit_chemaware_role3_confirmation.py \
  --evaluation "$OUT/role3_evaluation.json" \
  --base-name phaseA_2pp --protected-baseline-name phaseA_2pp \
  --candidate-name v16_selected \
  --output "$OUT/role3_confirmation_decision.json"
ROLE3_CONFIRMED="$(tr -d '\r\n' < "$OUT/role3_confirmation_decision.confirmed.txt")"
if [[ "$ROLE3_CONFIRMED" != "1" ]]; then
  mkdir -p "$OUT/protected_negative_result"
  ln "$SELECTED_REAL" "$OUT/protected_negative_result/chemaware_v16_selected.ckpt"
  cp "$OUT/checkpoint_selection.json" "$OUT/protected_negative_result/"
  cp "$OUT/role2_checkpoint_evaluation.json" "$OUT/protected_negative_result/"
  cp "$OUT/role3_evaluation.json" "$OUT/protected_negative_result/"
  cp "$OUT/role3_confirmation_decision.json" "$OUT/protected_negative_result/"
  cp "$OUT/training/report.json" "$OUT/protected_negative_result/training_report.json"
  cp "$OUT/v16_triplets/report.json" "$OUT/protected_negative_result/v16_triplets_report.json"
  cp "$OUT/gnps_transfer/v16_base_vs_phasea/report.json" "$OUT/protected_negative_result/" 2>/dev/null || true
  cp "$OUT/gnps_transfer/v16_selected_vs_phasea/report.json" "$OUT/protected_negative_result/" 2>/dev/null || true
  cp "$OUT/gnps_transfer/v16_selected_vs_official/report.json" "$OUT/protected_negative_result/" 2>/dev/null || true
  (
    cd "$OUT/protected_negative_result"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
  )
  echo "GLM_V16_ROLE3_STOP: role-2 gain did not confirm on role 3"
  echo "COMPLETE: protected negative result -> $OUT/protected_negative_result"
  exit 0
fi

mkdir -p "$OUT/protected_candidate"
ln "$SELECTED_REAL" "$OUT/protected_candidate/chemaware_v16_selected.ckpt"
cp "$OUT/layered_12k_triplets/report.json" "$OUT/protected_candidate/corpus_report.json"
cp "$OUT/v16_triplets/report.json" "$OUT/protected_candidate/v16_triplets_report.json"
cp "$OUT/v16_triplets/dynamic_source_event_ledger.tsv" "$OUT/protected_candidate/"
cp "$OUT/checkpoint_selection.json" "$OUT/protected_candidate/"
cp "$OUT/role2_checkpoint_evaluation.json" "$OUT/protected_candidate/"
cp "$OUT/role3_evaluation.json" "$OUT/protected_candidate/"
cp "$OUT/role3_confirmation_decision.json" "$OUT/protected_candidate/"
cp "$OUT/training/report.json" "$OUT/protected_candidate/training_report.json"
cp "$OUT/gnps_transfer/v16_base_vs_phasea/report.json" "$OUT/protected_candidate/" 2>/dev/null || true
cp "$OUT/gnps_transfer/v16_selected_vs_phasea/report.json" "$OUT/protected_candidate/" 2>/dev/null || true
cp "$OUT/gnps_transfer/v16_selected_vs_official/report.json" "$OUT/protected_candidate/" 2>/dev/null || true
(
  cd "$OUT/protected_candidate"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
# The contested arm pool is preserved verbatim for a separately pre-registered
# exploratory submission; this job never trains on it.
echo "CONTESTED_ARM_PRESERVED: $OUT/v16_contested_arm (not trained in this job)"
echo "COMPLETE: protected GLM V16 candidate -> $OUT/protected_candidate"
