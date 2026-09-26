#!/bin/bash
#SBATCH --job-name=chem_dense_ts
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=data/validation/chem_dense_ts_%j.out
#SBATCH --error=data/validation/chem_dense_ts_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OFFICIAL="data/e1/official_embedding_slim.pt"
BASE="data/validation/chemaware_high_coverage_native/run_2340524"
OUT="data/validation/chemaware_dense_true_support_native/run_${SLURM_JOB_ID}"
TRIPLETS="$OUT/triplets"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

mapfile -t phasea_candidates < <(
  find data/validation/chemaware_phasea_2pp_protected -type f \
    -path '*/protected_artifact/chemaware_phasea_max_boundary_step2000.ckpt' \
    -print 2>/dev/null | sort
)
[[ ${#phasea_candidates[@]} -gt 0 ]] || {
  echo "Missing protected +2.1266 pp Phase-A artifact" >&2
  exit 2
}
PHASEA="${phasea_candidates[$((${#phasea_candidates[@]} - 1))]}"
PHASEA_DIR="$(dirname "$PHASEA")"
(cd "$PHASEA_DIR" && sha256sum -c SHA256SUMS)
python - "$PHASEA_DIR/artifact_manifest.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
body = json.loads(p.read_text(encoding="utf-8"))
assert body["status"] == "CHEMAWARE_PHASEA_2PP_ROLE2_ARTIFACT_PROTECTED"
assert abs(float(body["delta_recall1_percentage_points"]) - 2.1265822784810124) <= 1e-12
assert int(body["corrected_at_1"]) == 54 and int(body["introduced_at_1"]) == 12
print(f"Verified protected Phase-A warm start: {p.parent}", flush=True)
PY
PHASEA_SHA="$(sha256sum "$PHASEA" | awk '{print $1}')"

required_files=(
  "$OFFICIAL"
  "$PHASEA"
  "$BASE/triplets/val_pool.npz"
  "$BASE/evidence/train_triplet_evidence.npz"
  "$BASE/evidence/selection_triplet_evidence.npz"
  "$BASE/evidence/confirmation_triplet_evidence.npz"
  data/e1/e1_train_triplet_pool_10ppm.npz
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  dreams/models/pretrained/ssl_model_server.pt
  tasks/encode_chemaware_formula_role_checkpoint_rows.py
  tasks/encode_chemaware_checkpoint_manifest_rows.py
  tasks/build_chemaware_dense_true_support_native_triplets.py
  tasks/build_chemaware_true_support_native_triplets.py
  tasks/build_chemaware_multicondition_max_boundary_triplets.py
  tasks/build_chemaware_max_boundary_native_triplets.py
  tasks/build_chemaware_dreams_native_triplets.py
  tasks/chemaware_numpy_sampling.py
  tasks/train_chemaware_dreams_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/evaluate_chemaware_full_role_native.py
  tasks/select_chemaware_residual_checkpoint.py
  tasks/freeze_chemaware_true_support_artifact.py
  tasks/test_chemaware_dense_true_support_native.py
  tasks/test_freeze_chemaware_true_support_artifact.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required dense true-support file: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_chemaware_dense_true_support_native.py
python -u tasks/test_freeze_chemaware_true_support_artifact.py

# Reuse only an exact-checksum Phase-A role-0/1 cache.  Otherwise encode it
# once in this run.  Formula roles 2--4 remain untouched during construction.
PHASEA_CACHE=""
mapfile -t cache_reports < <(
  find data/validation/chemaware_phasea_residual_consensus -type f \
    -path '*/phasea_role01_cache/report.json' -print 2>/dev/null | sort -r
)
for report in "${cache_reports[@]}"; do
  if python - "$report" "$PHASEA_SHA" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
body = json.loads(p.read_text(encoding="utf-8"))
ok = (
    body.get("status") == "CHEMAWARE_FORMULA_ROLE_CHECKPOINT_EMBEDDINGS_COMPLETE"
    and body.get("checkpoint_sha256") == sys.argv[2]
    and body.get("formula_roles") == [0, 1]
    and body.get("outer_roles_2_3_4_accessed") is False
    and (p.parent / "rows.npy").is_file()
    and (p.parent / "embeddings_f32.npy").is_file()
)
raise SystemExit(0 if ok else 1)
PY
  then
    PHASEA_CACHE="$(dirname "$report")"
    echo "Reusing checksum-matched Phase-A cache: $PHASEA_CACHE"
    break
  fi
done

if [[ -z "$PHASEA_CACHE" ]]; then
  PHASEA_CACHE="$OUT/phasea_role01_cache"
  srun --export=ALL --preserve-env python -u tasks/encode_chemaware_formula_role_checkpoint_rows.py \
    --checkpoint "$PHASEA" \
    --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
    --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
    --official-checkpoint "$OFFICIAL" \
    --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
    --formula-role 0 1 --output "$PHASEA_CACHE" \
    --batch-size 64 --n-highest-peaks 100 --device cuda
fi

python - "$PHASEA_CACHE/report.json" "$PHASEA_SHA" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
body = json.loads(p.read_text(encoding="utf-8"))
assert body["checkpoint_sha256"] == sys.argv[2]
assert body["formula_roles"] == [0, 1]
assert body["outer_roles_2_3_4_accessed"] is False
print(f"Verified Phase-A-current cache: {p.parent}", flush=True)
PY

python -u tasks/build_chemaware_dense_true_support_native_triplets.py \
  --evidence "$BASE/evidence/train_triplet_evidence.npz" \
  --phasea-cache "$PHASEA_CACHE" \
  --validation-pool "$BASE/triplets/val_pool.npz" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --dreams-replay-pool data/e1/e1_train_triplet_pool_10ppm.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --output "$TRIPLETS" \
  --margin 0.1 --minimum-rule-wins 2 \
  --max-error-anchors-per-identity 6 --max-safety-anchors-per-identity 2 \
  --negative-candidates-per-query 4 --negative-references-per-candidate 3 \
  --dreams-replay-events 1024 \
  --minimum-correction-triplets 1000 \
  --minimum-correction-queries 200 --minimum-distinct-boundaries 200

python - "$TRIPLETS/report.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
body = json.loads(p.read_text(encoding="utf-8"))
assert body["status"] == "CHEMAWARE_DENSE_TRUE_SUPPORT_NATIVE_TRIPLETS_COMPLETE"
assert body["cache_kind"] == "protected_phasea"
assert body["events"]["unique_correction_triplets"] >= 1000
assert body["coverage"]["correction_queries"] >= 200
assert body["coverage"]["distinct_identity_false_boundaries"] >= 200
assert body["sampler"]["custom_sampling_weight_present"] is False
assert all(body["gates"].values())
print(
    "Verified dense hard curriculum: "
    f"{body['events']['unique_correction_triplets']} correction triplets, "
    f"{body['coverage']['correction_queries']} queries",
    flush=True,
)
PY

# Exact DreaMS runtime: native shuffled DataLoader, native ContrastiveHead,
# cosine triplet margin and Adam.  There is no weighted replacement sampler.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --train-pool "$TRIPLETS/train_pool.npz" \
  --val-pool "$TRIPLETS/val_pool.npz" \
  --official-checkpoint "$PHASEA" \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --output "$OUT/training" \
  --seed 3407 --lr 1e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
  --max-epochs 4 --max-steps 1500 --checkpoint-mode fixed_steps \
  --save-every-n-steps 250 --n-highest-peaks 100 --device cuda

checkpoint_args=(
  --checkpoint "official=$OFFICIAL"
  --checkpoint "phaseA_base=$PHASEA"
)
mapfile -t trained_checkpoints < <(
  find "$OUT/training" -maxdepth 1 -type f -name 'step-*.ckpt' | sort
)
[[ ${#trained_checkpoints[@]} -eq 6 ]] || {
  echo "Expected six dense true-support checkpoints, found ${#trained_checkpoints[@]}" >&2
  exit 5
}
for checkpoint in "${trained_checkpoints[@]}"; do
  checkpoint_args+=(--checkpoint "true_support_$(basename "$checkpoint" .ckpt)=$checkpoint")
done

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/selection_triplet_evidence.npz" \
  "${checkpoint_args[@]}" \
  --paired-reference phaseA_base --formula-role 2 \
  --output "$OUT/role2_checkpoint_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

python -u tasks/select_chemaware_residual_checkpoint.py \
  --evaluation "$OUT/role2_checkpoint_evaluation.json" \
  --base-name phaseA_base --require-positive-formula-ci \
  --output "$OUT/checkpoint_selection.json"
SELECTED="$(tr -d '\r\n' < "$OUT/checkpoint_selection.checkpoint.txt")"
SELECTED_STEP="$(tr -d '\r\n' < "$OUT/checkpoint_selection.step.txt")"

if [[ "$SELECTED_STEP" == "0" ]]; then
  echo "CHEMAWARE_DENSE_TRUE_SUPPORT_ROLE2_STOP: no checkpoint safely beat protected Phase A"
  echo "PRESERVED: all checkpoints, triplets and audits remain under $OUT"
  exit 0
fi
SELECTED_REAL="$(realpath "$SELECTED")"

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/confirmation_triplet_evidence.npz" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "phaseA_base=$PHASEA" \
  --checkpoint "true_support=$SELECTED_REAL" \
  --paired-reference phaseA_base --formula-role 3 \
  --output "$OUT/role3_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_full_role_native.py \
  --formula-role 2 3 \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "phaseA_base=$PHASEA" \
  --checkpoint "true_support=$SELECTED_REAL" \
  --output "$OUT/role2_role3_full_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

python -u tasks/freeze_chemaware_true_support_artifact.py \
  --checkpoint "$SELECTED_REAL" \
  --selection "$OUT/checkpoint_selection.json" \
  --role2-evaluation "$OUT/role2_checkpoint_evaluation.json" \
  --role3-evaluation "$OUT/role3_evaluation.json" \
  --full-evaluation "$OUT/role2_role3_full_evaluation.json" \
  --triplet-report "$TRIPLETS/report.json" \
  --train-pool "$TRIPLETS/train_pool.npz" \
  --val-pool "$TRIPLETS/val_pool.npz" \
  --training-report "$OUT/training/report.json" \
  --output "$OUT/protected_artifact"

test -f "$OUT/protected_artifact/artifact_manifest.json"
test -f "$OUT/protected_artifact/SHA256SUMS"
echo "COMPLETE: ChemAware dense true-support native embedding -> $OUT"
