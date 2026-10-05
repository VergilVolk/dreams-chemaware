#!/bin/bash
#SBATCH --job-name=chem_exact_bnd
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/chem_exact_bnd_%j.out
#SBATCH --error=data/validation/chem_exact_bnd_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OFFICIAL="data/e1/official_embedding_slim.pt"
BASE="data/validation/chemaware_high_coverage_native/run_2340524"
OUT="data/validation/chemaware_exact_boundary_residual/run_${SLURM_JOB_ID}"
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
  tasks/build_chemaware_exact_boundary_residual_triplets.py
  tasks/audit_chemaware_dense_boundary_alignment.py
  tasks/encode_chemaware_formula_role_checkpoint_rows.py
  tasks/train_chemaware_dreams_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/evaluate_chemaware_full_role_native.py
  tasks/select_chemaware_residual_checkpoint.py
  tasks/freeze_chemaware_exact_boundary_artifact.py
  tasks/test_chemaware_exact_boundary_residual_triplets.py
  tasks/test_chemaware_dense_boundary_alignment.py
  tasks/test_freeze_chemaware_exact_boundary_artifact.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required exact-boundary file: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_chemaware_exact_boundary_residual_triplets.py
python -u tasks/test_chemaware_dense_boundary_alignment.py
python -u tasks/test_freeze_chemaware_exact_boundary_artifact.py

# Reuse only a cache whose checkpoint checksum and formula roles match the
# protected Phase-A weights.  Otherwise encode roles 0--1 once in this job.
PHASEA_CACHE=""
mapfile -t cache_reports < <(
  find data/validation -type f -path '*/phasea_role01_cache/report.json' \
    -print 2>/dev/null | sort -r
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

python -u tasks/build_chemaware_exact_boundary_residual_triplets.py \
  --evidence "$BASE/evidence/train_triplet_evidence.npz" \
  --phasea-cache "$PHASEA_CACHE" \
  --validation-pool "$BASE/triplets/val_pool.npz" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --dreams-replay-pool data/e1/e1_train_triplet_pool_10ppm.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --output "$TRIPLETS" \
  --margin 0.1 --minimum-baseline-wins 2 --minimum-alternate-wins 3 \
  --negative-references-per-correction 3 \
  --safety-events-per-correction 3 --dreams-replay-events 512 \
  --minimum-correction-events 100 --minimum-correction-queries 50 \
  --minimum-correction-formulas 40 --seed 3407

TRIPLET_STATUS="$(python - "$TRIPLETS/report.json" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["status"])
PY
)"
if [[ "$TRIPLET_STATUS" == "CHEMAWARE_EXACT_BOUNDARY_RESIDUAL_COVERAGE_STOP" ]]; then
  python - "$TRIPLETS/report.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
body = json.loads(p.read_text(encoding="utf-8"))
print(
    "CHEMAWARE_EXACT_BOUNDARY_COVERAGE_STOP: "
    f"{body['coverage']['correction_events']} events / "
    f"{body['coverage']['correction_queries']} queries / "
    f"{body['coverage']['correction_formulas']} formulas; "
    "native continuation is scientifically underpowered and will not run",
    flush=True,
)
PY
  echo "COMPLETE: exact-boundary route stopped at frozen coverage gate; no weights updated"
  exit 0
fi

python - "$TRIPLETS/report.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
body = json.loads(p.read_text(encoding="utf-8"))
assert body["status"] == "CHEMAWARE_EXACT_BOUNDARY_RESIDUAL_TRIPLETS_COMPLETE"
assert body["cache_kind"] == "protected_phasea"
assert body["coverage"]["correction_events"] >= 100
assert body["coverage"]["correction_queries"] >= 50
assert body["coverage"]["correction_formulas"] >= 40
assert body["coverage"]["selected_safety_events"] == body["coverage"]["requested_safety_events"]
assert body["gates"]["no_identity_broadcast"] is True
assert body["gates"]["no_custom_sampling_weights"] is True
assert all(body["gates"].values())
print(
    "Verified exact-boundary curriculum: "
    f"{body['coverage']['correction_events']} events / "
    f"{body['coverage']['correction_queries']} queries / "
    f"{body['coverage']['correction_formulas']} formulas",
    flush=True,
)
PY

# Native DreaMS continuation only: unchanged ContrastiveSpectraDataset,
# ContrastiveHead, cosine triplet margin, Adam and uniform shuffled loader.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_dreams_native.py \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --train-pool "$TRIPLETS/train_pool.npz" \
  --val-pool "$TRIPLETS/val_pool.npz" \
  --official-checkpoint "$PHASEA" \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --output "$OUT/training" \
  --seed 3407 --lr 1e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
  --max-epochs 4 --max-steps 500 --checkpoint-mode fixed_steps \
  --save-every-n-steps 100 --n-highest-peaks 100 --device cuda

checkpoint_args=(
  --checkpoint "official=$OFFICIAL"
  --checkpoint "phaseA_base=$PHASEA"
)
mapfile -t trained_checkpoints < <(
  find "$OUT/training" -maxdepth 1 -type f -name 'step-*.ckpt' | sort
)
[[ ${#trained_checkpoints[@]} -eq 5 ]] || {
  echo "Expected five exact-boundary checkpoints, found ${#trained_checkpoints[@]}" >&2
  exit 5
}
for checkpoint in "${trained_checkpoints[@]}"; do
  checkpoint_args+=(--checkpoint "exact_$(basename "$checkpoint" .ckpt)=$checkpoint")
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
  echo "CHEMAWARE_EXACT_BOUNDARY_ROLE2_STOP: no checkpoint safely beat protected Phase A"
  echo "PRESERVED: all checkpoints, exact triplets and audits remain under $OUT"
  exit 0
fi
SELECTED_REAL="$(realpath "$SELECTED")"

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/confirmation_triplet_evidence.npz" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "phaseA_base=$PHASEA" \
  --checkpoint "exact_boundary=$SELECTED_REAL" \
  --paired-reference phaseA_base --formula-role 3 \
  --output "$OUT/role3_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_full_role_native.py \
  --formula-role 2 3 \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "phaseA_base=$PHASEA" \
  --checkpoint "exact_boundary=$SELECTED_REAL" \
  --output "$OUT/role2_role3_full_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

python -u tasks/freeze_chemaware_exact_boundary_artifact.py \
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
echo "COMPLETE: ChemAware exact-boundary native embedding -> $OUT"
