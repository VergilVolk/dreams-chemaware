#!/bin/bash
#SBATCH --job-name=chem_pa_resid
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/chem_pa_resid_%j.out
#SBATCH --error=data/validation/chem_pa_resid_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OFFICIAL="data/e1/official_embedding_slim.pt"
BASE="data/validation/chemaware_high_coverage_native/run_2340524"
OUT="data/validation/chemaware_phasea_residual_consensus/run_${SLURM_JOB_ID}"
PHASEA_BANK="$OUT/phasea_bank"
PHASEA_CACHE="$OUT/phasea_role01_cache"
RESIDUAL_BANK="$OUT/residual_triplets"

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

required_files=(
  "$OFFICIAL"
  "$PHASEA"
  "$BASE/triplets/report.json"
  "$BASE/triplets/train_pool.npz"
  "$BASE/triplets/val_pool.npz"
  "$BASE/evidence/train_triplet_evidence.npz"
  "$BASE/evidence/selection_triplet_evidence.npz"
  "$BASE/evidence/confirmation_triplet_evidence.npz"
  data/e1/e1_train_triplet_pool_10ppm.npz
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  data/validation/chemaware_corrected_manifest_tokens_v1/rows.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/official_embeddings_f32.npy
  dreams/models/pretrained/ssl_model_server.pt
  tasks/build_chemaware_max_boundary_native_triplets.py
  tasks/encode_chemaware_formula_role_checkpoint_rows.py
  tasks/build_chemaware_phasea_residual_consensus_triplets.py
  tasks/train_chemaware_weighted_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/evaluate_chemaware_full_role_native.py
  tasks/select_chemaware_residual_checkpoint.py
  tasks/freeze_chemaware_multicondition_artifact.py
  tasks/test_chemaware_phasea_residual_consensus.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required Phase-A residual file: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_chemaware_phasea_residual_consensus.py

# Reconstruct the exact successful Phase-A triplet pool for immutable replay.
python -u tasks/build_chemaware_max_boundary_native_triplets.py \
  --base-bank "$BASE/triplets" \
  --evidence "$BASE/evidence/train_triplet_evidence.npz" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --embedding-rows data/validation/chemaware_corrected_manifest_tokens_v1/rows.npy \
  --official-embeddings data/validation/chemaware_corrected_manifest_tokens_v1/official_embeddings_f32.npy \
  --dreams-replay-pool data/e1/e1_train_triplet_pool_10ppm.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --output "$PHASEA_BANK" \
  --margin 0.1 --negative-references-per-error 2 \
  --chemical-candidates-per-error 2 --dreams-replay-events 1024 \
  --seed 3407 --min-error-event-fraction 0.20

# Encode every role-0/1 query and candidate reference with the protected
# Phase-A embedding.  Residual mining therefore targets current, not official,
# ranking errors.
srun --export=ALL --preserve-env python -u tasks/encode_chemaware_formula_role_checkpoint_rows.py \
  --checkpoint "$PHASEA" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --official-checkpoint "$OFFICIAL" \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --formula-role 0 1 --output "$PHASEA_CACHE" \
  --batch-size 64 --n-highest-peaks 100 --device cuda

python - "$PHASEA_DIR/artifact_manifest.json" "$PHASEA_CACHE/report.json" <<'PY'
import json, pathlib, sys
artifact = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
cache = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
assert cache["checkpoint_sha256"] == artifact["checkpoint_sha256"]
assert cache["formula_roles"] == [0, 1]
assert cache["outer_roles_2_3_4_accessed"] is False
print("Verified Phase-A-current role-0/1 embedding cache", flush=True)
PY

python -u tasks/build_chemaware_phasea_residual_consensus_triplets.py \
  --phasea-bank "$PHASEA_BANK" \
  --base-action-bank "$BASE/triplets" \
  --evidence "$BASE/evidence/train_triplet_evidence.npz" \
  --phasea-cache "$PHASEA_CACHE" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --output "$RESIDUAL_BANK" \
  --margin 0.1 --protection-margin 0.1 \
  --collision-energy-bin 10 --preselect-anchors-per-identity 8 \
  --max-residual-anchors-per-identity 2 --chemical-candidates-per-error 1 \
  --base-sampling-mass 0.70 --consensus-sampling-mass 0.18 \
  --isolated-sampling-mass 0.07 --sentinel-sampling-mass 0.05

# Native DreaMS continuation from Phase A.  The low learning rate and short
# checkpoints expose the earliest safe residual update before global drift.
srun --export=ALL --preserve-env python -u tasks/train_chemaware_weighted_native.py \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --train-pool "$RESIDUAL_BANK/train_pool.npz" \
  --val-pool "$RESIDUAL_BANK/val_pool.npz" \
  --official-checkpoint "$PHASEA" \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --output "$OUT/training" \
  --seed 3407 --lr 1e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
  --max-epochs 2 --max-steps 800 --checkpoint-mode fixed_steps \
  --save-every-n-steps 100 --n-highest-peaks 100 --device cuda

checkpoint_args=(
  --checkpoint "official=$OFFICIAL"
  --checkpoint "phaseA_base=$PHASEA"
)
mapfile -t trained_checkpoints < <(
  find "$OUT/training" -maxdepth 1 -type f -name 'step-*.ckpt' | sort
)
[[ ${#trained_checkpoints[@]} -eq 8 ]] || {
  echo "Expected eight Phase-A residual checkpoints, found ${#trained_checkpoints[@]}" >&2
  exit 5
}
for checkpoint in "${trained_checkpoints[@]}"; do
  checkpoint_args+=(--checkpoint "pa_resid_$(basename "$checkpoint" .ckpt)=$checkpoint")
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
  echo "CHEMAWARE_PHASEA_RESIDUAL_ROLE2_STOP: no short residual checkpoint safely beat Phase A"
  echo "PRESERVED: all checkpoints, current-geometry cache and audits remain under $OUT"
  exit 0
fi
SELECTED_REAL="$(realpath "$SELECTED")"

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/confirmation_triplet_evidence.npz" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "phaseA_base=$PHASEA" \
  --checkpoint "multicondition=$SELECTED_REAL" \
  --paired-reference phaseA_base --formula-role 3 \
  --output "$OUT/role3_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_full_role_native.py \
  --formula-role 2 3 \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "phaseA_base=$PHASEA" \
  --checkpoint "multicondition=$SELECTED_REAL" \
  --output "$OUT/role2_role3_full_evaluation.json" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

python -u tasks/freeze_chemaware_multicondition_artifact.py \
  --checkpoint "$SELECTED_REAL" \
  --selection "$OUT/checkpoint_selection.json" \
  --role2-evaluation "$OUT/role2_checkpoint_evaluation.json" \
  --role3-evaluation "$OUT/role3_evaluation.json" \
  --full-evaluation "$OUT/role2_role3_full_evaluation.json" \
  --triplet-report "$RESIDUAL_BANK/report.json" \
  --train-pool "$RESIDUAL_BANK/train_pool.npz" \
  --val-pool "$RESIDUAL_BANK/val_pool.npz" \
  --training-report "$OUT/training/report.json" \
  --output "$OUT/protected_artifact"

test -f "$OUT/protected_artifact/artifact_manifest.json"
test -f "$OUT/protected_artifact/SHA256SUMS"
echo "COMPLETE: ChemAware Phase-A residual-consensus embedding -> $OUT"
