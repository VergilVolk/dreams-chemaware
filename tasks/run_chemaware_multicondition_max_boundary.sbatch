#!/bin/bash
#SBATCH --job-name=chem_mcmax
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/chem_mcmax_%j.out
#SBATCH --error=data/validation/chem_mcmax_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OFFICIAL="data/e1/official_embedding_slim.pt"
STAGE1="data/validation/chemaware_high_coverage_native/run_2340721_resume_2340524/training/best.ckpt"
BASE="data/validation/chemaware_high_coverage_native/run_2340524"
OUT="data/validation/chemaware_multicondition_max_boundary/run_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

# The +2.1266 pp checkpoint is a required development comparator.  Resolve the
# newest protected, checksum-verified artifact automatically; never ask the
# operator to type a checkpoint path and never fall back to an unprotected run.
mapfile -t phasea_candidates < <(
  find data/validation/chemaware_phasea_2pp_protected -type f \
    -path '*/protected_artifact/chemaware_phasea_max_boundary_step2000.ckpt' \
    -print 2>/dev/null | sort
)
[[ ${#phasea_candidates[@]} -gt 0 ]] || {
  echo "Missing protected +2.1266 pp artifact. Finish tasks/run_chemaware_phasea_2pp_protect.sbatch first." >&2
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
print(f"Verified protected Phase-A comparator: {p.parent}", flush=True)
PY

required_files=(
  "$OFFICIAL"
  "$STAGE1"
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
  tasks/build_chemaware_multicondition_max_boundary_triplets.py
  tasks/train_chemaware_weighted_native.py
  tasks/evaluate_chemaware_v2_direct_triplet.py
  tasks/evaluate_chemaware_full_role_native.py
  tasks/select_chemaware_residual_checkpoint.py
  tasks/freeze_chemaware_multicondition_artifact.py
  tasks/test_chemaware_multicondition_max_boundary.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required multi-condition ChemAware file: $required_file" >&2
    exit 2
  }
done

python -u tasks/test_chemaware_multicondition_max_boundary.py

python -u tasks/build_chemaware_multicondition_max_boundary_triplets.py \
  --base-bank "$BASE/triplets" \
  --evidence "$BASE/evidence/train_triplet_evidence.npz" \
  --manifest data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz \
  --embedding-rows data/validation/chemaware_corrected_manifest_tokens_v1/rows.npy \
  --official-embeddings data/validation/chemaware_corrected_manifest_tokens_v1/official_embeddings_f32.npy \
  --dreams-replay-pool data/e1/e1_train_triplet_pool_10ppm.npz \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --output "$OUT/triplets" \
  --margin 0.1 --near-boundary-margin 0.1 \
  --max-anchors-per-identity 3 --preselect-anchors-per-identity 6 \
  --collision-energy-bin 10 \
  --negative-references-per-error 2 \
  --chemical-candidates-per-error 2 \
  --transferred-chemical-candidates-per-error 1 \
  --replay-fraction 0.20 --error-sampling-fraction 0.25 --seed 3407

srun --export=ALL --preserve-env python -u tasks/train_chemaware_weighted_native.py \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --train-pool "$OUT/triplets/train_pool.npz" \
  --val-pool "$OUT/triplets/val_pool.npz" \
  --official-checkpoint "$OFFICIAL" \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --output "$OUT/training" \
  --seed 3407 --lr 5e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --num-workers 0 \
  --max-epochs 4 --max-steps 3000 --checkpoint-mode fixed_steps \
  --save-every-n-steps 500 --n-highest-peaks 100 --device cuda

checkpoint_args=(
  --checkpoint "official=$OFFICIAL"
  --checkpoint "phaseA_base=$PHASEA"
)
mapfile -t trained_checkpoints < <(
  find "$OUT/training" -maxdepth 1 -type f -name 'step-*.ckpt' | sort
)
[[ ${#trained_checkpoints[@]} -eq 6 ]] || {
  echo "Expected six multi-condition checkpoints, found ${#trained_checkpoints[@]}" >&2
  exit 5
}
for checkpoint in "${trained_checkpoints[@]}"; do
  checkpoint_args+=(--checkpoint "mcmax_$(basename "$checkpoint" .ckpt)=$checkpoint")
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

# Never delete a scientific checkpoint automatically.  A non-advancing run is
# retained as complete negative evidence for later diagnosis and cleanup.
if [[ "$SELECTED_STEP" == "0" ]]; then
  echo "CHEMAWARE_MULTICONDITION_ROLE2_STOP: no checkpoint safely beat protected +2.1266 pp Phase A"
  echo "PRESERVED: all checkpoints and audits remain under $OUT"
  exit 0
fi
SELECTED_REAL="$(realpath "$SELECTED")"

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/confirmation_triplet_evidence.npz" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "stage1_base=$STAGE1" \
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
  --triplet-report "$OUT/triplets/report.json" \
  --train-pool "$OUT/triplets/train_pool.npz" \
  --val-pool "$OUT/triplets/val_pool.npz" \
  --training-report "$OUT/training/report.json" \
  --output "$OUT/protected_artifact"

test -f "$OUT/triplets/report.json"
test -f "$OUT/checkpoint_selection.json"
test -f "$OUT/role3_evaluation.json"
test -f "$OUT/role2_role3_full_evaluation.json"
test -f "$OUT/protected_artifact/artifact_manifest.json"
test -f "$OUT/protected_artifact/SHA256SUMS"
echo "COMPLETE: ChemAware multi-condition max-boundary embedding -> $OUT"
