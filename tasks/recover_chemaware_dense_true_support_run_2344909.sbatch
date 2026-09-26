#!/bin/bash
#SBATCH --job-name=chem_dense_eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_dense_eval_%j.out
#SBATCH --error=data/validation/chem_dense_eval_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RUN="data/validation/chemaware_dense_true_support_native/run_2344909"
BASE="data/validation/chemaware_high_coverage_native/run_2340524"
OFFICIAL="data/e1/official_embedding_slim.pt"

[[ -d "$RUN" ]] || { echo "Missing interrupted dense run: $RUN" >&2; exit 2; }
[[ -f "$RUN/triplets/report.json" ]] || { echo "Missing dense triplet report" >&2; exit 2; }

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

python - "$RUN/triplets/report.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
body = json.loads(p.read_text(encoding="utf-8"))
assert body["status"] == "CHEMAWARE_DENSE_TRUE_SUPPORT_NATIVE_TRIPLETS_COMPLETE"
assert body["cache_kind"] == "protected_phasea"
assert body["events"]["unique_correction_triplets"] == 2324
assert body["coverage"]["correction_queries"] == 553
assert body["coverage"]["distinct_identity_false_boundaries"] == 436
assert body["sampler"]["custom_sampling_weight_present"] is False
assert all(body["gates"].values())
print("Verified interrupted run triplet ledger", flush=True)
PY

# The step-1500 save raised EDQUOT during the cross-filesystem copy.  Any
# destination left by that failed transaction is incomplete and is not an
# evaluable checkpoint.  Remove only that known-incomplete final checkpoint;
# the five earlier checkpoints and every triplet/audit artifact are retained.
FAILED_FINAL="$RUN/training/step-001500.ckpt"
if [[ -e "$FAILED_FINAL" ]]; then
  echo "Removing known-incomplete final checkpoint: $FAILED_FINAL"
  rm -f -- "$FAILED_FINAL"
fi

steps=(000250 000500 000750 001000 001250)
checkpoint_args=(
  --checkpoint "official=$OFFICIAL"
  --checkpoint "phaseA_base=$PHASEA"
)
for step in "${steps[@]}"; do
  checkpoint="$RUN/training/step-${step}.ckpt"
  [[ -s "$checkpoint" ]] || {
    echo "Missing completed recovery checkpoint: $checkpoint" >&2
    exit 2
  }
  checkpoint_args+=(--checkpoint "true_support_step-${step}=$checkpoint")
done

ROLE2="$RUN/role2_checkpoint_evaluation_recovered.json"
SELECTION="$RUN/checkpoint_selection_recovered.json"
[[ ! -e "$ROLE2" && ! -e "$SELECTION" ]] || {
  echo "Recovery evaluation output already exists; refusing to overwrite" >&2
  exit 3
}

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/selection_triplet_evidence.npz" \
  "${checkpoint_args[@]}" \
  --paired-reference phaseA_base --formula-role 2 \
  --output "$ROLE2" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

python -u tasks/select_chemaware_residual_checkpoint.py \
  --evaluation "$ROLE2" \
  --base-name phaseA_base --require-positive-formula-ci \
  --output "$SELECTION"
SELECTED="$(tr -d '\r\n' < "$RUN/checkpoint_selection_recovered.checkpoint.txt")"
SELECTED_STEP="$(tr -d '\r\n' < "$RUN/checkpoint_selection_recovered.step.txt")"

if [[ "$SELECTED_STEP" == "0" ]]; then
  echo "CHEMAWARE_DENSE_TRUE_SUPPORT_RECOVERY_ROLE2_STOP: no saved checkpoint safely beat Phase A"
  echo "PRESERVED: five completed checkpoints and all audits remain under $RUN"
  exit 0
fi
SELECTED_REAL="$(realpath "$SELECTED")"

ROLE3="$RUN/role3_evaluation_recovered.json"
FULL="$RUN/role2_role3_full_evaluation_recovered.json"
srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --policy "$BASE/evidence/confirmation_triplet_evidence.npz" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "phaseA_base=$PHASEA" \
  --checkpoint "true_support=$SELECTED_REAL" \
  --paired-reference phaseA_base --formula-role 3 \
  --output "$ROLE3" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_full_role_native.py \
  --formula-role 2 3 \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "phaseA_base=$PHASEA" \
  --checkpoint "true_support=$SELECTED_REAL" \
  --output "$FULL" \
  --batch-size 64 --bootstrap-draws 5000 --device cuda

RECOVERY_DIR="$RUN/recovery"
mkdir -p "$RECOVERY_DIR"
python - "$RECOVERY_DIR/training_report.json" "$SELECTED_REAL" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
body = {
    "status": "CHEMAWARE_DREAMS_NATIVE_TRAINING_RECOVERED_AFTER_FINAL_CHECKPOINT_EDQUOT",
    "formal_training_completed_optimizer_steps": 1500,
    "completed_checkpoints_evaluated": [250, 500, 750, 1000, 1250],
    "failed_checkpoint_excluded": 1500,
    "selected_checkpoint": sys.argv[2],
    "weights_updated": True,
    "triplet_pool_changed": False,
}
p.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
PY

python -u tasks/freeze_chemaware_true_support_artifact.py \
  --checkpoint "$SELECTED_REAL" \
  --selection "$SELECTION" \
  --role2-evaluation "$ROLE2" \
  --role3-evaluation "$ROLE3" \
  --full-evaluation "$FULL" \
  --triplet-report "$RUN/triplets/report.json" \
  --train-pool "$RUN/triplets/train_pool.npz" \
  --val-pool "$RUN/triplets/val_pool.npz" \
  --training-report "$RECOVERY_DIR/training_report.json" \
  --output "$RUN/protected_artifact_recovered"

test -f "$RUN/protected_artifact_recovered/artifact_manifest.json"
test -f "$RUN/protected_artifact_recovered/SHA256SUMS"
echo "COMPLETE: recovered dense ChemAware evaluation -> $RUN"
