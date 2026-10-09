#!/bin/bash
#SBATCH --job-name=unified_all
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/grand_unified_all_%j.out
#SBATCH --error=data/validation/grand_unified_all_%j.err

# One explicit job for the evidence-qualified joint model. Bundle generation is
# kept outside this job so candidate keys, applicability and hashes are frozen.
set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

: "${TRAIN_BUNDLE:?export TRAIN_BUNDLE=<aligned all-module training bundle>}"
RUN_ROOT="${RUN_ROOT:-data/validation/grand_unified_all_modules/run_${SLURM_JOB_ID}}"

required=(
  "$TRAIN_BUNDLE"
  tasks/grand_unified_evidence_core.py
  tasks/train_grand_unified_evidence_model.py
  tasks/grand_unified_components_v2.json
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "GRAND_UNIFIED_MISSING: $path" >&2; exit 5; }
done

python - "$TRAIN_BUNDLE" <<'PY'
import json, os, sys, numpy as np
with open("tasks/grand_unified_components_v2.json") as handle:
    registry = json.load(handle)
status = {item["name"]: item["status"] for item in registry["modules"]}
required = {"official_dreams", "noise_v1", "chemaware_stage1_encoder",
            "weighted_spectral_entropy", "p2b_noise_v1_frozen"}
allowed = {"enabled"}
if os.environ.get("ALLOW_PROVISIONAL") == "1":
    allowed.add("provisional")
for position, path in enumerate(path for path in sys.argv[1:] if path):
    with np.load(path, allow_pickle=False) as z:
        ordered_names = list(map(str, z["module_names"]))
        names = set(ordered_names)
        availability = np.asarray(z["availability"])
        scores = np.asarray(z["scores"])
    missing = required - names
    if missing:
        raise SystemExit(f"GRAND_UNIFIED_MODULES_MISSING {path}: {sorted(missing)}")
    forbidden = [name for name in ordered_names if status.get(name, "unregistered") not in allowed]
    if forbidden:
        raise SystemExit(f"GRAND_UNIFIED_UNQUALIFIED_MODULES {path}: {forbidden}")
    p2b_raw = {"p2b_sqrt_cosine", "p2b_unweighted_entropy", "neutral_loss_sqrt_cosine"}
    if "p2b_noise_v1_frozen" in names and names & p2b_raw:
        raise SystemExit(
            f"GRAND_UNIFIED_P2B_DOUBLE_COUNT {path}: frozen P2b composite and raw P2b channels cannot co-occur"
        )
    if availability.shape != scores.shape:
        raise SystemExit(f"GRAND_UNIFIED_CANDIDATE_AVAILABILITY_SHAPE {path}: {availability.shape} != {scores.shape}")
    if np.any(availability.sum(axis=0) == 0):
        raise SystemExit(f"GRAND_UNIFIED_CANDIDATE_WITHOUT_EVIDENCE {path}")
    never_observed = [name for i, name in enumerate(ordered_names)
                      if availability[i].sum() == 0]
    if position == 0 and never_observed:
        raise SystemExit(
            f"GRAND_UNIFIED_TRAIN_MODULE_NEVER_OBSERVED {path}: {never_observed}"
        )
    print(path, "qualified_modules", len(names), "availability_shape", availability.shape,
          "contextually_unobserved", never_observed)
PY

mkdir -p "$RUN_ROOT"
train_args=(
  --train-bundle "$TRAIN_BUNDLE"
  --out "$RUN_ROOT"
  --device cuda
)
python tasks/train_grand_unified_evidence_model.py "${train_args[@]}"

echo "GRAND_UNIFIED_COMPLETE: $RUN_ROOT"
