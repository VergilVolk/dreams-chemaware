#!/usr/bin/env bash
#SBATCH --job-name=glm_orb_bnd
#SBATCH --gpus=1
#SBATCH --output=data/validation/GLM_orbit_boundary_factorial_%j.out
#SBATCH --error=data/validation/GLM_orbit_boundary_factorial_%j.err

# GLM: six-arm orbit-boundary factorial (audit section 11, file 5).
# STRICT CONTRACT:
#   - all six arms: same pool file, same steps/batch/lr/seed (same dose);
#   - arms differ ONLY through --arm (two boolean factors in the trainer);
#   - one shared checkpoint step chosen afterwards by
#     GLM_select_shared_factorial_step.py on the R arm ONLY, applied
#     mechanically to the other five;
#   - inputs hash-pinned before training starts; any hash change aborts.
# Prerequisites (server): candidate groups from the frozen Noise-V1
#   geometry, multi-condition ledger, ChemAware margin ledger -> pool built
#   by GLM_build_orbit_boundary_groups.py with its integrity gates green.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

POOL="${GLM_POOL:-data/validation/GLM_orbit_boundary_pools/pool.npz}"
OUT="data/validation/GLM_orbit_boundary_factorial/run_${SLURM_JOB_ID}"
STEPS="${GLM_STEPS:-2000}"
BATCH="${GLM_BATCH:-4}"
LR="${GLM_LR:-5e-6}"
SEED="3407"
mkdir -p "$OUT"

# hash pins (abort on drift)
sha256sum "$POOL" > "$OUT/pool.sha256"
POOL_SHA="$(cut -d' ' -f1 "$OUT/pool.sha256")"
echo "pool sha256: $POOL_SHA"

for ARM in "R" "O-null" "O-real" "C-null" "C-real" "OC"; do
  echo "=== training arm $ARM (steps=$STEPS batch=$BATCH lr=$LR seed=$SEED) ==="
  python -X utf8 tasks/GLM_train_orbit_boundary_encoder.py \
    --arm "$ARM" \
    --pool "$POOL" \
    --encoder dreams \
    --steps "$STEPS" --batch-size "$BATCH" --lr "$LR" --seed "$SEED" \
    --output-dir "$OUT/$ARM"
  # verify the arm's report records the SAME dose
  python - "$OUT/$ARM/${ARM}_report.json" "$STEPS" "$BATCH" "$LR" "$SEED" "$POOL_SHA" <<'PYEOF'
import json, sys, hashlib
rep = json.load(open(sys.argv[1], encoding="utf-8"))
steps, batch, lr, seed, pool_sha = sys.argv[2:7]
assert rep["dose"]["steps"] == int(steps), "dose drift: steps"
assert rep["dose"]["batch_size"] == int(batch), "dose drift: batch"
assert rep["dose"]["lr"] == float(lr), "dose drift: lr"
assert rep["dose"]["seed"] == int(seed), "dose drift: seed"
print(f"dose verified for {rep['arm']}")
PYEOF
done

echo "=== all six arms complete; shared-step selection runs on R only ==="
echo "next: GLM_select_shared_factorial_step.py --r-arm-metrics <R role-2 curve> --output $OUT/shared_step.json"
echo "then: frozen role-3 confirmation + GNPS external chain with the shared step applied to all arms"
echo "COMPLETE: six-arm factorial $OUT"
