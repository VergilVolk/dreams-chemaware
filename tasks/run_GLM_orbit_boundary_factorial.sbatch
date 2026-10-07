#!/usr/bin/env bash
#SBATCH --job-name=glm_orb_bnd
#SBATCH --gpus=1
#SBATCH --output=data/validation/GLM_orbit_boundary_factorial_%j.out
#SBATCH --error=data/validation/GLM_orbit_boundary_factorial_%j.err

# ============================================================================
# DO NOT SUBMIT (draft, 2026-10-07 post-audit).
# Blocked until, in order:
#   1. the real DreaMS loading path passes a 2-group GPU forward/backward
#      smoke (GLM_train_orbit_boundary_encoder.py --encoder dreams);
#   2. role-2 curve extraction + role-3 confirmation + GNPS chain wired;
#   3. the pool is built from the REAL server ledgers and its sha is
#      preregistered below.
# This header is removed only when all three hold.
# ============================================================================

# GLM: six-arm orbit-boundary factorial (audit section 11, file 5).
# STRICT CONTRACT:
#   - all six arms: same pool file, same steps/batch/lr/seed (same dose);
#   - arms differ ONLY through --arm;
#   - pool sha is PREREGISTERED in GLM_EXPECTED_POOL_SHA and every arm
#     ABORTS on mismatch (fail-closed);
#   - one shared checkpoint step chosen afterwards by
#     GLM_select_shared_factorial_step.py on the R arm ONLY.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

: "${GLM_EXPECTED_POOL_SHA:?preregistered pool sha required}"
POOL="${GLM_POOL:-data/validation/GLM_orbit_boundary_pools/pool.npz}"
OFFICIAL="${GLM_OFFICIAL_CKPT:?official checkpoint required}"
ARCHITECTURE="${GLM_ARCH_CKPT:?architecture checkpoint required}"
NOISE_V1="${GLM_NOISE_V1_CKPT:?noise V1 checkpoint required}"
OUT="data/validation/GLM_orbit_boundary_factorial/run_${SLURM_JOB_ID}"
STEPS="${GLM_STEPS:-2000}"
BATCH="${GLM_BATCH:-4}"
LR="${GLM_LR:-5e-6}"
SEED="3407"
mkdir -p "$OUT"

for ARM in "R" "O-null" "O-real" "C-null" "C-real" "OC"; do
  echo "=== training arm $ARM (steps=$STEPS batch=$BATCH lr=$LR seed=$SEED) ==="
  python -X utf8 tasks/GLM_train_orbit_boundary_encoder.py \
    --arm "$ARM" \
    --pool "$POOL" \
    --expected-pool-sha256 "$GLM_EXPECTED_POOL_SHA" \
    --encoder dreams \
    --official-checkpoint "$OFFICIAL" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --noise-v1-checkpoint "$NOISE_V1" \
    --steps "$STEPS" --batch-size "$BATCH" --lr "$LR" --seed "$SEED" \
    --output-dir "$OUT/$ARM"
  python - "$OUT/$ARM/${ARM}_report.json" "$STEPS" "$BATCH" "$LR" "$SEED" "$GLM_EXPECTED_POOL_SHA" <<'PYEOF'
import json, sys
rep = json.load(open(sys.argv[1], encoding="utf-8"))
steps, batch, lr, seed, sha = sys.argv[2:7]
assert rep["dose"]["steps"] == int(steps), "dose drift: steps"
assert rep["dose"]["batch_size"] == int(batch), "dose drift: batch"
assert rep["dose"]["lr"] == float(lr), "dose drift: lr"
assert rep["dose"]["seed"] == int(seed), "dose drift: seed"
assert rep["pool_sha256"] == sha, "dose drift: pool sha"
print(f"dose verified for {rep['arm']} (sha pinned)")
PYEOF
done

echo "=== all six arms complete; NEXT (not automatic yet): ==="
echo "  GLM_select_shared_factorial_step.py --r-arm-metrics <R role-2 curve> --output $OUT/shared_step.json"
echo "COMPLETE (draft run): six-arm factorial $OUT"
