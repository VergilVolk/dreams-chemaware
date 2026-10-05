#!/bin/bash
#SBATCH --job-name=noise_ckpt_cleanup
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=data/validation/noise_ckpt_cleanup_%j.out
#SBATCH --error=data/validation/noise_ckpt_cleanup_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
ROOT="/data02/run01/scv7tsl/DreaMS/data/validation"
RESOLVED_ROOT=$(realpath -e -- "$ROOT")
[[ "$RESOLVED_ROOT" == "$ROOT" ]] || {
  echo "Unexpected validation root: $RESOLVED_ROOT" >&2
  exit 2
}

# Exact paths and sizes from the 2026-09-28 read-only server inventory.
# These are superseded early native runs. Reports, evaluations, logs and all
# directories remain intact; only their eight large training checkpoints go.
ENTRIES=(
  "1241316890|noise_dreams_native_fold_0_run_2342162/arms/primary/last.ckpt"
  "1241316762|noise_dreams_native_fold_0_run_2342162/arms/primary/step-001899.ckpt"
  "1241316890|noise_dreams_native_fold_0_run_2342162/arms/replicate/last.ckpt"
  "1241316762|noise_dreams_native_fold_0_run_2342162/arms/replicate/step-001899.ckpt"
  "1241316890|noise_dreams_native_fold_0_run_2342223/arms/primary/last.ckpt"
  "1241316762|noise_dreams_native_fold_0_run_2342223/arms/primary/step-013412.ckpt"
  "1241316890|noise_dreams_native_fold_0_run_2342223/arms/replicate/last.ckpt"
  "1241316762|noise_dreams_native_fold_0_run_2342223/arms/replicate/step-013412.ckpt"
)

# Validate every target before deleting the first byte.
TOTAL=0
for entry in "${ENTRIES[@]}"; do
  expected=${entry%%|*}
  relative=${entry#*|}
  target="$ROOT/$relative"
  [[ -f "$target" ]] || { echo "Missing cleanup target: $target" >&2; exit 3; }
  resolved=$(realpath -e -- "$target")
  case "$resolved" in
    "$RESOLVED_ROOT"/*) ;;
    *) echo "Cleanup target escaped validation root: $resolved" >&2; exit 4 ;;
  esac
  actual=$(stat -c '%s' -- "$resolved")
  [[ "$actual" == "$expected" ]] || {
    echo "Cleanup target size drifted: $resolved expected=$expected actual=$actual" >&2
    exit 5
  }
  TOTAL=$((TOTAL + actual))
done

[[ "$TOTAL" == "9930534608" ]] || {
  echo "Registered cleanup total drifted: $TOTAL" >&2
  exit 6
}

RECEIPT="data/validation/noise_ckpt_cleanup_${SLURM_JOB_ID}.tsv"
printf 'bytes\tpath\tstatus\n' > "$RECEIPT"
for entry in "${ENTRIES[@]}"; do
  expected=${entry%%|*}
  relative=${entry#*|}
  target="$ROOT/$relative"
  rm -- "$target"
  [[ ! -e "$target" ]] || { echo "Deletion failed: $target" >&2; exit 7; }
  printf '%s\t%s\tdeleted\n' "$expected" "$target" >> "$RECEIPT"
done

echo "Deleted exactly $TOTAL bytes of superseded Noise checkpoints."
echo "Receipt: $RECEIPT"
df -h "$ROOT"
quota -s || true
