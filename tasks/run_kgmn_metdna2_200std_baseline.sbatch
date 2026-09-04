#!/bin/bash
#SBATCH --job-name=kgmn_200std
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=kgmn_200std_%j.out
#SBATCH --error=kgmn_200std_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

AUDIT=data/validation/metabolic_network_framework_reproducibility.json
OUT=data/validation/kgmn_metdna2_200std_author_baseline

python -u tasks/audit_metabolic_network_framework_reproducibility.py --output "$AUDIT"
python - "$AUDIT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if not report["kgmn_metdna2"]["reproducible_baseline_ready"]:
    raise SystemExit("KGMN/MetDNA2 public code or network assets are incomplete")
PY

command -v Rscript >/dev/null 2>&1 || { echo "Rscript is unavailable" >&2; exit 2; }
Rscript tasks/preflight_kgmn_metdna2_runtime.R third_party/MetDNA2

SOURCE_COMMIT=$(git -C third_party/MetDNA2 rev-parse HEAD)
RLIB="data/runtime/kgmn_metdna2_r_lib_${SOURCE_COMMIT}"
mkdir -p "$RLIB"
if [ ! -f "$RLIB/MetDNA2/DESCRIPTION" ]; then
  R CMD INSTALL --no-multiarch --with-keep.source --library="$RLIB" third_party/MetDNA2
fi
export R_LIBS_USER="$PWD/$RLIB${R_LIBS_USER:+:$R_LIBS_USER}"
Rscript tasks/preflight_kgmn_metdna2_dp.R

if [ -e "$OUT" ]; then
  echo "Refusing to overwrite existing baseline directory: $OUT" >&2
  exit 1
fi

TEMPORARY_LOG="$(mktemp)"
cleanup_log () {
  rm -f "$TEMPORARY_LOG"
}
trap cleanup_log EXIT
Rscript tasks/run_kgmn_metdna2_200std_baseline.R third_party/MetDNA2 "$OUT" 2>&1 | tee "$TEMPORARY_LOG"
if [ ! -d "$OUT" ]; then
  echo "MetDNA2 baseline returned without creating its formal output directory" >&2
  exit 1
fi
mv "$TEMPORARY_LOG" "$OUT/run.log.txt"
trap - EXIT
python -u tasks/freeze_kgmn_metdna2_baseline.py --run-dir "$OUT"
