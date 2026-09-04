#!/bin/bash
#SBATCH --job-name=kgmn_edge_cal
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=kgmn_edge_cal_%j.out
#SBATCH --error=kgmn_edge_cal_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

command -v Rscript >/dev/null 2>&1 || { echo "Rscript is unavailable" >&2; exit 2; }
command -v sha256sum >/dev/null 2>&1 || { echo "sha256sum is unavailable" >&2; exit 2; }

SOURCE_COMMIT=$(git -C third_party/MetDNA2 rev-parse HEAD)
RLIB="data/runtime/kgmn_metdna2_r_lib_${SOURCE_COMMIT}"
if [[ ! -s "$RLIB/MetDNA2/DESCRIPTION" ]]; then
  echo "Missing isolated author MetDNA2 runtime: $RLIB" >&2
  echo "Run the frozen author baseline first." >&2
  exit 1
fi
export R_LIBS_USER="$PWD/$RLIB${R_LIBS_USER:+:$R_LIBS_USER}"
Rscript -e 'stopifnot(as.character(packageVersion("MetDNA2")) == "1.2.10")'
Rscript tasks/preflight_kgmn_metdna2_dp.R

MANIFEST=data/validation/kgmn_dreams_edge_calibration_manifest_20260831
BASELINE=data/validation/kgmn_metdna2_200std_author_baseline/frozen_author_baseline.json
AUTHOR_INPUT=data/validation/kgmn_edge_calibration_exact_author_input_20260831
AUTHOR_SCORES=data/validation/kgmn_edge_calibration_exact_author_dp_20260831
OUT=data/validation/kgmn_dreams_edge_calibration_official_20260831

for output in "$AUTHOR_INPUT" "$AUTHOR_SCORES" "$OUT"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite formal output: $output" >&2
    exit 1
  fi
done

python -u tasks/export_kgmn_edge_calibration_author_input.py \
  --manifest-dir "$MANIFEST" \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --output-dir "$AUTHOR_INPUT"

Rscript tasks/score_kgmn_edge_calibration_author_dp.R \
  "$AUTHOR_INPUT" "$AUTHOR_SCORES"

python -u tasks/score_kgmn_dreams_edge_calibration.py \
  --manifest-dir "$MANIFEST" \
  --author-baseline "$BASELINE" \
  --author-scores "$AUTHOR_SCORES/author_dp_scores.csv.gz" \
  --output-dir "$OUT" \
  --device cuda \
  --batch-size 128 \
  --bootstrap-resamples 5000
