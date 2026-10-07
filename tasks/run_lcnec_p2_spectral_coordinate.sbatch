#!/bin/bash
#SBATCH --job-name=lcnec_p2
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=data/validation/lcnec_p2_%j.out
#SBATCH --error=data/validation/lcnec_p2_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Run 2353437 is an invalid single-platform pilot, not a formal P2 result.
# Preserve the old body only for exact forensic reproduction.  A formal P2
# must wait for P0-G no-match calibration and genuine LIPn MS/MS/effect inputs.
if [[ "${ALLOW_INVALID_LCNEC_P2_PILOT:-0}" != "1" ]]; then
  echo "LCNEC_P2_BLOCKED: invalid legacy pilot; do not resubmit" >&2
  echo "See docs/LCNEC_P2_SPECTRAL_COORDINATE_RUN_2353437_AUDIT.md" >&2
  exit 64
fi

STAGE1_RUN="data/validation/lcnec_global_remodeling_stage1/run_2352292"
LIPN_ZIP="data/validation/lcnec_zenodo19005638_preflight/MTB22_P073_LIPn_mzML_public.zip"
LIPN_OVERVIEW="data/validation/lcnec_zenodo19005638_preflight/04_MTB22_P073_LIPn_mzML_overview_v1.txt"
OUT="data/validation/lcnec_p2_spectral_coordinate/run_${SLURM_JOB_ID}"

mkdir -p "$OUT"

# Check inputs
for p in "$STAGE1_RUN/analysis/primary_decomposition.npz" \
         "$STAGE1_RUN/query/priority_dark_modules.mgf" \
         "data/validation/gnps_gold_silver_10ppm_benchmark_v1/manifest.csv.gz" \
         "data/validation/gnps_gold_silver_10ppm_benchmark_v1/spectra.mgf"; do
  [[ -s "$p" ]] || { echo "P2_MISSING: $p" >&2; exit 5; }
done

# Quantify LIPn if zip available
LIPN_MATRIX=""
if [[ -s "$LIPN_ZIP" && -s "$LIPN_OVERVIEW" ]]; then
  echo "=== Quantifying LIPn ==="
  python -u tasks/quantify_lcnec_hsst3n_all_qc_eic.py \
    --zip "$LIPN_ZIP" \
    --overview "$LIPN_OVERVIEW" \
    --targets "$STAGE1_RUN/qc_headroom/precursor_family_ledger.csv" \
    --output-dir "$OUT/lipn_eic" \
    --overwrite
  LIPN_MATRIX="$OUT/lipn_eic/all_qc_family_eic_matrix.npz"
else
  echo "=== LIPn not available; single-platform only ==="
fi

# Build LIPn patient effects if available
LIPN_EFFECTS=""
if [[ -n "$LIPN_MATRIX" && -s "$LIPN_MATRIX" ]]; then
  echo "=== Building LIPn effects ==="
  python -u - <<'PYEOF'
import numpy as np
from collections import defaultdict

lipn = np.load("{{}}".replace("{{}}", "PLACEHOLDER"))
# This will be handled by the projection script
PYEOF
  # For now, the projection script handles cross-platform internally
  LIPN_EFFECTS="$LIPN_MATRIX"
fi

# Run P2 projection (single platform if no LIPn, cross-platform if available)
echo "=== P2 spectral coordinate projection ==="
python -u tasks/GLM_p2_spectral_coordinate_projection.py \
  --query-mgf "$STAGE1_RUN/query/priority_dark_modules.mgf" \
  --query-effects "$STAGE1_RUN/analysis/primary_decomposition.npz" \
  ${LIPN_MATRIX:+--query-mgf-2 "$STAGE1_RUN/query/priority_dark_modules.mgf" --query-effects-2 "$LIPN_MATRIX"} \
  --output-dir "$OUT" \
  --method both \
  --max-coordinates 2000 \
  --score-threshold 0.3 \
  --allow-invalid-legacy-pilot \
  --overwrite

echo "P2_COMPLETE: $OUT"
