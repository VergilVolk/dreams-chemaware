#!/bin/bash
#SBATCH --job-name=lcnec_s2
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=data/validation/lcnec_s2_%j.out
#SBATCH --error=data/validation/lcnec_s2_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

STAGE1_RUN="data/validation/lcnec_global_remodeling_stage1/run_2352292"
RAW_ZIP="data/validation/lcnec_zenodo19005638_preflight/MTB22_P073_HSST3n_mzML_public.zip"
OVERVIEW="data/validation/lcnec_zenodo19005638_preflight/06_MTB22_P073_HSST3n_mzML_overview_v1.txt"
LIPN_ZIP="data/validation/lcnec_zenodo19005638_preflight/MTB22_P073_LIPn_mzML_public.zip"
LIPN_OVERVIEW="data/validation/lcnec_zenodo19005638_preflight/04_MTB22_P073_LIPn_mzML_overview_v1.txt"
GNPS_MANIFEST="data/validation/gnps_gold_silver_10ppm_benchmark_v1/manifest.csv.gz"
GNPS_MGF="data/validation/gnps_gold_silver_10ppm_benchmark_v1/spectra.mgf"
OUT="data/validation/lcnec_global_remodeling_stage2/run_${SLURM_JOB_ID}"

mkdir -p "$OUT"

# Required files check
required=("$STAGE1_RUN/analysis/primary_decomposition.npz" "$RAW_ZIP" "$OVERVIEW" "$GNPS_MANIFEST")
for p in "${required[@]}"; do
  [[ -s "$p" ]] || { echo "MISSING: $p" >&2; exit 5; }
done

# Quantify LIPn if available
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
  echo "=== LIPn not available; 2A will be skipped ==="
fi

# Run Stage-2 analysis
echo "=== Stage-2 analysis ==="
python -u tasks/analyze_lcnec_remodeling_stage2.py \
  --stage1-dir "$STAGE1_RUN/analysis" \
  ${LIPN_MATRIX:+--lipn-matrix "$LIPN_MATRIX"} \
  --gnps-manifest "$GNPS_MANIFEST" \
  --gnps-mgf "$GNPS_MGF" \
  --output-dir "$OUT/analysis" \
  --overwrite

echo "STAGE2_COMPLETE: $OUT"
