#!/bin/bash
#SBATCH --job-name=prep_gnps_e180
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=36:00:00
#SBATCH --output=data/validation/prepare_gnps_enveda_%j.out
#SBATCH --error=data/validation/prepare_gnps_enveda_%j.err

# Prepare, but do not score, the two labelled benchmark resources:
#   1) consumed GNPS clean benchmark -> development registry
#   2) previously unused Enveda-180 -> frozen cross-condition external panels
set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
fi

GNPS="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
GNPS_REBUILT="data/validation/GLM_gnps_identity_panel_reconstruction"
ENVDATA="data/external/enveda180_20260713"
ENAUDIT="data/validation/enveda180_scoreblind_manifest_v2_20261010"
ENBENCH="data/validation/enveda180_cross_condition_benchmark_v2_20261010"
E180_MGF="$ENVDATA/enveda-180-filtered.mgf.gz"
E180_EXPECTED_BYTES=285632591
E180_EXPECTED_MD5=6d06a2c916aefe1c0b2d86bf40628441

required=(
  "$GNPS/report.json"
  "$GNPS/spectra.mgf"
  "$GNPS/manifest.csv.gz"
  "$GNPS/panel_formula_disjoint.npz"
  "$GNPS/pairs_identity_disjoint.npz"
  "$GNPS/pairs_formula_disjoint.npz"
  "$GNPS_REBUILT/panel_identity_disjoint.npz"
  "$GNPS_REBUILT/report.json"
  "$E180_MGF"
  tasks/prepare_gnps_unified_development_asset.py
  tasks/prepare_enveda180_scoreblind_manifest.py
  tasks/build_enveda180_cross_condition_benchmark.py
  tasks/unified_consumed_sources_v1.json
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/models/mona_pos_full.mgf
  data/models/mona_neg_full.mgf
  data/validation/bioaware_b44_massbank_panel_localcheck_20260913_v4/reference_library.csv.gz
  data/massbank/massbank_full.mgf
  data/validation/chemaware_msnlib_external_manifest_v2/eligible_records.csv.gz
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "BENCHMARK_PREP_MISSING: $path" >&2; exit 5; }
done

actual_bytes=$(stat -c%s "$E180_MGF")
actual_md5=$(md5sum "$E180_MGF" | awk '{print $1}')
[[ "$actual_bytes" == "$E180_EXPECTED_BYTES" ]] || {
  echo "ENVEDA180_SIZE_MISMATCH: expected=$E180_EXPECTED_BYTES actual=$actual_bytes path=$E180_MGF" >&2
  exit 6
}
[[ "$actual_md5" == "$E180_EXPECTED_MD5" ]] || {
  echo "ENVEDA180_MD5_MISMATCH: expected=$E180_EXPECTED_MD5 actual=$actual_md5 path=$E180_MGF" >&2
  exit 7
}
echo "ENVEDA180_STAGED_AND_VERIFIED: bytes=$actual_bytes md5=$actual_md5"

[[ ! -e "$ENAUDIT" ]] || { echo "ENVEDA180_AUDIT_OUTPUT_EXISTS: $ENAUDIT" >&2; exit 8; }
[[ ! -e "$ENBENCH" ]] || { echo "ENVEDA180_BENCHMARK_OUTPUT_EXISTS: $ENBENCH" >&2; exit 9; }

python -u tasks/prepare_gnps_unified_development_asset.py \
  --benchmark "$GNPS" \
  --identity-reconstruction "$GNPS_REBUILT/panel_identity_disjoint.npz" \
  --identity-reconstruction-report "$GNPS_REBUILT/report.json" \
  --out data/validation/unified_benchmark_assets/gnps_development_v1.json

python -u tasks/prepare_enveda180_scoreblind_manifest.py \
  --mgf "$E180_MGF" \
  --out "$ENAUDIT" \
  --exclusion-registry tasks/unified_consumed_sources_v1.json

python -u tasks/build_enveda180_cross_condition_benchmark.py \
  --source-mgf "$E180_MGF" \
  --audit "$ENAUDIT" \
  --out "$ENBENCH" \
  --ppm 10 \
  --max-candidates 256 \
  --refs-per-identity 3 \
  --max-queries 25000 \
  --sampling-seed 20261008 \
  --minimum-queries 1000

echo "GNPS_DEVELOPMENT_READY: data/validation/unified_benchmark_assets/gnps_development_v1.json"
echo "ENVEDA180_EXTERNAL_FROZEN_UNSCORED: $ENBENCH"
