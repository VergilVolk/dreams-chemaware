#!/bin/bash
# Build the MultiMS2 external cross-condition confirmation panels without
# model scoring.  This is not represented as the sole large final benchmark.
# Primary: CID 60 eV -> CID 40 eV.  EAD -> CID is a separate OOD stress test.
#SBATCH --job-name=multims2_ext
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=data/validation/multims2_external_%j.out
#SBATCH --error=data/validation/multims2_external_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

COMMIT="659bd9b608408a75ecbb4b03295ed4c03ba52539"
SOURCE_DIR="data/external/MultiMS2_${COMMIT}"
SOURCE_MGF="$SOURCE_DIR/data/multims2_spectra.mgf"
SOURCE_SHA256="4ef86f75a12ad2fdadbd63732e2d6690f4e506a6ff08f171758c1808d1249673"
REGISTRY="tasks/unified_consumed_sources_v1.json"
OUT="data/validation/multims2_external_scoreblind_run_${SLURM_JOB_ID}"

required=(
  tasks/build_multims2_external_benchmark.py
  tasks/test_build_multims2_external_benchmark.py
  tasks/prepare_enveda180_scoreblind_manifest.py
  "$REGISTRY"
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/models/mona_pos_full.mgf
  data/models/mona_neg_full.mgf
  data/validation/gnps_gold_silver_10ppm_benchmark_v1/manifest.csv.gz
  data/validation/bioaware_b44_massbank_panel_localcheck_20260913_v4/reference_library.csv.gz
  data/validation/chemaware_msnlib_external_manifest_v2/eligible_records.csv.gz
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "MULTIMS2_EXTERNAL_MISSING: $path" >&2; exit 4; }
done
[[ ! -e "$OUT" ]] || { echo "MULTIMS2_EXTERNAL_OUTPUT_EXISTS: $OUT" >&2; exit 3; }

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/zamboni-lab/MultiMS2.git "$SOURCE_DIR"
  git -C "$SOURCE_DIR" sparse-checkout init --no-cone
  git -C "$SOURCE_DIR" sparse-checkout set data/multims2_spectra.mgf
fi
git -C "$SOURCE_DIR" fetch --depth=1 origin "$COMMIT"
git -C "$SOURCE_DIR" checkout --detach "$COMMIT"
actual_commit=$(git -C "$SOURCE_DIR" rev-parse HEAD)
[[ "$actual_commit" == "$COMMIT" ]] || { echo "MULTIMS2_COMMIT_MISMATCH: $actual_commit" >&2; exit 5; }
[[ -s "$SOURCE_MGF" ]] || { echo "MULTIMS2_SOURCE_MISSING: $SOURCE_MGF" >&2; exit 6; }
actual_sha256=$(sha256sum "$SOURCE_MGF" | awk '{print $1}')
[[ "$actual_sha256" == "$SOURCE_SHA256" ]] || {
  echo "MULTIMS2_SOURCE_HASH_MISMATCH: expected=$SOURCE_SHA256 actual=$actual_sha256" >&2
  exit 7
}

python -m py_compile tasks/build_multims2_external_benchmark.py
python -u tasks/test_build_multims2_external_benchmark.py
python -u tasks/build_multims2_external_benchmark.py \
  --source-mgf "$SOURCE_MGF" \
  --exclusion-registry "$REGISTRY" \
  --output "$OUT" \
  --expected-source-sha256 "$SOURCE_SHA256" \
  --min-peaks 10 \
  --precursor-ppm 30 \
  --candidate-ppm 10

echo "MULTIMS2_CROSS_CONDITION_CONFIRMATION_READY_NO_MODEL_SCORES: $OUT"
