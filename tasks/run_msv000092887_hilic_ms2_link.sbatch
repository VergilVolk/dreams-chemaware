#!/bin/bash
#SBATCH --job-name=covid_hilic_ms2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/msv000092887_hilic_ms2_%j.out
#SBATCH --error=data/validation/msv000092887_hilic_ms2_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

MANIFEST="data/validation/msv000092887_hilic_positive_mzxml_manifest_v2_20261019/mzxml_manifest.tsv"
RESPONSE="data/validation/contrast_atlas_msv000092887_paired_response_v1_20261019/paired_response_object.tsv.gz"
RAW="data/msv000092887/hilic_positive_mzxml"
RESULT="data/validation/contrast_atlas_msv000092887_raw_ms2_link_v1_20261022"
RUN_RESULT="${RESULT}.run_${SLURM_JOB_ID}"

test -s "$MANIFEST"
test -s "$RESPONSE"
python -c "import pyopenms, pandas, numpy"

if [[ ! -s "$RAW/report.json" || ! -s "$RAW/staged_manifest.tsv" ]]; then
  python -u tasks/stage_msv000092887_hilic_mzxml.py \
    --manifest "$MANIFEST" \
    --output-dir "$RAW" \
    --workers 3
fi

if [[ -e "$RESULT" ]]; then
  echo "Refusing to overwrite existing result: $RESULT" >&2
  exit 4
fi
if [[ -e "$RUN_RESULT" ]]; then
  echo "Refusing to overwrite this job's run directory: $RUN_RESULT" >&2
  exit 5
fi

python -u tasks/link_msv000092887_hilic_ms2.py \
  --response-object "$RESPONSE" \
  --staged-manifest "$RAW/staged_manifest.tsv" \
  --output-dir "$RUN_RESULT"

python - "$RUN_RESULT/report.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text())
assert report["status"] == "PHENOTYPE_BLIND_RAW_MS2_LINK_COMPLETE"
assert report["evidence_role"] == "development_consumed"
den = report["denominators"]
assert den["features"] == 3041
assert den["mzxml_files"] == 129
assert den["features_with_usable_raw_ms2"] > 0
print(json.dumps(den, indent=2, sort_keys=True))
PY

mv "$RUN_RESULT" "$RESULT"
