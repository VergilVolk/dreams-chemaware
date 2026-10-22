#!/bin/bash
#SBATCH --job-name=mtbls1066_kd
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/mtbls1066_kd_%j.out
#SBATCH --error=data/validation/mtbls1066_kd_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

QUAL="data/validation/contrast_atlas_mtbls1066_kd_qualification_v1_20261022"
RAW="data/mtbls1066/pos_rp_primary_mzml"
RESULT="data/validation/contrast_atlas_mtbls1066_kd_response_v1_20261022"

test -s "$QUAL/report.json"
test -s "$QUAL/primary_pos_rp_manifest.tsv"
python -c "import pyopenms, pandas, numpy"

if [[ ! -s "$RAW/report.json" ]]; then
  python -u tasks/stage_mtbls1066_primary_pos_rp.py \
    --manifest "$QUAL/primary_pos_rp_manifest.tsv" \
    --output-dir "$RAW" \
    --workers 3
fi

if [[ -e "$RESULT" ]]; then
  echo "Refusing to overwrite existing result: $RESULT" >&2
  exit 4
fi

python -u tasks/build_mtbls1066_ketogenic_response.py \
  --staged-manifest "$RAW/staged_manifest.tsv" \
  --work-dir data/mtbls1066/pos_rp_primary_work_v1 \
  --output-dir "$RESULT"

python - "$RESULT/report.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text())
assert report["status"] == "development_response_objects_complete"
assert report["denominators"]["biological_units"] == 15
assert report["denominators"]["entities_with_representative_ms2"] > 0
for contrast in report["denominators"]["responses"].values():
    assert contrast["eligible_ms2_linked"] > 0
print(json.dumps(report["denominators"], indent=2, sort_keys=True))
PY
