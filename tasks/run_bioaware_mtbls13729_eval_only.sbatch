#!/bin/bash
#SBATCH --job-name=bioaware_13729_eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/bioaware_mtbls13729_eval_%j.out
#SBATCH --error=data/validation/bioaware_mtbls13729_eval_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

if [[ -f "data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz" ]]; then
  RHEA="data/reference/bioaware_rhea_offline_20260827"
else
  RHEA="data/reference/bioaware_rhea"
fi
INPUT="data/mtbls13729/bioaware_v1_input"
OUTPUT="data/mtbls13729/bioaware_v1_eval"

for CODE in annotation/bioaware.py tasks/evaluate_bioaware_expert.py tasks/validate_bioaware_result.py tasks/validate_bioaware_rhea_cache.py; do
  [[ -s "$CODE" ]] || { echo "Missing required code: $CODE" >&2; exit 2; }
done
python -c "from annotation.bioaware import BioAwareConfig" || {
  echo "BioAware core import failed" >&2; exit 2;
}
for ARTIFACT in candidates.csv.gz seeds.csv.gz report.json; do
  [[ -s "$INPUT/$ARTIFACT" ]] || { echo "Missing prepared input: $INPUT/$ARTIFACT" >&2; exit 3; }
done
python -c "import json; from pathlib import Path; r=json.loads(Path('$INPUT/report.json').read_text()); assert r['status']=='mtbls13729_bioaware_v1_input_complete'; print('[input] PASS queries=',r['queries'],'seeds=',r['seed_rows'])"
[[ -s "$RHEA/rhea_participants.csv.gz" ]] || { echo "Missing Rhea cache: $RHEA" >&2; exit 4; }
if [[ -d "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite non-empty evaluation: $OUTPUT" >&2
  exit 5
fi
if [[ -e "$OUTPUT" && ! -d "$OUTPUT" ]]; then
  echo "Evaluation output path exists but is not a directory: $OUTPUT" >&2
  exit 5
fi

echo "=== BioAware MTBLS13729 evaluation-only started: $(date) ==="
python -u tasks/validate_bioaware_rhea_cache.py "$RHEA"
python -u tasks/evaluate_bioaware_expert.py \
  --candidates "$INPUT/candidates.csv.gz" \
  --participants "$RHEA/rhea_participants.csv.gz" \
  --seeds "$INPUT/seeds.csv.gz" \
  --output-dir "$OUTPUT" \
  --decoy-repeats 10 \
  --bootstrap-resamples 5000 \
  --seed 20260827
python -u tasks/validate_bioaware_result.py "$OUTPUT"
echo "=== BioAware MTBLS13729 evaluation-only completed: $(date) ==="
