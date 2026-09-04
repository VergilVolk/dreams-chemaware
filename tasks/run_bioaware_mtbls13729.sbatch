#!/bin/bash
#SBATCH --job-name=bioaware_13729
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/bioaware_mtbls13729_%j.out
#SBATCH --error=data/validation/bioaware_mtbls13729_%j.err

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
LINK="data/mtbls13729/ms1_ms2_link"
INPUT="data/mtbls13729/bioaware_v1_input"
OUTPUT="data/mtbls13729/bioaware_v1_eval"

REQUIRED_CODE=(
  annotation/__init__.py
  annotation/bioaware.py
  tasks/prepare_mtbls13729_bioaware_pilot.py
  tasks/evaluate_bioaware_expert.py
  tasks/validate_bioaware_result.py
  tasks/link_mtbls13729_ms2_annotations_to_ms1.py
  tasks/validate_bioaware_rhea_cache.py
)
for CODE in "${REQUIRED_CODE[@]}"; do
  [[ -s "$CODE" ]] || { echo "Missing required BioAware code: $CODE" >&2; exit 2; }
done
python -c "from annotation.bioaware import BioAwareConfig" || {
  echo "BioAware core import failed before data processing" >&2
  exit 2
}

[[ -f "$RHEA/rhea_participants.csv.gz" ]] || { echo "Missing Rhea cache: $RHEA" >&2; exit 3; }
INPUT_READY=0
if [[ -s "$INPUT/candidates.csv.gz" && -s "$INPUT/seeds.csv.gz" && -s "$INPUT/report.json" ]]; then
  python -c "import json; from pathlib import Path; r=json.loads(Path('$INPUT/report.json').read_text()); assert r['status']=='mtbls13729_bioaware_v1_input_complete'"
  INPUT_READY=1
  echo "[BioAware] Reusing validated prepared input: $INPUT"
elif [[ -e "$INPUT" && -n "$(find "$INPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite incomplete non-empty input: $INPUT" >&2
  exit 4
fi
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite: $OUTPUT" >&2; exit 5; }

echo "=== BioAware MTBLS13729 started: $(date) ==="
python -u tasks/validate_bioaware_rhea_cache.py "$RHEA"

# The BioAware input is downstream of the existing MS2-to-MS1 bridge.  Build
# it here when absent, but never silently trigger the much larger annotation or
# MS1 extraction pipelines.
MISSING_PANELS=()
for PANEL in neg_rp pos_rp; do
  if [[ ! -s "$LINK/${PANEL}__feature_annotation_candidates.csv.gz" || ! -s "$LINK/${PANEL}__feature_best_annotations.csv.gz" ]]; then
    MISSING_PANELS+=("$PANEL")
  fi
done
if [[ "${#MISSING_PANELS[@]}" -gt 0 ]]; then
  echo "[BioAware] MS1-MS2 link products are absent; checking frozen upstream inputs."
  MISSING=0
  for PANEL in "${MISSING_PANELS[@]}"; do
    if [[ ! -s "data/mtbls13729/annotation/$PANEL/annotations_fdr.csv" && ! -s "data/mtbls13729/annotation/$PANEL/annotations.csv" ]]; then
      echo "Missing annotation: data/mtbls13729/annotation/$PANEL/annotations_fdr.csv (or annotations.csv)" >&2
      MISSING=1
    fi
    [[ -s "data/mtbls13729/embeddings/$PANEL/manifest.csv" ]] || {
      echo "Missing embedding manifest: data/mtbls13729/embeddings/$PANEL/manifest.csv" >&2
      MISSING=1
    }
    [[ -s "data/mtbls13729/ms1_consensus/${PANEL}__requantification_targets.csv.gz" ]] || {
      echo "Missing MS1 targets: data/mtbls13729/ms1_consensus/${PANEL}__requantification_targets.csv.gz" >&2
      MISSING=1
    }
  done
  [[ "$MISSING" -eq 0 ]] || { echo "BioAware stopped before bridge construction; produce/sync the listed upstream files." >&2; exit 6; }
  python -u tasks/link_mtbls13729_ms2_annotations_to_ms1.py \
    --output-dir "$LINK" --panels "${MISSING_PANELS[@]}"
fi
for PANEL in neg_rp pos_rp; do
  [[ -s "$LINK/${PANEL}__feature_annotation_candidates.csv.gz" ]] || {
    echo "Bridge did not produce candidate table for $PANEL" >&2; exit 7;
  }
  [[ -s "$LINK/${PANEL}__feature_best_annotations.csv.gz" ]] || {
    echo "Bridge did not produce best-annotation table for $PANEL" >&2; exit 8;
  }
done

if [[ "$INPUT_READY" -eq 0 ]]; then
  python -u tasks/prepare_mtbls13729_bioaware_pilot.py \
    --link-dir "$LINK" \
    --participants "$RHEA/rhea_participants.csv.gz" \
    --output-dir "$INPUT"
fi
python -u tasks/evaluate_bioaware_expert.py \
  --candidates "$INPUT/candidates.csv.gz" \
  --participants "$RHEA/rhea_participants.csv.gz" \
  --seeds "$INPUT/seeds.csv.gz" \
  --output-dir "$OUTPUT" \
  --decoy-repeats 10 \
  --bootstrap-resamples 5000 \
  --seed 20260827
python -u tasks/validate_bioaware_result.py "$OUTPUT"
echo "=== BioAware MTBLS13729 completed: $(date) ==="
