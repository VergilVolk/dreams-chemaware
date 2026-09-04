#!/bin/bash
#SBATCH --job-name=mtbls13729_ms1_targets
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/mtbls13729_ms1_targets_%j.out
#SBATCH --error=data/validation/mtbls13729_ms1_targets_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

MZML="data/mtbls13729/mzml"
FEATURES="data/mtbls13729/ms1_features_full/features"
CONSENSUS="data/mtbls13729/ms1_consensus"
mkdir -p "$FEATURES" "$CONSENSUS"

REQUIRED_SCRIPTS=(
  tasks/extract_mtbls13729_ms1_features.py
  tasks/build_mtbls13729_ms1_consensus.py
  tasks/validate_mtbls13729_ms1_targets.py
)
for SCRIPT in "${REQUIRED_SCRIPTS[@]}"; do
  [[ -s "$SCRIPT" ]] || { echo "Missing required script: $SCRIPT" >&2; exit 2; }
done
python -c "import pyopenms, pandas, numpy" || {
  echo "The dreams environment is missing pyopenms/pandas/numpy" >&2
  exit 2
}

echo "=== MTBLS13729 MS1 target construction started: $(date) ==="
for PANEL in neg_rp pos_rp; do
  COUNT=$(find "$FEATURES" -maxdepth 1 -type f -name "${PANEL}__*__noise_10000.csv.gz" 2>/dev/null | wc -l)
  if [[ "$COUNT" -lt 50 ]]; then
    if [[ -d "$MZML/$PANEL" ]]; then
      MZML_COUNT=$(find "$MZML/$PANEL" -maxdepth 1 -type f -name '*.mzML' | wc -l)
    else
      MZML_COUNT=0
    fi
    [[ "$MZML_COUNT" -ge 50 ]] || {
      echo "Missing raw input: $MZML/$PANEL has only $MZML_COUNT mzML files" >&2
      exit 3
    }
    echo "[$PANEL] extracted features=$COUNT; checkpointed extraction will fill missing samples."
    python -u tasks/extract_mtbls13729_ms1_features.py --panels "$PANEL"
  else
    echo "[$PANEL] reusing $COUNT extracted feature files."
  fi
done

MISSING_PANELS=()
for PANEL in neg_rp pos_rp; do
  [[ -s "$CONSENSUS/${PANEL}__requantification_targets.csv.gz" ]] || MISSING_PANELS+=("$PANEL")
done
if [[ "${#MISSING_PANELS[@]}" -gt 0 ]]; then
  echo "Building consensus targets for: ${MISSING_PANELS[*]}"
  python -u tasks/build_mtbls13729_ms1_consensus.py --panels "${MISSING_PANELS[@]}"
else
  echo "Both consensus target tables already exist; no rebuild."
fi

python -u tasks/validate_mtbls13729_ms1_targets.py --panels neg_rp pos_rp
echo "=== MTBLS13729 MS1 target construction completed: $(date) ==="
