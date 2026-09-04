#!/bin/bash
#SBATCH --job-name=kgmn_oep_xcms_sens
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/kgmn_oep_xcms_%j.out
#SBATCH --error=logs/kgmn_oep_xcms_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
mkdir -p logs
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

RAW_ROOT=data/reference/OEP003284_raw
INPUT_ROOT=data/reference/OEP003284_xcms_sensitivity
INVENTORY=data/validation/kgmn_oep003284_node_inventory_xcms_sensitivity_20260831
PREFLIGHT=data/validation/kgmn_oep003284_input_preflight_xcms_sensitivity_20260831.json

if [[ -e "$INPUT_ROOT" ]]; then
  echo "Refusing to overwrite KGMN input root: $INPUT_ROOT" >&2
  exit 1
fi
if [[ -e "$INVENTORY" ]]; then
  echo "Refusing to overwrite NODE inventory: $INVENTORY" >&2
  exit 1
fi
if [[ -e "$PREFLIGHT" ]]; then
  echo "Refusing to overwrite input preflight: $PREFLIGHT" >&2
  exit 1
fi

python -u tasks/inventory_kgmn_oep003284_node.py \
  --output-dir "$INVENTORY" \
  --local-raw-root "$RAW_ROOT" \
  --require-local

mkdir "$INPUT_ROOT"
for polarity in positive negative; do
  mkdir "$INPUT_ROOT/$polarity"
  Rscript tasks/preprocess_kgmn_oep003284_xcms.R \
    "$RAW_ROOT" "$polarity" "$INPUT_ROOT/$polarity" \
    "$INPUT_ROOT/${polarity}_xcms_report.json"
done

python -u tasks/preflight_kgmn_oep003284_inputs.py \
  --input-root "$INPUT_ROOT" \
  --output "$PREFLIGHT" \
  --require-ready \
  --require-dreams-edge \
  --allow-author-mapped-mzxml

echo "=== KGMN OEP003284 XCMS reconstruction completed: $(date) ==="
