#!/bin/bash
#SBATCH --job-name=kgmn_oep_inputs
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/kgmn_oep_inputs_%j.out
#SBATCH --error=logs/kgmn_oep_inputs_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
mkdir -p logs
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

RAW_ROOT=data/reference/OEP003284_raw
INPUT_ROOT=data/reference/OEP003284
SOURCES=data/reference/kgmn_zenodo_7089991
INVENTORY=data/validation/kgmn_oep003284_node_inventory_20260831
CONTRACT=data/validation/kgmn_external_validation_contract_20260831
IDENTIFIER_CONTRACT=data/validation/kgmn_oep003284_author_identifier_contract_20260831
PREFLIGHT=data/validation/kgmn_oep003284_input_preflight_${SLURM_JOB_ID:-manual}.json

for output in "$INPUT_ROOT" "$PREFLIGHT"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite formal output: $output" >&2
    exit 1
  fi
done

python -u tasks/fetch_kgmn_external_validation_sources.py --output-dir "$SOURCES"
if [[ -s "$CONTRACT/report.json" ]]; then
  python - "$CONTRACT/report.json" <<'PY'
import json, pathlib, sys
r=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if r.get("status") != "kgmn_external_validation_contract_frozen": raise SystemExit("invalid existing external contract")
PY
elif [[ -e "$CONTRACT" ]]; then
  echo "Partial external contract exists: $CONTRACT" >&2; exit 1
else
  python -u tasks/freeze_kgmn_external_validation_contract.py --source-dir "$SOURCES" --output-dir "$CONTRACT"
fi
if [[ -s "$IDENTIFIER_CONTRACT/report.json" ]]; then
  python - "$IDENTIFIER_CONTRACT/report.json" <<'PY'
import json, pathlib, sys
r=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if r.get("status") != "kgmn_oep003284_author_identifier_contract_frozen": raise SystemExit("invalid existing identifier contract")
PY
elif [[ -e "$IDENTIFIER_CONTRACT" ]]; then
  echo "Partial identifier contract exists: $IDENTIFIER_CONTRACT" >&2; exit 1
else
  python -u tasks/audit_kgmn_oep003284_author_identifiers.py \
    --supplement "$SOURCES/Supplementary data3.xlsx" --contract-dir "$CONTRACT" --output-dir "$IDENTIFIER_CONTRACT"
fi
if [[ -s "$INVENTORY/report.json" && -s "$INVENTORY/node_files.csv" ]]; then
  python - "$INVENTORY/report.json" <<'PY'
import json, pathlib, sys
r=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if r.get("status") != "kgmn_oep003284_node_inventory_complete" or r.get("local_validation",{}).get("ready") is not True:
    raise SystemExit("invalid existing local NODE inventory")
PY
elif [[ -e "$INVENTORY" ]]; then
  echo "Partial NODE inventory exists: $INVENTORY" >&2; exit 1
else
  python -u tasks/inventory_kgmn_oep003284_node.py \
    --output-dir "$INVENTORY" --local-raw-root "$RAW_ROOT" --require-local
fi
python -u tasks/prepare_kgmn_oep003284_author_inputs.py \
  --supplements-dir "$SOURCES" \
  --inventory-dir "$INVENTORY" \
  --raw-root "$RAW_ROOT" \
  --output-root "$INPUT_ROOT"
python -u tasks/preflight_kgmn_oep003284_inputs.py \
  --input-root "$INPUT_ROOT" \
  --output "$PREFLIGHT" \
  --require-ready \
  --require-dreams-edge \
  --allow-author-mapped-mzxml

echo "=== KGMN OEP003284 author inputs completed: $(date) ==="
