#!/bin/bash
#SBATCH --job-name=kgmn_full_ext
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
# Slurm opens stdout/stderr before the script body runs. Use the existing
# absolute repository directory so launch never depends on a log subdirectory
# or on the directory from which sbatch is called.
#SBATCH --output=/data02/run01/scv7tsl/DreaMS/kgmn_full_external_%j.out
#SBATCH --error=/data02/run01/scv7tsl/DreaMS/kgmn_full_external_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
mkdir -p data/validation
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

JOB_TAG=${SLURM_JOB_ID:-manual}
PREFLIGHT=data/validation/kgmn_full_external_preflight_${JOB_TAG}.json
BASELINE_ROOT=data/validation/kgmn_metdna2_200std_author_baseline
BASELINE_REPORT=$BASELINE_ROOT/frozen_author_baseline.json
CAL_ROOT=data/validation/kgmn_dreams_edge_calibration_official_20260831
CAL_REPORT=$CAL_ROOT/report.json
CAL_AUTHOR_INPUT=data/validation/kgmn_edge_calibration_exact_author_input_20260831
CAL_AUTHOR_SCORES=data/validation/kgmn_edge_calibration_exact_author_dp_20260831
AUTHOR_INPUT=data/reference/OEP003284
AUTHOR_INPUT_REPORT=$AUTHOR_INPUT/author_input_report.json
HIDDEN_ROOT=data/validation/kgmn_oep003284_hidden_seed_20260831
FINAL_DECISION=$HIDDEN_ROOT/final_decision.json

json_status () {
  python - "$1" "$2" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
if not p.is_file(): raise SystemExit(1)
try: r=json.loads(p.read_text(encoding="utf-8"))
except Exception: raise SystemExit(1)
raise SystemExit(0 if r.get("status") == sys.argv[2] else 1)
PY
}

echo "=== KGMN full external pipeline dependency audit: $(date) ==="
python -u tasks/preflight_kgmn_full_external_pipeline.py \
  --raw-root data/reference/OEP003284_raw \
  --output "$PREFLIGHT" \
  --require-ready
Rscript tasks/preflight_kgmn_metdna2_runtime.R third_party/MetDNA2

echo "=== stage 1/4: untouched author 200STD baseline ==="
if json_status "$BASELINE_REPORT" kgmn_metdna2_200std_author_baseline_frozen; then
  echo "[resume] verified author baseline: $BASELINE_REPORT"
else
  if [[ -e "$BASELINE_ROOT" ]]; then
    echo "Partial or invalid author baseline exists: $BASELINE_ROOT" >&2
    exit 1
  fi
  bash tasks/run_kgmn_metdna2_200std_baseline.sbatch
  json_status "$BASELINE_REPORT" kgmn_metdna2_200std_author_baseline_frozen || {
    echo "Author baseline did not freeze correctly" >&2; exit 1;
  }
fi

echo "=== stage 2/4: component-isolated DreaMS edge calibration ==="
if json_status "$CAL_REPORT" kgmn_dreams_edge_calibration_complete; then
  echo "[resume] verified edge calibration: $CAL_REPORT"
else
  for partial in "$CAL_ROOT" "$CAL_AUTHOR_INPUT" "$CAL_AUTHOR_SCORES"; do
    if [[ -e "$partial" ]]; then
      echo "Partial or invalid edge-calibration output exists: $partial" >&2
      exit 1
    fi
  done
  bash tasks/run_kgmn_dreams_edge_calibration.sbatch
  json_status "$CAL_REPORT" kgmn_dreams_edge_calibration_complete || {
    echo "DreaMS edge calibration did not freeze correctly" >&2; exit 1;
  }
fi
python - "$CAL_REPORT" <<'PY'
import json, pathlib, sys
r=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if r.get("gates",{}).get("official_dreams_eligible_for_dynamic_propagation_test") is not True:
    raise SystemExit("STOP: official DreaMS did not pass the pre-registered dynamic-propagation eligibility gate")
PY

echo "=== stage 3/4: exact author OEP003284 inputs and identifier bridge ==="
if json_status "$AUTHOR_INPUT_REPORT" kgmn_oep003284_author_inputs_complete; then
  echo "[resume] verified exact author OEP inputs: $AUTHOR_INPUT_REPORT"
else
  if [[ -e "$AUTHOR_INPUT" ]]; then
    echo "Partial or invalid exact-author input tree exists: $AUTHOR_INPUT" >&2
    exit 1
  fi
  bash tasks/run_kgmn_oep003284_author_inputs.sbatch
  json_status "$AUTHOR_INPUT_REPORT" kgmn_oep003284_author_inputs_complete || {
    echo "Exact author OEP inputs did not freeze correctly" >&2; exit 1;
  }
fi

echo "=== stage 4/4: frozen OEP003284 hidden-seed external test ==="
if [[ -s "$FINAL_DECISION" ]]; then
  python - "$FINAL_DECISION" <<'PY'
import json, pathlib, sys
r=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
allowed={"kgmn_oep003284_hidden_seed_external_passed","kgmn_oep003284_hidden_seed_external_failed"}
if r.get("status") not in allowed: raise SystemExit("invalid existing final decision")
print("[resume] completed scientific result:", r["status"])
PY
else
  if [[ -e "$HIDDEN_ROOT" ]]; then
    echo "Partial hidden-seed tree exists without a final decision: $HIDDEN_ROOT" >&2
    exit 1
  fi
  bash tasks/run_kgmn_oep003284_hidden_seed.sbatch
fi

python - "$FINAL_DECISION" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
if not p.is_file(): raise SystemExit("missing final_decision.json")
r=json.loads(p.read_text(encoding="utf-8"))
allowed={"kgmn_oep003284_hidden_seed_external_passed","kgmn_oep003284_hidden_seed_external_failed"}
if r.get("status") not in allowed: raise SystemExit("invalid final decision status")
print(json.dumps({
  "status": r["status"],
  "primary_arm": r.get("primary_arm"),
  "pass_to_network_teacher_embedding_distillation": r.get("decision",{}).get("pass_to_network_teacher_embedding_distillation"),
  "claim_limit": r.get("claim_limit"),
}, indent=2))
PY

echo "=== KGMN full external pipeline completed: $(date) ==="
