#!/bin/bash
#SBATCH --job-name=netid_public_audit
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/data02/run01/scv7tsl/DreaMS/netid_public_audit_%j.out
#SBATCH --error=/data02/run01/scv7tsl/DreaMS/netid_public_audit_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SOURCE=data/external/netid_v1/source
REPORT=data/validation/netid_public_release_audit_v2_20260831/report.json

echo "=== NetID public release audit started: $(date) ==="
python -u -m pytest -q \
  tests/test_bioaware_netid_external.py \
  tests/test_netid_public_release_audit.py

python -u tasks/audit_netid_public_release.py --source-dir "$SOURCE"

python - "$REPORT" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("status") != "netid_public_release_audit_complete":
    raise SystemExit("unexpected NetID public audit status")
gates = report.get("gates", {})
required = [
    "source_integrity",
    "author_output_rows_align",
    "mapped_confident_manual_rows_eq_314",
    "author_formula_accuracy_ge_0_90",
    "mouse_liver_ms2_spectra_ge_3_peaks_ge_750",
    "mouse_liver_unique_features_ge_3_peaks_ge_750",
    "pass_to_component_isolated_ms2_edge_stage",
]
failed = [name for name in required if gates.get(name) is not True]
if failed:
    raise SystemExit(f"NetID public audit failed gates: {failed}")
if gates.get("exact_author_solver_reproduction_ready") is not False:
    raise SystemExit("exact-solver boundary was lost")
if gates.get("direct_dreams_structure_benchmark_ready") is not False:
    raise SystemExit("structure-benchmark boundary was lost")
print(json.dumps({
    "status": report["status"],
    "yeast": report["yeast_author_output_audit"],
    "mouse_liver_targeted_ms2": report["mouse_liver_targeted_ms2"],
    "gates": gates,
    "claim_limit": report["claim_limit"],
}, indent=2))
PY

echo "=== NetID public release audit completed: $(date) ==="
