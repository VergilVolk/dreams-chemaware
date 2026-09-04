#!/bin/bash
#SBATCH --job-name=netid_dreams_edge
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/data02/run01/scv7tsl/DreaMS/netid_dreams_edge_%j.out
#SBATCH --error=/data02/run01/scv7tsl/DreaMS/netid_dreams_edge_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== public NetID + official DreaMS edge stage started: $(date) ==="
python - <<'PY'
import importlib

required = ("h5py", "numpy", "openpyxl", "pandas", "pytest", "scipy", "sklearn", "torch")
missing = []
for package in required:
    try:
        importlib.import_module(package)
    except Exception as exc:
        missing.append(f"{package}: {exc}")
if missing:
    raise SystemExit("Missing Python dependencies:\n" + "\n".join(missing))
print("[dependency preflight] PASS: " + ", ".join(required), flush=True)
PY

python -u -m pytest -q \
  tests/test_bioaware_netid_external.py \
  tests/test_netid_public_release_audit.py \
  tests/test_encode_netid_mouse_liver_dreams.py \
  tests/test_netid_dreams_edge_signal.py \
  tests/test_diagnose_netid_edge_signal_modalities.py \
  tests/test_netid_public_dreams_edge_sbatch.py

python -u tasks/fetch_bioaware_netid_external.py
python -u tasks/audit_netid_public_release.py
python -u tasks/encode_netid_mouse_liver_dreams.py \
  --device cuda \
  --batch-size 128 \
  --minimum-peaks 3
python -u tasks/audit_netid_dreams_edge_signal.py

EDGE_STATUS=$(python -c 'import json; print(json.load(open("data/validation/netid_dreams_edge_signal_20260831/report.json"))["status"])')
if [ "$EDGE_STATUS" = "netid_dreams_edge_signal_failed" ]; then
  python -u tasks/diagnose_netid_edge_signal_modalities.py
fi

python - <<'PY'
import json
import pathlib

path = pathlib.Path("data/validation/netid_dreams_edge_signal_20260831/report.json")
report = json.loads(path.read_text(encoding="utf-8"))
allowed = {"netid_dreams_edge_signal_passed", "netid_dreams_edge_signal_failed"}
if report.get("status") not in allowed:
    raise SystemExit("invalid NetID DreaMS edge result")
print(json.dumps({
    "status": report["status"],
    "features_with_embeddings": report["features_with_embeddings"],
    "author_edges_with_both_spectra": report["author_edges_with_both_spectra"],
    "families": report["families"],
    "gates": report["gates"],
    "claim_limit": report["claim_limit"],
}, indent=2))
PY
echo "=== public NetID + official DreaMS edge stage completed: $(date) ==="
