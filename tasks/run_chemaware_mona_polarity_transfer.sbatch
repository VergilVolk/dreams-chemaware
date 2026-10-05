#!/bin/bash
#SBATCH --job-name=chem_mona_xfer
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/chem_mona_xfer_%j.out
#SBATCH --error=data/validation/chem_mona_xfer_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
POSITIVE_MGF="data/models/mona_pos_full.mgf"
NEGATIVE_MGF="data/models/mona_neg_full.mgf"
DEVELOPMENT="data/models/MassSpecGym_MurckoHist_split.hdf5"
PHASEA="${CHEMAWARE_CHECKPOINT:-data/validation/chemaware_online_role_budget_native/run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt}"
EXPECTED_PHASEA_SHA256="${CHEMAWARE_CHECKPOINT_SHA256:-a8428329ca1d12bfe735f3f8b848ed020a8db18d0a654d4f62cdbeb006e61135}"
OUT="data/validation/chemaware_mona_polarity_transfer/run_${SLURM_JOB_ID}"

required_files=(
  "$OFFICIAL"
  "$ARCHITECTURE"
  "$POSITIVE_MGF"
  "$NEGATIVE_MGF"
  "$DEVELOPMENT"
  "$PHASEA"
  tasks/chemaware_mona_transfer_core.py
  tasks/build_chemaware_mona_polarity_transfer_panel.py
  tasks/evaluate_chemaware_mona_polarity_transfer.py
  tasks/test_chemaware_mona_polarity_transfer.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required MoNA transfer input: $required_file" >&2
    exit 2
  }
done
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }

export PHASEA EXPECTED_PHASEA_SHA256
python - <<'PY'
import hashlib
import os
from pathlib import Path

path = Path(os.environ["PHASEA"])
digest = hashlib.sha256()
with path.open("rb") as handle:
    for block in iter(lambda: handle.read(8 << 20), b""):
        digest.update(block)
actual = digest.hexdigest()
expected = os.environ["EXPECTED_PHASEA_SHA256"]
if actual != expected:
    raise RuntimeError(
        f"ChemAware checkpoint hash mismatch: expected={expected} actual={actual} path={path}"
    )
print(f"ChemAware checkpoint verified: {actual}", flush=True)
PY

python -u tasks/test_chemaware_mona_polarity_transfer.py

python -u tasks/build_chemaware_mona_polarity_transfer_panel.py \
  --positive-mgf "$POSITIVE_MGF" \
  --negative-mgf "$NEGATIVE_MGF" \
  --development-hdf5 "$DEVELOPMENT" \
  --output-dir "$OUT/panel" \
  --ppm 20 --queries-per-identity 4 --references-per-candidate 20 \
  --n-highest-peaks 100 \
  --minimum-positive-queries 700 \
  --minimum-negative-queries 1500 \
  --minimum-combined-queries 2500

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_mona_polarity_transfer.py \
  --panel-dir "$OUT/panel" \
  --positive-mgf "$POSITIVE_MGF" \
  --negative-mgf "$NEGATIVE_MGF" \
  --official-checkpoint "$OFFICIAL" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "chemaware_phasea_2pp=$PHASEA" \
  --output "$OUT/evaluation.json" \
  --batch-size 64 --n-highest-peaks 100 \
  --bootstrap-draws 5000 --seed 20260928 --device cuda

test -f "$OUT/panel/report.json"
test -f "$OUT/evaluation.json"
test -f "$OUT/evaluation_paired_outcomes.npz"
echo "COMPLETE: ChemAware paired transfer evaluation on sealed positive+negative MoNA -> $OUT/evaluation.json"
