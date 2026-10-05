#!/bin/bash
#SBATCH --job-name=chem_ice_resid
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_ice_resid_%j.out
#SBATCH --error=data/validation/chem_ice_resid_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ROOT="data/validation/chemaware_iceberg_residual_shared_phase_a/run_${SLURM_JOB_ID}"
[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 3; }
mkdir -p "$RUN_ROOT"

python tasks/test_chemaware_direct_prior_objective.py
python tasks/test_chemaware_direct_prior_torch.py

LEDGER_DIR="$RUN_ROOT/strict_corrective_ledger"
UTILITY_DIR="$RUN_ROOT/strict_corrective_utility"
python tasks/build_chemaware_iceberg_corrective_residual_ledger.py \
  --output "$LEDGER_DIR" --fold-seed 20260905
python tasks/audit_chemaware_direct_chemical_prior_utility.py \
  --graph data/validation/chemaware_full_manifest_iceberg_graph_v1/graph.npz \
  --ledger "$LEDGER_DIR/ledger.npz" --output "$UTILITY_DIR" \
  --folds 0 1 2 --alpha-grid 0.10 0.25 0.50 1.0 \
  --bootstrap-draws 10000 --seed 20260907
python - "$LEDGER_DIR/report.json" "$UTILITY_DIR/report.json" <<'PY'
import json
import pathlib
import sys

ledger = json.loads(pathlib.Path(sys.argv[1]).read_text())
utility = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert ledger["strict_corrective_actions"] == 272
assert utility["status"] == "CHEMAWARE_DIRECT_PRIOR_UTILITY_DEVELOPMENT_PASS"
row = next(value for value in utility["grid"] if value["alpha"] == 0.5)
assert row["passes_development_gate"] is True
assert row["arms"]["correct"]["introduced"] == 0
assert row["arms"]["correct"]["global_delta_recall1"] >= 0.04
PY

LABELS=(clean_duplicate correct_alpha050 structure_alpha050 peak_alpha050)
PRIOR_ARMS=(none correct structure_swapped peak_permuted)
ALPHAS=(0.0 0.50 0.50 0.50)

for INDEX in 0 1 2 3; do
  LABEL="${LABELS[$INDEX]}"
  PRIOR_ARM="${PRIOR_ARMS[$INDEX]}"
  ALPHA="${ALPHAS[$INDEX]}"
  OUT="$RUN_ROOT/$LABEL"
  echo "START: ChemAware residual arm=$LABEL alpha=$ALPHA"
  python -u tasks/train_chemaware_iceberg_residual_shared.py \
    --output "$OUT" --prior-arm "$PRIOR_ARM" --alpha "$ALPHA" \
    --action-ledger-dir "$LEDGER_DIR" \
    --inner-fold 3 --outer-fold 4 --seed 20260907 \
    --epochs 2 --batch-queries 2 --references-per-molecule 2 \
    --safety-identities 272 --max-eval-identities 2000 \
    --unfreeze-blocks 1 --backbone-lr 2e-6 --head-lr 1e-5 \
    --weight-decay 1e-4 --temperature 0.10 \
    --lambda-listwise 1.0 --lambda-inbatch 0.25 \
    --lambda-margin-floor 2.0 --margin-floor-slack 0.005 \
    --lambda-preserve 5.0 --lambda-chemical 1.0 \
    --chemical-huber 0.02 --chemical-gradient-ratio 0.25 \
    --chemical-gradient-scale-cap 4.0 --grad-clip 1.0 \
    --n-highest-peaks 100 --eval-batch-size 96 \
    --bootstrap-draws 10000 --device cuda --no-amp --torch-threads 8
  test -f "$OUT/report.json"
  test -f "$OUT/inner_per_query.npz"
  test -f "$OUT/final_shared_encoder.pt"
  echo "COMPLETE: ChemAware residual arm=$LABEL"
done

python tasks/summarize_chemaware_iceberg_residual_shared_phase_a.py \
  --root "$RUN_ROOT" --bootstrap-draws 10000 --seed 20260907
test -f "$RUN_ROOT/phase_a_summary.json"
echo "COMPLETE: ChemAware ICEBERG residual shared-embedding Phase A"
