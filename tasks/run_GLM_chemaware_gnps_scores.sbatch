#!/usr/bin/env bash
#SBATCH --job-name=glm_chem_gnps
#SBATCH --gpus=1
#SBATCH --output=data/validation/GLM_chemaware_gnps_scores_%j.out
#SBATCH --error=data/validation/GLM_chemaware_gnps_scores_%j.err

# GLM: score the ChemAware encoder on both frozen GNPS panels.
# Output: a 16-method bundle copy + embeddings, ready for the Gate-0
# complementarity ledger (GLM_gate0_complementarity.py) and the unified
# G_ab (alpha_C slot).
#
# Prerequisites (server):
#   - ChemAware encoder checkpoint (frozen, hash below)
#   - the GNPS benchmark spectra + panels (same frozen assets the
#     15-method ladder used)
# Usage:
#   GLM_CHEM_CKPT=<path> sbatch tasks/run_GLM_chemaware_gnps_scores.sbatch
set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CHEM_CKPT="${GLM_CHEM_CKPT:-data/validation/chemaware_online_role_budget_native/run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt}"
if [[ ! -e "$CHEM_CKPT" ]]; then
  echo "ERROR: ChemAware checkpoint not found: $CHEM_CKPT" >&2
  find data/validation -name '*phasea*step2000*' -o -name '*phasea*max*boundary*' 2>/dev/null | head -5
  exit 3
fi
CHEM_SHA="$(sha256sum "$CHEM_CKPT" | cut -d' ' -f1)"
echo "ChemAware checkpoint: $CHEM_CKPT"
echo "ChemAware checkpoint sha256: $CHEM_SHA"

OUT="data/validation/GLM_chemaware_gnps_scores/run_${SLURM_JOB_ID}"
mkdir -p "$OUT"

# 1. embed all panel spectra with the ChemAware encoder
python -X utf8 tasks/GLM_embed_panel_spectra.py \
  --checkpoint "$CHEM_CKPT" \
  --encoder dreams \
  --official-checkpoint data/e1/official_embedding_slim.pt \
  --architecture-checkpoint dreams/models/pretrained/ssl_model_server.pt \
  --output "$OUT/chemaware_panel_embeddings.npz" \
  --batch-size 2048

# 2. merge as a new method row -> frozen evaluation -> ladder row
python -X utf8 tasks/GLM_arm_scores_to_ladder.py \
  --embeddings "$OUT/chemaware_panel_embeddings.npz" \
  --method-name chemaware_encoder \
  --run-dir "$OUT/run_chemaware"

# 3. Gate-0 complementarity ledger (alpha_C verdict)
python -X utf8 tasks/GLM_gate0_complementarity.py \
  --bundle "$OUT/run_chemaware/bundle/method_scores.npz" \
  --chem-method chemaware_encoder

echo "COMPLETE: ChemAware GNPS scores + Gate-0 ledger in $OUT"
