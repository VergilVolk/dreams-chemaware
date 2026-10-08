#!/bin/bash
#SBATCH --job-name=glm_s4_smoke
#SBATCH --gpus=1
#SBATCH --output=data/validation/GLM_s4_smoke_%j.out
#SBATCH --error=data/validation/GLM_s4_smoke_%j.err

# GLM: S4 real-weights smoke for the orbit-boundary trainer.
# Runs the REAL DreaMS 116M backbone forward/backward on 2 groups of real
# GNPS spectra. This is sbatch-unbanning condition #1.
set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OFFICIAL="data/e1/official_embedding_slim.pt"
ARCH="dreams/models/pretrained/ssl_model_server.pt"
NOISE_V1="data/validation/chemaware_online_role_budget_native/run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt"

for f in "$OFFICIAL" "$ARCH" "$NOISE_V1"; do
  [[ -e "$f" ]] || { echo "MISSING: $f" >&2; exit 3; }
done

python -X utf8 tasks/GLM_real_smoke_orbit_boundary.py
echo "S4 SMOKE COMPLETE"
