#!/bin/bash
#SBATCH --job-name=bio_b47_u1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=bioaware_b47_u1_program_%j.out
#SBATCH --error=bioaware_b47_u1_program_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

GRAPH=data/validation/noise_corrected_candidate_graph_v1_20260906
SEEDS=data/validation/bioaware_b47_truthblind_seeds_20260914_v1
PARTICIPANTS=data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz
U1=data/validation/bioaware_b47_u1_spectrum_unary_dev_20260921_v1
U1B=data/validation/bioaware_b47_u1b_safe_veto_dev_20260921_v2
U1C=data/validation/bioaware_b47_u1c_seed_denominator_20260921_v1

echo "=== BioAware B47 U1 focused programme started: $(date) ==="
required=(
  "$GRAPH/candidate_graph.npz" "$GRAPH/report.json"
  "$SEEDS/report.json" "$SEEDS/candidate_scores.csv.gz"
  "$SEEDS/query_summaries.csv.gz" "$SEEDS/feature_consensus.csv.gz"
  "$SEEDS/seeds_primary.csv.gz" "$PARTICIPANTS"
  tasks/bioaware_b47_u1_core.py
  tasks/develop_bioaware_b47_u1_spectrum_unary.py
  tasks/audit_bioaware_b47_u1b_safe_veto.py
  tasks/audit_bioaware_b47_u1c_seed_denominator.py
  tasks/test_bioaware_b47_u1_spectrum_unary.py
  tasks/test_bioaware_b47_u1c_seed_denominator.py
  tasks/validate_bioaware_b47_u1_spectrum_unary.py
  tasks/validate_bioaware_b47_u1b_safe_veto.py
  tasks/validate_bioaware_b47_u1c_seed_denominator.py
)
for path in "${required[@]}"; do
  test -s "$path" || { echo "Missing required dependency: $path" >&2; exit 2; }
done
python -c "import numpy,pandas; print('[BioAware B47 U1 dependencies] PASS')"
python -m py_compile \
  tasks/bioaware_b47_u1_core.py \
  tasks/develop_bioaware_b47_u1_spectrum_unary.py \
  tasks/audit_bioaware_b47_u1b_safe_veto.py \
  tasks/audit_bioaware_b47_u1c_seed_denominator.py \
  tasks/validate_bioaware_b47_u1_spectrum_unary.py \
  tasks/validate_bioaware_b47_u1b_safe_veto.py \
  tasks/validate_bioaware_b47_u1c_seed_denominator.py
python -u tasks/test_bioaware_b47_u1_spectrum_unary.py
python -u tasks/test_bioaware_b47_u1c_seed_denominator.py

if test -d "$U1"; then
  echo "[resume] validating frozen U1 development result"
  python -u tasks/validate_bioaware_b47_u1_spectrum_unary.py --input "$U1"
else
  U1_WORK="${U1}.partial.${SLURM_JOB_ID}"
  test ! -e "$U1_WORK" || { echo "Refusing stale U1 partial: $U1_WORK" >&2; exit 3; }
  python -u tasks/develop_bioaware_b47_u1_spectrum_unary.py \
    --graph-dir "$GRAPH" --output "$U1_WORK" --preflight-only
  test ! -e "$U1_WORK" || { echo "U1 preflight unexpectedly wrote output" >&2; exit 4; }
  python -u tasks/develop_bioaware_b47_u1_spectrum_unary.py \
    --graph-dir "$GRAPH" --output "$U1_WORK" \
    --folds 5 --fold-seed 20260921 --bootstrap-resamples 5000
  python -u tasks/validate_bioaware_b47_u1_spectrum_unary.py --input "$U1_WORK"
  mv "$U1_WORK" "$U1"
fi

if test -d "$U1B"; then
  echo "[resume] validating frozen U1b safety result"
  python -u tasks/validate_bioaware_b47_u1b_safe_veto.py --input "$U1B"
else
  U1B_WORK="${U1B}.partial.${SLURM_JOB_ID}"
  test ! -e "$U1B_WORK" || { echo "Refusing stale U1b partial: $U1B_WORK" >&2; exit 3; }
  python -u tasks/audit_bioaware_b47_u1b_safe_veto.py \
    --graph-dir "$GRAPH" --u1-dir "$U1" --output "$U1B_WORK" \
    --bootstrap-resamples 5000 --seed 20260923
  python -u tasks/validate_bioaware_b47_u1b_safe_veto.py --input "$U1B_WORK"
  mv "$U1B_WORK" "$U1B"
fi

if test -d "$U1C"; then
  echo "[resume] validating frozen U1c denominator audit"
  python -u tasks/validate_bioaware_b47_u1c_seed_denominator.py --input "$U1C"
else
  U1C_WORK="${U1C}.partial.${SLURM_JOB_ID}"
  test ! -e "$U1C_WORK" || { echo "Refusing stale U1c partial: $U1C_WORK" >&2; exit 3; }
  python -u tasks/audit_bioaware_b47_u1c_seed_denominator.py \
    --seed-dir "$SEEDS" --participants "$PARTICIPANTS" \
    --output "$U1C_WORK" --maximum-seed-degree 250
  python -u tasks/validate_bioaware_b47_u1c_seed_denominator.py --input "$U1C_WORK"
  mv "$U1C_WORK" "$U1C"
fi

echo "=== BioAware B47 U1 focused programme completed: $(date) ==="
echo "U1:  $U1/report.json"
echo "U1b: $U1B/report.json"
echo "U1c: $U1C/report.json"
