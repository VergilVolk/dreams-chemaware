#!/bin/bash
#SBATCH --job-name=b47_gate_repair
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/b47_gate_repair_%j.out
#SBATCH --error=data/validation/b47_gate_repair_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

: "${B47_U3_DIR:?Set B47_U3_DIR to the frozen server U3-v3 output directory}"
OUT="data/validation/bioaware_b47_gate_remediation_${SLURM_JOB_ID}_v1"
[[ -f "$B47_U3_DIR/report.json" ]] || { echo "Missing U3 report" >&2; exit 5; }
[[ -f "$B47_U3_DIR/query_event_opportunities.csv.gz" ]] || { echo "Missing U3 opportunities" >&2; exit 5; }
[[ -f "$B47_U3_DIR/candidate_event_features.csv.gz" ]] || { echo "Missing U3 candidate features" >&2; exit 5; }

python -m py_compile tasks/audit_bioaware_b47_gate_remediation.py
python -u tasks/test_audit_bioaware_b47_gate_remediation.py
python -u tasks/audit_bioaware_b47_gate_remediation.py \
  --u3-dir "$B47_U3_DIR" --output "$OUT"
echo "B47 gate remediation complete: $OUT"

