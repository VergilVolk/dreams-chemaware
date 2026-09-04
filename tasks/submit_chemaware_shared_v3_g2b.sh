#!/bin/bash
# Submit the independent targeted frozen-probe G2b branch after the formal
# cache, Morgan teacher, and G1 summary artifacts already exist.
set -euo pipefail

[[ -f tasks/run_chemaware_shared_v3_g2b_pilot.sbatch ]] || {
  echo "Run this script from the DreaMS repository root." >&2
  exit 2
}
cat >&2 <<'EOF'
ChemAware G2b submission is paused fail-closed.
This launcher is bound to the historical real_train_primary candidate graph
and to a rejected structure/frozen-probe branch. It is retained only as an
audit trail and must not authorize new ChemAware training.
EOF
exit 2

command -v sbatch >/dev/null 2>&1 || {
  echo "sbatch is unavailable; submit this chain on the configured cluster." >&2
  exit 2
}
required=(
  data/validation/g8r_error_atlas_listwise_cache.npz
  data/validation/g8r_chemaware_shared_v3_morgan/report.json
  data/validation/g8r_chemaware_shared_v3_peft_g1_summary.json
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "MISSING prerequisite: $file" >&2; exit 2; }
done

chemaware_g2b_pilot_job=$(sbatch --parsable \
  tasks/run_chemaware_shared_v3_g2b_pilot.sbatch)
chemaware_g2b_decision_job=$(sbatch --parsable \
  --dependency="afterok:${chemaware_g2b_pilot_job}" \
  tasks/run_chemaware_shared_v3_g2b_pilot_decision.sbatch)
chemaware_g2b_full_job=$(sbatch --parsable \
  --dependency="afterok:${chemaware_g2b_decision_job}" \
  tasks/run_chemaware_shared_v3_g2b.sbatch)
chemaware_g2b_summary_job=$(sbatch --parsable \
  --dependency="afterok:${chemaware_g2b_full_job}" \
  tasks/run_chemaware_shared_v3_g2b_summary.sbatch)

printf '%s\n' \
  "g2b_pilot=${chemaware_g2b_pilot_job}" \
  "g2b_pilot_decision=${chemaware_g2b_decision_job}" \
  "g2b_full_matrix=${chemaware_g2b_full_job}" \
  "g2b_summary=${chemaware_g2b_summary_job}"
