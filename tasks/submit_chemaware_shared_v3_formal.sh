#!/bin/bash
# Submit the fail-closed ChemAware-v3 formal dependency chain on the server.
# The seed-17/fold-0 G1 task is run first as the resource/protocol pilot; the
# remaining fourteen G1 tasks start only after it succeeds.
set -euo pipefail

[[ -f tasks/run_chemaware_shared_v3_g1.sbatch ]] || {
  echo "Run this script from the DreaMS repository root." >&2
  exit 2
}

cat >&2 <<'EOF'
ChemAware formal submission is paused fail-closed.
The frozen candidate graph was built from the obsolete real_train_primary
interpretation of MassSpecGym SIMULATION_CHALLENGE. That field is benchmark
subset membership, not spectrum provenance. Rebuild and re-audit the
P3-disjoint graph from train_primary_all before enabling this submission chain.
EOF
exit 2

command -v sbatch >/dev/null 2>&1 || {
  echo "sbatch is unavailable; submit this chain on the configured cluster." >&2
  exit 2
}

chemaware_cache_job=$(sbatch --parsable tasks/run_chemaware_shared_v2_cache.sbatch)
chemaware_morgan_job=$(sbatch --parsable \
  --dependency="afterok:${chemaware_cache_job}" \
  tasks/run_chemaware_shared_v3_morgan.sbatch)
chemaware_g1_pilot_job=$(sbatch --parsable \
  --dependency="afterok:${chemaware_cache_job}" \
  --array=0 \
  tasks/run_chemaware_shared_v3_g1.sbatch)
chemaware_g1_rest_job=$(sbatch --parsable \
  --dependency="afterok:${chemaware_g1_pilot_job}" \
  --array=1-14%3 \
  tasks/run_chemaware_shared_v3_g1.sbatch)
chemaware_g1_summary_job=$(sbatch --parsable \
  --dependency="afterok:${chemaware_g1_pilot_job}:${chemaware_g1_rest_job}" \
  tasks/run_chemaware_shared_v3_g1_summary.sbatch)
chemaware_g2_pilot_job=$(sbatch --parsable \
  --dependency="afterok:${chemaware_morgan_job}:${chemaware_g1_summary_job}" \
  tasks/run_chemaware_shared_v3_g2_pilot.sbatch)
chemaware_g2_decision_job=$(sbatch --parsable \
  --dependency="afterok:${chemaware_g2_pilot_job}" \
  tasks/run_chemaware_shared_v3_g2_pilot_decision.sbatch)
chemaware_g2_full_job=$(sbatch --parsable \
  --dependency="afterok:${chemaware_g2_decision_job}" \
  tasks/run_chemaware_shared_v3_g2.sbatch)
chemaware_g2_summary_job=$(sbatch --parsable \
  --dependency="afterok:${chemaware_g2_full_job}" \
  tasks/run_chemaware_shared_v3_g2_summary.sbatch)

printf '%s\n' \
  "cache=${chemaware_cache_job}" \
  "morgan=${chemaware_morgan_job}" \
  "g1_pilot=${chemaware_g1_pilot_job}" \
  "g1_remaining=${chemaware_g1_rest_job}" \
  "g1_summary=${chemaware_g1_summary_job}" \
  "g2_pilot=${chemaware_g2_pilot_job}" \
  "g2_pilot_decision=${chemaware_g2_decision_job}" \
  "g2_full_matrix=${chemaware_g2_full_job}" \
  "g2_summary=${chemaware_g2_summary_job}"
