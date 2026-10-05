#!/bin/bash
set -euo pipefail
echo "BLOCKED: historical official-only continuation is not a ChemAware causal experiment" >&2
exit 64
[[ -f tasks/run_chemaware_full_candidate_official_cache.sbatch ]] || {
  echo "Run from the DreaMS repository root" >&2; exit 2;
}
cache_job=$(sbatch --parsable tasks/run_chemaware_full_candidate_official_cache.sbatch)
pilot_job=$(sbatch --parsable --dependency="afterok:${cache_job}" \
  tasks/run_chemaware_full_candidate_official_pilot.sbatch)
direct_job=$(sbatch --parsable --dependency="afterok:${pilot_job}" \
  --export="ALL,ADAPTER_RUN=data/validation/chemaware_full_candidate_official_pilot/run_${pilot_job}" \
  tasks/run_chemaware_full_candidate_direct_official.sbatch)
printf '%s\n' "cache=${cache_job}" "official_pilot=${pilot_job}" \
  "direct_official=${direct_job}"
