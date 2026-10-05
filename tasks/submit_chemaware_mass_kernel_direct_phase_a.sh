#!/bin/bash
set -euo pipefail
echo "BLOCKED: rejected Phase-A rule kernel is retained only for provenance" >&2
exit 64
[[ -f tasks/run_chemaware_mass_kernel_direct_arm.sbatch ]] || {
  echo "Run from the DreaMS repository root" >&2; exit 2;
}
arms_job=$(sbatch --parsable tasks/run_chemaware_mass_kernel_direct_arm.sbatch)
summary_job=$(sbatch --parsable --dependency="afterok:${arms_job}" \
  --export="ALL,ARMS_JOB=${arms_job}" \
  tasks/run_chemaware_mass_kernel_direct_summary.sbatch)
printf '%s\n' "eight_arm_array=${arms_job}" "summary=${summary_job}"
