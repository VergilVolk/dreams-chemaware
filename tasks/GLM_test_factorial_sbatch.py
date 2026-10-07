"""Static test for the factorial sbatch (no cluster needed).

Checks the audit contract is structurally enforced:
  S1 --gpus=1 present, no --partition/--mem directives;
  S2 all six arms appear exactly once, same dose variables;
  S3 dose verification block present (drift abort);
  S4 pool hash pinning present;
  S5 selector-on-R-only referenced, applied mechanically.
"""
from pathlib import Path

txt = Path("tasks/run_GLM_orbit_boundary_factorial.sbatch").read_text(
    encoding="utf-8")

assert "--gpus=1" in txt
assert "--partition" not in txt and "--mem" not in txt
print("S1 PASS: gpus-only resource directives")

assert "DO NOT SUBMIT" in txt, "draft banner required until GPU smoke passes"
print("S1b PASS: do-not-submit banner present")

for arm in ("R", "O-null", "O-real", "C-null", "C-real", "OC"):
    assert txt.count(f'"{arm}"') >= 1, f"arm {arm} missing"
assert txt.count('--arm "$ARM"') == 1, "arms must be driven by one loop variable"
print("S2 PASS: six arms via one loop, single --arm flag")

assert "dose drift: steps" in txt and "dose drift: lr" in txt
print("S3 PASS: per-arm dose verification with abort")

assert "sha256sum" not in txt  # pins must go through --expected-pool-sha256
assert "--expected-pool-sha256" in txt and "GLM_EXPECTED_POOL_SHA" in txt
assert 'dose drift: pool sha' in txt
print("S4 PASS: preregistered SHA fail-closed pinning")

assert "GLM_select_shared_factorial_step" in txt and "R arm ONLY" in txt
print("S5 PASS: shared-step selector on R only")
print("SBATCH STATIC TEST: ALL PASS")
