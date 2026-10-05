#!/bin/bash
# Resume ONLY post-training GNPS evaluation and sealing for run 2347317.
# Training, step selection, role-2/3 evaluation, and the three 52,871-row GNPS
# embedding archives already completed. This job never retrains or encodes;
# one GPU is declared solely because this cluster's submission policy requires it.
#SBATCH --job-name=glm_lw_gnps_fix
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/glm_lw_gnps_fix_%j.out
#SBATCH --error=data/validation/glm_lw_gnps_fix_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT="data/validation/GLM_chemaware_listwise_two_arm/run_2347317"
GNPS_BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
PROTECTED="$OUT/protected_corpus"
PROTECTED_STAGE="$OUT/.protected_corpus_recovery_staging"

if [[ -f "$PROTECTED/SHA256SUMS" ]]; then
  echo "GLM_LISTWISE_GNPS_RECOVERY_ALREADY_COMPLETE: $PROTECTED"
  exit 0
fi
[[ ! -e "$PROTECTED" ]] || {
  echo "Incomplete protected corpus already exists: $PROTECTED" >&2; exit 3;
}

required=(
  "$OUT/listwise_pool/groups.npz"
  "$OUT/listwise_pool/groups.json"
  "$OUT/arm1_training/report.json"
  "$OUT/arm2_training/report.json"
  "$OUT/eval_selection.json"
  "$OUT/eval_confirmation.json"
  "$OUT/eval_arm_pair_selection.json"
  "$OUT/eval_arm_pair_confirmation.json"
  "$OUT/shared_step_selection.json"
  "$OUT/shared_step_selection.step.txt"
  "$OUT/shared_step_selection.arm1_checkpoint.txt"
  "$OUT/shared_step_selection.arm2_checkpoint.txt"
  "$OUT/spectrum_free_baseline.json"
  "$OUT/gnps_embeddings/phasea_embeddings.npy"
  "$OUT/gnps_embeddings/arm1_selected_embeddings.npy"
  "$OUT/gnps_embeddings/arm2_selected_embeddings.npy"
  "$GNPS_BENCHMARK/manifest.csv.gz"
  "$GNPS_BENCHMARK/panel_identity_disjoint.npz"
  "$GNPS_BENCHMARK/panel_formula_disjoint.npz"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "GLM_LISTWISE_RECOVERY_MISSING: $path" >&2; exit 2; }
done

ARM1_CKPT="$(tr -d '\r\n' < "$OUT/shared_step_selection.arm1_checkpoint.txt")"
ARM2_CKPT="$(tr -d '\r\n' < "$OUT/shared_step_selection.arm2_checkpoint.txt")"
[[ -f "$ARM1_CKPT" && -f "$ARM2_CKPT" ]] || {
  echo "Selected checkpoint is absent" >&2; exit 2;
}

python -m py_compile tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
python - "$OUT" <<'PYEOF'
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1]) / "gnps_embeddings"
for name in ("phasea", "arm1_selected", "arm2_selected"):
    path = root / f"{name}_embeddings.npy"
    with np.load(path, allow_pickle=False) as body:
        if set(body.files) != {"rows", "embeddings"}:
            raise RuntimeError(f"malformed recovered GNPS archive: {path}")
        if body["embeddings"].shape != (52871, 1024):
            raise RuntimeError(f"unexpected recovered GNPS shape: {path}")
print("GLM_LISTWISE_RECOVERY_INPUTS_PASS: three complete mis-suffixed NPZ archives")
PYEOF

mkdir -p "$PROTECTED_STAGE"
link_once() {
  local source="$1" destination="$2"
  if [[ -e "$destination" ]]; then
    [[ "$source" -ef "$destination" ]] || {
      echo "Protection collision: $destination" >&2; exit 4;
    }
  else
    ln "$source" "$destination"
  fi
}
link_once "$OUT/listwise_pool/groups.npz" "$PROTECTED_STAGE/listwise_groups.npz"
link_once "$ARM1_CKPT" "$PROTECTED_STAGE/arm1_selected.ckpt"
link_once "$ARM2_CKPT" "$PROTECTED_STAGE/arm2_selected.ckpt"

run_eval() {
  local destination="$1" baseline="$2" candidate="$3"
  if [[ -f "$destination/report.json" ]]; then
    python - "$destination/report.json" <<'PYEOF'
import json, sys
body = json.load(open(sys.argv[1], encoding="utf-8"))
if body.get("status") != "gnps_gold_silver_10ppm_embedding_evaluation_complete":
    raise RuntimeError(f"incomplete existing GNPS report: {sys.argv[1]}")
PYEOF
    echo "REUSE_COMPLETE_GNPS_EVALUATION: $destination"
    return
  fi
  [[ ! -e "$destination" ]] || {
    echo "Incomplete GNPS destination exists: $destination" >&2; exit 5;
  }
  srun --export=ALL --preserve-env python -u \
    tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
    --benchmark "$GNPS_BENCHMARK" \
    --baseline-embeddings "$baseline" \
    --candidate-embeddings "$candidate" \
    --output "$destination"
}

run_eval "$OUT/gnps_arm1_vs_phasea" \
  "$OUT/gnps_embeddings/phasea_embeddings.npy" \
  "$OUT/gnps_embeddings/arm1_selected_embeddings.npy"
run_eval "$OUT/gnps_arm2_vs_phasea" \
  "$OUT/gnps_embeddings/phasea_embeddings.npy" \
  "$OUT/gnps_embeddings/arm2_selected_embeddings.npy"
run_eval "$OUT/gnps_arm2_vs_arm1" \
  "$OUT/gnps_embeddings/arm1_selected_embeddings.npy" \
  "$OUT/gnps_embeddings/arm2_selected_embeddings.npy"

python - "$OUT" <<'PYEOF'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])

def paired(path, name, reference):
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["name"]: row for row in report["results"]}
    return rows[name][f"paired_vs_{reference}"]

def ci_lower(panel, name, reference):
    return float(paired(panel, name, reference)["formula_cluster_bootstrap_delta_recall1_ci95"][0])

selection = out / "eval_selection.json"
confirmation = out / "eval_confirmation.json"
pair_selection = out / "eval_arm_pair_selection.json"
pair_confirmation = out / "eval_arm_pair_confirmation.json"
selected_meta = json.loads((out / "shared_step_selection.json").read_text(encoding="utf-8"))
selected_arm1_name = selected_meta["selected"]["arm1_name"]
gates = {
    "G1_objective_benefit_arm1": {
        "selection_ci_lower": ci_lower(selection, selected_arm1_name, "phaseA_2pp"),
        "confirmation_ci_lower": ci_lower(confirmation, "arm1_selected", "phaseA_2pp"),
        "passed": ci_lower(selection, selected_arm1_name, "phaseA_2pp") > 0.0
        and ci_lower(confirmation, "arm1_selected", "phaseA_2pp") > 0.0,
    },
    "G2_chemical_attribution_arm2_vs_arm1": {
        "selection_ci_lower": ci_lower(pair_selection, "arm2_selected", "arm1_selected"),
        "confirmation_ci_lower": ci_lower(pair_confirmation, "arm2_selected", "arm1_selected"),
        "passed": ci_lower(pair_selection, "arm2_selected", "arm1_selected") > 0.0
        and ci_lower(pair_confirmation, "arm2_selected", "arm1_selected") > 0.0,
    },
    "G3_protection_role3_regression": {
        "arm1_confirmation_ci_lower": ci_lower(confirmation, "arm1_selected", "phaseA_2pp"),
        "arm2_confirmation_ci_lower": ci_lower(confirmation, "arm2_selected", "phaseA_2pp"),
        "passed": ci_lower(confirmation, "arm1_selected", "phaseA_2pp") >= -0.005
        and ci_lower(confirmation, "arm2_selected", "phaseA_2pp") >= -0.005,
    },
}
summary = {
    "status": "GLM_LISTWISE_TWO_ARM_DECISION_SUMMARY",
    "recovered_after_format_dispatch_fix": True,
    "training_repeated": False,
    "gnps_encoding_repeated": False,
    "gates": gates,
    "interpretation": "gates are reported, never enforced post hoc",
}
(out / "decision_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PYEOF

python - "$OUT/recovery_record.json" <<'PYEOF'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "status": "GLM_LISTWISE_GNPS_FORMAT_RECOVERY_COMPLETE",
    "source_run": "run_2347317",
    "root_cause": "NPZ payload was named .npy and evaluator dispatched by suffix",
    "fix": "content-based NumPy format detection",
    "training_repeated": False,
    "embedding_encoding_repeated": False,
}, indent=2), encoding="utf-8")
PYEOF

cp "$OUT/listwise_pool/groups.json" "$PROTECTED_STAGE/pool_report.json"
cp "$OUT/arm1_training/report.json" "$PROTECTED_STAGE/arm1_training_report.json"
cp "$OUT/arm2_training/report.json" "$PROTECTED_STAGE/arm2_training_report.json"
cp "$OUT/eval_selection.json" "$PROTECTED_STAGE/"
cp "$OUT/eval_confirmation.json" "$PROTECTED_STAGE/"
cp "$OUT/eval_arm_pair_selection.json" "$PROTECTED_STAGE/"
cp "$OUT/eval_arm_pair_confirmation.json" "$PROTECTED_STAGE/"
cp "$OUT/shared_step_selection.json" "$OUT/shared_step_selection.step.txt" "$PROTECTED_STAGE/"
cp "$OUT/shared_step_selection.arm1_checkpoint.txt" "$PROTECTED_STAGE/"
cp "$OUT/shared_step_selection.arm2_checkpoint.txt" "$PROTECTED_STAGE/"
cp "$OUT/decision_summary.json" "$OUT/spectrum_free_baseline.json" "$PROTECTED_STAGE/"
cp "$OUT/recovery_record.json" "$PROTECTED_STAGE/"
cp -a "$OUT/gnps_arm1_vs_phasea" "$PROTECTED_STAGE/"
cp -a "$OUT/gnps_arm2_vs_phasea" "$PROTECTED_STAGE/"
cp -a "$OUT/gnps_arm2_vs_arm1" "$PROTECTED_STAGE/"
(
  cd "$PROTECTED_STAGE"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
mv "$PROTECTED_STAGE" "$PROTECTED"
echo "GLM_LISTWISE_GNPS_RECOVERY_COMPLETE: $PROTECTED"
