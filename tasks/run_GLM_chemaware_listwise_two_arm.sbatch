#!/bin/bash
# GLM listwise two-arm decisive experiment (pre-registered 2026-09-30).
# docs/GLM_CHEMAWARE_LISTWISE_TWO_ARM_PREREGISTRATION_20260930.md froze every
# parameter below BEFORE this submission; the no-amendment rule applies.
# Arm 1 (zero margins) and Arm 2 (sealed V16 chemical relation margins) share
# one hash-anchored pool, one protected Phase-A initialization, one seed and
# one schedule; they differ only in the margin vector.  The chain then runs
# role-2 selection, role-3 confirmation, the arm-pair attribution evaluation,
# GNPS gold/silver transfer and the spectrum-free candidate baseline.  No
# release decision is automated: G1/G2/G3 are reported, not enforced here.
#SBATCH --job-name=glm_listwise_2arm
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/glm_listwise_2arm_%j.out
#SBATCH --error=data/validation/glm_listwise_2arm_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
BASE="data/validation/chemaware_high_coverage_native/run_2340524"
TRAIN_EVIDENCE="$BASE/evidence/train_triplet_evidence.npz"
SELECTION_EVIDENCE="$BASE/evidence/selection_triplet_evidence.npz"
CONFIRMATION_EVIDENCE="$BASE/evidence/confirmation_triplet_evidence.npz"
PHASEA_RUN="data/validation/chemaware_online_role_budget_native/run_2345481"
PHASEA="$PHASEA_RUN/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt"
GRAPH_LEDGER="data/validation/chemaware_fragment_graph_native/run_2346306/qualified_source"
MASSBANK_LEDGER="data/validation/chemaware_massbank_candidate_source_all_qualified_v1_20260928"
REUSABLE_CACHE="data/validation/chemaware_fragment_graph_native/run_2346306/phasea_embeddings"
DIAGNOSTIC="data/validation/GLM_chemaware_v16_union/run_2347164/evidence_class_diagnostic_v2.json"
GNPS_BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
MANIFEST="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
OUT="data/validation/GLM_chemaware_listwise_two_arm/run_${SLURM_JOB_ID}"

[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT"

required_files=(
  "$OFFICIAL" "$ARCHITECTURE" "$DATA" "$MANIFEST" "$PHASEA"
  "$TRAIN_EVIDENCE" "$SELECTION_EVIDENCE" "$CONFIRMATION_EVIDENCE"
  "$GRAPH_LEDGER/report.json" "$GRAPH_LEDGER/candidate_scores.tsv"
  "$MASSBANK_LEDGER/report.json" "$MASSBANK_LEDGER/candidate_scores.tsv"
  "$REUSABLE_CACHE/rows.npy" "$REUSABLE_CACHE/embeddings_f32.npy"
  "$REUSABLE_CACHE/report.json" "$DIAGNOSTIC"
  "$GNPS_BENCHMARK/manifest.csv.gz"
  "$GNPS_BENCHMARK/panel_identity_disjoint.npz"
  "$GNPS_BENCHMARK/panel_formula_disjoint.npz"
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "GLM_LISTWISE_MISSING_INPUT: $path" >&2; exit 2; }
done

# Contract tests (tasks/test_GLM_*) are a LOCAL pre-submission discipline;
# the server dreams environment intentionally carries no pytest.  The server
# preflight below is syntax-only; every semantic contract (panel disjointness,
# sealed V16 relation anchors, pool/trainer hash identity) is enforced
# fail-closed inside the builders and trainers themselves.
python -m py_compile \
  tasks/GLM_build_chemaware_listwise_two_arm_pools.py \
  tasks/GLM_chemaware_listwise_loss.py \
  tasks/GLM_train_chemaware_listwise.py \
  tasks/GLM_select_chemaware_listwise_shared_step.py \
  tasks/GLM_spectrum_free_candidate_baseline.py

python -u tasks/GLM_build_chemaware_listwise_two_arm_pools.py \
  --manifest "$MANIFEST" --data "$DATA" \
  --train-evidence "$TRAIN_EVIDENCE" \
  --selection-evidence "$SELECTION_EVIDENCE" \
  --confirmation-evidence "$CONFIRMATION_EVIDENCE" \
  --source-ledger "$GRAPH_LEDGER" --source-ledger "$MASSBANK_LEDGER" \
  --embedding-rows "$REUSABLE_CACHE/rows.npy" \
  --phasea-embeddings "$REUSABLE_CACHE/embeddings_f32.npy" \
  --embedding-report "$REUSABLE_CACHE/report.json" \
  --geometry-checkpoint "$PHASEA" \
  --evidence-class-diagnostic "$DIAGNOSTIC" \
  --delta-main 0.05 --delta-contested 0.025 \
  --maximum-candidates-per-query 8 \
  --maximum-reference-spectra-per-molecule 3 \
  --expected-qualified-relations 1684 \
  --expected-admitted-relations 1202 \
  --expected-uncontested-relations 602 \
  --expected-recovered-relations 600 \
  --expected-contested-relations 482 \
  --output "$OUT/listwise_pool/groups.npz"

POOL_META="$(python -c \
  "import json; r=json.load(open('$OUT/listwise_pool/groups.json')); print(r['shared_arrays_sha256'], r['margins']['arm2_molecules_with_main_margin'] + r['margins']['arm2_molecules_with_contested_margin'])")"
read -r SHARED_SHA NONZERO_MARGINS <<< "$POOL_META"
echo "GLM_LISTWISE_POOL: shared_sha=$SHARED_SHA nonzero_arm2_margins=$NONZERO_MARGINS"

for ARM in arm1 arm2; do
  srun --export=ALL --preserve-env python -u tasks/GLM_train_chemaware_listwise.py \
    --pool "$OUT/listwise_pool/groups.npz" \
    --arm "$ARM" \
    --data "$DATA" \
    --official-checkpoint "$PHASEA" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$OUT/${ARM}_training" \
    --expected-shared-sha256 "$SHARED_SHA" \
    --expected-nonzero-margins "$NONZERO_MARGINS" \
    --tau 0.1 --temperature 0.05 \
    --lr 2e-6 --weight-decay 0 \
    --batch-size 1 --max-steps 3600 --max-epochs 2 \
    --save-every-n-steps 900 --maximum-spectra-per-batch 32 \
    --n-highest-peaks 100 \
    --num-workers 0 --seed 3407 --device cuda
done

checkpoint_args=(
  --checkpoint "official=$OFFICIAL"
  --checkpoint "phaseA_2pp=$PHASEA"
  --checkpoint "arm1_step900=$OUT/arm1_training/step-000900.ckpt"
  --checkpoint "arm1_step1800=$OUT/arm1_training/step-001800.ckpt"
  --checkpoint "arm1_step2700=$OUT/arm1_training/step-002700.ckpt"
  --checkpoint "arm1_step3600=$OUT/arm1_training/step-003600.ckpt"
  --checkpoint "arm2_step900=$OUT/arm2_training/step-000900.ckpt"
  --checkpoint "arm2_step1800=$OUT/arm2_training/step-001800.ckpt"
  --checkpoint "arm2_step2700=$OUT/arm2_training/step-002700.ckpt"
  --checkpoint "arm2_step3600=$OUT/arm2_training/step-003600.ckpt"
)

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --manifest "$MANIFEST" \
  --policy "$SELECTION_EVIDENCE" \
  --data "$DATA" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --official-checkpoint "$OFFICIAL" \
  "${checkpoint_args[@]}" \
  --formula-role 2 \
  --output "$OUT/eval_selection.json" \
  --paired-reference phaseA_2pp \
  --device cuda

python -u tasks/GLM_select_chemaware_listwise_shared_step.py \
  --evaluation "$OUT/eval_selection.json" \
  --expected-step 900 --expected-step 1800 \
  --expected-step 2700 --expected-step 3600 \
  --output "$OUT/shared_step_selection.json"
ARM1_CKPT="$(tr -d '\r\n' < "$OUT/shared_step_selection.arm1_checkpoint.txt")"
ARM2_CKPT="$(tr -d '\r\n' < "$OUT/shared_step_selection.arm2_checkpoint.txt")"
SHARED_STEP="$(tr -d '\r\n' < "$OUT/shared_step_selection.step.txt")"
echo "GLM_LISTWISE_SELECTED: shared_step=$SHARED_STEP arm1=$ARM1_CKPT arm2=$ARM2_CKPT"

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --manifest "$MANIFEST" \
  --policy "$CONFIRMATION_EVIDENCE" \
  --data "$DATA" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --official-checkpoint "$OFFICIAL" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "phaseA_2pp=$PHASEA" \
  --checkpoint "arm1_selected=$ARM1_CKPT" \
  --checkpoint "arm2_selected=$ARM2_CKPT" \
  --formula-role 3 \
  --output "$OUT/eval_confirmation.json" \
  --paired-reference phaseA_2pp \
  --device cuda

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --manifest "$MANIFEST" \
  --policy "$SELECTION_EVIDENCE" \
  --data "$DATA" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --official-checkpoint "$OFFICIAL" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "arm1_selected=$ARM1_CKPT" \
  --checkpoint "arm2_selected=$ARM2_CKPT" \
  --formula-role 2 \
  --output "$OUT/eval_arm_pair_selection.json" \
  --paired-reference arm1_selected \
  --device cuda

srun --export=ALL --preserve-env python -u tasks/evaluate_chemaware_v2_direct_triplet.py \
  --manifest "$MANIFEST" \
  --policy "$CONFIRMATION_EVIDENCE" \
  --data "$DATA" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --official-checkpoint "$OFFICIAL" \
  --checkpoint "official=$OFFICIAL" \
  --checkpoint "arm1_selected=$ARM1_CKPT" \
  --checkpoint "arm2_selected=$ARM2_CKPT" \
  --formula-role 3 \
  --output "$OUT/eval_arm_pair_confirmation.json" \
  --paired-reference arm1_selected \
  --device cuda

python -u tasks/GLM_spectrum_free_candidate_baseline.py \
  --graph-panel "gnps_identity_disjoint=$GNPS_BENCHMARK/panel_identity_disjoint.npz" \
  --graph-panel "gnps_formula_disjoint=$GNPS_BENCHMARK/panel_formula_disjoint.npz" \
  --manifest "$MANIFEST" \
  --manifest-panel "role2_selection=$SELECTION_EVIDENCE" \
  --manifest-panel "role3_confirmation=$CONFIRMATION_EVIDENCE" \
  --output "$OUT/spectrum_free_baseline.json"

mkdir -p "$OUT/gnps_embeddings"
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$ARM1_CKPT" --output "$OUT/gnps_embeddings/arm1_selected_slim.pt"
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$ARM2_CKPT" --output "$OUT/gnps_embeddings/arm2_selected_slim.pt"
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$PHASEA" --output "$OUT/gnps_embeddings/phasea_slim.pt"

for MODEL in arm1_selected arm2_selected phasea; do
  srun --export=ALL --preserve-env python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
    --benchmark "$GNPS_BENCHMARK" \
    --checkpoint "$OUT/gnps_embeddings/${MODEL}_slim.pt" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$OUT/gnps_embeddings/${MODEL}_embeddings.npz" \
    --label "$MODEL"
done

srun --export=ALL --preserve-env python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --benchmark "$GNPS_BENCHMARK" \
  --baseline-embeddings "$OUT/gnps_embeddings/phasea_embeddings.npz" \
  --candidate-embeddings "$OUT/gnps_embeddings/arm1_selected_embeddings.npz" \
  --output "$OUT/gnps_arm1_vs_phasea"
srun --export=ALL --preserve-env python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --benchmark "$GNPS_BENCHMARK" \
  --baseline-embeddings "$OUT/gnps_embeddings/phasea_embeddings.npz" \
  --candidate-embeddings "$OUT/gnps_embeddings/arm2_selected_embeddings.npz" \
  --output "$OUT/gnps_arm2_vs_phasea"
srun --export=ALL --preserve-env python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --benchmark "$GNPS_BENCHMARK" \
  --baseline-embeddings "$OUT/gnps_embeddings/arm1_selected_embeddings.npz" \
  --candidate-embeddings "$OUT/gnps_embeddings/arm2_selected_embeddings.npz" \
  --output "$OUT/gnps_arm2_vs_arm1"

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
selected_meta = json.loads(
    (out / "shared_step_selection.json").read_text(encoding="utf-8")
)
selected_arm1_name = selected_meta["selected"]["arm1_name"]

gates = {
    "G1_objective_benefit_arm1": {
        "selection_ci_lower": ci_lower(selection, selected_arm1_name, "phaseA_2pp"),
        "confirmation_ci_lower": ci_lower(confirmation, "arm1_selected", "phaseA_2pp"),
        "passed": (
            ci_lower(selection, selected_arm1_name, "phaseA_2pp") > 0.0
            and ci_lower(confirmation, "arm1_selected", "phaseA_2pp") > 0.0
        ),
    },
    "G2_chemical_attribution_arm2_vs_arm1": {
        "selection_ci_lower": ci_lower(pair_selection, "arm2_selected", "arm1_selected"),
        "confirmation_ci_lower": ci_lower(
            pair_confirmation, "arm2_selected", "arm1_selected",
        ),
        "passed": (
            ci_lower(pair_selection, "arm2_selected", "arm1_selected") > 0.0
            and ci_lower(pair_confirmation, "arm2_selected", "arm1_selected") > 0.0
        ),
    },
    "G3_protection_role3_regression": {
        "arm1_confirmation_ci_lower": ci_lower(
            confirmation, "arm1_selected", "phaseA_2pp",
        ),
        "arm2_confirmation_ci_lower": ci_lower(
            confirmation, "arm2_selected", "phaseA_2pp",
        ),
        "passed": (
            ci_lower(confirmation, "arm1_selected", "phaseA_2pp") >= -0.005
            and ci_lower(confirmation, "arm2_selected", "phaseA_2pp") >= -0.005
        ),
    },
}
summary = {
    "status": "GLM_LISTWISE_TWO_ARM_DECISION_SUMMARY",
    "preregistration": (
        "docs/GLM_CHEMAWARE_LISTWISE_TWO_ARM_PREREGISTRATION_20260930.md"
    ),
    "gates": gates,
    "interpretation": (
        "gates are reported, never enforced post hoc; G1 attributes objective "
        "alignment, G2 attributes chemical evidence, G3 protects role 3"
    ),
}
(out / "decision_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8",
)
print(json.dumps(summary, indent=2))
PYEOF

PROTECTED_STAGE="$OUT/.protected_corpus_staging"
[[ ! -e "$PROTECTED_STAGE" && ! -e "$OUT/protected_corpus" ]] || {
  echo "Protected-corpus output collision" >&2; exit 4;
}
mkdir "$PROTECTED_STAGE"
ln "$OUT/listwise_pool/groups.npz" "$PROTECTED_STAGE/listwise_groups.npz"
ln "$ARM1_CKPT" "$PROTECTED_STAGE/arm1_selected.ckpt"
ln "$ARM2_CKPT" "$PROTECTED_STAGE/arm2_selected.ckpt"
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
cp "$OUT/decision_summary.json" "$PROTECTED_STAGE/"
cp "$OUT/spectrum_free_baseline.json" "$PROTECTED_STAGE/"
cp -a "$OUT/gnps_arm1_vs_phasea" "$PROTECTED_STAGE/"
cp -a "$OUT/gnps_arm2_vs_phasea" "$PROTECTED_STAGE/"
cp -a "$OUT/gnps_arm2_vs_arm1" "$PROTECTED_STAGE/"
(
  cd "$PROTECTED_STAGE"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
mv "$PROTECTED_STAGE" "$OUT/protected_corpus"
echo "GLM_LISTWISE_TWO_ARM_COMPLETE: $OUT"
