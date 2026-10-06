#!/bin/bash
# GLM: internal protection-gate evaluation of the seven task-vector merge arms
# (run 2349962) on the corrected-graph fold-0 panel, per the pre-registered
# grand-fusion parameter-layer gates:
#   gate A (non-inferiority): a merge arm must not trail the best constituent
#          by more than 0.10 pp on Recall@1;
#   gate B (advancement):     at least one pre-registered arm must beat the
#          best constituent by >= 0.30 pp with formula-cluster CI lower
#          bound > 0.
# Ten rows: 7 merge checkpoints + official + noise (T1/T3 champion slim) +
# chem (stage-1 normalized slim).  Pure encoding+evaluation; no training.
#SBATCH --job-name=glm_taskvec_gate
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/glm_taskvec_gate_%j.out
#SBATCH --error=data/validation/glm_taskvec_gate_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SCAN="data/validation/grand_fusion_task_vector_scan_2349962"
GRAPH_DIR="data/validation/noise_corrected_candidate_graph_v1_20260906"
GRAPH="$GRAPH_DIR/candidate_graph.npz"
EMBEDDING_CACHE="$GRAPH_DIR/official_embeddings.npz"
MANIFEST="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
MATURE_E8="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_0/final_shared_encoder.pt"
NOISE_SLIM="data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt"
CHEM_SLIM="$SCAN/chem_normalized_slim.pt"

OUT="data/validation/GLM_taskvec_internal_gate/run_${SLURM_JOB_ID}"
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }
mkdir -p "$OUT/evaluation"

required_files=(
  "$GRAPH" "$EMBEDDING_CACHE" "$MANIFEST" "$DATA" "$OFFICIAL"
  "$ARCHITECTURE" "$MATURE_E8" "$NOISE_SLIM" "$CHEM_SLIM"
  "$SCAN/checkpoints/linear_n050_c050.pt"
  "$SCAN/checkpoints/linear_n075_c025.pt"
  "$SCAN/checkpoints/linear_n025_c075.pt"
  "$SCAN/checkpoints/ties_d010_s050.pt"
  "$SCAN/checkpoints/ties_d020_s050.pt"
  "$SCAN/checkpoints/ties_d020_s100.pt"
  "$SCAN/checkpoints/ties_d050_s050.pt"
)
for path in "${required_files[@]}"; do
  [[ -e "$path" ]] || { echo "Missing gate input: $path" >&2; exit 2; }
done

# sha pin of the two constituents used by the scan (fail-closed against drift)
python - "$NOISE_SLIM" "$CHEM_SLIM" <<'PY'
import hashlib, sys
pins = {
    sys.argv[1]: "01ac8c7bfa75d89862e0b1eb2b5ffb297955cffae471fd483ad6c38d13f33beb",
    sys.argv[2]: "83160374c963b8fafbee1552d2e85400ae94ec821cc80112bd2f0e76dd98e201",
}
for path, expected in pins.items():
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise SystemExit(f"constituent drift: {path} {actual} != {expected}")
print("constituent sha pins verified")
PY

evaluate_row() {
  local name="$1" checkpoint="$2"
  python -u tasks/evaluate_noise_dreams_native.py \
    --graph "$GRAPH" --embedding-cache "$EMBEDDING_CACHE" \
    --official-slim-checkpoint "$OFFICIAL" \
    --source-manifest "$MANIFEST" --data "$DATA" \
    --checkpoint "$checkpoint" --architecture-checkpoint "$ARCHITECTURE" \
    --output "$OUT/evaluation/$name" --arm targeted \
    --outer-fold 0 --formula-fold-seed 20260825 \
    --batch-size 64 --bootstrap-resamples 10000 --bootstrap-seed 20261006 \
    --official-finetuned-checkpoint "$OFFICIAL" \
    --mature-e8-checkpoint "$MATURE_E8" \
    --e8-architecture-checkpoint "$ARCHITECTURE" \
    >"$OUT/evaluate_${name}.out" 2>"$OUT/evaluate_${name}.err"
}

evaluate_row official "$OFFICIAL"
evaluate_row noise_constituent "$NOISE_SLIM"
evaluate_row chem_constituent "$CHEM_SLIM"
for arm in linear_n050_c050 linear_n075_c025 linear_n025_c075 \
           ties_d010_s050 ties_d020_s050 ties_d020_s100 ties_d050_s050; do
  evaluate_row "$arm" "$SCAN/checkpoints/${arm}.pt"
done

python -u - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = {}
for name in ("official", "noise_constituent", "chem_constituent",
             "linear_n050_c050", "linear_n075_c025", "linear_n025_c075",
             "ties_d010_s050", "ties_d020_s050", "ties_d020_s100",
             "ties_d050_s050"):
    report = json.loads((out / "evaluation" / name / "report.json")
                        .read_text(encoding="utf-8"))
    rows[name] = report
def recall(name):
    block = rows[name]
    for key in ("retrieval", "metrics", "correct"):
        if isinstance(block, dict) and key in block and "recall@1" in block[key]:
            return float(block[key]["recall@1"])
    raise SystemExit(f"cannot locate recall@1 in {name}")
noise, chem = recall("noise_constituent"), recall("chem_constituent")
best = max(noise, chem)
summary = {"status": "GLM_TASKVEC_INTERNAL_GATE_COMPLETE",
           "best_constituent_recall1": best,
           "noise_constituent_recall1": noise,
           "chem_constituent_recall1": chem,
           "arms": {}}
for name in ("linear_n050_c050", "linear_n075_c025", "linear_n025_c075",
             "ties_d010_s050", "ties_d020_s050", "ties_d020_s100",
             "ties_d050_s050"):
    value = recall(name)
    summary["arms"][name] = {
        "recall1": value,
        "delta_vs_best_constituent_pp": (value - best) * 100.0,
        "non_inferior_within_0p10pp": (value - best) * 100.0 > -0.10,
        "advancement_ge_0p30pp": (value - best) * 100.0 >= 0.30,
    }
advancing = [n for n, a in summary["arms"].items() if a["advancement_ge_0p30pp"]]
surviving = [n for n, a in summary["arms"].items() if a["non_inferior_within_0p10pp"]]
summary["arms_passing_non_inferiority"] = surviving
summary["arms_meeting_advancement_gate"] = advancing
summary["pre_registered_decision"] = (
    "advancement gate unmet: retain two independent experts for routing"
    if not advancing else
    "candidate arms advance to the one-shot GNPS confirmation; CI on the "
    "advancing arm is then computed from its per-query table before any "
    "checkpoint is selected")
(out / "gate_summary.json").write_text(json.dumps(summary, indent=2),
                                       encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

find "$OUT" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SHA256SUMS"
echo "COMPLETE: GLM task-vector internal gate -> $OUT"
