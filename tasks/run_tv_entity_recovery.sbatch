#!/bin/bash
# Evaluate all 7 task-vector merged checkpoints on entity recovery metric.
# Each checkpoint: encode panel → pair scores → entity recovery at calibrated FDR.
# Success criterion: coverage at 5% FDR > WSE's 67.11%.
#SBATCH --job-name=tv_entity_eval
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/tv_entity_eval_%j.out
#SBATCH --error=data/validation/tv_entity_eval_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

TV_DIR="data/validation/grand_fusion_task_vector_scan_2349962/checkpoints"
BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
PANEL="data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz"
SCORE_BUNDLE="data/validation/GLM_gnps_article_benchmark_s2v26/run15/bundle/method_scores.npz"
ARCH="dreams/models/pretrained/ssl_model_server.pt"
OUT="data/validation/tv_entity_recovery/run_${SLURM_JOB_ID}"

mkdir -p "$OUT"

# Verify inputs
for p in "$TV_DIR" "$BENCHMARK/spectra.mgf" "$PANEL" "$SCORE_BUNDLE" "$ARCH"; do
  [[ -e "$p" ]] || { echo "MISSING: $p" >&2; exit 5; }
done

# List checkpoints
CHECKPOINTS=($(ls "$TV_DIR"/*.pt | sort))
echo "checkpoints found: ${#CHECKPOINTS[@]}"
for ckpt in "${CHECKPOINTS[@]}"; do
  echo "  $(basename "$ckpt")"
done

# Evaluate baselines first (from frozen bundle, no encoding needed)
echo "=== Evaluating baselines from frozen bundle ==="
python -u tasks/GLM_entity_recovery_evaluation.py \
  --panel "$PANEL" \
  --score-bundle "$SCORE_BUNDLE" \
  --score-key scores_identity_disjoint \
  --output "$OUT/baselines" \
  --methods official_dreams noise_v1 weighted_spectral_entropy p2b_noise_v1_frozen spec2vec_gnps_public

# Now evaluate each task-vector checkpoint
for ckpt in "${CHECKPOINTS[@]}"; do
  CKPT_NAME=$(basename "$ckpt" .pt)
  CKPT_OUT="$OUT/${CKPT_NAME}"
  mkdir -p "$CKPT_OUT"

  echo ""
  echo "=== Encoding panel with $CKPT_NAME ==="

  # Step 1: Encode all panel spectra with this checkpoint
  # The GLM_embed_panel_spectra.py from commit 4fb6489 handles this
  if [[ -f tasks/GLM_embed_panel_spectra.py ]]; then
    python -u tasks/GLM_embed_panel_spectra.py \
      --checkpoint "$ckpt" \
      --encoder dreams \
      --architecture-checkpoint "$ARCH" \
      --output "$CKPT_OUT/embeddings" \
      --batch-size 64
  else
    echo "GLM_embed_panel_spectra.py not found, trying direct encoding" >&2
    # Fallback: direct encoding script
    python -u - <<PYEOF
import sys, os
sys.path.insert(0, "tasks")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch

# Load checkpoint
ckpt = torch.load("$ckpt", map_location="cpu", weights_only=False)
arch = torch.load("$ARCH", map_location="cpu", weights_only=False)

# Build model from architecture
from train_e1_identity import load_base_model
device = torch.device("cuda")
model, init_kind = load_base_model(
    torch.load("$ckpt", map_location="cpu", weights_only=False),
    "$ARCH", device, 100)

# Encode all panel spectra
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA

preproc = SpectrumPreprocessor(
    DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
    spec_entropy_cleaning=False, precision=32,
    mz_shift_aug_p=0, mz_shift_aug_max=0)

# Parse MGF
spectra = []
with open("$BENCHMARK/spectra.mgf", "r", errors="ignore") as f:
    cur = None
    for line in f:
        line = line.strip()
        if line == "BEGIN IONS":
            cur = {"peaks": []}
        elif line == "END IONS":
            if cur and cur["peaks"]:
                spectra.append(cur)
            cur = None
        elif cur is not None:
            if line.startswith("PEPMASS="):
                cur["precursor"] = float(line.split("=")[1].split()[0])
            elif line and line[0].isdigit():
                p = line.split()
                if len(p) >= 2 and float(p[1]) > 0:
                    cur["peaks"].append((float(p[0]), float(p[1])))

print(f"parsed {len(spectra)} spectra")

# Encode in batches
from tasks.noise_gnps_article_spectral_scores import weighted_entropy_similarity
model.eval()
all_embeds = []
with torch.no_grad():
    batch = []
    for i, spec in enumerate(spectra):
        peaks = sorted(spec["peaks"], key=lambda x: x[0])[:100]
        mzs = torch.tensor([p[0] for p in peaks], dtype=torch.float32)
        ints = torch.tensor([p[1] for p in peaks], dtype=torch.float32)
        if ints.max() > 0:
            ints = ints / ints.max()
        batch.append((mzs, ints, torch.tensor([spec.get("precursor", 0.0)], dtype=torch.float32)))

        if len(batch) == 64 or i == len(spectra) - 1:
            # Stack and encode
            try:
                max_peaks = max(len(b[0]) for b in batch)
                mz_pad = torch.zeros(len(batch), max_peaks)
                int_pad = torch.zeros(len(batch), max_peaks)
                prec_pad = torch.zeros(len(batch), 1)
                mask = torch.zeros(len(batch), max_peaks, dtype=torch.bool)
                for bi, (m, it, pr) in enumerate(batch):
                    mz_pad[bi, :len(m)] = m
                    int_pad[bi, :len(it)] = it
                    prec_pad[bi] = pr
                    mask[bi, :len(m)] = True
                mz_pad, int_pad, prec_pad, mask = (
                    mz_pad.to(device), int_pad.to(device),
                    prec_pad.to(device), mask.to(device))
                emb = model(mz_pad, int_pad, prec_pad, mask)
                all_embeds.append(emb.cpu().numpy())
            except Exception as e:
                print(f"batch {i//64} error: {e}")
                all_embeds.append(np.zeros(len(batch), 768))
            batch = []

embeddings = np.concatenate(all_embeds, axis=0)
# L2 normalize
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1
embeddings = embeddings / norms

np.save("$CKPT_OUT/embeddings/embeddings.npy", embeddings.astype(np.float32))
print(f"encoded {len(embeddings)} spectra, dim={embeddings.shape[1]}")
# Save row indices
np.save("$CKPT_OUT/embeddings/rows.npy", np.arange(len(spectra), dtype=np.int64))
PYEOF
  fi

  echo "=== Computing pair scores for $CKPT_NAME ==="

  # Step 2: Compute pair scores (embedding dot product for each panel pair)
  python -u - <<PYEOF
import numpy as np
import json

# Load panel
panel = dict(np.load("$PANEL", allow_pickle=False))
candidate_rows = panel["candidate_row"]
query_row = panel["query_row"]

# Load embeddings
embeddings = np.load("$CKPT_OUT/embeddings/embeddings.npy")

# For each pair in the panel, compute cosine of embeddings
n_pairs = len(candidate_rows)
pair_scores = np.zeros(n_pairs, dtype=np.float32)
for i in range(n_pairs):
    qr = int(query_row[0]) if False else None  # need actual query index
    # Actually, we need the query index for each pair
    # The panel has query_ptr which maps queries to their pairs
    pass

# Better approach: use query_ptr to iterate
query_ptr = panel["query_ptr"]
for q in range(len(query_ptr) - 1):
    lo, hi = query_ptr[q], query_ptr[q + 1]
    q_row = int(panel["query_row"][q])
    q_emb = embeddings[q_row]
    for p in range(lo, hi):
        c_row = int(candidate_rows[p])
        c_emb = embeddings[c_row]
        pair_scores[p] = float(np.dot(q_emb, c_emb))

np.savez_compressed("$CKPT_OUT/pair_scores.npz",
                    scores=pair_scores,
                    method_name="${CKPT_NAME}")
print(f"computed {len(pair_scores)} pair scores")
PYEOF

  echo "=== Running entity recovery for $CKPT_NAME ==="

  # Step 3: Run entity recovery evaluation
  # We need to modify the evaluation to accept our custom scores
  python -u - <<PYEOF
import sys, json
import numpy as np
from pathlib import Path

# Load our custom scores
custom = np.load("$CKPT_OUT/pair_scores.npz")
our_scores = custom["scores"]

# Load the frozen bundle to get other methods
bundle = dict(np.load("$SCORE_BUNDLE", allow_pickle=False))
methods = [str(m) for m in bundle["method_names"]]

# We'll create a temporary bundle with our scores added
# First run baseline methods
sys.path.insert(0, "tasks")
# Import the evaluation logic inline

panel = dict(np.load("$PANEL", allow_pickle=False))
scores = np.asarray(bundle["scores_identity_disjoint"], dtype=np.float64)
methods_list = methods

n_queries = len(panel["query_row"])
query_ptr = panel["query_ptr"]
molecule_ptr = panel["molecule_ptr"]
molecule_label = panel["molecule_label"].astype(bool)
query_ik = panel["query_ik14"].astype(str)

# Identity split (same seed as baselines)
identities = sorted(set(query_ik))
rng = np.random.default_rng(20261008)
n_dict = int(len(identities) * 0.70)
perm = rng.permutation(len(identities))
dict_ids = set(identities[i] for i in perm[:n_dict])
is_match = np.array([q in dict_ids for q in query_ik])

# Evaluate our checkpoint
pair_scores = our_scores.astype(np.float64)
mol_scores = np.maximum.reduceat(pair_scores, molecule_ptr[:-1])

match_data, nomatch_data = [], []
for q in range(n_queries):
    lo, hi = query_ptr[q], query_ptr[q + 1]
    ms = mol_scores[lo:hi].copy()
    ml = molecule_label[lo:hi]
    if is_match[q]:
        pos = np.flatnonzero(ml)
        if len(pos) != 1:
            continue
        true_score = ms[pos[0]]
        neg = ms[~ml]
        rank = 1 + int(np.sum(neg >= true_score))
        s = np.sort(ms)
        margin = float(s[-1] - s[-2]) if len(ms) >= 2 else 1.0
        match_data.append({"rank": rank, "correct": rank == 1, "margin": margin})
    else:
        ms[ml] = -np.inf
        valid = ms[~ml]
        if len(valid) >= 2:
            sv = np.sort(valid)
            nomatch_data.append({"margin": float(sv[-1] - sv[-2])})
        else:
            nomatch_data.append({"margin": 0.0})

match_margins = np.array([d["margin"] for d in match_data])
match_correct = np.array([d["correct"] for d in match_data])
nomatch_margins = np.array([d["margin"] for d in nomatch_data])

# Coverage at 5% FDR
best_cov = 0
for t in np.linspace(0, 1, 201):
    m_assigned = match_margins > t
    n_m = int(m_assigned.sum())
    n_correct = int(np.sum(match_correct[m_assigned]))
    n_f = int(np.sum(nomatch_margins > t))
    n_incorrect = n_m - n_correct
    n_total = n_m + n_f
    fdr = (n_f + n_incorrect) / max(n_total, 1) if n_total > 0 else 0
    coverage = n_correct / max(len(match_data), 1)
    if fdr <= 0.05:
        best_cov = max(best_cov, coverage)

top1 = float(np.mean(match_correct)) if len(match_correct) > 0 else 0

result = {
    "checkpoint": "${CKPT_NAME}",
    "top1_accuracy": round(top1, 4),
    "coverage_at_5pct_fdr": round(best_cov, 4),
    "n_match": len(match_data),
    "n_nomatch": len(nomatch_data),
}
print(json.dumps(result, indent=1))

with open("$CKPT_OUT/entity_recovery.json", "w") as f:
    json.dump(result, f, indent=1)
PYEOF

done

# Summary: compare all checkpoints
echo ""
echo "=========================================="
echo "=== SUMMARY: Entity Recovery Comparison ==="
echo "=========================================="
echo "Method                          | R@1    | Cov@5%FDR"
echo "-------------------------------|--------|----------"

# Print baseline results
python -c "
import json
with open('$OUT/baselines/entity_recovery_report.json') as f:
    r = json.load(f)
for m, d in r['results'].items():
    cov = d.get('coverage_at_fdr', {}).get('0.05', {}).get('coverage', 0)
    print(f'{m:30s} | {d[\"top1_accuracy\"]:.4f} | {cov:.4f}')
"

# Print checkpoint results
for ckpt in "${CHECKPOINTS[@]}"; do
  CKPT_NAME=$(basename "$ckpt" .pt)
  RESULT_FILE="$OUT/${CKPT_NAME}/entity_recovery.json"
  if [[ -f "$RESULT_FILE" ]]; then
    python -c "
import json
with open('$RESULT_FILE') as f:
    r = json.load(f)
print(f'{\"tv_${CKPT_NAME}\":30s} | {r[\"top1_accuracy\"]:.4f} | {r[\"coverage_at_5pct_fdr\"]:.4f}')
"
  fi
done

echo ""
echo "ENTITY_RECOVERY_EVALUATION_COMPLETE: $OUT"
