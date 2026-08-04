#!/bin/bash
#SBATCH --gpus=1
#SBATCH -o triplet_sweep_%j.out
#SBATCH -e triplet_sweep_%j.err
#SBATCH -t 01:00:00

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
git pull
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== T1 Triplet Parameter Sweep ====="
echo "Started at $(date)"
echo "Hostname: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

for combo in \
    "conservative:0.01:0.05:0.1" \
    "v5_experience:0.05:0.02:0.2" \
    "moderate:0.1:0.02:0.2"
do
    IFS=':' read -r name alpha beta margin <<< "$combo"
    dir="./triplet_sweep/${name}"
    mkdir -p "$dir"

    echo ""
    echo "============================================"
    echo "  $name  α=$alpha β=$beta margin=$margin"
    echo "============================================"

    python -u -m dreams.models.chem_aware.train_triplet_t1 \
        --ckpt_path ./dreams/models/pretrained/ssl_model_server.pt \
        --epochs 3 \
        --batch_size 32 \
        --lr 5e-6 \
        --alpha $alpha \
        --beta $beta \
        --margin $margin \
        --save_dir "$dir" \
        --val_every 1 \
        --save_best_only \
        --auc_pairs 500

    echo "  $name done: $(date)"
done

echo ""
echo "===== Sweep finished at $(date) ====="
echo ""
echo "Results:"
for combo in conservative v5_experience moderate; do
    echo -n "  $combo: "
    python -c "
import torch, json
try:
    ckpt = torch.load('./triplet_sweep/${combo}/best.pt', map_location='cpu', weights_only=False)
    h = ckpt.get('history', {})
    best_auc = max(h.get('val_auc', [0]))
    best_sep = max(h.get('val_sep', [0]))
    print(f'AUC={best_auc:.4f}  Sep={best_sep:.4f}  (epoch {h[\"val_auc\"].index(best_auc)+1 if best_auc>0 else \"?\"})')
except Exception as e:
    print(f'ERROR: {e}')
"
done
echo ""
echo "Pick the best combo and run: sbatch run_triplet_t1.sbatch"
