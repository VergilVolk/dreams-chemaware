#!/bin/bash
#SBATCH --gpus=1
#SBATCH -o baseline_%j.out
#SBATCH -e baseline_%j.err
#SBATCH -t 01:00:00

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== Zero-shot Similarity (Figure 4a) ====="
python -u baseline_zeroshot.py --n_spectra 2000 --n_pairs 4000 \
    --output_dir ./baseline_results

echo "===== Done at $(date) ====="
