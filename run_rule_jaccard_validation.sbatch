#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o rule_jaccard_%j.out
#SBATCH -e rule_jaccard_%j.err
#SBATCH -t 06:00:00

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export MPLBACKEND=Agg

cd /data02/run01/scv7tsl/DreaMS
git pull
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== Task 0: Rule Jaccard vs MCES Validation ====="
echo "Started at $(date)"
echo "Hostname: $(hostname)"
echo ""

# 验证 335 条化学规则的 Jaccard 与分子结构 MCES 是否正相关
# 输出: data/validation/rule_mces_correlation/

python -u tasks/validate_rule_jaccard_mces.py

echo ""
echo "===== Job finished at $(date) ====="
echo ""
echo "Output files:"
echo "  data/validation/rule_mces_correlation/sampled_pairs.csv"
echo "  data/validation/rule_mces_correlation/pair_mces_jaccard.csv"
echo "  data/validation/rule_mces_correlation/correlation_report.json"
echo "  data/validation/rule_mces_correlation/rule_jaccard_vs_mces.png"
echo ""
echo "Check key result:"
echo "  cat data/validation/rule_mces_correlation/correlation_report.json | python -c 'import json,sys; d=json.load(sys.stdin); print(f\"Pearson r={d[\"pearson_r\"]:.4f} Spearman rho={d[\"spearman_rho\"]:.4f}\")'"
