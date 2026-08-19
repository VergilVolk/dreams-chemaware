#!/bin/bash
#SBATCH --gpus=2
#SBATCH --time=02:00:00
#SBATCH --output=std_eval_%j.out
#SBATCH --error=std_eval_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ============================================================================
# Dreams 标准评估器：12 个 checkpoint vs Dreams 官方，统一出分。
#   指标（复用之前 eval，非重造）：
#     AUC / Pearson r / Spearman rho —— dreams_zeroshot_baseline.py 协议
#     R@1 / MRR / macro-AUC           —— 严格 10ppm 同 adduct 检索（噪声 pilot 的 retrieval_metrics）
#   两类 embedding 约定不可混比（结果表 emb 列标注）：
#     head = head(CLS)：官方微调 + 跨条件 M3/M4 + 噪声 pilot（base 在官方之上，直接可比）
#     cls  = raw CLS ：零样本 backbone + B 线 triplet（backbone-only，无 head）
#
# 输入依赖（必须已同步到服务器，否则 exit 2）：
#   1. data/models/MassSpecGym_MurckoHist_split.hdf5   # 谱+元数据
#   2. dreams/models/pretrained/ssl_model_server.pt    # 零样本 backbone -> 重建模型 args
#   3. data/e1/official_embedding_slim.pt              # 官方 backbone+head
# 待比较的 checkpoint（data/、triplet_* 是 gitignored，须已在服务器；缺则 SKIP 不致命）：
#   data/validation/cross_condition_m{3,4,4_v3}/seed_*/best_m*.pt
#   data/validation/noise_consistency_pilot/seed_1/best_noise_consistency.pt
#   triplet_sweep/{v5_experience,conservative}/best.pt
#   triplet_t1_checkpoints/best_model.pt
# 生成物：
#   tasks/standard_evaluator/eval_sets.npz   # 评估集缓存（Tanimoto ~500M 对，一次性，可复用）
#   tasks/standard_evaluator/results*.json   # 出分
# ============================================================================

echo "===== Dreams 标准评估器（12 checkpoint vs 官方）====="
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

for REQUIRED in \
    data/models/MassSpecGym_MurckoHist_split.hdf5 \
    dreams/models/pretrained/ssl_model_server.pt \
    data/e1/official_embedding_slim.pt ; do
    if [[ ! -f "$REQUIRED" ]]; then
        echo "MISSING DEPENDENCY: $REQUIRED  (see 输入依赖 above)" >&2
        exit 2
    fi
done
echo "核心 3 依赖齐。"
echo ""

# ---- (0) 一次性建评估集缓存（Tanimoto ~500M 对，纯 CPU，~10-15min；已存在则秒加载）----
python -u tasks/standard_evaluator/evaluate.py --build-only --device cpu

# ---- (1) 12 checkpoint 拆两半，2 GPU 并行（复用上面缓存）----
CUDA_VISIBLE_DEVICES=0 python -u tasks/standard_evaluator/evaluate.py --device cuda \
    --out tasks/standard_evaluator/results_gpu0.json \
    --only dreams_zeroshot=dreams/models/pretrained/ssl_model_server.pt \
    --only dreams_official=data/e1/official_embedding_slim.pt \
    --only cc_m4_v3_s1=data/validation/cross_condition_m4_v3/seed_1/best_m4.pt \
    --only cc_m4_s1=data/validation/cross_condition_m4/seed_1/best_m4.pt \
    --only cc_m3_s0=data/validation/cross_condition_m3/seed_0/best_m3.pt \
    --only triplet_v5_experience=triplet_sweep/v5_experience/best.pt &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python -u tasks/standard_evaluator/evaluate.py --device cuda \
    --out tasks/standard_evaluator/results_gpu1.json \
    --only cc_m3_s1=data/validation/cross_condition_m3/seed_1/best_m3.pt \
    --only cc_m3_s2=data/validation/cross_condition_m3/seed_2/best_m3.pt \
    --only cc_m3_s3=data/validation/cross_condition_m3/seed_3/best_m3.pt \
    --only noise_pilot_s1=data/validation/noise_consistency_pilot/seed_1/best_noise_consistency.pt \
    --only triplet_t1=triplet_t1_checkpoints/best_model.pt \
    --only triplet_conservative=triplet_sweep/conservative/best.pt &
PID1=$!

RC0=0; RC1=0
wait $PID0 || RC0=$?
wait $PID1 || RC1=$?
if [ "$RC0" != "0" ] || [ "$RC1" != "0" ]; then
    echo "评估 FAILED: rc0=$RC0 rc1=$RC1 (见 .out/.err)" >&2
    exit 1
fi

# ---- (2) 合并两 GPU 结果 + 统一出表 ----
python -u - <<'PY'
import json
from pathlib import Path
d = Path("tasks/standard_evaluator")
rows, meta = [], {}
for fn in ("results_gpu0.json", "results_gpu1.json"):
    p = d / fn
    if not p.exists():
        print(f"[merge] WARN missing {fn}"); continue
    j = json.loads(p.read_text(encoding="utf-8"))
    rows += j["results"]
    meta.setdefault("n_peaks", j.get("n_peaks"))
    meta.setdefault("paper_baseline", j.get("paper_baseline"))
(d / "results.json").write_text(
    json.dumps({"n_peaks": meta.get("n_peaks"), "paper_baseline": meta.get("paper_baseline"),
                "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n{'='*80}\n合并汇总（{len(rows)} checkpoint, n_peaks={meta.get('n_peaks')}）")
print(f"论文零样本基线: {meta.get('paper_baseline')}")
print("  emb=head(CLS) 与 emb=raw CLS 是两种约定，只应在同约定内比较")
print(f"{'='*80}")
print(f"{'checkpoint':24s} {'emb':>4s} {'AUC':>6s} {'Pearson':>8s} {'Spearman':>8s} {'R@1':>6s} {'MRR':>6s} {'m-AUC':>6s}")
for r in rows:
    e = "head" if r["has_head"] else "cls"
    rt = r["retrieval"]
    print(f"{r['name']:24s} {e:>4s} {r['auc']:6.4f} {r['pearson_r']:8.4f} {r['spearman_rho']:8.4f} "
          f"{rt['recall1']:6.4f} {rt['mrr']:6.4f} {rt['macro_auc']:6.4f}")
print(f"\nSaved -> tasks/standard_evaluator/results.json")
PY

echo ""
echo "===== DONE $(date) ====="
echo "合并结果: tasks/standard_evaluator/results.json"
