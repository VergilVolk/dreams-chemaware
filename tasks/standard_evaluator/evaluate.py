"""
Dreams 标准评估器 — 任意 checkpoint vs Dreams 官方，统一出分。

复现之前 eval（忠实复用公式，不重造轮子）：

  ① 相似度三元组（dreams_zeroshot_baseline.py 协议）
     - AUC      : ROC on same-molecule vs different-molecule cosine
     - Pearson r: cos∈[0,1] vs Morgan Tanimoto (r=2, nBits=4096)
     - Spearman ρ: 同上的 rank 相关
  ② 严格 10ppm 检索（step3_noise_consistency_pilot.py 的 retrieval_metrics 协议）
     - 同 adduct + 前体 m/z ±10ppm + 分子(IK14)聚合 → recall@1/5/10 + MRR + macro-AUC

约定：
  - 相似度统一用 F.cosine_similarity（L2 归一化余弦，同 dreams_zeroshot_baseline.py）
  - backbone-only ckpt（68 键）→ emb = 最后一层 CLS token
  - official slim（backbone+head）→ emb = head(CLS)，1024→1024 线性头
  - 谱预处理 = model.spec_preproc（带 ssl 真实 dformat + 用户指定 n_peaks），缓存只存原始 (2,128) 谱

用法（GPU 服务器，conda dreams）：
  python tasks/standard_evaluator/evaluate.py --device cuda                # 全量
  python tasks/standard_evaluator/evaluate.py --smoke --device cpu          # 本地冒烟（快）

评估集（Tanimoto 对 / AUC 对 / 检索集）只依赖数据+seed，与 checkpoint 无关，首次计算后
缓存到 tasks/standard_evaluator/eval_sets.npz（存原始谱），多 checkpoint / 多次运行复用。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

EVAL_DIR = REPO_ROOT / "tasks" / "standard_evaluator"
HDF5_PATH = REPO_ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
SSL_PATH = REPO_ROOT / "dreams/models/pretrained/ssl_model_server.pt"
OFFICIAL_SLIM = REPO_ROOT / "data/e1/official_embedding_slim.pt"

# 默认对比清单。注意两类 embedding 约定不可混比：
#   - head 约定（emb = head(CLS)）：官方微调 + 本次会话的跨条件 M3/M4 + 噪声 pilot
#   - cls  约定（emb = raw CLS，无 head）：零样本 backbone + B 线 triplet
DEFAULT_CHECKPOINTS = [
    ("dreams_zeroshot", str(SSL_PATH)),                    # 零样本 backbone（论文基线，raw CLS）
    ("dreams_official", str(OFFICIAL_SLIM)),               # 官方微调 embedding（head，基准）
    # 本次会话的跨条件微调（head 约定，与官方直接可比）
    ("cc_m4_v3_s1", "data/validation/cross_condition_m4_v3/seed_1/best_m4.pt"),
    ("cc_m4_s1", "data/validation/cross_condition_m4/seed_1/best_m4.pt"),
    ("cc_m3_s0", "data/validation/cross_condition_m3/seed_0/best_m3.pt"),
    ("cc_m3_s1", "data/validation/cross_condition_m3/seed_1/best_m3.pt"),
    ("cc_m3_s2", "data/validation/cross_condition_m3/seed_2/best_m3.pt"),
    ("cc_m3_s3", "data/validation/cross_condition_m3/seed_3/best_m3.pt"),
    ("noise_pilot_s1", "data/validation/noise_consistency_pilot/seed_1/best_noise_consistency.pt"),
    # B 线 hard-negative triplet（backbone-only，raw CLS，与 head 约定不同，单列）
    ("triplet_v5_experience", "triplet_sweep/v5_experience/best.pt"),
    ("triplet_t1", "triplet_t1_checkpoints/best_model.pt"),
    ("triplet_conservative", "triplet_sweep/conservative/best.pt"),
]

# 论文基线（用于 sanity check）
PAPER = {"pearson": 0.634, "spearman": 0.629, "auc": 0.85}

# 评估协议参数（同 dreams_zeroshot_baseline.py）
BIN_SIZE = 0.025
N_BINS = 40
THLD = 2500
RESERVOIR_PER_BIN = 100000
FP_BITS = 4096
AUC_N = 3000
RETRIEVAL_N = 5000   # 严格 10ppm 检索集谱数（val fold [M+H]+ 抽样，固定 seed）

# smoke 参数（本地 CPU 快速自检）
SMOKE = {"n_mol_cap": 3000, "thld": 500, "auc_n": 1000, "retrieval_n": 500}

rng_np = np.random.RandomState(42)
rng_py = random.Random(42)


# --------------------------------------------------------------------------- #
# 模型加载
# --------------------------------------------------------------------------- #
def build_backbone(n_peaks: int, device: torch.device):
    """从 ssl_model_server.pt 重建 DreaMS backbone（同 dreams_zeroshot_baseline.py L60-78）。"""
    from argparse import Namespace
    from dreams.utils.dformats import DataFormatA
    from dreams.utils.data import SpectrumPreprocessor
    from dreams.models.dreams.dreams import DreaMS

    pkg = torch.load(SSL_PATH, map_location="cpu", weights_only=False)
    recon_args = Namespace(**pkg["args"])
    recon_args.dformat = DataFormatA()
    for da in ["max_mz", "max_peaks_n", "max_tbxic_stdev", "min_peaks_n", "min_charge",
               "max_charge", "max_prec_mz", "high_intensity_thld", "min_intensity_ampl",
               "max_ms_level"]:
        if da in pkg["args"]:
            setattr(recon_args.dformat, da, pkg["args"][da])
    recon_args.d_graphormer_params = 0
    sp = SpectrumPreprocessor(dformat=recon_args.dformat, n_highest_peaks=n_peaks)
    model = DreaMS(recon_args, sp)

    state = model.state_dict()
    for k in state:
        if k in pkg["state_dict"] and state[k].shape == pkg["state_dict"][k].shape:
            state[k] = pkg["state_dict"][k].clone()
    model.load_state_dict(state, strict=False)
    return model


def _overlay(model, sd: dict) -> int:
    state = model.state_dict()
    n = 0
    for k in state:
        if k in sd and state[k].shape == sd[k].shape:
            state[k] = sd[k].clone()
            n += 1
    model.load_state_dict(state, strict=False)
    return n


def load_checkpoint(ckpt_path: str, n_peaks: int, device: torch.device):
    """返回 (model, has_head)。支持三种 checkpoint 格式：
      - {state_dict, args}                        ssl_model_server.pt（零样本 backbone）
      - {model_state_dict}                        triplet/contrastive 微调（backbone-only，68 键）
      - {backbone_state_dict, head_state_dict}    official_embedding_slim.pt（backbone+head）
    """
    model = build_backbone(n_peaks, device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    has_head = False
    if "backbone_state_dict" in ck:
        _overlay(model, ck["backbone_state_dict"])
        if "head_state_dict" in ck:
            hsd = ck["head_state_dict"]
            w = hsd["weight"]
            head = nn.Linear(w.shape[1], w.shape[0])
            head.load_state_dict(hsd)
            model.head = head
            has_head = True
    elif "model_state_dict" in ck:
        _overlay(model, ck["model_state_dict"])
    elif "state_dict" in ck:
        _overlay(model, ck["state_dict"])
    else:
        raise ValueError(f"unrecognized checkpoint format: {ckpt_path}")

    model.eval().to(device)
    return model, has_head


@torch.no_grad()
def embed(model, has_head: bool, raw_specs: np.ndarray, device: torch.device, batch_size: int = 32):
    """raw_specs: (N, 2, 128) float32 原始谱 -> (N, d_model) 嵌入。

    预处理用 model.spec_preproc（正确 dformat + n_peaks），同 dreams_zeroshot_baseline.py。
    """
    sp = model.spec_preproc
    out = []
    for s in range(0, len(raw_specs), batch_size):
        batch = []
        for r in raw_specs[s:s + batch_size]:
            pp = sp(r.astype(np.float32), high_form=False)   # (n_peaks, 2)
            batch.append(torch.as_tensor(pp, dtype=torch.float32))
        b = torch.stack(batch).to(device)
        z = model(b, None)[:, 0, :]
        if has_head:
            z = model.head(z)
        out.append(z.cpu())
    return torch.cat(out, dim=0)


# --------------------------------------------------------------------------- #
# 评估集构建（只依赖数据 + seed，与 checkpoint 无关）
# --------------------------------------------------------------------------- #
def decode(arr):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def build_eval_sets(smoke: bool) -> dict:
    cache = EVAL_DIR / ("eval_sets_smoke.npz" if smoke else "eval_sets.npz")
    if cache.exists():
        print(f"[eval-sets] loading cached {cache.name}", flush=True)
        with np.load(cache, allow_pickle=True) as d:
            return {k: d[k] for k in d.files}

    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import AllChem, DataStructs
    RDLogger.DisableLog("rdApp.*")   # 抑制每个分子的 DEPRECATION 刷屏

    n_mol_cap = SMOKE["n_mol_cap"] if smoke else 10**9
    thld = SMOKE["thld"] if smoke else THLD
    auc_n = SMOKE["auc_n"] if smoke else AUC_N
    ret_n = SMOKE["retrieval_n"] if smoke else RETRIEVAL_N

    f = h5py.File(HDF5_PATH, "r")
    all_adducts = decode(f["adduct"][:])
    all_smiles = decode(f["smiles"][:])
    all_inchi = decode(f["INCHIKEY"][:])
    all_folds = decode(f["fold"][:])
    all_pmz = np.array(f["precursor_mz"][:], dtype=np.float64)
    all_spec = f["spectrum"][:].astype(np.float32)   # (N, 2, 128) 一次读入内存，避免压缩 HDF5 随机读
    f.close()

    mh_mask = np.array([a == "[M+H]+" for a in all_adducts])
    mh_idx = np.where(mh_mask)[0]
    mh_smiles = [all_smiles[i] for i in mh_idx]
    mh_inchi = [all_inchi[i][:14] for i in mh_idx]

    smiles_to_spec = defaultdict(list)
    for idx, smi in zip(mh_idx, mh_smiles):
        smiles_to_spec[smi].append(idx)
    unique_smiles = sorted(smiles_to_spec.keys())

    # --- Morgan Tanimoto 分层采样（同 dreams_zeroshot_baseline.py L128-247）---
    fp_map = {}
    for i, smi in enumerate(unique_smiles):
        if i >= n_mol_cap:
            break
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp_map[smi] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, FP_BITS)
    valid_smiles = sorted(fp_map.keys())
    N = len(valid_smiles)
    fp_list = [fp_map[s] for s in valid_smiles]

    bins = {i: [] for i in range(N_BINS)}
    for i in range(N - 1):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[i + 1:]), dtype=np.float64)
        bi = np.minimum((sims / BIN_SIZE).astype(int), N_BINS - 1)
        for b in range(N_BINS):
            if len(bins[b]) >= RESERVOIR_PER_BIN:
                continue
            mask = bi == b
            if not mask.any():
                continue
            idxs = np.where(mask)[0]
            n_space = RESERVOIR_PER_BIN - len(bins[b])
            if len(idxs) > n_space:
                idxs = rng_np.choice(idxs, n_space, replace=False)
            for jo in idxs:
                bins[b].append((valid_smiles[i], valid_smiles[i + 1 + jo], float(sims[jo])))

    # same-molecule pairs (Tanimoto=1.0) -> bin 39
    ik_to_spec = defaultdict(list)
    for idx, ik in zip(mh_idx, mh_inchi):
        ik_to_spec[ik].append(idx)
    multi_iks = {ik: v for ik, v in ik_to_spec.items() if len(v) >= 2}
    for ik, spec_idxs in multi_iks.items():
        if len(bins[39]) >= RESERVOIR_PER_BIN:
            break
        smi = all_smiles[spec_idxs[0]]
        if smi not in fp_map:
            continue
        for si in range(min(len(spec_idxs), 5)):
            for sj in range(si + 1, min(len(spec_idxs), 5)):
                if len(bins[39]) < RESERVOIR_PER_BIN:
                    bins[39].append((smi, smi, 1.0))

    sampled = []
    for b in range(N_BINS):
        cand = bins[b]
        if not cand:
            continue
        idxs = rng_np.choice(len(cand), thld, replace=(len(cand) < thld))
        for i in idxs:
            sampled.append(cand[i])
    rng_np.shuffle(sampled)

    # pair -> (smi, h5_idx) 唯一谱集合
    spec_map = {}          # (smi, h5_idx) -> 原始谱 index
    pair_rows = []         # (spec_i, spec_j, tanimoto)
    for smi_a, smi_b, tan in sampled:
        ia = smiles_to_spec.get(smi_a, [])
        ib = smiles_to_spec.get(smi_b, [])
        if smi_a == smi_b:
            if len(ia) < 2:
                continue
            sa, sb = rng_np.choice(ia, 2, replace=False)
        else:
            if not ia or not ib:
                continue
            sa = rng_np.choice(ia)
            sb = rng_np.choice(ib)
        for key in ((smi_a, int(sa)), (smi_b, int(sb))):
            if key not in spec_map:
                spec_map[key] = len(spec_map)
        pair_rows.append((spec_map[(smi_a, int(sa))], spec_map[(smi_b, int(sb))], tan))

    corr_raw = np.stack([all_spec[h5_idx] for (_, h5_idx) in spec_map])
    corr_i = np.array([r[0] for r in pair_rows])
    corr_j = np.array([r[1] for r in pair_rows])
    corr_tani = np.array([r[2] for r in pair_rows], dtype=np.float64)

    # --- AUC 对（同 dreams_zeroshot_baseline.py L323-360）---
    auc_iks = rng_np.choice(sorted(multi_iks.keys()), min(300, len(multi_iks)), replace=False)
    auc_raw_list = []
    auc_ik_idx = {}
    for ik in auc_iks:
        auc_ik_idx[ik] = []
        for si in multi_iks[ik][:3]:
            try:
                auc_ik_idx[ik].append(len(auc_raw_list))
                auc_raw_list.append(all_spec[si])
            except Exception:
                pass
    ml = [ik for ik in auc_iks if len(auc_ik_idx[ik]) >= 2]
    al = [ik for ik in auc_iks if len(auc_ik_idx[ik]) >= 1]
    pi, pj, lb = [], [], []
    for _ in range(auc_n):
        if ml:
            ik = ml[rng_np.randint(0, len(ml))]
            a, b = rng_np.choice(len(auc_ik_idx[ik]), 2, replace=False)
            pi.append(auc_ik_idx[ik][a]); pj.append(auc_ik_idx[ik][b]); lb.append(1)
    for _ in range(auc_n):
        if len(al) >= 2:
            ia, ib = rng_np.choice(len(al), 2, replace=False)
            if ia != ib:
                pi.append(rng_np.choice(auc_ik_idx[al[ia]]))
                pj.append(rng_np.choice(auc_ik_idx[al[ib]]))
                lb.append(0)

    auc_raw = np.stack(auc_raw_list)

    # --- 检索集：val fold [M+H]+ 抽样 ret_n 张 ---
    val_mask = mh_mask & np.array([fl == "val" for fl in all_folds])
    val_idx = np.where(val_mask)[0]
    if len(val_idx) > ret_n:
        val_idx = rng_np.choice(val_idx, ret_n, replace=False)
    ret_raw = np.stack([all_spec[i] for i in val_idx])
    ret_ik14 = np.array([all_inchi[i][:14] for i in val_idx])
    ret_pmz = all_pmz[val_idx]
    ret_adduct = np.array([all_adducts[i] for i in val_idx])

    out = {
        "corr_raw": corr_raw, "corr_i": corr_i, "corr_j": corr_j, "corr_tani": corr_tani,
        "auc_raw": auc_raw, "auc_pi": np.array(pi), "auc_pj": np.array(pj), "auc_labels": np.array(lb),
        "ret_raw": ret_raw, "ret_ik14": ret_ik14, "ret_pmz": ret_pmz, "ret_adduct": ret_adduct,
    }
    np.savez_compressed(cache, **out)
    print(f"[eval-sets] cached -> {cache.name}", flush=True)
    return out


def query_auc(labels, scores):
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    return float((np.count_nonzero(diff > 0) + 0.5 * np.count_nonzero(diff == 0)) / diff.size)


def retrieval_metrics(emb, pmzs, iks, adducts, ppm_tol=10.0):
    """严格 10ppm 同 adduct 检索（复刻 step3_noise_consistency_pilot.retrieval_metrics）。"""
    aucs, recalls1, mrrs = [], [], []
    for qi in range(len(emb)):
        ppm_da = ppm_tol * 1e-6 * pmzs[qi]
        cand = (np.abs(pmzs - pmzs[qi]) <= ppm_da) & (np.arange(len(emb)) != qi) & (adducts == adducts[qi])
        idx = np.where(cand)[0]
        if len(idx) == 0:
            continue
        labels = (iks[idx] == iks[qi]).astype(int)
        if labels.sum() == 0 or (labels == 0).sum() == 0:
            continue
        scores = (emb[qi:qi + 1] * emb[idx]).sum(axis=1)
        aucs.append(query_auc(labels, scores))
        best = {}
        for j, s in zip(idx, scores):
            ik = iks[j]
            if ik not in best or s > best[ik]:
                best[ik] = float(s)
        order = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        ranks = [ik for ik, _ in order]
        rank = ranks.index(iks[qi]) + 1 if iks[qi] in ranks else len(ranks) + 1
        recalls1.append(1.0 if rank <= 1 else 0.0)
        mrrs.append(1.0 / rank)
    return {
        "n_eligible": len(aucs),
        "macro_auc": float(np.mean(aucs)) if aucs else 0.5,
        "recall1": float(np.mean(recalls1)) if recalls1 else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-peaks", type=int, default=100,
                    help="模型 n_highest_peaks（DreaMS 生产默认 100；论文零样本图用 60）")
    ap.add_argument("--checkpoint", action="append", default=None,
                    help="追加 checkpoint（可多次，格式 name=path）")
    ap.add_argument("--only", action="append", default=None,
                    help="只跑这些 checkpoint（name=path，替换默认清单；可多次）")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--build-only", action="store_true",
                    help="只建评估集缓存（不加载任何 checkpoint 嵌入）就退出；供多 GPU 并行进程复用缓存")
    ap.add_argument("--no-cache", action="store_true", help="重建评估集缓存")
    ap.add_argument("--out", default=str(EVAL_DIR / "results.json"))
    args = ap.parse_args()

    if args.no_cache:
        for fn in ("eval_sets.npz", "eval_sets_smoke.npz"):
            (EVAL_DIR / fn).unlink(missing_ok=True)

    device = torch.device(args.device)
    smoke = args.smoke
    if smoke:
        print("== SMOKE MODE (tiny eval sets, CPU fast check) ==", flush=True)

    def _parse(c):
        if "=" in c:
            return c.split("=", 1)
        return Path(c).stem, c

    if args.only:
        checkpoints = [_parse(c) for c in args.only]
    else:
        checkpoints = list(DEFAULT_CHECKPOINTS)
        for c in (args.checkpoint or []):
            checkpoints.append(_parse(c))

    print(f"[eval-sets] building (smoke={smoke})...", flush=True)
    t0 = time.time()
    es = build_eval_sets(smoke)
    print(f"[eval-sets] done in {time.time()-t0:.0f}s | corr={len(es['corr_i'])} "
          f"auc={len(es['auc_pi'])} ret={len(es['ret_raw'])}", flush=True)

    if args.build_only:
        print("[eval-sets] build-only: cache ready, exiting.", flush=True)
        return

    results = []
    for name, path in checkpoints:
        p = Path(path)
        p = p if p.is_absolute() else (REPO_ROOT / p)
        if not p.exists():
            print(f"\n[{name}] SKIP (missing: {p})", flush=True)
            continue
        print(f"\n[{name}] loading {p.name} ...", flush=True)
        model, has_head = load_checkpoint(str(p), args.n_peaks, device)
        emb_auc = embed(model, has_head, es["auc_raw"], device)
        emb_corr = embed(model, has_head, es["corr_raw"], device)
        emb_ret = embed(model, has_head, es["ret_raw"], device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # AUC
        auc_cos = F.cosine_similarity(emb_auc[es["auc_pi"]], emb_auc[es["auc_pj"]], dim=-1).numpy()
        fpr, tpr, _ = metrics.roc_curve(es["auc_labels"], auc_cos)
        auc = float(metrics.auc(fpr, tpr))
        cp = float(auc_cos[es["auc_labels"] == 1].mean())
        cn = float(auc_cos[es["auc_labels"] == 0].mean())

        # Pearson / Spearman
        corr_cos = F.cosine_similarity(emb_corr[es["corr_i"]], emb_corr[es["corr_j"]], dim=-1).numpy()
        cos_clip = np.clip(corr_cos, 0, 1)
        r, _ = pearsonr(cos_clip, es["corr_tani"])
        rho, _ = spearmanr(cos_clip, es["corr_tani"])

        # 检索
        ret = retrieval_metrics(F.normalize(emb_ret, dim=-1).numpy(),
                                es["ret_pmz"], es["ret_ik14"], es["ret_adduct"])

        row = {
            "name": name, "checkpoint": str(p), "has_head": has_head,
            "auc": auc, "pearson_r": float(r), "spearman_rho": float(rho),
            "cos_pos": cp, "cos_neg": cn, "retrieval": ret,
        }
        results.append(row)
        print(f"  AUC={auc:.4f}  Pearson={r:.4f}  Spearman={rho:.4f}  "
              f"cos+={cp:.4f} cos-={cn:.4f} | R@1={ret['recall1']:.4f} MRR={ret['mrr']:.4f} "
              f"macroAUC={ret['macro_auc']:.4f} (eligible={ret['n_eligible']})", flush=True)

    print(f"\n{'='*78}")
    print(f"Dreams 标准评估器 — 汇总（n_peaks={args.n_peaks}, smoke={smoke}）")
    print(f"论文基线: Pearson={PAPER['pearson']} Spearman={PAPER['spearman']} AUC={PAPER['auc']}")
    print(f"  emb=head(CLS) 与 emb=raw CLS 是两种约定，只应在同约定内比较")
    print(f"{'='*78}")
    print(f"{'checkpoint':24s} {'emb':>4s} {'AUC':>6s} {'Pearson':>8s} {'Spearman':>8s} {'R@1':>6s} {'MRR':>6s} {'m-AUC':>6s}")
    for row in results:
        r = row["retrieval"]
        emb = "head" if row["has_head"] else "cls"
        print(f"{row['name']:24s} {emb:>4s} {row['auc']:6.4f} {row['pearson_r']:8.4f} {row['spearman_rho']:8.4f} "
              f"{r['recall1']:6.4f} {r['mrr']:6.4f} {r['macro_auc']:6.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_peaks": args.n_peaks, "smoke": smoke, "paper_baseline": PAPER,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
