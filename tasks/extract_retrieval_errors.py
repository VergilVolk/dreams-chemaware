"""从 G6/G7 检索里提取真实错误实例（全量 eval，成千上万条），写成文档。

对每个 query（全量 eval 锚）做 10ppm + 同 adduct 检索，逐 query 记录：
  - 正确分子（同 ik14）在 baseline/G6/G7 下的排名 rank 与 top1
  - 若 G7（或 G6）top1 判错，记录「错排到第 1 的分子」是谁、结构、与 query 的关系
  - 关系判定：是否同分子式（FORMULA）、Morgan Tanimoto、错误类型（近异构体/同式异构/diff-formula m/z 巧合）
  - 同分子谱一致性（同 ik14 谱的 pairwise cosine 均值，检验「同分子谱差异大」）
  - query 与错排分子的生物类别标签（嘌呤/氨基酸/含硫/色氨酸/泛酸/核苷酸）

输出：
  data/validation/retrieval_errors/retrieval_errors.csv   —— 全部错误行（G7 top1 错，含 G6 对照）
  data/validation/retrieval_errors/summary.json           —— 按生物类/错误类型的计数

用法（CPU 可跑，全量 10706 锚，3 模型嵌入约 1.5h，请后台跑）：
  python tasks/extract_retrieval_errors.py --device cpu
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402
from step5_gate_eval import load_trained, embed  # noqa: E402
from bio_class import bio_tags  # noqa: E402

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_MANIFEST = ROOT / "tasks/massspecgym_isomers/dataset_manifest.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
G6 = ROOT / "data/validation/noise_isomer_infonce_g6/seed_0/best_infonce.pt"
G7 = ROOT / "data/validation/noise_isomer_infonce_g7/seed_0/best_infonce.pt"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    ap.add_argument("--g6", type=Path, default=G6)
    ap.add_argument("--g7", type=Path, default=G7)
    ap.add_argument("--n-highest-peaks", type=int, default=100)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--ppm-tol", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/validation/retrieval_errors")
    ap.add_argument("--max-eval", type=int, default=0, help="锚子集（0=全量，用于 smoke 测试）")
    return ap.parse_args()


def _decode(x):
    return x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)


def mol_agg_rank(iks, idx, scores, query_ik):
    """分子聚合排名：每候选 ik14 取 max cosine，返回 (correct_rank, top1_ik, top1_row, cos_correct, cos_top1)。"""
    best = {}   # ik -> (cos, row)
    for j, s in zip(idx, scores):
        ik = iks[j]
        if ik not in best or s > best[ik][0]:
            best[ik] = (float(s), int(j))
    order = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
    ranks = [ik for ik, _ in order]
    r = ranks.index(query_ik) + 1 if query_ik in ranks else len(ranks) + 1
    top1_ik, (top1_cos, top1_row) = order[0]
    cos_correct = best[query_ik][0] if query_ik in best else float("nan")
    return r, top1_ik, top1_row, cos_correct, top1_cos


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    t_start = time.time()

    manifest = json.load(open(args.manifest))
    entries = manifest["eval"]
    if args.max_eval > 0 and len(entries) > args.max_eval:
        entries = entries[: args.max_eval]
    n = len(entries)
    print(f"[data] eval 锚 = {n}", flush=True)

    with h5py.File(args.data, "r") as f:
        pmz_all = np.array(f["precursor_mz"][:], dtype=float)
        smiles_all = [_decode(x) for x in f["smiles"][:]]
        formula_all = [_decode(x) for x in f["FORMULA"][:]]
        clean_specs = []
        for e in entries:
            r = e["anchor_row"]
            raw = np.asarray(f["spectrum"][r])
            clean_specs.append(preprocess_spectrum(raw, float(pmz_all[r]), args.n_highest_peaks))

    anchor_rows = [e["anchor_row"] for e in entries]
    iks = np.array([e["ik14"] for e in entries])
    pmzs = np.array([e["precursor_mz"] for e in entries], dtype=float)
    adducts = np.array([e["adduct"] for e in entries])
    smis = [smiles_all[r] for r in anchor_rows]
    formulas = [formula_all[r] for r in anchor_rows]
    tags = [bio_tags(s) for s in smis]

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_cache = {}
    def fp(sm):
        if sm not in fp_cache:
            m = Chem.MolFromSmiles(sm)
            fp_cache[sm] = fpgen.GetFingerprint(m) if m is not None else None
        return fp_cache[sm]

    print("[1] 加载基线...", flush=True)
    base_model, _ = load_base_model(args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks)
    base_model.eval()
    print("[2] 加载 G6...", flush=True)
    g6_model, _ = load_trained(args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks, args.g6)
    print("[3] 加载 G7...", flush=True)
    g7_model, _ = load_trained(args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks, args.g7)

    print("[4] 嵌入基线 (10706)...", flush=True)
    base_emb = embed(base_model, clean_specs, device, args.batch_size).numpy()
    print("[5] 嵌入 G6...", flush=True)
    g6_emb = embed(g6_model, clean_specs, device, args.batch_size).numpy()
    print("[6] 嵌入 G7...", flush=True)
    g7_emb = embed(g7_model, clean_specs, device, args.batch_size).numpy()
    print(f"[embed] {time.time()-t_start:.0f}s", flush=True)

    rows = []
    n_valid = 0
    for qi in range(n):
        ppm_da = args.ppm_tol * 1e-6 * pmzs[qi]
        cand = (np.abs(pmzs - pmzs[qi]) <= ppm_da) & (np.arange(n) != qi) & (adducts == adducts[qi])
        idx = np.where(cand)[0]
        if len(idx) == 0:
            continue
        labels = (iks[idx] == iks[qi]).astype(int)
        if labels.sum() == 0 or (labels == 0).sum() == 0:
            continue
        n_valid += 1

        s_base = (base_emb[qi:qi + 1] * base_emb[idx]).sum(axis=1)
        s_g6 = (g6_emb[qi:qi + 1] * g6_emb[idx]).sum(axis=1)
        s_g7 = (g7_emb[qi:qi + 1] * g7_emb[idx]).sum(axis=1)

        rb, t1b_ik, t1b_row, cb_cos, tb_cos = mol_agg_rank(iks, idx, s_base, iks[qi])
        r6, t16_ik, t16_row, c6_cos, t6_cos = mol_agg_rank(iks, idx, s_g6, iks[qi])
        r7, t17_ik, t17_row, c7_cos, t7_cos = mol_agg_rank(iks, idx, s_g7, iks[qi])

        # 同分子谱一致性：同 ik14 候选谱两两 cosine 均值
        same_rows = idx[labels == 1]
        sm_pair_cos = float("nan")
        if len(same_rows) >= 2:
            sub = base_emb[same_rows]
            cm = sub @ sub.T
            iu = np.triu_indices(len(same_rows), 1)
            sm_pair_cos = float(cm[iu].mean())

        # Record either model's error: otherwise a report of G7 errors alone
        # cannot distinguish corrected baseline errors from newly introduced ones.
        if rb > 1 or r7 > 1:
            if rb == 1 and r7 > 1:
                transition = "new_error_base_correct_g7_wrong"
                ref_ik, ref_row = t17_ik, t17_row
            elif rb > 1 and r7 == 1:
                transition = "corrected_error_base_wrong_g7_correct"
                ref_ik, ref_row = t1b_ik, t1b_row
            elif rb > 1 and r7 > 1:
                transition = "persistent_error"
                ref_ik, ref_row = t17_ik, t17_row
            else:
                raise AssertionError("unreachable retrieval transition")

            t1_formula = formulas[ref_row]
            t1_smiles = smis[ref_row]
            same_formula = (formulas[qi] == t1_formula)
            ta = fp(smis[qi]); tb = fp(t1_smiles)
            tanimoto = float(DataStructs.TanimotoSimilarity(ta, tb)) if (ta is not None and tb is not None) else float("nan")
            if same_formula:
                if tanimoto >= 0.7:
                    err_type = "near_isomer"
                else:
                    err_type = "distant_isomer_same_formula"
            else:
                err_type = "diff_formula_mz_coincidence"

            rows.append({
                "query_ik14": iks[qi], "query_smiles": smis[qi], "query_formula": formulas[qi],
                "query_adduct": adducts[qi], "query_pmz": float(pmzs[qi]), "query_bio": tags[qi],
                "correct_rank_base": rb, "correct_rank_g6": r6, "correct_rank_g7": r7,
                "base_top1_correct": bool(rb == 1), "g6_top1_correct": bool(r6 == 1), "g7_top1_correct": bool(r7 == 1),
                "transition": transition,
                "baseline_top1_ik14": t1b_ik, "g6_top1_ik14": t16_ik, "g7_top1_ik14": t17_ik,
                "reference_wrong_ik14": ref_ik, "reference_wrong_smiles": t1_smiles, "reference_wrong_formula": t1_formula,
                "wrong_top1_bio": bio_tags(t1_smiles),
                "same_formula": bool(same_formula), "morgan_tanimoto": tanimoto, "error_type": err_type,
                "cos_query_correct_base": float(cb_cos), "cos_query_top1_base": float(tb_cos),
                "cos_query_correct_g7": float(c7_cos), "cos_query_top1_g7": float(t7_cos),
                "margin_g7": float(c7_cos - t7_cos),
                "n_same_mol_spectra": int(labels.sum()),
                "same_mol_pairwise_cos_base": sm_pair_cos,
            })

        if qi % 2000 == 0:
            print(f"[retr] {qi}/{n} valid={n_valid} errors={len(rows)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "retrieval_errors.csv"
    # Use the CSV writer: canonical SMILES commonly contain commas.
    cols = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    # 汇总：按 query 生物类 / 错误类型计数
    from collections import Counter
    qbio = Counter(r["query_bio"] for r in rows)
    err = Counter(r["error_type"] for r in rows)
    transitions = Counter(r["transition"] for r in rows)
    wbio = Counter(r["wrong_top1_bio"] for r in rows)
    flip_bad = sum(1 for r in rows if r["base_top1_correct"] and not r["g7_top1_correct"])
    summary = {
        "n_eval_anchors": n, "n_valid_queries": n_valid,
        "n_g7_top1_errors": len(rows),
        "n_flip_bad_base_correct_g7_wrong": flip_bad,
        "query_bio_counts": dict(qbio),
        "wrong_top1_bio_counts": dict(wbio),
        "error_type_counts": dict(err),
        "transition_counts": dict(transitions),
        "error_rate_g7_top1": len(rows) / n_valid if n_valid else 0.0,
        "elapsed_seconds": time.time() - t_start,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 汇总 =====", flush=True)
    print(f"valid queries={n_valid}  G7 top1 错误={len(rows)}  (错误率 {summary['error_rate_g7_top1']:.2%})", flush=True)
    print(f"flip-bad(基线对→G7错)={flip_bad}", flush=True)
    print("query 生物类分布:", dict(qbio), flush=True)
    print("错误类型分布:", dict(err), flush=True)
    print("模型转换分布:", dict(transitions), flush=True)
    print("错排分子生物类分布:", dict(wbio), flush=True)
    print(f"CSV -> {csv_path}", flush=True)
    print(f"summary -> {args.out_dir / 'summary.json'}", flush=True)

    del base_model, g6_model, g7_model
    gc.collect()


if __name__ == "__main__":
    main()
