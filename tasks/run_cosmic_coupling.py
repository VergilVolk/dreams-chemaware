"""Step-1 COSMIC 耦合 —— 单谱化学自洽置信度（谱空间 decoy + 真值校准）端到端。

闭环 = 分数 -> 谱空间 decoy E-value -> 自检索真值 FDR 校准 -> 可比置信度。
模块在 annotation/cosmic/{score,decoys,calibration}.py。

分数 (rule_coherence_scores)：冻结 DreaMS embedding 解码出的 266 概念概率，
与实测质量模式命中的 266 位规则向量做单谱 AUROC（"embedding 的化学读出是否把
实际存在的规则排在不存在之前"）。

decoy：谱空间两种（shuffle_intensity 打乱强度保留 m/z 轴 / shuffle_mz 打乱 m/z
破坏质量模式），结构空间 StructureSpaceDecoy 已预留接口（完全复刻 COSMIC）。

真值校准：全库 MassSpecGym 留一自检索（m/z<=20ppm 硬约束，top-1 InChIKey 是否
等于 query 自身），按分数分桶算 FDR。

**严谨性门（fail-fast，不合就崩，绝不允许静默错误）：**
  1. 规则向量重算一致：本脚本从 hdf5 重算的 335 位规则向量必须与 cache 逐位相等。
  2. embedding 空间一致：模型前向 target embedding 必须复现检索空间缓存
     retrieval_embeddings（cosine >= 0.999），否则 decoy 与 target 不在同一空间、E-value 无效。

数据/资产全部在**检索空间**（headed、100 峰，与 M1/M2 检索同空间；见
tasks/embed_massspecgym_retrieval.py 关于「两套模型不可混用」的说明）：
  probe       = data/validation/cosmic_retrieval/frozen_concept_probe/concept_probe.pt
  labels      = data/validation/double_mapping/spectrum_rule_labels.npz
  embedding   = data/validation/cosmic_retrieval/retrieval_embeddings.npy
  谱图         = data/models/MassSpecGym_MurckoHist_split.hdf5
  模型         = load_embedder(official_embedding_slim.pt, ssl_model_server.pt)

用法 (conda dreams_env)：
    python tasks/run_cosmic_coupling.py --n 200 --n-decoys 1            # 本机 smoke
    python tasks/run_cosmic_coupling.py --n 5000 --n-decoys 20 --device cuda  # 全量

诚实的边界（不承诺必然提高）：
  * 分数是化学自洽度，不是结构注释；高分数 ≠ 库注释正确。
  * shuffle decoy 对 m/z 主导的 DreaMS embedding 是弱 decoy，E-value 可能不竞争
    —— 这正是本脚本要**实测**并如实报告的。
  * 校准曲线若平（分数不携带正确性信号），如实在报告里写出来。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from annotation import embed  # noqa: E402
from annotation._inference import torch_load_compat  # noqa: E402
from annotation.cosmic import (  # noqa: E402
    DECOYS,
    build_truth_fdr_curve,
    decoy_evalue,
    decoy_fraction_at_least,
    rule_coherence_scores,
)
from annotation.params import DEFAULT  # noqa: E402
from annotation.rule_evidence import load_main_rules, spectrum_rule_vector  # noqa: E402

PPM = DEFAULT.ppm_tolerance
EMBED_COS_THRESHOLD = 0.999  # fail-fast: model forward must reproduce the retrieval cache


def load_probe_numpy(probe_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the frozen concept probe as numpy (weight, bias, mean, std, rule_indices)."""
    probe = torch_load_compat(probe_dir / "concept_probe.pt", map_location="cpu")
    state = probe["state_dict"]
    weight = state["weight"].numpy().astype(np.float32)
    bias = state["bias"].numpy().astype(np.float32)
    mean = probe["embedding_mean"].numpy().astype(np.float32)
    std = probe["embedding_std"].numpy().astype(np.float32)
    rule_indices = probe["rule_indices"].numpy().astype(np.int64)
    return weight, bias, mean, std, rule_indices


def self_retrieval_correctness(
    query_emb: np.ndarray,      # [N, D] normalized
    query_mz: np.ndarray,       # [N]
    query_ik: list[str],        # [N]
    lib_emb: np.ndarray,        # [M, D] normalized
    lib_mz: np.ndarray,         # [M]
    lib_ik: list[str],          # [M]
    query_lib_row: np.ndarray,  # [N] each query's own row in the library (exclude)
    ppm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Self-retrieval truth: top-1 InChIKey == query's own, with/without m/z mask.

    Returns (correct_mz, valid_mz, correct_cos). correct_mz/valid_mz use the m/z
    hard constraint (queries with no m/z-matched neighbour are ``valid_mz=False``);
    correct_cos is cosine-only (always valid).
    """
    n = query_emb.shape[0]
    sim = query_emb @ lib_emb.T  # [N, M]
    for k in range(n):
        sim[k, int(query_lib_row[k])] = -np.inf  # leave-one-out
    dppm = np.abs(query_mz[:, None] - lib_mz[None, :]) / np.maximum(
        np.abs(lib_mz[None, :]), 1e-9) * 1e6
    mz_ok = dppm <= ppm
    masked = np.where(mz_ok, sim, -np.inf)
    top1_mz = masked.argmax(axis=1)
    top1_cos = sim.argmax(axis=1)
    valid_mz = np.isfinite(masked[np.arange(n), top1_mz])
    correct_mz = np.array(
        [query_ik[k] == lib_ik[int(top1_mz[k])] for k in range(n)], dtype=bool
    )
    correct_cos = np.array(
        [query_ik[k] == lib_ik[int(top1_cos[k])] for k in range(n)], dtype=bool
    )
    return correct_mz, valid_mz, correct_cos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="number of query spectra (smoke)")
    parser.add_argument("--n-decoys", type=int, default=1, help="decoys per spectrum per type")
    parser.add_argument("--decoy-types", nargs="+", default=["shuffle_intensity", "shuffle_mz"])
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-truth-bins", type=int, default=10)
    parser.add_argument("--verify-n", type=int, default=20,
                        help="target spectra to forward for the embedding-consistency gate")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/validation/double_mapping/spectrum_rule_labels.npz")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/validation/cosmic_retrieval/retrieval_embeddings.npy")
    parser.add_argument("--probe-dir", type=Path, default=ROOT / "data/validation/cosmic_retrieval/frozen_concept_probe")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/cosmic_confidence/smoke")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── 1. load labels cache + embeddings + probe ──
    cache = np.load(args.labels, allow_pickle=False)
    labels_all = cache["labels"].astype(np.uint8)        # [M, 335]
    ik14 = cache["ik14"].astype(str)                     # [M]
    hdf5_row = cache["hdf5_row"].astype(np.int64)        # [M]
    embeddings = np.load(args.embeddings, mmap_mode="r")
    m = int(embeddings.shape[0])                          # cache-order embedded spectra
    labels_all = labels_all[:m]
    ik14 = ik14[:m]
    hdf5_row = hdf5_row[:m]
    lib_emb = np.asarray(embeddings, dtype=np.float32)    # [m, D] cache order
    weight, bias, mean, std, rule_indices = load_probe_numpy(args.probe_dir)
    print(f"[cosmic] labels={m} probe_rules={len(rule_indices)} "
          f"emb={lib_emb.shape} ({time.time()-t0:.0f}s)", flush=True)

    with h5py.File(args.data, "r") as h:
        prec_all = np.asarray(h["precursor_mz"][:], dtype=np.float64)[hdf5_row]
        rng = np.random.default_rng(args.seed)
        subset = np.sort(rng.choice(m, size=min(args.n, m), replace=False))
        # Strip zero-padding (m/z<=0) so shuffle_mz permutes real peaks only. Stripping
        # is embedding-equivalent for spectra with >=100 real peaks (top-100 by intensity
        # excludes padding either way); the consistency gate below proves it.
        spec_rows = [
            np.asarray(h["spectrum"][r], dtype=np.float32)[:, np.asarray(h["spectrum"][r])[0] > 0]
            for r in hdf5_row[subset]
        ]
    lib_mz = prec_all
    lib_ik = ik14.tolist()
    q_emb = lib_emb[subset]
    q_mz = prec_all[subset]
    q_ik = [ik14[i] for i in subset]
    q_lib_row = subset
    q_hits = labels_all[subset]                          # [N, 335]
    print(f"[cosmic] n_query={len(subset)} ({time.time()-t0:.0f}s)", flush=True)

    # ── 2. gate: rule recompute == cache (bit-for-bit) ──
    rules = load_main_rules()
    recomputed = np.stack([
        spectrum_rule_vector(spec_rows[i][0], float(q_mz[i]), rules)
        for i in range(len(subset))
    ]).astype(np.uint8)
    mismatched_bits = int((recomputed != q_hits).sum())
    if mismatched_bits:
        raise RuntimeError(f"rule recompute != cache: {mismatched_bits} bits differ")
    print(f"[cosmic] gate OK: rule recompute == cache ({len(subset)} spectra)", flush=True)

    # ── 3. score (rule coherence) ──
    scores = rule_coherence_scores(
        q_emb, q_hits, rule_indices, weight, bias, mean, std
    )
    n_pos = q_hits[:, rule_indices].sum(axis=1)
    n_degenerate = int((n_pos == 0).sum())
    n_low_evidence = int((n_pos < 5).sum())
    print(f"[cosmic] score percentiles[10,50,90] = "
          f"{np.percentile(scores, [10, 50, 90]).round(4)} "
          f"(0-positive: {n_degenerate}, <5-positive: {n_low_evidence})", flush=True)

    # ── 4. decoy E-value (spectrum-space) ──
    model, w_head, b_head = embed.load_embedder(args.device)

    # gate: model forward must reproduce the retrieval cache (decoy/target same space)
    verify_n = min(args.verify_n, len(subset))
    verify_records = [
        {"peaks": spec_rows[i], "precursor_mz": float(q_mz[i])} for i in range(verify_n)
    ]
    verify_emb = embed.embed_records(
        verify_records, model, w_head, b_head, args.device, batch_size=args.batch_size
    )
    verify_cos = (verify_emb * q_emb[:verify_n]).sum(axis=1)  # both L2-normalized
    if float(verify_cos.min()) < EMBED_COS_THRESHOLD:
        raise RuntimeError(
            f"embedding forward does not reproduce retrieval cache (min cosine {verify_cos.min():.6f} < "
            f"{EMBED_COS_THRESHOLD}); decoy/target spaces inconsistent"
        )
    print(f"[cosmic] gate OK: forward==retrieval cache min cosine {verify_cos.min():.6f} "
          f"median {np.median(verify_cos):.6f} ({verify_n} spectra)", flush=True)

    evalue = {}
    for decoy_name in args.decoy_types:
        gen = DECOYS[decoy_name]
        decoy_records = []
        for i, row in enumerate(spec_rows):
            # per-spectrum seed: decoys independent across spectra (no shared permutation)
            decoy_records.extend(gen.generate(row, float(q_mz[i]), n=args.n_decoys, seed=args.seed + i))
        decoy_emb = embed.embed_records(
            decoy_records, model, w_head, b_head, args.device, batch_size=args.batch_size
        )
        # Recompute the observed rule vector from each decoy's own peaks. For
        # shuffle_intensity the m/z axis is unchanged so this equals the target's
        # vector (correct, just redundant); for shuffle_mz it changes.
        decoy_hits = np.stack([
            spectrum_rule_vector(r["peaks"][0], r["precursor_mz"], rules)
            for r in decoy_records
        ]).astype(np.uint8)
        decoy_scores = rule_coherence_scores(
            decoy_emb, decoy_hits, rule_indices, weight, bias, mean, std
        ).reshape(len(subset), args.n_decoys)
        evalues = np.array([decoy_evalue(scores[k], decoy_scores[k]) for k in range(len(subset))])
        fracs = np.array([decoy_fraction_at_least(scores[k], decoy_scores[k]) for k in range(len(subset))])
        evalue[decoy_name] = {
            "evalue_mean": float(evalues.mean()),
            "evalue_gt_1_fraction": float((evalues > 1).mean()),
            "decoy_score_median": float(np.median(decoy_scores)),
            "decoy_fraction_at_least_mean": float(fracs.mean()),
        }
        print(f"[cosmic] decoy={decoy_name}: decoy score median={np.median(decoy_scores):.4f} "
              f"E-value mean={evalues.mean():.3f} P(decoy>=target)={fracs.mean():.4f}", flush=True)

    # ── 5. truth calibration (full-library self-retrieval) ──
    correct_mz, valid_mz, correct_cos = self_retrieval_correctness(
        q_emb, q_mz, q_ik, lib_emb, lib_mz, lib_ik, q_lib_row, PPM
    )
    n_valid = int(valid_mz.sum())
    print(f"[cosmic] self-retrieval: m/z-valid={n_valid}/{len(subset)} "
          f"FDR_mz={float((~correct_mz[valid_mz]).mean()) if n_valid else float('nan'):.4f} "
          f"FDR_cos={float((~correct_cos).mean()):.4f}", flush=True)

    truth_mz = build_truth_fdr_curve(scores[valid_mz], correct_mz[valid_mz], args.n_truth_bins)
    truth_cos = build_truth_fdr_curve(scores, correct_cos, args.n_truth_bins)
    # gate: no spectrum may be dropped by the binning
    if int(np.sum(truth_mz["count"])) != n_valid:
        raise RuntimeError("truth binning dropped spectra (calibration regression)")
    if int(np.sum(truth_cos["count"])) != len(subset):
        raise RuntimeError("truth binning dropped spectra (calibration regression)")

    # ── 6. report ──
    report = {
        "status": "cosmic_coupling_complete",
        "n_query": int(len(subset)),
        "n_library": int(m),
        "n_decoys_per_type": args.n_decoys,
        "decoy_types": args.decoy_types,
        "gates": {"rule_recompute_matches_cache": True,
                   "embedding_forward_reproduces_retrieval_cache": True},
        "score": {
            "percentiles_10_50_90": np.percentile(scores, [10, 50, 90]).round(4).tolist(),
            "mean": float(scores.mean()),
            "degenerate_0_positive_spectra": n_degenerate,
            "low_evidence_lt5_positive_spectra": n_low_evidence,
        },
        "decoy_evalue": evalue,
        "truth": {
            "mz_constrained": truth_mz,
            "cosine_only": truth_cos,
            "n_mz_valid": n_valid,
            "fdr_mz_overall": float((~correct_mz[valid_mz]).mean()) if n_valid else None,
            "fdr_cos_overall": float((~correct_cos).mean()),
        },
        "sources": {
            "cosmic": "Hoffmann et al. 2022, DOI 10.1038/s41587-021-01045-9",
            "decoy": "Elias & Gygi 2007 DOI 10.1038/nmeth1013; passatutto 2017 DOI 10.1038/s41467-017-01318-5",
        },
        "claim_limit": (
            "score is chemical self-consistency, not structure annotation; "
            "a flat calibration curve (score carries no correctness signal) is reported as-is."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(args.output_dir / "scores.npy", scores.astype(np.float32))
    np.save(args.output_dir / "correct_mz.npy", correct_mz)
    np.save(args.output_dir / "valid_mz.npy", valid_mz)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
