#!/usr/bin/env python
"""Effect-stage tests T1-T3 for MSV000096214 (contract 4).

RUNS ONLY AFTER tasks/audit_msv000096214_feature_table.py reports
pass_to_effect_stage=true. This is the first stage that joins genotype
labels, by tube number, from the frozen readiness manifest.

T1 epistasis collapse: for each tt7xtt4 double D, Spearman(S(D), S(tt4))
vs Spearman(S(D), S(tt7-allele-mean)); prediction holds 3/3.
T2 held-out allele recovery: 15 folds (3 alleles x 5 held-out WT choices);
signature from remaining two alleles vs remaining WT over ALL qualified
features (no selection); 6 binary decisions per fold = 90 total, chance 50%.
T3 topology: paired-allele epistasis deltas and branch separation.

Permutation/bootstrap: feature bootstrap (2000) for d-rho CIs; genotype-label
permutation (2000) keeping 15+5 block structure for T2 null.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DOUBLES = {"tt7_115__tt4_11": "tt7_115_EMS",
           "tt7_8__tt4_11": "tt7_8_SALK_039417",
           "tt7_9__tt4_11": "tt7_9_SALK_seq_064306"}
TT7 = ["tt7_115_EMS", "tt7_8_SALK_039417", "tt7_9_SALK_seq_064306"]
N_BOOT = 2000
N_PERM = 2000
RNG_SEED = 20261012


def genotype_map(manifest: pd.DataFrame) -> dict[str, str]:
    mapping = {}
    study = manifest.loc[manifest["role"] == "study"]
    for row in study.itertuples(index=False):
        mapping[Path(row.filename).stem] = row.genotype
    return mapping


def qualified_features(matrix: pd.DataFrame, genotypes: dict[str, str],
                       min_rep: dict[str, int]) -> np.ndarray:
    wt_cols = [c for c, g in genotypes.items() if g == "WT_Col0" and c in matrix.columns]
    wt_ok = matrix[wt_cols].notna().sum(axis=1) >= 4
    any_genotype_ok = pd.Series(False, index=matrix.index)
    for genotype, threshold in min_rep.items():
        cols = [c for c, g in genotypes.items() if g == genotype and c in matrix.columns]
        if cols:
            any_genotype_ok |= matrix[cols].notna().sum(axis=1) >= threshold
    return (wt_ok & any_genotype_ok).to_numpy()


def genotype_signature(log2: pd.DataFrame, genotype: str, genotypes: dict[str, str],
                       wt_cols: list[str]) -> pd.Series:
    cols = [c for c, g in genotypes.items() if g == genotype and c in log2.columns]
    mutant = np.log2(log2[cols]).median(axis=1)
    wt = np.log2(log2[wt_cols]).median(axis=1)
    return (mutant - wt)


def spearman_pairwise(a: pd.Series, b: pd.Series) -> float:
    both = a.notna() & b.notna()
    if both.sum() < 10:
        return float("nan")
    rho, _ = stats.spearmanr(a[both], b[both])
    return float(rho)


def bootstrap_delta(first: pd.Series, second: pd.Series, third: pd.Series,
                    n_boot: int, rng: np.random.Generator) -> dict[str, float]:
    """Bootstrap CI for rho(first, second) - rho(first, third) over features."""
    frame = pd.concat([first, second, third], axis=1).dropna()
    deltas = []
    n = len(frame)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = frame.iloc[idx]
        r1, _ = stats.spearmanr(sample.iloc[:, 0], sample.iloc[:, 1])
        r2, _ = stats.spearmanr(sample.iloc[:, 0], sample.iloc[:, 2])
        deltas.append(r1 - r2)
    deltas = np.asarray(deltas)
    return {"delta_median": float(np.median(deltas)),
            "delta_ci_low": float(np.percentile(deltas, 2.5)),
            "delta_ci_high": float(np.percentile(deltas, 97.5)),
            "delta_gt_0_frac": float(np.mean(deltas > 0))}


def t2_experiment(log2: pd.DataFrame, genotypes: dict[str, str], rng: np.random.Generator) -> dict:
    wt_cols = [c for c, g in genotypes.items() if g == "WT_Col0" and c in log2.columns]
    tt7_cols = {allele: [c for c, g in genotypes.items() if g == allele and c in log2.columns]
                for allele in TT7}
    decisions: list[bool] = []
    per_fold = []
    for held_allele in TT7:
        train_alleles = [a for a in TT7 if a != held_allele]
        train_tt7 = [c for a in train_alleles for c in tt7_cols[a]]
        for held_wt in wt_cols:
            train_wt = [c for c in wt_cols if c != held_wt]
            tt7_mat = np.log2(log2[train_tt7]).median(axis=1)
            wt_mat = np.log2(log2[train_wt]).median(axis=1)
            w = tt7_mat - wt_mat
            usable = w.notna()
            correct_fold = []
            for sample in tt7_cols[held_allele] + [held_wt]:
                x = np.log2(log2[sample])
                delta = x - wt_mat
                both = delta.notna() & usable
                if both.sum() < 10:
                    continue
                score, _ = stats.spearmanr(delta[both], w[both])
                is_tt7 = sample in tt7_cols[held_allele]
                decisions.append(bool((score > 0) == is_tt7))
                correct_fold.append(bool((score > 0) == is_tt7))
            per_fold.append({"held_allele": held_allele, "held_wt": held_wt,
                             "fold_correct": sum(correct_fold),
                             "fold_total": len(correct_fold)})
    observed = float(np.mean(decisions)) if decisions else float("nan")

    # permutation null: shuffle WT membership among the 20 involved samples
    involved = wt_cols + [c for a in TT7 for c in tt7_cols[a]]
    sub = log2[involved]
    null = []
    labels = np.array([1] * len(wt_cols) + [0] * (len(involved) - len(wt_cols)))
    for _ in range(N_PERM):
        permuted = rng.permutation(labels)
        perm_map = {sample: ("WT" if permuted[i] else "TT7")
                    for i, sample in enumerate(involved)}
        perm_wt = [c for c in involved if perm_map[c] == "WT"]
        perm_tt7 = [c for c in involved if perm_map[c] == "TT7"]
        perm_wt_mat = np.log2(sub[perm_wt]).median(axis=1)
        perm_tt7_mat = np.log2(sub[perm_tt7]).median(axis=1)
        w = perm_tt7_mat - perm_wt_mat
        usable = w.notna()
        perm_decisions = []
        for sample in involved:
            x = np.log2(sub[sample])
            delta = x - perm_wt_mat
            both = delta.notna() & usable
            if both.sum() < 10:
                continue
            score, _ = stats.spearmanr(delta[both], w[both])
            perm_decisions.append(bool((score > 0) == (perm_map[sample] == "TT7")))
        null.append(np.mean(perm_decisions) if perm_decisions else np.nan)
    null = np.asarray(null, dtype=float)
    null = null[~np.isnan(null)]
    p_right = float(np.mean(null >= observed)) if len(null) else float("nan")
    return {"n_decisions": len(decisions),
            "accuracy": observed,
            "binomial_ci95": [float(x) for x in stats.binom.interval(
                0.95, len(decisions), 0.5)] if decisions else [float("nan")] * 2,
            "per_fold": per_fold,
            "permutation_null_mean": float(np.mean(null)),
            "permutation_null_p_right": p_right,
            "n_permutations_used": int(len(null))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/external/msv000096214"))
    parser.add_argument("--consensus-dir", type=Path, default=Path("data/external/msv000096214/consensus"))
    parser.add_argument("--readiness-manifest", type=Path,
                        default=Path("data/validation/msv000096214_readiness_v2/sample_manifest.csv.gz"))
    parser.add_argument("--audit", type=Path, default=Path("data/validation/msv000096214_table_audit/report.json"))
    parser.add_argument("--output", type=Path, default=Path("data/validation/msv000096214_effect_stage"))
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit.get("pass_to_effect_stage"):
        raise SystemExit("EFFECT_STAGE_BLOCKED: feature-table audit did not pass")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    rng = np.random.default_rng(RNG_SEED)
    manifest = pd.read_csv(args.readiness_manifest)
    genotypes = genotype_map(manifest)
    report: dict[str, object] = {
        "status": "MSV000096214_EFFECT_STAGE_T1_T3",
        "data_role": "development/consumed",
        "contract": "docs/GLM_MSV000096214_STRUCTURE_HIDDEN_CONTRACT_20261012.md",
        "n_bootstrap": N_BOOT, "n_permutations": N_PERM, "rng_seed": RNG_SEED,
        "polarity": {},
    }
    for polarity in ("positive", "negative"):
        matrix = pd.read_csv(args.consensus_dir / polarity / "matrix_filled.csv.gz",
                             index_col="group_id")
        genotypes_p = {c: g for c, g in genotypes.items() if c in matrix.columns}
        wt_cols = [c for c, g in genotypes_p.items() if g == "WT_Col0"]
        min_rep = {g: (3 if g == "fls1_3_EMS" else 4)
                   for g in set(genotypes_p.values()) if g != "WT_Col0"}
        qualified = qualified_features(matrix, genotypes_p, min_rep)
        matrix_q = matrix.loc[qualified]
        log2 = np.log2(matrix_q)

        signatures = {g: genotype_signature(log2, g, genotypes_p, wt_cols)
                      for g in set(genotypes_p.values()) if g != "WT_Col0"}
        tt7_mean = pd.concat([signatures[a] for a in TT7], axis=1).mean(axis=1)

        t1 = {}
        for double, paired_allele in DOUBLES.items():
            r_tt4 = spearman_pairwise(signatures[double], signatures["tt4_11_SALK_020583"])
            r_tt7mean = spearman_pairwise(signatures[double], tt7_mean)
            boot = bootstrap_delta(signatures[double],
                                   signatures["tt4_11_SALK_020583"], tt7_mean, N_BOOT, rng)
            t1[double] = {"rho_vs_tt4": r_tt4, "rho_vs_tt7mean": r_tt7mean,
                          "rho_delta": r_tt4 - r_tt7mean,
                          "paired_allele": paired_allele,
                          "bootstrap": boot}
        t1_pass = all(v["rho_delta"] > 0 for v in t1.values())

        t2 = t2_experiment(log2, genotypes_p, rng)

        names = list(signatures.keys())
        pair_rho = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pair_rho[f"{names[i]}|{names[j]}"] = spearman_pairwise(
                    signatures[names[i]], signatures[names[j]])
        tt7_pairs = [v for k, v in pair_rho.items()
                     if k.split("|")[0] in TT7 and k.split("|")[1] in TT7]
        tt7_tt3 = [v for k, v in pair_rho.items()
                   if (k.split("|")[0] in TT7) != (k.split("|")[1] in TT7)
                   and ("tt3_SALK_099848C" in k)]
        tt7_fls1 = [v for k, v in pair_rho.items()
                    if (k.split("|")[0] in TT7) != (k.split("|")[1] in TT7)
                    and ("fls1_3_EMS" in k)]
        t3_paired = {d: {"rho_vs_tt4": spearman_pairwise(signatures[d], signatures["tt4_11_SALK_020583"]),
                         "rho_vs_paired_tt7": spearman_pairwise(signatures[d], signatures[a])}
                     for d, a in DOUBLES.items()}
        t3 = {"paired_allele_deltas": {d: v["rho_vs_tt4"] - v["rho_vs_paired_tt7"]
                                       for d, v in t3_paired.items()},
              "min_paired_delta_gt_0": all(v > 0 for v in
                                           (x["rho_vs_tt4"] - x["rho_vs_paired_tt7"]
                                            for x in t3_paired.values())),
              "median_rho_tt7_pairs": float(np.median(tt7_pairs)) if tt7_pairs else float("nan"),
              "median_rho_tt7_tt3": float(np.nanmedian(tt7_tt3)) if tt7_tt3 else float("nan"),
              "median_rho_tt7_fls1": float(np.nanmedian(tt7_fls1)) if tt7_fls1 else float("nan"),
              "pairwise_rho": pair_rho}

        report["polarity"][polarity] = {
            "n_qualified_features": int(qualified.sum()),
            "n_total_features": int(len(matrix)),
            "T1_epistasis": t1, "T1_pass_3of3": bool(t1_pass),
            "T2_heldout_allele": t2,
            "T3_topology": t3,
        }

    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({p: {"T1_pass": body["T1_pass_3of3"],
                          "T2_acc": body["T2_heldout_allele"]["accuracy"],
                          "T2_p": body["T2_heldout_allele"]["permutation_null_p_right"],
                          "T3_min_delta_gt0": body["T3_topology"]["min_paired_delta_gt_0"]}
                      for p, body in report["polarity"].items()}, indent=2))


if __name__ == "__main__":
    main()
