"""Router with raw-spectrum features: the last untried honest channel.

router_v3's features were score-derived (gaps, top1 percentiles, agreement,
consensus, n_mol, precursor mz). This experiment adds features computed
from the query spectrum ITSELF (peak count, intensity distribution, m/z
coverage) -- an information channel that does not depend on any method's
output. Feature-set ablation under the identical leak-free protocol:

  S   spectrum features only
  C   score-derived features only (router_v3 baseline)
  S+C both

If S+C does not beat C beyond noise, the routing negative result is closed
across information channels. Splits, model, and CIs identical to router_v3.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GLM_RUN_DIR",
                      "data/validation/GLM_gnps_article_benchmark_s2v26/run15")
from GLM_truthblind_fusion_analysis import load_panel, per_query_stats, r1  # noqa: E402
from GLM_router_v2 import build_features  # noqa: E402
from GLM_router_v3 import top1_molecule  # noqa: E402
from GLM_score_challenger_models_on_gnps import parse_mgf_used  # noqa: E402

BENCH = ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1"
PANELS = ("identity_disjoint", "formula_disjoint")
BOOT_N = 10000
RNG = np.random.default_rng(20261007)


def spectrum_features(mz: np.ndarray, inten: np.ndarray,
                      precursor: float) -> list[float]:
    n = len(mz)
    w = inten / max(inten.sum(), 1e-12)
    ent = -float(np.sum(w[w > 0] * np.log(w[w > 0]))) / max(np.log(n), 1e-9)
    order = np.argsort(-inten)
    top1_share = float(inten[order[0]] / max(inten.sum(), 1e-12))
    top5_share = float(inten[order[:5]].sum() / max(inten.sum(), 1e-12))
    span = float(mz.max() - mz.min()) if n else 0.0
    frac_below_prec = float((mz < precursor).mean()) if n else 0.0
    wmean_mz = float((mz * w).sum())
    med_mz = float(np.median(mz)) if n else 0.0
    med_inten = float(np.median(inten)) if n else 0.0
    max_inten = float(inten.max()) if n else 0.0
    lo = float((mz < precursor / 3).mean()) if n else 0.0
    mid = float(((mz >= precursor / 3) & (mz < 2 * precursor / 3)).mean()) if n else 0.0
    return [float(n), np.log10(max(inten.sum(), 1e-12)), ent, top1_share,
            top5_share, span, frac_below_prec, wmean_mz, med_mz, med_inten,
            max_inten, lo, mid]


def paired_bootstrap(a: np.ndarray, b: np.ndarray) -> list[float]:
    n = len(a)
    idx = RNG.integers(0, n, size=(BOOT_N, n))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return [round(float(lo * 100), 2), round(float(hi * 100), 2)]


def main() -> None:
    # ---- panel data -----------------------------------------------------
    data = {}
    wanted_rows = set()
    for panel in PANELS:
        g, methods, mol = load_panel(panel)
        rank, gap, top1 = per_query_stats(g, mol)
        mz_arr = np.load(Path("data/validation/"
                              "GLM_gnps_identity_panel_reconstruction")
                         / f"panel_{panel}.npz")["query_precursor_mz"]
        data[panel] = dict(graph=g, methods=methods, rank=rank, gap=gap,
                           top1=top1, mol=mol.astype(np.float64),
                           n_mol=np.diff(g.query_ptr).astype(np.float64),
                           mz=mz_arr.astype(np.float64),
                           t1m=top1_molecule(g, mol),
                           q_rows=g.query_row.astype(int))
        wanted_rows.update(data[panel]["q_rows"].tolist())

    print(f"query spectra needed: {len(wanted_rows):,}")
    spectra = parse_mgf_used(BENCH / "spectra.mgf", wanted_rows)
    print(f"parsed: {len(spectra):,}")
    for panel in PANELS:
        feats = np.zeros((len(data[panel]["q_rows"]), 13), dtype=np.float64)
        for i, row in enumerate(data[panel]["q_rows"]):
            mz, inten, prec = spectra[int(row)]
            feats[i] = spectrum_features(np.asarray(mz, dtype=np.float64),
                                         np.asarray(inten, dtype=np.float64),
                                         float(prec))
        data[panel]["S"] = feats
        data[panel]["C"] = build_features(data[panel], data[panel]["t1m"])
        data[panel]["SC"] = np.column_stack([data[panel]["S"],
                                             data[panel]["C"]])

    zi = np.load(Path("data/validation/GLM_gnps_identity_panel_reconstruction/"
                      "panel_identity_disjoint.npz"))
    zf = np.load(Path("data/validation/GLM_gnps_identity_panel_reconstruction/"
                      "panel_formula_disjoint.npz"))
    f_rows = set(map(int, zf["query_row"]))
    f_iks = set(map(str, zf["query_ik14"]))
    id_rows = zi["query_row"]
    id_iks = zi["query_ik14"].astype(str)
    unseen = np.array([(int(r) not in f_rows and k not in f_iks)
                       for r, k in zip(id_rows, id_iks)])
    uniq = np.array(sorted(set(id_iks)))
    half = set(uniq[RNG.permutation(len(uniq))[: len(uniq) // 2]])
    test_c = np.isin(id_iks, list(half))
    splits = {
        "A": ("formula_disjoint",
              np.ones(len(zf["query_row"]), dtype=bool),
              "identity_disjoint", unseen),
        "B": ("identity_disjoint", unseen,
              "formula_disjoint", np.ones(len(zf["query_row"]), dtype=bool)),
        "C": ("identity_disjoint", ~test_c, "identity_disjoint", test_c),
    }

    report = {"status": "GLM_SPECTRUM_ROUTER", "splits": {}}
    for sname, (tr_p, tr_m, te_p, te_m) in splits.items():
        tr, te = data[tr_p], data[te_p]
        methods = tr["methods"]
        M = tr["rank"].shape[0]
        ytr = (tr["rank"] == 1).T[tr_m]
        correct_te = (te["rank"] == 1)[:, te_m]
        best_i = int(np.argmax((te["rank"] == 1).mean(axis=1)))
        best_mask = correct_te[best_i]
        N = int(te_m.sum())
        res = {"_best_single": {"method": methods[best_i],
                                "recall1": r1(best_mask)}}
        for fs in ("S", "C", "SC"):
            Xtr = tr[fs][tr_m]
            Xte = te[fs][te_m]
            probs = np.zeros((N, M))
            for m_i in range(M):
                clf = HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.06, min_samples_leaf=40,
                    l2_regularization=1.0, random_state=20261007)
                clf.fit(Xtr, ytr[:, m_i])
                probs[:, m_i] = clf.predict_proba(Xte)[:, 1]
            pick = probs.argmax(axis=1)
            mask = correct_te[pick, np.arange(N)]
            res[f"router_{fs}"] = {
                "recall1": r1(mask),
                "delta_vs_best_single_pp": round(r1(mask) - r1(best_mask), 2),
                "ci95_vs_best_single": paired_bootstrap(mask, best_mask)}
        report["splits"][sname] = res
        print(f"=== {sname} ===")
        for fs in ("S", "C", "SC"):
            r = res[f"router_{fs}"]
            print(f"  router_{fs}: {r['recall1']:6.2f}%  "
                  f"delta {r['delta_vs_best_single_pp']:+.2f}  "
                  f"CI [{r['ci95_vs_best_single'][0]:+.2f},"
                  f"{r['ci95_vs_best_single'][1]:+.2f}]")

    out = Path("deliverables/GLM_gnps_article_ladder/run15/"
               "spectrum_router.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("written:", out)


if __name__ == "__main__":
    main()
