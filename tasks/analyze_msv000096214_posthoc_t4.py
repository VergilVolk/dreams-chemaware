#!/usr/bin/env python
"""Post-hoc biochemical consistency opening T4 for MSV000096214 (contract 4).

RUNS LAST, after T1-T3. This is the only stage that opens MS2 fragment
content. Sets are defined exactly as in the frozen contract:

F (flavonoid-predicted): log2FC(tt4 vs WT) <= -1 with BH-FDR < 0.05
   (Mann-Whitney 5v5) AND log2FC <= -0.5 in >= 2 of 3 tt7xtt4 doubles.
C (unchanged controls): |log2FC| <= 0.25 in every genotype, FDR > 0.5,
   matched to F features within +/-50 Da m/z (up to 2 per F feature).

MS2 linkage: DDA spectra whose precursor matches a feature within 5 ppm and
whose RT (inverse-warped to file scale) is within 20 s of the group median.
Glycosidic neutral losses searched at +/-20 ppm (min 0.01 Da) of
162.0528 / 146.0579 / 308.1107 / 176.0321. Enrichment: Fisher exact,
two-sided. Branch evidence: among tt7-down features (log2FC <= -1, FDR<0.05,
not in F) count those with a +15.9949 Da partner (5 ppm) whose tt7
log2FC >= 0; compare to the same count among C.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms
from scipy import stats
from scipy.stats import mannwhitneyu

LOSSES = [162.0528, 146.0579, 308.1107, 176.0321]
BRANCH_DELTA = 15.9949
PPM = 5.0
RT_LINK = 20.0
LOSS_TOL_PPM = 20.0


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def load_ms2_spectra(path: Path) -> list[dict[str, object]]:
    experiment = oms.MSExperiment()
    loader = oms.MzXMLFile()
    try:
        options = loader.getOptions()
        options.setMSLevels([2])
        loader.setOptions(options)
    except AttributeError:
        pass
    loader.load(str(path), experiment)
    spectra = []
    for spectrum in experiment:
        if spectrum.getMSLevel() != 2:
            continue
        precursor_mz = np.nan
        for precursor in spectrum.getPrecursors():
            precursor_mz = float(precursor.getMZ())
            break
        if np.isnan(precursor_mz):
            continue
        mzs, intensities = spectrum.get_peaks()
        if len(mzs) == 0:
            continue
        keep = np.argsort(-np.asarray(intensities))[:60]
        mzs = np.sort(np.asarray(mzs, dtype=np.float64)[keep])
        spectra.append({"precursor_mz": precursor_mz,
                        "rt": float(spectrum.getRT()),
                        "frag_mz": mzs})
    return spectra


def loss_present(frag_mz: np.ndarray, precursor_mz: float) -> bool:
    for loss in LOSSES:
        tol = max(loss * LOSS_TOL_PPM * 1e-6, 0.01)
        target = precursor_mz - loss
        lo = np.searchsorted(frag_mz, target - tol)
        hi = np.searchsorted(frag_mz, target + tol)
        if hi > lo:
            return True
    return False


def prepare_log2(matrix: pd.DataFrame, median_center: bool) -> pd.DataFrame:
    """v1.3 preprocessing: log2 of positive intensities, optional per-sample median centering."""
    numeric = matrix.astype(float).where(matrix.astype(float) > 0)
    transformed = np.log2(numeric)
    if median_center:
        transformed = transformed.subtract(transformed.median(axis=0, skipna=True), axis=1)
    return transformed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/external/msv000096214"))
    parser.add_argument("--consensus-dir", type=Path, default=Path("data/external/msv000096214/consensus"))
    parser.add_argument("--readiness-manifest", type=Path,
                        default=Path("data/validation/msv000096214_readiness_v2/sample_manifest.csv.gz"))
    parser.add_argument("--effect-stage", type=Path, default=Path("data/validation/msv000096214_effect_stage/report.json"))
    parser.add_argument("--output", type=Path, default=Path("data/validation/msv000096214_posthoc_t4"))
    args = parser.parse_args()
    if not args.effect_stage.exists():
        raise SystemExit("T4_RUNS_LAST: effect stage report missing")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    manifest = pd.read_csv(args.readiness_manifest)
    genotype_of = {Path(r.filename).stem: r.genotype for r in
                   manifest.loc[manifest["role"] == "study"].itertuples(index=False)}
    report: dict[str, object] = {
        "status": "MSV000096214_POSTHOC_T4",
        "data_role": "development/consumed",
        "contract": "docs/GLM_MSV000096214_STRUCTURE_HIDDEN_CONTRACT_20261012.md",
        "losses": LOSSES, "branch_delta_da": BRANCH_DELTA,
        "polarity": {},
    }

    for polarity in ("positive", "negative"):
        cons = args.consensus_dir / polarity
        matrix = pd.read_csv(cons / "matrix_filled.csv.gz", index_col="group_id")
        cons_report = json.loads((cons / "report.json").read_text(encoding="utf-8"))
        warps = cons_report["warp_params"]
        samples = [c for c in matrix.columns]
        genotype_cols = {g: [c for c in samples if genotype_of.get(c) == g]
                         for g in ["WT_Col0", "tt4_11_SALK_020583", "tt7_115_EMS",
                                   "tt7_8_SALK_039417", "tt7_9_SALK_seq_064306",
                                   "tt7_115__tt4_11", "tt7_8__tt4_11", "tt7_9__tt4_11"]}
        log2 = prepare_log2(matrix, True)
        log2_sensitivity = prepare_log2(matrix, False)
        wt_med = log2[genotype_cols["WT_Col0"]].median(axis=1)

        fc, pvalues = {}, np.ones(len(matrix))
        for genotype, cols in genotype_cols.items():
            if genotype == "WT_Col0" or not cols:
                continue
            group_med = log2[cols].median(axis=1)
            fc[genotype] = group_med - wt_med
        # Mann-Whitney per feature for tt4 vs WT
        tt4_cols = genotype_cols["tt4_11_SALK_020583"]
        wt_cols = genotype_cols["WT_Col0"]
        valid = matrix[tt4_cols].notna().all(axis=1) & matrix[wt_cols].notna().all(axis=1)
        pvals = np.ones(len(matrix))
        idx = np.flatnonzero(valid.to_numpy())
        if len(idx):
            a = log2[tt4_cols].to_numpy()[idx]
            b = log2[wt_cols].to_numpy()[idx]
            from scipy.stats import mannwhitneyu
            res = [mannwhitneyu(a[k], b[k], alternative="two-sided").pvalue for k in range(len(idx))]
            pvals[idx] = res
        fdr = bh_fdr(pvals)
        frame = pd.DataFrame({"mz": matrix["mz"] if "mz" in matrix else np.nan,
                              "fc_tt4": fc["tt4_11_SALK_020583"].to_numpy(),
                              "fdr_tt4": fdr})
        frame.index = matrix.index
        frame["mz"] = pd.read_csv(cons / "groups.csv").set_index("group_id").loc[frame.index, "mz"]
        frame["rt_sec"] = pd.read_csv(cons / "groups.csv").set_index("group_id").loc[frame.index, "rt_sec"]
        doubles = ["tt7_115__tt4_11", "tt7_8__tt4_11", "tt7_9__tt4_11"]
        for d in doubles:
            frame[f"fc_{d}"] = fc[d].to_numpy()

        # v1.3 raw-log2 sensitivity frame: same FC definitions without centering
        wt_med_s = log2_sensitivity[genotype_cols["WT_Col0"]].median(axis=1)
        a_s = log2_sensitivity[tt4_cols].to_numpy()[idx] if len(idx) else np.zeros((0, len(tt4_cols)))
        b_s = log2_sensitivity[wt_cols].to_numpy()[idx] if len(idx) else np.zeros((0, len(wt_cols)))
        pvals_s = np.ones(len(matrix))
        if len(idx):
            pvals_s[idx] = [mannwhitneyu(a_s[k], b_s[k], alternative="two-sided").pvalue
                            for k in range(len(idx))]
        frame_s = frame.copy()
        frame_s["fc_tt4"] = (log2_sensitivity[tt4_cols].median(axis=1) - wt_med_s).to_numpy()
        frame_s["fdr_tt4"] = bh_fdr(pvals_s)
        for d in doubles:
            frame_s[f"fc_{d}"] = (log2_sensitivity[genotype_cols[d]].median(axis=1) - wt_med_s).to_numpy()

        f_mask = (frame["fc_tt4"] <= -1) & (frame["fdr_tt4"] < 0.05)
        double_down = sum((frame[f"fc_{d}"] <= -0.5).to_numpy() for d in doubles)
        f_mask &= (double_down >= 2)
        all_fc_cols = ["fc_tt4"] + [f"fc_{d}" for d in doubles] + [
            f"fc_{g}" for g in ["tt7_115_EMS", "tt7_8_SALK_039417", "tt7_9_SALK_seq_064306"]
            if f"fc_{g}" in frame.columns]
        for genotype in ("tt3_SALK_099848C", "fls1_3_EMS"):
            cols = [c for c in samples if genotype_of.get(c) == genotype]
            if cols:
                frame[f"fc_{genotype}"] = log2[cols].median(axis=1).to_numpy() - wt_med.to_numpy()
                all_fc_cols.append(f"fc_{genotype}")
        stable = np.ones(len(frame), dtype=bool)
        for col in all_fc_cols:
            stable &= np.abs(frame[col].to_numpy()) <= 0.25
        c_mask = stable & (frame["fdr_tt4"] > 0.5)

        f_ids = frame.index[f_mask]
        c_pool = frame.index[c_mask]

        # matched controls within +/-50 Da
        matched_c: dict[object, list[object]] = {}
        c_mz = frame.loc[c_pool, "mz"].to_numpy()
        c_order = np.argsort(c_mz)
        c_sorted = c_mz[c_order]
        rng = np.random.default_rng(20261012)
        for fid in f_ids:
            mz = frame.loc[fid, "mz"]
            lo = np.searchsorted(c_sorted, mz - 50.0)
            hi = np.searchsorted(c_sorted, mz + 50.0)
            if hi <= lo:
                continue
            picks = rng.choice(c_order[lo:hi], size=min(2, hi - lo), replace=False)
            matched_c[fid] = [c_pool[p] for p in picks]
        used_c = sorted({c for picks in matched_c.values() for c in picks})

        # MS2 linkage over all files of this polarity
        loss_flags: dict[object, bool] = {fid: False for fid in list(f_ids) + used_c}
        linkage_counts: dict[object, int] = {fid: 0 for fid in loss_flags}
        mz_arr = frame["mz"].to_numpy()
        rt_arr = frame["rt_sec"].to_numpy()
        mz_order = np.argsort(mz_arr)
        mz_sorted = mz_arr[mz_order]
        for name in samples:
            source = args.root / "mzxml" / polarity / f"{name}.mzXML"
            if not source.exists():
                continue
            warp = warps.get(name, {"intercept": 0.0, "slope": 0.0})
            denom = 1.0 + warp["slope"]
            for spectrum in load_ms2_spectra(source):
                precursor = float(spectrum["precursor_mz"])
                lo = np.searchsorted(mz_sorted, precursor * (1 - PPM * 1e-6) - 1e-9)
                hi = np.searchsorted(mz_sorted, precursor * (1 + PPM * 1e-6) + 1e-9)
                if hi <= lo:
                    continue
                rt_file = float(spectrum["rt"])
                rt_warped = rt_file + warp["intercept"] + warp["slope"] * rt_file
                frag = spectrum["frag_mz"]
                for pos in mz_order[lo:hi]:
                    gid = frame.index[pos]
                    if gid not in loss_flags:
                        continue
                    if abs(rt_warped - rt_arr[pos]) > RT_LINK:
                        continue
                    linkage_counts[gid] += 1
                    if loss_flags[gid] is False and loss_present(frag, precursor):
                        loss_flags[gid] = True

        f_linked = [g for g in f_ids if linkage_counts[g] > 0]
        c_linked = [g for g in used_c if linkage_counts[g] > 0]
        fisher_p = float("nan")
        if f_linked and c_linked:
            table = [[sum(loss_flags[g] for g in f_linked), len(f_linked) - sum(loss_flags[g] for g in f_linked)],
                     [sum(loss_flags[g] for g in c_linked), len(c_linked) - sum(loss_flags[g] for g in c_linked)]]
            fisher_p = float(stats.fisher_exact(table)[1])
        else:
            table = None

        # branch evidence: tt7-down set with +15.9949 partner
        tt7_mean_fc = (frame.get("fc_tt7_115_EMS", pd.Series(np.nan, index=frame.index)).to_numpy()
                       + frame.get("fc_tt7_8_SALK_039417", pd.Series(np.nan, index=frame.index)).to_numpy()
                       + frame.get("fc_tt7_9_SALK_seq_064306", pd.Series(np.nan, index=frame.index)).to_numpy()) / 3.0
        tt7_down_mask = (tt7_mean_fc <= -1) & (frame["fdr_tt4"].to_numpy() > 0.05) & (~f_mask.to_numpy())

        def branch_partner_count(mask: np.ndarray) -> tuple[int, int]:
            total, with_partner = 0, 0
            ids = frame.index[mask]
            for gid in ids:
                mz = frame.loc[gid, "mz"]
                target = mz + BRANCH_DELTA
                lo = np.searchsorted(mz_sorted, target * (1 - PPM * 1e-6))
                hi = np.searchsorted(mz_sorted, target * (1 + PPM * 1e-6))
                total += 1
                for pos in mz_order[lo:hi]:
                    pid = frame.index[pos]
                    if tt7_mean_fc[pos] >= 0:
                        with_partner += 1
                        break
            return with_partner, total

        tt7_partner = branch_partner_count(tt7_down_mask)
        c_partner = branch_partner_count(np.isin(frame.index, used_c))

        # v1.3 sensitivity: repeat the F/C/Fisher computation on raw log2
        f_mask_s = (frame_s["fc_tt4"] <= -1) & (frame_s["fdr_tt4"] < 0.05)
        double_down_s = sum((frame_s[f"fc_{d}"] <= -0.5).to_numpy() for d in doubles)
        f_mask_s &= (double_down_s >= 2)
        sensitivity_summary = {
            "n_F_raw_sensitivity": int(f_mask_s.sum()),
            "F_ids_primary_minus_raw": int((f_mask & ~f_mask_s).sum()),
            "F_ids_raw_minus_primary": int((f_mask_s & ~f_mask).sum()),
        }

        report["polarity"][polarity] = {
            "n_F": int(f_mask.sum()), "n_C_pool": int(c_mask.sum()),
            "sensitivity_raw_log2": sensitivity_summary,
            "n_matched_C": len(used_c),
            "F_linked_ms2": len(f_linked), "C_linked_ms2": len(c_linked),
            "F_with_glycoloss": int(sum(loss_flags[g] for g in f_linked)),
            "C_with_glycoloss": int(sum(loss_flags[g] for g in c_linked)),
            "fisher_table": table, "fisher_p_two_sided": fisher_p,
            "F_in_400_900Da_frac": float(np.mean(
                (frame.loc[f_ids, "mz"].between(400, 900)).to_numpy())) if len(f_ids) else float("nan"),
            "C_in_400_900Da_frac": float(np.mean(
                (frame.loc[used_c, "mz"].between(400, 900)).to_numpy())) if used_c else float("nan"),
            "tt7down_with_O16_partner": tt7_partner[0],
            "tt7down_total": tt7_partner[1],
            "C_with_O16_partner": c_partner[0],
            "C_total": c_partner[1],
        }
        frame.to_csv(args.output / f"{polarity}_feature_frame.csv.gz", compression="gzip")

    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["polarity"], indent=2))


if __name__ == "__main__":
    main()
