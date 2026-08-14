"""Test whether validated spectral factors mark official DreaMS failures.

The unit of inference is a molecule pair (two experimental views), not a peak
or a spectrum.  Tests are conditioned on presence of the same exact-mass peak,
so they ask whether the learned factor activation adds information beyond the
peak's mere existence.  Discovery and confirmation molecule sets are audited
separately and only directionally replicated results can pass the gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu


def bh_adjust(values: list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def odds_ratio(a: int, b: int, c: int, d: int) -> float:
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def bootstrap_rate_difference(active: np.ndarray, outcome: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(active)
    draws = []
    for _ in range(5000):
        idx = rng.integers(0, n, n)
        x = active[idx]
        y = outcome[idx]
        if x.any() and (~x).any():
            draws.append(float(y[x].mean() - y[~x].mean()))
    if not draws:
        return math.nan, math.nan
    return tuple(np.quantile(draws, [0.025, 0.975]))


def load_evidence(activation_dir: Path, codes_path: Path, targets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mask = np.load(activation_dir / "peak_mask.npy")
    values = np.load(activation_dir / "peak_values.npy")
    codes = np.load(codes_path, mmap_mode="r")
    spectra = json.loads((activation_dir / "spectra.json").read_text(encoding="utf-8"))
    counts = mask.sum(axis=1)
    spectrum_index = np.repeat(np.arange(len(spectra)), counts)
    mz = values[:, :, 0][mask].astype(np.float64)
    if len(mz) != len(codes):
        raise RuntimeError(f"Misaligned peak codes in {activation_dir}")
    precursor = np.asarray([record["precursor_mz"] for record in spectra], dtype=float)
    frame = pd.DataFrame({
        "spectrum_index": np.arange(len(spectra)),
        "query_row": [int(record["hdf5_row"]) for record in spectra],
        "ik14": [record["ik14"] for record in spectra],
    })
    metadata: dict[int, dict] = {}
    for target_row in targets.itertuples(index=False):
        factor = int(target_row.factor)
        kind = str(target_row.spectral_kind)
        center = float(target_row.theoretical_mass)
        tolerance = max(center * 10e-6, 5e-5)
        observed = mz if kind == "fragment_mz" else precursor[spectrum_index] - mz
        target = np.abs(observed - center) <= tolerance
        active = target & (np.asarray(codes[:, factor], dtype=np.float32) > 0)
        presence_by_spectrum = np.bincount(spectrum_index[target], minlength=len(spectra)) > 0
        active_by_spectrum = np.bincount(spectrum_index[active], minlength=len(spectra)) > 0
        score_by_spectrum = np.zeros(len(spectra), dtype=np.float32)
        np.maximum.at(score_by_spectrum, spectrum_index[target], np.asarray(codes[:, factor], dtype=np.float32)[target])
        frame[f"f{factor}_mass_present"] = presence_by_spectrum
        frame[f"f{factor}_active"] = active_by_spectrum
        frame[f"f{factor}_score"] = score_by_spectrum
        metadata[factor] = {"spectral_kind": kind, "exact_mass_da": center, "tolerance_da": tolerance}
    return frame, metadata


def molecule_level(taxonomy: pd.DataFrame, evidence: pd.DataFrame, factors: list[int]) -> pd.DataFrame:
    taxonomy = taxonomy.copy()
    taxonomy["official_failure"] = as_bool(taxonomy["official_failure"])
    merged = taxonomy.merge(evidence, on="query_row", how="left", validate="one_to_one", suffixes=("", "_evidence"))
    if merged[[f"f{factor}_active" for factor in factors]].isna().any().any():
        raise RuntimeError("Taxonomy rows failed to map one-to-one to extracted spectra")
    rows: list[dict] = []
    for ik14, group in merged.groupby("ik14", sort=False):
        failures = group["official_failure"].to_numpy(bool)
        failed_rows = group.loc[group["official_failure"]]
        item = {
            "ik14": str(ik14),
            "source_pairs": int(group["pair_id"].nunique()),
            "official_failure_any": bool(failures.any()),
            "official_failure_both": bool(failures.all()),
            "official_margin_min": float(group["official_margin"].astype(float).min()),
            "official_margin_mean": float(group["official_margin"].astype(float).mean()),
            "failure_same_formula_any": bool(as_bool(failed_rows["same_formula"]).any()) if len(failed_rows) else False,
            "failure_same_scaffold_any": bool((failed_rows["scaffold_relation"] == "same_scaffold").any()) if len(failed_rows) else False,
            "failure_near_structure_any": bool(failed_rows["mces_bin"].isin(["0-2", "3-5"]).any()) if len(failed_rows) else False,
        }
        for factor in factors:
            item[f"f{factor}_mass_present"] = bool(group[f"f{factor}_mass_present"].any())
            item[f"f{factor}_active"] = bool(group[f"f{factor}_active"].any())
            item[f"f{factor}_score"] = float(group[f"f{factor}_score"].max())
        rows.append(item)
    return pd.DataFrame(rows)


def audit_split(split: str, pairs: pd.DataFrame, factors: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tests: list[dict] = []
    subtype_rows: list[dict] = []
    for factor in factors:
        subset = pairs.loc[pairs[f"f{factor}_mass_present"]].copy()
        active = subset[f"f{factor}_active"].to_numpy(bool)
        failure = subset["official_failure_any"].to_numpy(bool)
        a = int(np.sum(active & failure))
        b = int(np.sum(active & ~failure))
        c = int(np.sum(~active & failure))
        d = int(np.sum(~active & ~failure))
        _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided") if min(len(subset), active.sum(), (~active).sum()) > 0 else (np.nan, 1.0)
        active_rate = float(failure[active].mean()) if active.any() else np.nan
        inactive_rate = float(failure[~active].mean()) if (~active).any() else np.nan
        rng = np.random.default_rng(20260812 + factor)
        boot = []
        for _ in range(5000):
            idx = rng.integers(0, len(subset), len(subset)) if len(subset) else np.array([], dtype=int)
            x, y = active[idx], failure[idx]
            if x.any() and (~x).any():
                boot.append(float(y[x].mean() - y[~x].mean()))
        ci = np.quantile(boot, [0.025, 0.975]) if boot else [np.nan, np.nan]
        margin_active = subset.loc[active, "official_margin_min"].to_numpy(float)
        margin_inactive = subset.loc[~active, "official_margin_min"].to_numpy(float)
        margin_p = (mannwhitneyu(margin_active, margin_inactive, alternative="two-sided").pvalue
                    if len(margin_active) and len(margin_inactive) else 1.0)
        tests.append({
            "split": split,
            "factor": factor,
            "target_present_pairs": len(subset),
            "active_pairs": int(active.sum()),
            "inactive_pairs": int((~active).sum()),
            "active_failure_pairs": a,
            "inactive_failure_pairs": c,
            "active_failure_rate": active_rate,
            "inactive_failure_rate": inactive_rate,
            "failure_rate_difference": active_rate - inactive_rate,
            "failure_rate_difference_ci_low": float(ci[0]),
            "failure_rate_difference_ci_high": float(ci[1]),
            "failure_odds_ratio": odds_ratio(a, b, c, d),
            "failure_fisher_p": float(p),
            "margin_active_mean": float(np.mean(margin_active)) if len(margin_active) else np.nan,
            "margin_inactive_mean": float(np.mean(margin_inactive)) if len(margin_inactive) else np.nan,
            "margin_mannwhitney_p": float(margin_p),
        })
        failed = subset.loc[subset["official_failure_any"]]
        for subtype in ("failure_same_formula_any", "failure_same_scaffold_any", "failure_near_structure_any"):
            if len(failed) == 0:
                continue
            x = failed[f"f{factor}_active"].to_numpy(bool)
            y = failed[subtype].to_numpy(bool)
            aa, bb = int(np.sum(x & y)), int(np.sum(x & ~y))
            cc, dd = int(np.sum(~x & y)), int(np.sum(~x & ~y))
            _, subtype_p = fisher_exact([[aa, bb], [cc, dd]], alternative="two-sided") if x.any() and (~x).any() else (np.nan, 1.0)
            subtype_rows.append({
                "split": split, "factor": factor, "failure_subtype": subtype,
                "active_subtype_rate": float(y[x].mean()) if x.any() else np.nan,
                "inactive_subtype_rate": float(y[~x].mean()) if (~x).any() else np.nan,
                "odds_ratio": odds_ratio(aa, bb, cc, dd), "fisher_p": float(subtype_p),
                "failed_pairs": len(failed),
            })
    result = pd.DataFrame(tests)
    result["failure_fisher_bh_q"] = bh_adjust(result["failure_fisher_p"].tolist())
    result["margin_mannwhitney_bh_q"] = bh_adjust(result["margin_mannwhitney_p"].tolist())
    subtype = pd.DataFrame(subtype_rows)
    if len(subtype):
        subtype["fisher_bh_q"] = bh_adjust(subtype["fisher_p"].tolist())
    return result, subtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, default=Path("data/validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/validated_factor_failure_enrichment_v2"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    consensus = pd.read_csv(args.validation_root / "validated_factor_exact_mass_refinement/formula_consensus.csv")
    dominant = consensus.loc[consensus["consensus_rank"] == 1].copy()
    factors = dominant["factor"].astype(int).tolist()
    split_config = {
        "discovery": ("mass_dense_all_peak_discovery", "peak_token_centered_sae_seed201/discovery_codes.npy", "mass_dense_failure_taxonomy/discovery_taxonomy.csv"),
        "confirmation": ("mass_dense_all_peak_confirmation", "peak_token_centered_sae_seed201/confirmation_codes.npy", "mass_dense_failure_taxonomy/confirmation_taxonomy.csv"),
    }
    all_tests, all_subtypes = [], []
    split_reports = {}
    for split, (activation_name, codes_name, taxonomy_name) in split_config.items():
        evidence, metadata = load_evidence(args.validation_root / activation_name, args.validation_root / codes_name, dominant)
        taxonomy = pd.read_csv(args.validation_root / taxonomy_name)
        molecules = molecule_level(taxonomy, evidence, factors)
        molecules.to_csv(args.output_dir / f"{split}_molecule_factor_evidence.csv", index=False)
        tests, subtypes = audit_split(split, molecules, factors)
        all_tests.append(tests)
        all_subtypes.append(subtypes)
        split_reports[split] = {
            "molecules": len(molecules),
            "source_pairs": int(molecules["source_pairs"].sum()),
            "failure_any_molecules": int(molecules["official_failure_any"].sum()),
            "failure_all_views_molecules": int(molecules["official_failure_both"].sum()),
        }
    tests = pd.concat(all_tests, ignore_index=True)
    tests.to_csv(args.output_dir / "factor_failure_enrichment.csv", index=False)
    pd.concat(all_subtypes, ignore_index=True).to_csv(args.output_dir / "failure_subtype_enrichment.csv", index=False)
    discovery = tests.loc[tests["split"] == "discovery"].set_index("factor")
    confirmation = tests.loc[tests["split"] == "confirmation"].set_index("factor")
    replicated = []
    for factor in factors:
        d, c = discovery.loc[factor], confirmation.loc[factor]
        same_direction = np.sign(d["failure_rate_difference"]) == np.sign(c["failure_rate_difference"])
        enough_support = min(d["active_pairs"], d["inactive_pairs"], c["active_pairs"], c["inactive_pairs"]) >= 10
        confirmation_significant = c["failure_fisher_bh_q"] < 0.05
        replicated.append({
            "factor": factor,
            "spectral_kind": dominant.set_index("factor").loc[factor, "spectral_kind"],
            "candidate_formula": dominant.set_index("factor").loc[factor, "candidate_formula"],
            "exact_mass_da": float(dominant.set_index("factor").loc[factor, "theoretical_mass"]),
            "discovery_failure_rate_difference": d["failure_rate_difference"],
            "discovery_bh_q": d["failure_fisher_bh_q"],
            "confirmation_failure_rate_difference": c["failure_rate_difference"],
            "confirmation_bh_q": c["failure_fisher_bh_q"],
            "same_direction": bool(same_direction),
            "enough_conditioned_support": bool(enough_support),
            "passes_failure_enrichment_gate": bool(same_direction and enough_support and confirmation_significant),
        })
    replicated_frame = pd.DataFrame(replicated)
    replicated_frame.to_csv(args.output_dir / "replicated_failure_enrichment.csv", index=False)
    report = {
        "status": "validated_factor_failure_enrichment_audit",
        "unit_of_inference": "unique IK14 molecule (all sampled experimental views and repeated pairs aggregated)",
        "conditioning": "Each factor is compared only among molecules containing its dominant exact-mass peak; active versus inactive tests learned-factor information beyond peak presence.",
        "split_reports": split_reports,
        "factors_tested": len(factors),
        "factors_passing_failure_enrichment_gate": replicated_frame.loc[replicated_frame["passes_failure_enrichment_gate"], "factor"].astype(int).tolist(),
        "microtuning_decision": "Eligible only if direction agrees across molecule-disjoint splits, confirmation BH q < 0.05, and both active/inactive conditioned groups have at least 10 molecules.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
