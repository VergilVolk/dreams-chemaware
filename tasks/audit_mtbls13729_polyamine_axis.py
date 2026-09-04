"""Adversarial audit of the MTBLS13729 feature-1717 polyamine axis.

This audit deliberately separates three questions that earlier summaries mixed:

1. Is the targeted MS1 feature quantitatively robust in the Rmu pairs?
2. Is its exact identity supported by experimental MS/MS?
3. Does an independent paired CRC cohort support the broader polyamine axis?

The script does not upgrade an exact-mass candidate to MSI Level 2 and does not
use external pathway agreement as identity evidence.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


FEATURE_ID = 1717
PROTON_MASS = 1.007276466621
EXTERNAL_NAMES = {
    "N1-Acetylspermidine",
    "N1-Acetylspermine",
    "Spermidine",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signflip_p(values: np.ndarray, seed: int, draws: int = 200_000) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, draws, 10_000):
        size = min(10_000, draws - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(values)))
        exceed += int((np.abs((signs * values).mean(axis=1)) >= observed).sum())
    return float((exceed + 1) / (draws + 1))


def read_gzip_csv(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return pd.read_csv(handle)


def bh_qvalues(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def external_polyamine_effects(maf: Path, samples: Path, seed: int) -> list[dict[str, object]]:
    matrix = pd.read_csv(maf, sep="\t")
    metadata = pd.read_csv(samples, sep="\t")
    tumour = metadata.loc[
        metadata["Factor Value[Disease]"].eq("colorectal carcinoma"), "Sample Name"
    ].tolist()
    normal = metadata.loc[
        metadata["Factor Value[Disease]"].eq("control"), "Sample Name"
    ].tolist()
    if len(tumour) != 35 or {name + "_1" for name in tumour} != set(normal):
        raise RuntimeError("MTBLS8090 paired-sample contract changed")
    selected = matrix.loc[matrix["metabolite_identification"].isin(EXTERNAL_NAMES)].copy()
    if selected.empty:
        return []
    abundance = selected[tumour + normal].apply(pd.to_numeric, errors="coerce")
    positive = abundance.to_numpy(float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    pseudocount = float(np.percentile(positive, 1) / 2)
    log_abundance = np.log2(abundance + pseudocount)
    results: list[dict[str, object]] = []
    for position, (index, row) in enumerate(selected.iterrows()):
        delta = np.asarray(
            [log_abundance.loc[index, name] - log_abundance.loc[index, name + "_1"] for name in tumour],
            dtype=float,
        )
        results.append(
            {
                "source_maf": maf.name,
                "metabolite": str(row["metabolite_identification"]),
                "formula": str(row.get("chemical_formula", "")),
                "n_pairs": int(len(delta)),
                "mean_log2fc": float(delta.mean()),
                "median_log2fc": float(np.median(delta)),
                "positive_fraction": float((delta > 0).mean()),
                "wilcoxon_p": float(wilcoxon(delta).pvalue),
                "signflip_p": signflip_p(delta, seed + position),
                "pseudocount": pseudocount,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--closure-dir", type=Path, default=Path("data/mtbls13729/biology_closure_analysis_v1")
    )
    parser.add_argument(
        "--link-dir", type=Path, default=Path("data/mtbls13729/ms1_ms2_link")
    )
    parser.add_argument(
        "--mtbls8090-dir", type=Path, default=Path("data/external/MTBLS8090")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/mtbls13729/polyamine_axis_adversarial_audit_v1")
    )
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    inputs = {
        "candidate": args.closure_dir / "candidate_identity_and_abundance.csv",
        "paired": args.closure_dir / "paired_abundance_by_normalization.csv",
        "quality": args.closure_dir / "peak_quality_audit.csv",
        "linked_ms2": args.link_dir / "pos_rp__linked_ms2.csv.gz",
        "best_annotations": args.link_dir / "pos_rp__feature_best_annotations.csv.gz",
        "external_hilic": args.mtbls8090_dir / "hilic.maf.tsv",
        "external_rp": args.mtbls8090_dir / "reverse_phase.maf.tsv",
        "external_samples": args.mtbls8090_dir / "s_MTBLS8090.txt",
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing audit inputs: {missing}")

    candidate_frame = pd.read_csv(inputs["candidate"])
    row = candidate_frame.loc[candidate_frame["feature_id"].eq(FEATURE_ID)]
    if len(row) != 1:
        raise RuntimeError(f"expected exactly one feature {FEATURE_ID} row")
    candidate = row.iloc[0]
    paired = pd.read_csv(inputs["paired"])
    paired = paired.loc[paired["feature_id"].eq(FEATURE_ID)].copy()
    quality = pd.read_csv(inputs["quality"])
    quality = quality.loc[quality["feature_id"].eq(FEATURE_ID)]
    if len(paired) != 4 or len(quality) != 1:
        raise RuntimeError("feature-1717 local robustness inputs are incomplete")

    linked = read_gzip_csv(inputs["linked_ms2"])
    best = read_gzip_csv(inputs["best_annotations"])
    linked_1717 = linked.loc[pd.to_numeric(linked["feature_id"], errors="coerce").eq(FEATURE_ID)]
    best_1717 = best.loc[pd.to_numeric(best["feature_id"], errors="coerce").eq(FEATURE_ID)]

    theoretical_mh = float(candidate["neutral_exact_mass"]) + PROTON_MASS
    measured_mz = float(candidate["mz"])
    mass_error_ppm = (measured_mz - theoretical_mh) / theoretical_mh * 1e6

    local_rows = []
    for _, item in paired.iterrows():
        local_rows.append(
            {
                "normalization": str(item["normalization"]),
                "rmu_n": int(item["rmu_n"]),
                "rmu_mean_log2fc": float(item["rmu_mean_log2fc"]),
                "rmu_exact_signflip_p": float(item["rmu_exact_signflip_p"]),
                "rmu_positive_fraction": float(item["rmu_positive_fraction"]),
                "rmu_loo_min_mean_log2fc": float(item["rmu_loo_min_mean_log2fc"]),
                "rtu_mean_log2fc": float(item["rtu_mean_log2fc"]),
                "interaction_log2fc": float(item["interaction_log2fc"]),
                "interaction_exact_permutation_p": float(item["interaction_exact_permutation_p"]),
            }
        )

    external = []
    for offset, maf in enumerate((inputs["external_hilic"], inputs["external_rp"])):
        external.extend(external_polyamine_effects(maf, inputs["external_samples"], args.seed + 100 * offset))
    if len(external) != 3:
        raise RuntimeError(f"expected three external polyamine rows, observed {len(external)}")
    q_values = bh_qvalues([float(item["wilcoxon_p"]) for item in external])
    for item, q_value in zip(external, q_values, strict=True):
        item["wilcoxon_bh_q_three_metabolites"] = q_value

    peak = quality.iloc[0]
    report = {
        "status": "mtbls13729_polyamine_axis_adversarial_audit_complete",
        "formal": False,
        "feature_id": FEATURE_ID,
        "identity": {
            "label": "N1,N8-diacetylspermidine-like",
            "measured_mz": measured_mz,
            "rt_sec": float(candidate["rt_sec"]),
            "candidate_formula": str(candidate["candidate_formula"]),
            "candidate_inchikey": str(candidate["inchikey"]),
            "candidate_neutral_exact_mass": float(candidate["neutral_exact_mass"]),
            "theoretical_mh_mz": theoretical_mh,
            "mass_error_ppm": mass_error_ppm,
            "hmdb_exact_formula_match_count_in_frozen_local_table": int(
                candidate["hmdb_exact_formula_match_count"]
            ),
            "experimental_ms2_linked_spectra": int(len(linked_1717)),
            "accepted_best_annotation_rows": int(len(best_1717)),
            "identity_decision": (
                "exact-mass/formula-consistent candidate only; no accepted experimental MS/MS bridge, "
                "no authentic-standard retention time, and therefore not MSI Level 2"
            ),
        },
        "local_targeted_eic": {
            "detected_fraction": float(peak["detected_fraction"]),
            "median_abs_apex_delta_sec": float(peak["median_abs_apex_delta_sec"]),
            "p95_abs_apex_delta_sec": float(peak["p95_abs_apex_delta_sec"]),
            "median_snr": float(peak["median_snr"]),
            "normalizations": local_rows,
            "direction_positive_all_normalizations": bool(
                all(item["rmu_mean_log2fc"] > 0 for item in local_rows)
            ),
            "exact_signflip_p_le_0_05_all_normalizations": bool(
                all(item["rmu_exact_signflip_p"] <= 0.05 for item in local_rows)
            ),
            "leave_one_out_mean_positive_all_normalizations": bool(
                all(item["rmu_loo_min_mean_log2fc"] > 0 for item in local_rows)
            ),
            "selection_boundary": (
                "eight previously selected targeted candidates; these p values are not a fresh untargeted FDR screen"
            ),
        },
        "external_mtbls8090_pathway_context": {
            "n_pairs": 35,
            "metabolites": external,
            "interpretation": (
                "independent paired CRC tumour-normal pathway context only; neither a direct replication of "
                "N1,N8-diacetylspermidine nor evidence of mucinous specificity"
            ),
        },
        "decision": {
            "abundance_axis": "strong local candidate signal",
            "exact_identity": "unconfirmed",
            "mucinous_specificity": "suggestive in the targeted subset but not established after global multiplicity",
            "paper_role": (
                "third exploratory orthogonal axis; promote to a named metabolite only after authentic-standard "
                "RT and MS/MS confirmation"
            ),
        },
        "forbidden_claims": [
            "MSI Level 2 identity",
            "N1,N8-diacetylspermidine-specific enzyme mechanism",
            "polyamine flux",
            "mucinous-subtype specificity without multiplicity-controlled confirmation",
        ],
        "provenance": {name: sha256(path) for name, path in inputs.items()},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(local_rows).to_csv(args.output_dir / "local_normalization_audit.csv", index=False)
    pd.DataFrame(external).to_csv(args.output_dir / "mtbls8090_polyamine_context.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
