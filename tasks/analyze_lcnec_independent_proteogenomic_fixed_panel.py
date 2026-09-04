"""Run the frozen independent-LCNEC proteogenomic fixed-panel audit.

The primary endpoint is pure-LCNEC paired tumor-minus-NAT protein abundance.
The 22-protein panel and all decision rules were frozen before the matrix was
opened. Missing proteins receive no substitutes and contribute p=1 to the
22-test Benjamini-Hochberg correction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def bh_adjust(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def sign(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def stable_direction(differences: np.ndarray) -> dict[str, object]:
    finite = differences[np.isfinite(differences)]
    mean_effect = float(np.mean(finite)) if finite.size else float("nan")
    median_effect = float(np.median(finite)) if finite.size else float("nan")
    mean_sign = sign(mean_effect) if finite.size else 0
    median_sign = sign(median_effect) if finite.size else 0
    mean_median_same = mean_sign != 0 and mean_sign == median_sign

    if finite.size >= 2 and mean_sign != 0:
        leave_one_out_means = (finite.sum() - finite) / (finite.size - 1)
        leave_one_out_stable = bool(
            np.all(np.asarray([sign(value) for value in leave_one_out_means]) == mean_sign)
        )
    else:
        leave_one_out_stable = False

    nonzero = finite[np.abs(finite) > 1e-12]
    if nonzero.size and mean_sign != 0:
        direction_fraction = float(np.mean(np.sign(nonzero) == mean_sign))
    else:
        direction_fraction = float("nan")

    direction_stable = bool(
        mean_median_same
        and leave_one_out_stable
        and np.isfinite(direction_fraction)
        and direction_fraction >= 0.60
    )
    return {
        "mean_effect": mean_effect,
        "median_effect": median_effect,
        "direction": "up" if mean_sign > 0 else "down" if mean_sign < 0 else "zero",
        "mean_median_same_nonzero_sign": mean_median_same,
        "leave_one_pair_out_mean_sign_unchanged": leave_one_out_stable,
        "nonzero_pair_direction_fraction": direction_fraction,
        "direction_stable": direction_stable,
    }


def paired_test(differences: np.ndarray) -> float:
    finite = differences[np.isfinite(differences)]
    if finite.size == 0 or np.all(np.abs(finite) <= 1e-12):
        return 1.0
    return float(
        wilcoxon(
            finite,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        ).pvalue
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protein-matrix",
        type=Path,
        default=Path(
            "data/external/LCNEC_proteogenomic_2026/LCNEC_2026-SA_extracted/"
            "LCNEC_2026-SA/data/Updated_LCNEC-omics-supp_data_1-6/SuppData5.xlsx"
        ),
    )
    parser.add_argument(
        "--clinical",
        type=Path,
        default=Path(
            "data/external/LCNEC_proteogenomic_2026/LCNEC_2026-SA_extracted/"
            "LCNEC_2026-SA/data/Updated_LCNEC-omics-supp_data_1-6/SuppData1.xlsx"
        ),
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path(
            "data/external/LCNEC_proteogenomic_2026/fixed_panel_preregistration_v1.json"
        ),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "data/external/LCNEC_proteogenomic_2026/fixed_panel_analysis_contract_v1.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/external/LCNEC_proteogenomic_2026/fixed_panel_patient_audit_v1"
        ),
    )
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    axes: dict[str, list[str]] = preregistration["axes"]
    panel = [gene for genes in axes.values() for gene in genes]
    if len(panel) != 22 or len(set(panel)) != 22:
        raise RuntimeError("frozen panel must contain exactly 22 unique proteins")

    clinical = pd.read_excel(args.clinical, dtype={"Sample.ID": str})
    proteins = pd.read_excel(args.protein_matrix)
    if proteins.columns[0] != "Protein/Sample.ID":
        raise RuntimeError(f"unexpected protein identifier column: {proteins.columns[0]}")
    proteins["Protein/Sample.ID"] = proteins["Protein/Sample.ID"].astype(str)
    if proteins["Protein/Sample.ID"].duplicated().any():
        raise RuntimeError("protein identifiers are not unique")
    proteins = proteins.set_index("Protein/Sample.ID")

    tumor_columns = {column[1:]: column for column in proteins.columns if column.startswith("T")}
    normal_columns = {column[1:]: column for column in proteins.columns if column.startswith("N")}
    paired_ids = sorted(set(tumor_columns) & set(normal_columns))
    if len(paired_ids) != 103:
        raise RuntimeError(f"expected 103 protein pairs, found {len(paired_ids)}")

    clinical["Sample.ID"] = clinical["Sample.ID"].astype(str)
    clinical_by_id = clinical.set_index("Sample.ID")
    missing_clinical = sorted(set(paired_ids) - set(clinical_by_id.index))
    if missing_clinical:
        raise RuntimeError(f"paired protein IDs absent from clinical data: {missing_clinical}")
    pure_ids = [
        patient
        for patient in paired_ids
        if str(clinical_by_id.loc[patient, "Histologic.type"]).strip() == "LCNEC"
    ]
    combined_ids = [
        patient
        for patient in paired_ids
        if str(clinical_by_id.loc[patient, "Histologic.type"]).strip() == "combined LCNEC"
    ]
    if len(pure_ids) + len(combined_ids) != len(paired_ids):
        raise RuntimeError("unrecognized histologic type among protein pairs")

    rows: list[dict[str, object]] = []
    patient_rows: list[dict[str, object]] = []
    for axis, genes in axes.items():
        for gene in genes:
            measured = gene in proteins.index
            if measured:
                pure_tumor = pd.to_numeric(
                    proteins.loc[gene, [tumor_columns[patient] for patient in pure_ids]],
                    errors="coerce",
                ).to_numpy(float)
                pure_normal = pd.to_numeric(
                    proteins.loc[gene, [normal_columns[patient] for patient in pure_ids]],
                    errors="coerce",
                ).to_numpy(float)
                pure_differences = pure_tumor - pure_normal
                finite = np.isfinite(pure_differences)
                stability = stable_direction(pure_differences)
                p_value = paired_test(pure_differences)

                combined_tumor = pd.to_numeric(
                    proteins.loc[gene, [tumor_columns[patient] for patient in combined_ids]],
                    errors="coerce",
                ).to_numpy(float)
                pure_tumor_finite = pure_tumor[np.isfinite(pure_tumor)]
                combined_tumor_finite = combined_tumor[np.isfinite(combined_tumor)]
                if pure_tumor_finite.size and combined_tumor_finite.size:
                    secondary_p = float(
                        mannwhitneyu(
                            combined_tumor_finite,
                            pure_tumor_finite,
                            alternative="two-sided",
                            method="auto",
                        ).pvalue
                    )
                    secondary_median_delta = float(
                        np.median(combined_tumor_finite) - np.median(pure_tumor_finite)
                    )
                else:
                    secondary_p = 1.0
                    secondary_median_delta = float("nan")

                for index, patient in enumerate(pure_ids):
                    patient_rows.append(
                        {
                            "axis": axis,
                            "gene": gene,
                            "patient_id": patient,
                            "histologic_type": "LCNEC",
                            "tumor": pure_tumor[index],
                            "normal_adjacent": pure_normal[index],
                            "tumor_minus_normal": pure_differences[index],
                        }
                    )
            else:
                finite = np.asarray([], dtype=bool)
                stability = {
                    "mean_effect": float("nan"),
                    "median_effect": float("nan"),
                    "direction": "missing",
                    "mean_median_same_nonzero_sign": False,
                    "leave_one_pair_out_mean_sign_unchanged": False,
                    "nonzero_pair_direction_fraction": float("nan"),
                    "direction_stable": False,
                }
                p_value = 1.0
                secondary_p = 1.0
                secondary_median_delta = float("nan")
                pure_tumor_finite = np.asarray([], dtype=float)
                combined_tumor_finite = np.asarray([], dtype=float)

            rows.append(
                {
                    "axis": axis,
                    "gene": gene,
                    "measured": measured,
                    "primary_pairs": int(np.sum(finite)),
                    **stability,
                    "primary_wilcoxon_p": p_value,
                    "combined_tumors": int(combined_tumor_finite.size),
                    "pure_tumors": int(pure_tumor_finite.size),
                    "secondary_combined_minus_pure_median": secondary_median_delta,
                    "secondary_mannwhitney_p": secondary_p,
                }
            )

    results = pd.DataFrame(rows)
    results["primary_bh_q_22"] = bh_adjust(results["primary_wilcoxon_p"].tolist())
    results["secondary_bh_q_22"] = bh_adjust(results["secondary_mannwhitney_p"].tolist())
    results["primary_protein_gate"] = (
        results["measured"]
        & (results["primary_pairs"] >= int(contract["minimum_pairs"]))
        & results["direction_stable"]
        & (results["primary_bh_q_22"] < 0.10)
    )

    axis_rows: list[dict[str, object]] = []
    for axis, genes in axes.items():
        subset = results[results["axis"] == axis]
        passing = subset[subset["primary_protein_gate"]]
        up = passing[passing["direction"] == "up"]["gene"].tolist()
        down = passing[passing["direction"] == "down"]["gene"].tolist()
        if len(up) >= 2 and len(down) >= 2:
            axis_direction = "mixed_both_directions_pass"
            axis_gate = False
        elif len(up) >= 2:
            axis_direction = "up"
            axis_gate = True
        elif len(down) >= 2:
            axis_direction = "down"
            axis_gate = True
        else:
            axis_direction = "insufficient"
            axis_gate = False
        axis_rows.append(
            {
                "axis": axis,
                "frozen_proteins": len(genes),
                "measured_proteins": int(subset["measured"].sum()),
                "passing_proteins": int(passing.shape[0]),
                "passing_up": ";".join(up),
                "passing_down": ";".join(down),
                "axis_direction": axis_direction,
                "primary_axis_gate": axis_gate,
            }
        )
    axis_results = pd.DataFrame(axis_rows)

    report = {
        "status": "lcnec_independent_proteogenomic_fixed_panel_complete",
        "formal": True,
        "protocol": "frozen 22-protein panel; pure-LCNEC paired tumor-minus-NAT primary endpoint",
        "cohort": {
            "clinical_patients": int(clinical.shape[0]),
            "protein_pairs": len(paired_ids),
            "pure_lcnec_protein_pairs": len(pure_ids),
            "combined_lcnec_protein_pairs": len(combined_ids),
        },
        "panel": {
            "frozen_proteins": len(panel),
            "measured_proteins": int(results["measured"].sum()),
            "missing_proteins": results.loc[~results["measured"], "gene"].tolist(),
            "primary_passing_proteins": results.loc[
                results["primary_protein_gate"], "gene"
            ].tolist(),
        },
        "axes": axis_results.to_dict(orient="records"),
        "primary_axis_gates_passed": axis_results.loc[
            axis_results["primary_axis_gate"], "axis"
        ].tolist(),
        "secondary_status": (
            "exploratory combined-versus-pure tumor-only Mann-Whitney results are reported "
            "separately and cannot rescue the primary paired endpoint"
        ),
        "keap1_status": (
            "not evaluated: the deposited bundle did not provide an author-curated binary label "
            "and the raw mutation table does not exactly reproduce the article percentage"
        ),
        "claim_limit": contract["claim_limit"],
        "provenance": {
            "protein_matrix": str(args.protein_matrix.resolve()),
            "protein_matrix_sha256": sha256(args.protein_matrix),
            "clinical": str(args.clinical.resolve()),
            "clinical_sha256": sha256(args.clinical),
            "preregistration_sha256": sha256(args.preregistration),
            "analysis_contract_sha256": sha256(args.contract),
            "script_sha256": sha256(Path(__file__)),
        },
    }

    results.to_csv(args.output_dir / "protein_results.csv", index=False)
    axis_results.to_csv(args.output_dir / "axis_results.csv", index=False)
    pd.DataFrame(patient_rows).to_csv(
        args.output_dir / "pure_lcnec_patient_pair_differences.csv", index=False
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    readme = [
        "# Independent LCNEC proteogenomic fixed-panel audit",
        "",
        f"- Protein pairs: {len(paired_ids)} ({len(pure_ids)} pure, {len(combined_ids)} combined)",
        f"- Frozen proteins: 22; measured: {int(results['measured'].sum())}",
        f"- Missing without substitution: {', '.join(report['panel']['missing_proteins'])}",
        f"- Primary passing proteins: {', '.join(report['panel']['primary_passing_proteins']) or 'none'}",
        f"- Primary axis gates: {', '.join(report['primary_axis_gates_passed']) or 'none'}",
        "",
        "Protein abundance is context evidence only. It is not metabolite replication, identity validation, enzyme activity, flux, causality, or therapeutic vulnerability.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
