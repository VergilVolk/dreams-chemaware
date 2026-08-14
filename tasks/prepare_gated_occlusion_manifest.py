"""Convert gated reranker peak localization into the occlusion audit schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def joined(values: pd.Series) -> str:
    return ";".join(f"{value:.5f}" for value in values.astype(float))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--localization-dir", type=Path, default=Path("data/validation/gated_reranker_peak_localization"))
    parser.add_argument("--manifest", type=Path, default=Path("data/validation/large_observability_embeddings_confirmation/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/gated_reranker_occlusion_manifest"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(args.localization_dir / "changed_query_peak_summary.csv")
    peaks = pd.read_csv(args.localization_dir / "changed_query_peak_table.csv")
    manifest = pd.read_csv(args.manifest)
    rows = []
    for case in cases.itertuples(index=False):
        selected = peaks.loc[peaks["query_index"] == case.query_index]
        identity = selected.loc[selected["evidence_class"] == "identity_only"]
        confounder = selected.loc[selected["evidence_class"] == "confounder_only"]
        if identity.empty and confounder.empty:
            continue
        query = manifest.loc[int(case.query_index)]
        rows.append({
            "split": "confirmation", "query_index": int(case.query_index),
            "query_hdf5_row": int(query.hdf5_row), "query_precursor_mz": float(query.precursor_mz),
            "ik14": case.ik14, "formula": case.formula, "ring_class": query.ring_class,
            "audit_quadrant": case.transition, "robust_model_residual_candidate": case.transition == "fixed",
            "fragment_identity_support_mz": joined(identity["mz"]),
            "fragment_confounder_support_mz": joined(confounder["mz"]),
            "fragment_all_confounder_support_mz": joined(confounder["mz"]),
        })
    output = pd.DataFrame(rows)
    output.to_csv(args.output_dir / "confirmation_peak_evidence.csv", index=False)
    print(f"Prepared {len(output)} changed queries for paired occlusion")


if __name__ == "__main__":
    main()
