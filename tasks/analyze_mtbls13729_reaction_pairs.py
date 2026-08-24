#!/usr/bin/env python
"""Test reaction-level lipid ratios in MTBLS13729.

The analysis asks whether matched substrate/product abundance ratios shift in
Rmu, which is closer to an enzymatic hypothesis than pathway enrichment. It is
still a steady-state abundance analysis and must not be called flux.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.stats import ttest_1samp, ttest_ind

from analyze_mtbls13729_paired_ms1 import bh_adjust, parse_samples


SPHINGO_RE = re.compile(r"ceramide|\bcer\b|sphing|hexcer|glccer|galcer|lactosyl|ganglios", re.I)
FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_counts(formula: str) -> dict[str, int]:
    return {element: int(count or 1) for element, count in FORMULA_RE.findall(formula)}


def formula_key(counts: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((element, count) for element, count in counts.items() if count))


def shifted(counts: dict[str, int], delta: dict[str, int]) -> tuple[tuple[str, int], ...] | None:
    out = dict(counts)
    for element, value in delta.items():
        out[element] = out.get(element, 0) + value
        if out[element] < 0:
            return None
    return formula_key(out)


def annotate_formulas(best: pd.DataFrame) -> pd.DataFrame:
    formulas = []
    for smiles in best["best_smiles"].fillna(""):
        mol = Chem.MolFromSmiles(str(smiles)) if smiles else None
        formulas.append(rdMolDescriptors.CalcMolFormula(mol) if mol is not None else "")
    out = best.copy()
    out["neutral_formula"] = formulas
    out["sphingolipid_name_support"] = out["best_name"].fillna("").str.contains(SPHINGO_RE)
    return out


def reaction_candidates(annotated: pd.DataFrame) -> pd.DataFrame:
    lookup: dict[tuple[tuple[str, int], ...], list[pd.Series]] = defaultdict(list)
    for _, row in annotated.iterrows():
        if row.neutral_formula and bool(row.sphingolipid_name_support):
            lookup[formula_key(formula_counts(row.neutral_formula))].append(row)
    rows = []
    seen = set()
    reactions = {
        "putative_desaturation_minus_H2": {"H": -2},
        "putative_hexosylation_plus_C6H10O5": {"C": 6, "H": 10, "O": 5},
    }
    for _, substrate in annotated.iterrows():
        if not substrate.neutral_formula or not bool(substrate.sphingolipid_name_support):
            continue
        counts = formula_counts(substrate.neutral_formula)
        for reaction, delta in reactions.items():
            target_key = shifted(counts, delta)
            if target_key is None:
                continue
            for product in lookup.get(target_key, []):
                key = (reaction, int(substrate.feature_id), int(product.feature_id))
                if key in seen or substrate.feature_id == product.feature_id:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "reaction": reaction,
                        "substrate_feature_id": int(substrate.feature_id),
                        "product_feature_id": int(product.feature_id),
                        "substrate_name": substrate.best_name,
                        "product_name": product.best_name,
                        "substrate_formula": substrate.neutral_formula,
                        "product_formula": product.neutral_formula,
                        "substrate_cosine": substrate.max_cosine,
                        "product_cosine": product.max_cosine,
                        "substrate_evidence_tier": substrate.annotation_evidence_tier,
                        "product_evidence_tier": product.annotation_evidence_tier,
                    }
                )
    return pd.DataFrame(rows)


def paired_ratio_deltas(ratio: pd.Series, meta: pd.DataFrame, tumor_suffix: str, normal_suffix: str) -> np.ndarray:
    indexed = meta.set_index(["patient", "suffix"])["sample_name"]
    patients = sorted(set(meta.loc[meta["suffix"] == tumor_suffix, "patient"]))
    values = []
    for patient in patients:
        try:
            tumor, normal = indexed.loc[(patient, tumor_suffix)], indexed.loc[(patient, normal_suffix)]
        except KeyError:
            continue
        if np.isfinite(ratio.get(tumor, math.nan)) and np.isfinite(ratio.get(normal, math.nan)):
            values.append(float(ratio[tumor] - ratio[normal]))
    return np.asarray(values, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eic-dir", type=Path, default=Path("data/mtbls13729/ms1_eic_requant"))
    parser.add_argument("--annotation-dir", type=Path, default=Path("data/mtbls13729/ms1_ms2_link"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/reaction_pair_analysis"))
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    parser.add_argument("--min-pairs", type=int, default=6)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "complete", "panels": {}}
    for panel in args.panels:
        best_path = args.annotation_dir / f"{panel}__feature_best_annotations.csv.gz"
        auc_path = args.eic_dir / f"{panel}__eic_auc_matrix.csv.gz"
        detect_path = args.eic_dir / f"{panel}__eic_detection_matrix.csv.gz"
        if not best_path.exists() or not auc_path.exists() or not detect_path.exists():
            report["panels"][panel] = {"status": "missing_inputs"}
            continue
        best = annotate_formulas(pd.read_csv(best_path))
        best.to_csv(out / f"{panel}__annotations_with_formula.csv.gz", index=False)
        candidates = reaction_candidates(best)
        candidates.to_csv(out / f"{panel}__reaction_candidates.csv", index=False)
        if candidates.empty:
            report["panels"][panel] = {"status": "no_reaction_candidates", "n_sphingolipid_annotations": int(best["sphingolipid_name_support"].sum())}
            continue

        auc = pd.read_csv(auc_path).set_index("feature_id")
        detected = pd.read_csv(detect_path).set_index("feature_id").astype(bool)
        auc = auc.where(detected & (auc > 0))
        meta = parse_samples(list(auc.columns))
        positive = auc.stack()
        pseudo = float(np.percentile(positive, 1) / 2) if len(positive) else 1.0
        log_auc = np.log2(auc + pseudo)
        stats_rows = []
        for row in candidates.itertuples(index=False):
            if row.substrate_feature_id not in log_auc.index or row.product_feature_id not in log_auc.index:
                continue
            ratio = log_auc.loc[row.product_feature_id] - log_auc.loc[row.substrate_feature_id]
            rmu = paired_ratio_deltas(ratio, meta, "Rmu", "RN")
            rtu = paired_ratio_deltas(ratio, meta, "Rtu", "RN")
            record = row._asdict()
            record.update({"n_rmu_pairs": len(rmu), "n_rtu_pairs": len(rtu)})
            if len(rmu) >= args.min_pairs:
                record.update(
                    {
                        "rmu_ratio_delta_mean": float(np.mean(rmu)),
                        "rmu_ratio_delta_median": float(np.median(rmu)),
                        "rmu_ratio_p": float(ttest_1samp(rmu, 0).pvalue) if np.std(rmu) > 0 else 1.0,
                        "rmu_loo_sign_stability": float(np.mean([np.sign(np.mean(np.delete(rmu, i))) == np.sign(np.mean(rmu)) for i in range(len(rmu))])),
                    }
                )
            if len(rmu) >= args.min_pairs and len(rtu) >= args.min_pairs:
                record.update(
                    {
                        "subtype_interaction": float(np.mean(rmu) - np.mean(rtu)),
                        "subtype_interaction_p": float(ttest_ind(rmu, rtu, equal_var=False).pvalue),
                    }
                )
            stats_rows.append(record)
        stats = pd.DataFrame(stats_rows)
        if "rmu_ratio_p" not in stats:
            stats["rmu_ratio_p"] = math.nan
        if "subtype_interaction_p" not in stats:
            stats["subtype_interaction_p"] = math.nan
        stats["rmu_ratio_q"] = bh_adjust(stats["rmu_ratio_p"])
        stats["subtype_interaction_q"] = bh_adjust(stats["subtype_interaction_p"])
        stats.to_csv(out / f"{panel}__reaction_pair_stats.csv", index=False)
        report["panels"][panel] = {
            "status": "complete",
            "n_sphingolipid_annotations": int(best["sphingolipid_name_support"].sum()),
            "n_reaction_candidates": len(candidates),
            "n_tested_rmu": int(stats["rmu_ratio_p"].notna().sum()),
            "boundary": "Steady-state substrate/product ratios; not flux. Candidate formulas/names require fragment and standard review.",
        }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
