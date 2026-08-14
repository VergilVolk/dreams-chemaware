"""P0: systematic failure-case audit for the cached strict E0 retrieval.

This script does not run or train DreaMS.  It reconstructs every eligible
query from ``e0_pair_arrays.npz``, aggregates candidate spectra by molecule,
and records which chemical/experimental strata are enriched for Top-1 errors.

Ground truth is always molecular identity (14-character InChIKey block).
Chemical-rule overlap is used only as a diagnostic feature.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Top-1 failure modes in cached E0 retrieval outputs.")
    parser.add_argument(
        "--e0-dir", default="data/validation/e0_baseline",
        help="Directory produced by tasks/eval_e0_baseline.py.")
    parser.add_argument(
        "--output-dir", default="data/validation/e0_failure_audit",
        help="Directory for P0 audit outputs.")
    parser.add_argument(
        "--protocol", default="primary",
        help="NPZ prefix, normally 'primary' for [M+H]+.")
    parser.add_argument(
        "--rule-vectors", default="tasks/_cache/rule_vectors/ik_to_rvec.npz",
        help="Optional IK14 -> rule-vector NPZ. Missing file disables rule diagnostics.")
    parser.add_argument(
        "--compute-mces", action="store_true",
        help="Compute thresholded MCES for failures plus sampled correct controls.")
    parser.add_argument(
        "--mces-controls-per-error", type=float, default=1.0,
        help="Number of randomly sampled correct controls per error for MCES.")
    parser.add_argument(
        "--mces-max-pairs", type=int, default=0,
        help="Cap unique MCES pairs (0 means no cap). Useful for a quick pilot.")
    parser.add_argument(
        "--mces-threshold", type=float, default=10.0,
        help="myopic-MCES exact-computation threshold; values above may be bounds.")
    parser.add_argument(
        "--mces-time-limit", type=float, default=10.0,
        help="Per-pair CBC time limit in seconds.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def safe_float(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def clean_text(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return "missing"
    text = str(value).strip()
    return "missing" if not text or text.lower() in {"nan", "none", "null"} else text


def mean_or_none(values: Iterable[Any]) -> Optional[float]:
    vals = [v for v in (safe_float(x) for x in values) if v is not None]
    return float(np.mean(vals)) if vals else None


def wilson_interval(errors: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = errors / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def fmt_bin(value: float, cuts: Sequence[float], labels: Sequence[str]) -> str:
    for cut, label in zip(cuts, labels):
        if value < cut:
            return label
    return labels[-1]


def candidate_count_bin(value: int) -> str:
    if value <= 2:
        return "2"
    if value == 3:
        return "3"
    if value <= 5:
        return "4-5"
    if value <= 10:
        return "6-10"
    return ">10"


def positive_count_bin(value: int) -> str:
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 7:
        return "4-7"
    return ">=8"


def peak_count_bin(value: int) -> str:
    if value <= 10:
        return "<=10"
    if value <= 30:
        return "11-30"
    if value <= 64:
        return "31-64"
    return "65-128"


def mz_bin(value: float) -> str:
    if value < 250:
        return "<250"
    if value < 400:
        return "250-399"
    if value < 600:
        return "400-599"
    return ">=600"


def similarity_bin(value: Optional[float]) -> str:
    if value is None:
        return "missing"
    if value < 0.25:
        return "<0.25"
    if value < 0.50:
        return "0.25-0.49"
    if value < 0.75:
        return "0.50-0.74"
    return ">=0.75"


def jaccard_bin(value: Optional[float]) -> str:
    if value is None:
        return "missing"
    if value < 0.20:
        return "<0.20"
    if value < 0.50:
        return "0.20-0.49"
    if value < 0.75:
        return "0.50-0.74"
    return ">=0.75"


def molecule_features(smiles: str, fpgen: Any) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {
            "valid": False, "formula": "", "canonical_smiles": "",
            "scaffold": "", "ring_class": "invalid", "fingerprint": None,
            "exact_mw": None, "heavy_atoms": None, "rings": None,
            "aromatic_rings": None, "rotatable_bonds": None,
        }
    rings = int(rdMolDescriptors.CalcNumRings(mol))
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    if rings == 0:
        ring_class = "acyclic"
    elif rings == 1:
        ring_class = "one-ring"
    else:
        ring_class = "multi-ring"
    return {
        "valid": True,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "canonical_smiles": Chem.MolToSmiles(mol, isomericSmiles=False),
        "scaffold": scaffold,
        "ring_class": ring_class,
        "fingerprint": fpgen.GetFingerprint(mol),
        "exact_mw": float(Descriptors.ExactMolWt(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "rings": rings,
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
    }


def pair_structure_features(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    if not a["valid"] or not b["valid"]:
        return {
            "same_formula": "missing", "scaffold_relation": "missing",
            "formula_scaffold_group": "missing", "morgan_tanimoto": None,
        }
    same_formula = bool(a["formula"] == b["formula"])
    if not a["scaffold"] and not b["scaffold"]:
        scaffold_relation = "both_acyclic"
    elif a["scaffold"] and a["scaffold"] == b["scaffold"]:
        scaffold_relation = "same_scaffold"
    else:
        scaffold_relation = "different_scaffold"
    formula_scaffold_group = (
        ("same_formula" if same_formula else "different_formula")
        + "__" + scaffold_relation
    )
    tanimoto = float(DataStructs.TanimotoSimilarity(a["fingerprint"], b["fingerprint"]))
    return {
        "same_formula": str(same_formula).lower(),
        "scaffold_relation": scaffold_relation,
        "formula_scaffold_group": formula_scaffold_group,
        "morgan_tanimoto": tanimoto,
    }


def rule_jaccard(rule_npz: Any, available: set, ik_a: str, ik_b: str) -> Optional[float]:
    if rule_npz is None or ik_a not in available or ik_b not in available:
        return None
    a = np.asarray(rule_npz[ik_a], dtype=bool)
    b = np.asarray(rule_npz[ik_b], dtype=bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return None
    return float(np.logical_and(a, b).sum() / union)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def enrichment_rows(records: List[Dict[str, Any]], dimensions: Sequence[str]) -> List[Dict[str, Any]]:
    n_all = len(records)
    n_errors = sum(int(r["is_top1_error"]) for r in records)
    base_error_rate = n_errors / n_all
    output: List[Dict[str, Any]] = []
    for dimension in dimensions:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(record.get(dimension, "missing"))].append(record)
        for category, members in groups.items():
            total = len(members)
            errors = sum(int(r["is_top1_error"]) for r in members)
            error_rate = errors / total
            ci_low, ci_high = wilson_interval(errors, total)
            output.append({
                "dimension": dimension,
                "category": category,
                "n_queries": total,
                "n_errors": errors,
                "error_rate": error_rate,
                "error_rate_ci_low": ci_low,
                "error_rate_ci_high": ci_high,
                "error_share": errors / n_errors if n_errors else 0.0,
                "query_share": total / n_all if n_all else 0.0,
                "enrichment_vs_overall": error_rate / base_error_rate if base_error_rate else None,
            })
    output.sort(key=lambda x: (x["dimension"], -x["enrichment_vs_overall"], -x["n_queries"]))
    return output


def group_enrichment(
    records: List[Dict[str, Any]], field: str, minimum_queries: int = 5
) -> List[Dict[str, Any]]:
    n_all = len(records)
    n_errors = sum(int(r["is_top1_error"]) for r in records)
    base = n_errors / n_all
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(field, "missing"))].append(record)
    rows = []
    for value, members in grouped.items():
        if len(members) < minimum_queries:
            continue
        errors = sum(int(r["is_top1_error"]) for r in members)
        low, high = wilson_interval(errors, len(members))
        rows.append({
            field: value,
            "n_queries": len(members),
            "n_errors": errors,
            "error_rate": errors / len(members),
            "error_rate_ci_low": low,
            "error_rate_ci_high": high,
            "enrichment_vs_overall": (errors / len(members)) / base if base else None,
        })
    rows.sort(key=lambda x: (-x["n_errors"], -x["enrichment_vs_overall"]))
    return rows


def molecule_level_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse replicate query spectra so prolific molecules do not dominate."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["query_ik14"]].append(record)
    output = []
    for ik14, members in grouped.items():
        errors = sum(int(row["is_top1_error"]) for row in members)
        instruments = sorted({row["query_instrument"] for row in members})
        output.append({
            "query_ik14": ik14,
            "query_smiles": members[0]["query_smiles"],
            "query_formula": members[0]["query_formula"],
            "query_scaffold": members[0]["query_scaffold"],
            "n_query_spectra": len(members),
            "n_top1_errors": errors,
            "spectrum_error_rate": errors / len(members),
            "any_top1_error": errors > 0,
            "majority_top1_error": errors > len(members) / 2,
            "all_top1_error": errors == len(members),
            "instruments": "|".join(instruments),
            "mean_candidate_molecules": mean_or_none(row["candidate_molecules"] for row in members),
        })
    output.sort(key=lambda row: (-row["n_top1_errors"], -row["spectrum_error_rate"], row["query_ik14"]))
    return output


def molecule_group_rows(
    molecule_rows: List[Dict[str, Any]], field: str, minimum_molecules: int = 3
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in molecule_rows:
        grouped[str(row[field])].append(row)
    output = []
    overall_macro = float(np.mean([row["spectrum_error_rate"] for row in molecule_rows]))
    for value, members in grouped.items():
        if len(members) < minimum_molecules:
            continue
        mean_rate = float(np.mean([row["spectrum_error_rate"] for row in members]))
        output.append({
            field: value,
            "n_unique_molecules": len(members),
            "n_molecules_with_any_error": sum(bool(row["any_top1_error"]) for row in members),
            "fraction_molecules_with_any_error": float(np.mean([row["any_top1_error"] for row in members])),
            "molecule_macro_spectrum_error_rate": mean_rate,
            "enrichment_vs_molecule_macro": mean_rate / overall_macro if overall_macro else None,
            "n_query_spectra": sum(row["n_query_spectra"] for row in members),
            "n_top1_errors": sum(row["n_top1_errors"] for row in members),
        })
    output.sort(key=lambda row: (-row["n_top1_errors"], -row["enrichment_vs_molecule_macro"]))
    return output


def mces_bin(value: Optional[float]) -> str:
    if value is None or value < 0:
        return "missing"
    if value <= 2:
        return "0-2"
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    return ">10_or_bound"


def compute_mces_subset(
    records: List[Dict[str, Any]], output_dir: Path, seed: int,
    controls_per_error: float, max_pairs: int, threshold: float,
    time_limit: float,
) -> List[Dict[str, Any]]:
    from myopic_mces import MCES

    failures = [r for r in records if r["is_top1_error"]]
    correct = [r for r in records if not r["is_top1_error"]]
    rng = random.Random(seed)
    n_controls = min(len(correct), int(round(len(failures) * controls_per_error)))
    controls = rng.sample(correct, n_controls) if n_controls else []
    selected = failures + controls
    rng.shuffle(selected)

    cache_path = output_dir / "mces_cache.json"
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            cache = json.load(handle)
    else:
        cache = {}

    unique: Dict[str, Tuple[str, str]] = {}
    for row in selected:
        iks = sorted([row["query_ik14"], row["best_negative_ik14"]])
        key = "|".join(iks)
        unique.setdefault(key, (row["query_smiles"], row["best_negative_smiles"]))

    missing = [key for key in unique if key not in cache]
    if max_pairs > 0 and len(missing) > max_pairs:
        rng.shuffle(missing)
        missing = missing[:max_pairs]
    print(f"MCES: {len(unique):,} unique selected pairs; {len(missing):,} to compute")
    for i, key in enumerate(missing, 1):
        smiles_a, smiles_b = unique[key]
        try:
            _, distance, elapsed, mode = MCES(
                smiles_a, smiles_b, threshold=threshold,
                solver_options={"msg": False, "timeLimit": time_limit, "threads": 1},
                catch_errors=True,
            )
            cache[key] = {
                "distance": float(distance), "mode": int(mode),
                "elapsed_seconds": float(elapsed),
            }
        except Exception as exc:  # keep the audit resumable on malformed structures
            cache[key] = {"distance": -1.0, "mode": -1, "error": str(exc)}
        if i % 100 == 0 or i == len(missing):
            print(f"  computed {i:,}/{len(missing):,}")
            with cache_path.open("w", encoding="utf-8") as handle:
                json.dump(cache, handle, indent=2)

    output = []
    for row in selected:
        key = "|".join(sorted([row["query_ik14"], row["best_negative_ik14"]]))
        result = cache.get(key)
        if result is None:
            continue
        output.append({
            "query_spectrum_id": row["query_spectrum_id"],
            "is_top1_error": row["is_top1_error"],
            "query_ik14": row["query_ik14"],
            "best_negative_ik14": row["best_negative_ik14"],
            "same_formula": row["same_formula"],
            "scaffold_relation": row["scaffold_relation"],
            "morgan_tanimoto": row["morgan_tanimoto"],
            "mces": result.get("distance"),
            "mces_bin": mces_bin(result.get("distance")),
            "mces_mode": result.get("mode"),
            "mces_elapsed_seconds": result.get("elapsed_seconds"),
            "query_smiles": row["query_smiles"],
            "best_negative_smiles": row["best_negative_smiles"],
        })
    return output


def case_control_enrichment(rows: List[Dict[str, Any]], field: str) -> List[Dict[str, Any]]:
    """Odds ratios for a categorical feature in errors vs sampled controls."""
    cases = [row for row in rows if row["is_top1_error"]]
    controls = [row for row in rows if not row["is_top1_error"]]
    categories = sorted({str(row[field]) for row in rows if str(row[field]) != "missing"})
    output = []
    for category in categories:
        a = sum(str(row[field]) == category for row in cases)
        b = len(cases) - a
        c = sum(str(row[field]) == category for row in controls)
        d = len(controls) - c
        # Haldane-Anscombe correction keeps zero cells finite.
        aa, bb, cc, dd = (value + 0.5 for value in (a, b, c, d))
        odds_ratio = (aa * dd) / (bb * cc)
        se = math.sqrt(1.0 / aa + 1.0 / bb + 1.0 / cc + 1.0 / dd)
        output.append({
            "dimension": field,
            "category": category,
            "n_errors_in_category": a,
            "n_errors_total": len(cases),
            "n_controls_in_category": c,
            "n_controls_total": len(controls),
            "error_fraction": a / len(cases) if cases else None,
            "control_fraction": c / len(controls) if controls else None,
            "case_control_odds_ratio": odds_ratio,
            "odds_ratio_ci_low": math.exp(math.log(odds_ratio) - 1.959963984540054 * se),
            "odds_ratio_ci_high": math.exp(math.log(odds_ratio) + 1.959963984540054 * se),
        })
    output.sort(key=lambda row: -row["case_control_odds_ratio"])
    return output


def make_plot(records: List[Dict[str, Any]], mces_rows: List[Dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [
        (axes[0, 0], "candidate_molecule_bin", "Candidate molecules per query", ["2", "3", "4-5", "6-10", ">10"]),
        (axes[0, 1], "formula_scaffold_group", "Formula / scaffold relation to best negative", None),
        (axes[1, 0], "morgan_tanimoto_bin", "Morgan similarity to best negative", ["<0.25", "0.25-0.49", "0.50-0.74", ">=0.75"]),
    ]
    for ax, field, title, preferred_order in panels:
        grouped = defaultdict(list)
        for row in records:
            grouped[str(row[field])].append(int(row["is_top1_error"]))
        labels = [label for label, values in grouped.items() if len(values) >= 50]
        if preferred_order is not None:
            labels = [label for label in preferred_order if label in labels]
        rates = [np.mean(grouped[label]) for label in labels]
        bars = ax.bar(range(len(labels)), rates, color="#6b3fa0")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Top-1 error rate")
        ax.set_title(title)
        ax.axhline(np.mean([r["is_top1_error"] for r in records]), color="black", ls="--", lw=1)
        for bar, label in zip(bars, labels):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"n={len(grouped[label]):,}", ha="center", va="bottom", fontsize=8)

    ax = axes[1, 1]
    if mces_rows:
        order = ["0-2", "3-5", "6-10", ">10_or_bound"]
        failures = Counter(r["mces_bin"] for r in mces_rows if r["is_top1_error"])
        controls = Counter(r["mces_bin"] for r in mces_rows if not r["is_top1_error"])
        x = np.arange(len(order))
        nf = max(sum(failures.values()), 1)
        nc = max(sum(controls.values()), 1)
        ax.bar(x - 0.2, [failures[k] / nf for k in order], 0.4, label="Top-1 errors", color="#c44e52")
        ax.bar(x + 0.2, [controls[k] / nc for k in order], 0.4, label="Correct controls", color="#4c72b0")
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_ylabel("Within-group fraction")
        ax.set_title("MCES to best negative (case-control subset)")
        ax.legend(frameon=False)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Run with --compute-mces\nfor MCES case-control panel", ha="center", va="center")

    fig.suptitle("DreaMS E0: systematic Top-1 failure audit", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    e0_dir = Path(args.e0_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (e0_dir / "e0_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    with (e0_dir / "e0_report.json").open(encoding="utf-8") as handle:
        e0_report = json.load(handle)
    embeddings = np.load(e0_dir / "e0_embeddings.npy", mmap_mode="r")
    arrays = np.load(e0_dir / "e0_pair_arrays.npz")
    prefix = args.protocol
    required = [f"{prefix}__{suffix}" for suffix in ("pair_i", "pair_j", "labels", "scores", "query_ids")]
    missing = [name for name in required if name not in arrays.files]
    if missing:
        raise KeyError(f"Protocol {prefix!r} is missing arrays: {missing}")

    pair_i = arrays[f"{prefix}__pair_i"]
    pair_j = arrays[f"{prefix}__pair_j"]
    labels = arrays[f"{prefix}__labels"]
    scores = arrays[f"{prefix}__scores"]
    query_ids = arrays[f"{prefix}__query_ids"]

    rule_path = Path(args.rule_vectors)
    rule_npz = np.load(rule_path, allow_pickle=False) if rule_path.exists() else None
    rule_keys = set(rule_npz.files) if rule_npz is not None else set()
    print(f"Rule diagnostics: {'enabled' if rule_npz is not None else 'disabled'}")

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mol_cache: Dict[str, Dict[str, Any]] = {}

    def mol_features(smiles: str) -> Dict[str, Any]:
        if smiles not in mol_cache:
            mol_cache[smiles] = molecule_features(smiles, fpgen)
        return mol_cache[smiles]

    counts = np.bincount(query_ids)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    records: List[Dict[str, Any]] = []
    for qid in range(len(counts)):
        start, end = int(offsets[qid]), int(offsets[qid + 1])
        qi_values = pair_i[start:end]
        if len(qi_values) == 0 or not np.all(qi_values == qi_values[0]):
            raise RuntimeError(f"Non-contiguous or inconsistent query group {qid}")
        qi = int(qi_values[0])
        qmeta = manifest[qi]
        qik = qmeta["inchikey_14"]
        # Match evaluate_query_level exactly.  Its multiply-then-sum reduction
        # can differ by a few ulps from the einsum scores cached for pair AUC;
        # those ulps decide a handful of exact-score ties.
        query_scores = (
            embeddings[qi:qi + 1] * embeddings[pair_j[start:end]]
        ).sum(axis=-1)

        # score, best spectrum index, label, first spectrum index.  The final
        # field reproduces Python's stable sort in E0 when molecule scores tie:
        # candidate spectra were originally visited in ascending global index.
        candidate_by_ik: Dict[str, Tuple[float, int, int, int]] = {}
        positive_spectra = 0
        negative_spectra = 0
        for pj, label, score in zip(pair_j[start:end], labels[start:end], query_scores):
            pj = int(pj)
            label = int(label)
            score = float(score)
            cik = manifest[pj]["inchikey_14"]
            previous = candidate_by_ik.get(cik)
            if previous is None:
                candidate_by_ik[cik] = (score, pj, label, pj)
            elif score > previous[0]:
                candidate_by_ik[cik] = (score, pj, label, min(previous[3], pj))
            elif pj < previous[3]:
                candidate_by_ik[cik] = (previous[0], previous[1], previous[2], pj)
            if label:
                positive_spectra += 1
            else:
                negative_spectra += 1

        if qik not in candidate_by_ik:
            raise RuntimeError(f"Eligible query {qid} has no molecule-level positive")
        correct_score, correct_idx, _, _ = candidate_by_ik[qik]
        sorted_molecules = sorted(
            candidate_by_ik.items(), key=lambda item: (-item[1][0], item[1][3]))
        rank = next(i for i, (ik, _) in enumerate(sorted_molecules, 1) if ik == qik)
        is_error = rank > 1
        negatives = [
            (score, idx, ik, first_idx)
            for ik, (score, idx, _, first_idx) in candidate_by_ik.items() if ik != qik
        ]
        best_negative_score, best_negative_idx, best_negative_ik, _ = min(
            negatives, key=lambda x: (-x[0], x[3]))

        cmeta = manifest[int(correct_idx)]
        nmeta = manifest[int(best_negative_idx)]
        qfeat = mol_features(qmeta.get("smiles", ""))
        nfeat = mol_features(nmeta.get("smiles", ""))
        struct = pair_structure_features(qfeat, nfeat)
        rj = rule_jaccard(rule_npz, rule_keys, qik, best_negative_ik)
        qce = safe_float(qmeta.get("ce"))
        cce = safe_float(cmeta.get("ce"))
        query_instrument = clean_text(qmeta.get("instrument"))
        correct_instrument = clean_text(cmeta.get("instrument"))
        if query_instrument == "missing" or correct_instrument == "missing":
            same_instrument = "missing"
        else:
            same_instrument = str(query_instrument == correct_instrument).lower()
        record = {
            "query_id": qid,
            "query_spectrum_id": qmeta["spectrum_id"],
            "query_embedding_idx": qi,
            "query_ik14": qik,
            "query_smiles": qmeta.get("smiles", ""),
            "query_formula": qfeat["formula"],
            "query_scaffold": qfeat["scaffold"],
            "query_ring_class": qfeat["ring_class"],
            "query_precursor_mz": qmeta["precursor_mz"],
            "query_adduct": qmeta["adduct"],
            "query_instrument": query_instrument,
            "query_ce": qce,
            "query_n_peaks": qmeta.get("n_peaks"),
            "correct_best_spectrum_id": cmeta["spectrum_id"],
            "correct_best_score": correct_score,
            "correct_best_instrument": correct_instrument,
            "correct_best_ce": cce,
            "best_negative_spectrum_id": nmeta["spectrum_id"],
            "best_negative_ik14": best_negative_ik,
            "best_negative_smiles": nmeta.get("smiles", ""),
            "best_negative_formula": nfeat["formula"],
            "best_negative_scaffold": nfeat["scaffold"],
            "best_negative_score": best_negative_score,
            "score_margin_correct_minus_negative": correct_score - best_negative_score,
            "precursor_delta_da": abs(float(qmeta["precursor_mz"]) - float(nmeta["precursor_mz"])),
            "rank": rank,
            "is_top1_error": is_error,
            "candidate_molecules": len(candidate_by_ik),
            "positive_spectra": positive_spectra,
            "negative_spectra": negative_spectra,
            "same_formula": struct["same_formula"],
            "scaffold_relation": struct["scaffold_relation"],
            "formula_scaffold_group": struct["formula_scaffold_group"],
            "morgan_tanimoto": struct["morgan_tanimoto"],
            "rule_jaccard_335": rj,
            "same_instrument_as_best_positive": same_instrument,
            "ce_delta_to_best_positive": abs(qce - cce) if qce is not None and cce is not None else None,
        }
        record.update({
            "candidate_molecule_bin": candidate_count_bin(record["candidate_molecules"]),
            "positive_spectra_bin": positive_count_bin(record["positive_spectra"]),
            "query_peak_count_bin": peak_count_bin(int(record["query_n_peaks"])),
            "query_mz_bin": mz_bin(float(record["query_precursor_mz"])),
            "morgan_tanimoto_bin": similarity_bin(record["morgan_tanimoto"]),
            "rule_jaccard_bin": jaccard_bin(record["rule_jaccard_335"]),
            "ce_available": "known" if qce is not None else "missing",
        })
        records.append(record)
        if (qid + 1) % 5000 == 0 or qid + 1 == len(counts):
            print(f"Reconstructed {qid + 1:,}/{len(counts):,} eligible queries")

    failures = [row for row in records if row["is_top1_error"]]
    write_csv(output_dir / "e0_query_audit.csv", records)
    write_csv(output_dir / "e0_top1_failures.csv", failures)

    dimensions = [
        "candidate_molecule_bin", "positive_spectra_bin", "query_peak_count_bin",
        "query_mz_bin", "query_instrument", "ce_available", "query_ring_class",
        "same_formula", "scaffold_relation", "formula_scaffold_group",
        "morgan_tanimoto_bin", "rule_jaccard_bin",
        "same_instrument_as_best_positive",
    ]
    enrichment = enrichment_rows(records, dimensions)
    write_csv(output_dir / "error_enrichment.csv", enrichment)
    formula_rows = group_enrichment(records, "query_formula", minimum_queries=5)
    scaffold_rows = group_enrichment(records, "query_scaffold", minimum_queries=5)
    write_csv(output_dir / "formula_error_enrichment.csv", formula_rows)
    write_csv(output_dir / "scaffold_error_enrichment.csv", scaffold_rows)
    molecule_rows = molecule_level_rows(records)
    write_csv(output_dir / "molecule_error_summary.csv", molecule_rows)
    write_csv(
        output_dir / "formula_molecule_error_summary.csv",
        molecule_group_rows(molecule_rows, "query_formula", minimum_molecules=3),
    )
    write_csv(
        output_dir / "scaffold_molecule_error_summary.csv",
        molecule_group_rows(molecule_rows, "query_scaffold", minimum_molecules=3),
    )

    mces_rows: List[Dict[str, Any]] = []
    if args.compute_mces:
        mces_rows = compute_mces_subset(
            records, output_dir, args.seed, args.mces_controls_per_error,
            args.mces_max_pairs, args.mces_threshold, args.mces_time_limit,
        )
        write_csv(output_dir / "mces_case_control.csv", mces_rows)
        write_csv(
            output_dir / "mces_case_control_enrichment.csv",
            case_control_enrichment(mces_rows, "mces_bin"),
        )

    n_queries = len(records)
    n_errors = len(failures)
    observed_recall1 = 1.0 - n_errors / n_queries
    expected_recall1 = e0_report["query_results"][prefix]["recall@1"]
    if abs(observed_recall1 - expected_recall1) > 1e-8:
        raise RuntimeError(
            f"Reconstructed Recall@1 {observed_recall1:.10f} does not match E0 {expected_recall1:.10f}")

    eligible_enrichment = [row for row in enrichment if row["n_queries"] >= 50]
    top_enriched = sorted(
        eligible_enrichment,
        key=lambda x: (-x["enrichment_vs_overall"], -x["n_errors"]),
    )[:15]
    mces_summary = {}
    if mces_rows:
        for status, label in [(True, "errors"), (False, "correct_controls")]:
            subset = [r for r in mces_rows if bool(r["is_top1_error"]) == status]
            mces_summary[label] = {
                "n": len(subset),
                "by_bin": dict(Counter(r["mces_bin"] for r in subset)),
                "mean_exact_or_bound": mean_or_none(r["mces"] for r in subset),
            }
        mces_summary["case_control_enrichment"] = case_control_enrichment(
            mces_rows, "mces_bin")

    summary = {
        "p0_version": "1.0",
        "source_e0_dir": str(e0_dir),
        "protocol": prefix,
        "ground_truth": "same 14-character InChIKey block",
        "candidate_protocol": "same adduct, precursor window <= 10 ppm, duplicate peak hashes excluded",
        "candidate_aggregation": "maximum cosine similarity per candidate IK14",
        "n_eligible_queries": n_queries,
        "n_top1_errors": n_errors,
        "top1_error_rate": n_errors / n_queries,
        "n_unique_query_molecules": len(molecule_rows),
        "n_molecules_with_any_top1_error": sum(bool(row["any_top1_error"]) for row in molecule_rows),
        "molecule_macro_spectrum_error_rate": float(np.mean([
            row["spectrum_error_rate"] for row in molecule_rows])),
        "reconstructed_recall_at_1": observed_recall1,
        "e0_reported_recall_at_1": expected_recall1,
        "mean_candidate_molecules": mean_or_none(r["candidate_molecules"] for r in records),
        "median_candidate_molecules": float(np.median([r["candidate_molecules"] for r in records])),
        "same_formula_best_negative_fraction": float(np.mean([r["same_formula"] == "true" for r in records])),
        "rule_pair_coverage_fraction": float(np.mean([r["rule_jaccard_335"] is not None for r in records])),
        "top_enriched_strata_min_50_queries": top_enriched,
        "mces_case_control": mces_summary,
        "limitations": [
            "IK14 collapses stereochemical variants; this audit cannot measure stereo-isomer retrieval separately.",
            "MCES is computed for all failures plus random correct controls, not every eligible query.",
            "Rule Jaccard is diagnostic only and never defines ground-truth labels.",
            "Formula and scaffold labels are derived from the SMILES stored in the E0 manifest.",
        ],
    }
    with (output_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    make_plot(records, mces_rows, output_dir / "failure_audit.png")

    readme = f"""# E0 failure-case audit (P0)

Protocol: `{prefix}`; same-adduct 10-ppm candidates; molecule scores are the
maximum cosine similarity across spectra sharing the same 14-character InChIKey.

- Eligible queries: **{n_queries:,}**
- Top-1 errors: **{n_errors:,}** ({n_errors / n_queries:.2%})
- Reconstructed Recall@1: **{observed_recall1:.6f}**
- Rule overlap: diagnostic feature only; it is not a label.

Files:

- `e0_query_audit.csv`: all eligible queries and their strongest negative.
- `e0_top1_failures.csv`: Top-1 errors only.
- `error_enrichment.csv`: error rates and enrichment by predefined strata.
- `formula_error_enrichment.csv`: formulas with at least five eligible queries.
- `scaffold_error_enrichment.csv`: scaffolds with at least five eligible queries.
- `molecule_error_summary.csv`: query results collapsed by IK14.
- `formula_molecule_error_summary.csv`: formula audit weighted by unique molecules.
- `scaffold_molecule_error_summary.csv`: scaffold audit weighted by unique molecules.
- `mces_case_control.csv`: failures plus sampled correct controls when MCES is enabled.
- `mces_case_control_enrichment.csv`: MCES-bin case-control odds ratios.
- `audit_summary.json`: machine-readable summary and limitations.
- `failure_audit.png`: compact four-panel overview.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "n_eligible_queries": n_queries,
        "n_top1_errors": n_errors,
        "top1_error_rate": n_errors / n_queries,
        "recall_at_1": observed_recall1,
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
