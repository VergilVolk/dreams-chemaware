"""Complete P0.2-P0.6 after ``audit_e0_failures.py``.

Outputs a deduplicated error-pair taxonomy, a balanced 30-case manual-review
pack with structures/spectra/rule evidence, and a training decision table.
Automated chemical edit labels are deliberately named ``*_candidate`` because
they require expert review before use as scientific ground truth.
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Fragments, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.Scaffolds import MurckoScaffold

def load_rule_engine_class() -> Any:
    """Load the rule engine without importing the full DreaMS model package."""
    module_path = Path(__file__).resolve().parents[1] / "dreams" / "models" / "chem_aware" / "chem_rules.py"
    spec = importlib.util.spec_from_file_location("p0_chem_rules", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load chemical rule engine from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ChemicalRuleEngine


ChemicalRuleEngine = load_rule_engine_class()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete P0 failure taxonomy and manual review pack")
    parser.add_argument("--audit-dir", default="data/validation/e0_failure_audit")
    parser.add_argument("--hdf5", default="data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--manual-cases", type=int, default=30)
    parser.add_argument("--peak-tolerance", type=float, default=0.02)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def as_int(value: Any) -> int:
    return int(float(value))


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def decode(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def pair_key(ik_a: str, ik_b: str) -> str:
    return "|".join(sorted((ik_a, ik_b)))


def mol_info(smiles: str, fpgen: Any) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "mol": None}
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    fg = {}
    for name in dir(Fragments):
        if not name.startswith("fr_"):
            continue
        function = getattr(Fragments, name)
        if callable(function):
            value = int(function(mol))
            if value:
                fg[name] = value
    return {
        "valid": True,
        "mol": mol,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "canonical": Chem.MolToSmiles(mol, isomericSmiles=False),
        "scaffold": scaffold,
        "fingerprint": fpgen.GetFingerprint(mol),
        "atoms": mol.GetNumAtoms(),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "bonds": mol.GetNumBonds(),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "tpsa": Descriptors.TPSA(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "functional_groups": fg,
    }


def changed_functional_groups(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    keys = sorted(set(a.get("functional_groups", {})) | set(b.get("functional_groups", {})))
    return [
        f"{key}:{a['functional_groups'].get(key, 0)}->{b['functional_groups'].get(key, 0)}"
        for key in keys
        if a["functional_groups"].get(key, 0) != b["functional_groups"].get(key, 0)
    ]


def classify_edit(a: Dict[str, Any], b: Dict[str, Any], mces: Optional[float]) -> Tuple[str, List[str]]:
    if not a.get("valid") or not b.get("valid"):
        return "invalid_structure", []
    same_formula = a["formula"] == b["formula"]
    same_graph = a["canonical"] == b["canonical"]
    if not a["scaffold"] and not b["scaffold"]:
        scaffold_relation = "both_acyclic"
    elif a["scaffold"] and a["scaffold"] == b["scaffold"]:
        scaffold_relation = "same_scaffold"
    else:
        scaffold_relation = "different_scaffold"
    fg_changes = changed_functional_groups(a, b)

    if same_graph:
        label = "connectivity_equivalent_label_conflict"
    elif same_formula and mces is not None and mces <= 2:
        if scaffold_relation == "same_scaffold" and not fg_changes:
            label = "positional_or_local_connectivity_candidate"
        elif scaffold_relation == "same_scaffold":
            label = "same_scaffold_functional_group_candidate"
        elif a["rings"] != b["rings"] or a["aromatic_rings"] != b["aromatic_rings"]:
            label = "local_ring_topology_candidate"
        else:
            label = "close_connectivity_different_scaffold_candidate"
    elif same_formula and mces is not None and mces <= 5:
        label = "moderate_constitutional_isomer_candidate"
    elif same_formula:
        label = "different_scaffold_constitutional_isomer_candidate"
    elif scaffold_relation == "same_scaffold":
        label = "same_scaffold_substituent_formula_candidate"
    else:
        label = "near_mass_different_formula_candidate"
    return label, fg_changes


def build_pair_rows(
    failures: List[Dict[str, str]], mces_cache: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in failures:
        grouped[pair_key(row["query_ik14"], row["best_negative_ik14"])].append(row)

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mol_cache: Dict[str, Dict[str, Any]] = {}

    def info(smiles: str) -> Dict[str, Any]:
        if smiles not in mol_cache:
            mol_cache[smiles] = mol_info(smiles, fpgen)
        return mol_cache[smiles]

    pair_rows = []
    representatives = {}
    for key, members in grouped.items():
        representative = min(
            members, key=lambda row: float(row["score_margin_correct_minus_negative"]))
        representatives[key] = representative
        a = info(representative["query_smiles"])
        b = info(representative["best_negative_smiles"])
        mces_entry = mces_cache.get(key, {})
        mces = as_float(mces_entry.get("distance"))
        edit_class, fg_changes = classify_edit(a, b, mces)
        tanimoto = (
            float(DataStructs.TanimotoSimilarity(a["fingerprint"], b["fingerprint"]))
            if a.get("valid") and b.get("valid") else None
        )
        pair_rows.append({
            "pair_key": key,
            "ik_a": key.split("|")[0],
            "ik_b": key.split("|")[1],
            "n_error_query_spectra": len(members),
            "n_error_directions": len({row["query_ik14"] for row in members}),
            "n_query_instruments": len({row["query_instrument"] for row in members}),
            "mean_wrong_score_advantage": float(np.mean([
                -float(row["score_margin_correct_minus_negative"]) for row in members])),
            "max_wrong_score_advantage": max(
                -float(row["score_margin_correct_minus_negative"]) for row in members),
            "mean_candidate_molecules": float(np.mean([
                int(row["candidate_molecules"]) for row in members])),
            "mces": mces,
            "mces_mode": mces_entry.get("mode"),
            "mces_bin": (
                "missing" if mces is None or mces < 0 else
                "0-2" if mces <= 2 else "3-5" if mces <= 5 else
                "6-10" if mces <= 10 else ">10_or_bound"
            ),
            "edit_class_candidate": edit_class,
            "same_formula": a.get("formula") == b.get("formula") if a.get("valid") and b.get("valid") else None,
            "scaffold_relation": (
                "both_acyclic" if a.get("valid") and b.get("valid") and not a["scaffold"] and not b["scaffold"] else
                "same_scaffold" if a.get("valid") and b.get("valid") and a["scaffold"] and a["scaffold"] == b["scaffold"] else
                "different_scaffold"
            ),
            "morgan_tanimoto": tanimoto,
            "rule_jaccard_335_cached": as_float(representative.get("rule_jaccard_335")),
            "functional_group_changes": "|".join(fg_changes[:12]),
            "ring_delta": abs(a.get("rings", 0) - b.get("rings", 0)) if a.get("valid") and b.get("valid") else None,
            "aromatic_ring_delta": abs(a.get("aromatic_rings", 0) - b.get("aromatic_rings", 0)) if a.get("valid") and b.get("valid") else None,
            "hbd_delta": abs(a.get("hbd", 0) - b.get("hbd", 0)) if a.get("valid") and b.get("valid") else None,
            "hba_delta": abs(a.get("hba", 0) - b.get("hba", 0)) if a.get("valid") and b.get("valid") else None,
            "representative_query_spectrum": representative["query_spectrum_id"],
            "representative_correct_spectrum": representative["correct_best_spectrum_id"],
            "representative_wrong_spectrum": representative["best_negative_spectrum_id"],
            "representative_query_ik14": representative["query_ik14"],
            "representative_wrong_ik14": representative["best_negative_ik14"],
            "query_formula": representative["query_formula"],
            "wrong_formula": representative["best_negative_formula"],
            "query_smiles": representative["query_smiles"],
            "wrong_smiles": representative["best_negative_smiles"],
        })
    pair_rows.sort(key=lambda row: (-row["n_error_query_spectra"], -row["max_wrong_score_advantage"]))
    return pair_rows, representatives


def taxonomy_summary(pair_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        groups[row["edit_class_candidate"]].append(row)
    result = []
    for label, rows in groups.items():
        result.append({
            "edit_class_candidate": label,
            "n_unique_error_pairs": len(rows),
            "n_error_query_spectra": sum(row["n_error_query_spectra"] for row in rows),
            "fraction_unique_pairs": len(rows) / len(pair_rows),
            "fraction_error_queries": sum(row["n_error_query_spectra"] for row in rows) /
                                      sum(row["n_error_query_spectra"] for row in pair_rows),
            "median_mces": float(np.median([row["mces"] for row in rows if row["mces"] is not None and row["mces"] >= 0])),
            "median_morgan_tanimoto": float(np.median([row["morgan_tanimoto"] for row in rows if row["morgan_tanimoto"] is not None])),
            "fraction_cached_rule_jaccard_ge_0_75": float(np.mean([
                row["rule_jaccard_335_cached"] is not None and row["rule_jaccard_335_cached"] >= 0.75
                for row in rows])),
            "median_wrong_score_advantage": float(np.median([
                row["mean_wrong_score_advantage"] for row in rows])),
        })
    result.sort(key=lambda row: -row["n_error_query_spectra"])
    return result


def select_cases(pair_rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    used = set()
    # Preserve the most reproducible failures irrespective of taxonomy.
    for row in pair_rows[: min(10, n)]:
        selected.append(row)
        used.add(row["pair_key"])
    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        if row["pair_key"] not in used:
            by_class[row["edit_class_candidate"]].append(row)
    labels = sorted(by_class, key=lambda label: -sum(r["n_error_query_spectra"] for r in by_class[label]))
    while len(selected) < n:
        added = False
        for label in labels:
            if by_class[label] and len(selected) < n:
                row = by_class[label].pop(0)
                selected.append(row)
                used.add(row["pair_key"])
                added = True
        if not added:
            break
    return selected


def hdf5_rows_for_ids(hdf5_path: Path, wanted: Sequence[str]) -> Dict[str, int]:
    wanted_set = set(wanted)
    found = {}
    with h5py.File(hdf5_path, "r") as handle:
        identifiers = handle["IDENTIFIER"][:]
        for index, raw in enumerate(identifiers):
            identifier = decode(raw)
            if identifier in wanted_set:
                found[identifier] = index
    missing = wanted_set - set(found)
    if missing:
        raise KeyError(f"Missing {len(missing)} selected spectrum identifiers from HDF5: {sorted(missing)[:5]}")
    return found


def load_spectrum(handle: h5py.File, index: int) -> Dict[str, Any]:
    spectrum = np.asarray(handle["spectrum"][index], dtype=np.float64)
    mask = spectrum[0] > 0
    mz = spectrum[0, mask]
    intensity = spectrum[1, mask]
    if len(intensity) and intensity.max() > 0:
        intensity = intensity / intensity.max()
    return {
        "mz": mz,
        "intensity": intensity,
        "precursor_mz": float(handle["precursor_mz"][index]),
        "instrument": decode(handle["INSTRUMENT_TYPE"][index]),
        "ce": as_float(handle["COLLISION_ENERGY"][index]),
        "identifier": decode(handle["IDENTIFIER"][index]),
    }


def match_mask(query_mz: np.ndarray, target_mz: np.ndarray, tolerance: float) -> np.ndarray:
    if len(query_mz) == 0 or len(target_mz) == 0:
        return np.zeros(len(query_mz), dtype=bool)
    target = np.sort(target_mz)
    positions = np.searchsorted(target, query_mz)
    left = np.clip(positions - 1, 0, len(target) - 1)
    right = np.clip(positions, 0, len(target) - 1)
    error = np.minimum(np.abs(query_mz - target[left]), np.abs(query_mz - target[right]))
    return error <= tolerance


def peak_diagnostics(query: Dict[str, Any], correct: Dict[str, Any], wrong: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    match_correct = match_mask(query["mz"], correct["mz"], tolerance)
    match_wrong = match_mask(query["mz"], wrong["mz"], tolerance)
    true_only = match_correct & ~match_wrong
    wrong_only = match_wrong & ~match_correct

    def top_peaks(mask: np.ndarray) -> str:
        indices = np.where(mask)[0]
        if not len(indices):
            return ""
        indices = indices[np.argsort(query["intensity"][indices])[::-1][:6]]
        return "|".join(f"{query['mz'][i]:.4f}:{query['intensity'][i]:.3f}" for i in indices)

    union_correct = len(query["mz"]) + len(correct["mz"]) - int(match_correct.sum())
    union_wrong = len(query["mz"]) + len(wrong["mz"]) - int(match_wrong.sum())
    true_intensity = float(query["intensity"][true_only].sum())
    wrong_intensity = float(query["intensity"][wrong_only].sum())
    if true_intensity > wrong_intensity + 0.10:
        direction = "peak_evidence_may_rescue"
    elif wrong_intensity > true_intensity + 0.10:
        direction = "peak_evidence_reinforces_error"
    else:
        direction = "peak_evidence_ambiguous"
    return {
        "query_peak_count": len(query["mz"]),
        "correct_peak_count": len(correct["mz"]),
        "wrong_peak_count": len(wrong["mz"]),
        "query_correct_peak_jaccard": int(match_correct.sum()) / union_correct if union_correct else None,
        "query_wrong_peak_jaccard": int(match_wrong.sum()) / union_wrong if union_wrong else None,
        "true_only_query_peak_count": int(true_only.sum()),
        "wrong_only_query_peak_count": int(wrong_only.sum()),
        "true_only_query_intensity": true_intensity,
        "wrong_only_query_intensity": wrong_intensity,
        "top_true_only_query_peaks": top_peaks(true_only),
        "top_wrong_only_query_peaks": top_peaks(wrong_only),
        "peak_evidence_direction": direction,
        "true_only_mask": true_only,
        "wrong_only_mask": wrong_only,
    }


def exact_rule_vector(engine: ChemicalRuleEngine, spectrum: Dict[str, Any]) -> np.ndarray:
    import torch
    mz = torch.as_tensor(spectrum["mz"], dtype=torch.float32).unsqueeze(0)
    diffs = torch.abs(mz.unsqueeze(-1) - mz.unsqueeze(-2))
    precursor = torch.tensor([[spectrum["precursor_mz"]]], dtype=torch.float32)
    padding = torch.zeros_like(mz, dtype=torch.bool)
    with torch.no_grad():
        vector = engine.get_rule_match_vectors(
            diffs, mz_values=mz, precursor_mz=precursor, padding_mask=padding)
    return (vector[0].cpu().numpy() > 0)


def rule_diagnostics(
    engine: ChemicalRuleEngine, query_vec: np.ndarray, correct_vec: np.ndarray,
    wrong_vec: np.ndarray,
) -> Dict[str, Any]:
    names = [rule.name for rule in engine.rules]
    categories = [rule.category for rule in engine.rules]
    informative = np.array([category in {"CF", "ISO", "NL", "HR"} for category in categories])
    true_mask = query_vec & correct_vec & ~wrong_vec & informative
    wrong_mask = query_vec & wrong_vec & ~correct_vec & informative
    true_names = [names[i] for i in np.where(true_mask)[0]]
    wrong_names = [names[i] for i in np.where(wrong_mask)[0]]
    true_categories = Counter(categories[i] for i in np.where(true_mask)[0])
    wrong_categories = Counter(categories[i] for i in np.where(wrong_mask)[0])
    if len(true_names) >= len(wrong_names) + 2:
        direction = "rule_evidence_may_rescue"
    elif len(wrong_names) >= len(true_names) + 2:
        direction = "rule_evidence_reinforces_error"
    else:
        direction = "rule_evidence_ambiguous"
    union = int(np.logical_or(query_vec, wrong_vec).sum())
    return {
        "actual_query_wrong_rule_jaccard": float(np.logical_and(query_vec, wrong_vec).sum() / union) if union else None,
        "true_support_rule_count": len(true_names),
        "wrong_support_rule_count": len(wrong_names),
        "true_support_rule_categories": json.dumps(true_categories, ensure_ascii=False),
        "wrong_support_rule_categories": json.dumps(wrong_categories, ensure_ascii=False),
        "top_true_support_rules": "|".join(true_names[:10]),
        "top_wrong_support_rules": "|".join(wrong_names[:10]),
        "rule_evidence_direction": direction,
    }


def save_structure_image(case_no: int, row: Dict[str, Any], out_dir: Path) -> Path:
    mols = [Chem.MolFromSmiles(row["query_smiles"]), Chem.MolFromSmiles(row["wrong_smiles"])]
    legends = [
        f"Query / true identity\n{row['representative_query_ik14']}\n{row['query_formula']}",
        f"Top-1 wrong candidate\n{row['representative_wrong_ik14']}\n{row['wrong_formula']}",
    ]
    image = MolsToGridImage(mols, molsPerRow=2, subImgSize=(380, 280), legends=legends, useSVG=False)
    path = out_dir / f"case_{case_no:02d}_structures.png"
    image.save(path)
    return path


def save_spectrum_image(
    case_no: int, query: Dict[str, Any], correct: Dict[str, Any], wrong: Dict[str, Any],
    diagnostics: Dict[str, Any], out_dir: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    entries = [(query, "Query spectrum", "#222222"), (correct, "Best same-identity spectrum", "#4c72b0"), (wrong, "Top-1 wrong spectrum", "#c44e52")]
    for ax, (spectrum, title, color) in zip(axes, entries):
        ax.vlines(spectrum["mz"], 0, spectrum["intensity"], color=color, linewidth=0.8)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("rel. I")
        ax.set_title(f"{title} | {spectrum['instrument']} | CE={spectrum['ce']}", loc="left", fontsize=10)
    true_mask = diagnostics["true_only_mask"]
    wrong_mask = diagnostics["wrong_only_mask"]
    axes[0].scatter(query["mz"][true_mask], query["intensity"][true_mask], s=18, color="#55a868", label="matches true only")
    axes[0].scatter(query["mz"][wrong_mask], query["intensity"][wrong_mask], s=18, color="#c44e52", label="matches wrong only")
    axes[0].legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("m/z")
    fig.tight_layout()
    path = out_dir / f"case_{case_no:02d}_spectra.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def build_manual_review(
    selected: List[Dict[str, Any]], hdf5_path: Path, output_dir: Path,
    tolerance: float,
) -> List[Dict[str, Any]]:
    case_dir = output_dir / "manual_cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    wanted_ids = []
    for row in selected:
        wanted_ids.extend([
            row["representative_query_spectrum"], row["representative_correct_spectrum"],
            row["representative_wrong_spectrum"],
        ])
    index_by_id = hdf5_rows_for_ids(hdf5_path, wanted_ids)
    engine = ChemicalRuleEngine(tolerance=tolerance, use_massbank=False)
    rule_cache = {}
    spectrum_cache = {}
    review_rows = []
    with h5py.File(hdf5_path, "r") as handle:
        for case_no, row in enumerate(selected, 1):
            ids = [
                row["representative_query_spectrum"], row["representative_correct_spectrum"],
                row["representative_wrong_spectrum"],
            ]
            spectra = []
            vectors = []
            for identifier in ids:
                if identifier not in spectrum_cache:
                    spectrum_cache[identifier] = load_spectrum(handle, index_by_id[identifier])
                spectrum = spectrum_cache[identifier]
                spectra.append(spectrum)
                if identifier not in rule_cache:
                    rule_cache[identifier] = exact_rule_vector(engine, spectrum)
                vectors.append(rule_cache[identifier])
            query, correct, wrong = spectra
            peak_info = peak_diagnostics(query, correct, wrong, tolerance)
            rule_info = rule_diagnostics(engine, vectors[0], vectors[1], vectors[2])
            structure_path = save_structure_image(case_no, row, case_dir)
            spectrum_path = save_spectrum_image(case_no, query, correct, wrong, peak_info, case_dir)
            clean_peak_info = {key: value for key, value in peak_info.items() if not key.endswith("_mask")}
            review = dict(row)
            review.update({
                "case_no": case_no,
                "query_instrument_actual": query["instrument"],
                "correct_instrument_actual": correct["instrument"],
                "wrong_instrument_actual": wrong["instrument"],
                "query_ce_actual": query["ce"],
                "correct_ce_actual": correct["ce"],
                "wrong_ce_actual": wrong["ce"],
                **clean_peak_info,
                **rule_info,
                "structure_image": structure_path.name,
                "spectrum_image": spectrum_path.name,
                "expert_edit_class": "",
                "expert_key_evidence": "",
                "expert_keep_for_training": "",
                "expert_notes": "",
            })
            review_rows.append(review)
    write_csv(output_dir / "manual_review_30.csv", review_rows)
    return review_rows


def write_review_html(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    cards = []
    for row in rows:
        cards.append(f"""
        <section class="case">
          <h2>Case {row['case_no']:02d}: {html.escape(row['edit_class_candidate'])}</h2>
          <p><b>Repeated errors:</b> {row['n_error_query_spectra']} &nbsp;
             <b>MCES:</b> {row['mces']} &nbsp; <b>Morgan:</b> {row['morgan_tanimoto']:.3f} &nbsp;
             <b>Wrong-score advantage:</b> {row['max_wrong_score_advantage']:.3f}</p>
          <img src="manual_cases/{html.escape(row['structure_image'])}" class="structure">
          <img src="manual_cases/{html.escape(row['spectrum_image'])}" class="spectrum">
          <table>
            <tr><th>Peak evidence</th><td>{html.escape(row['peak_evidence_direction'])}; true-only intensity={row['true_only_query_intensity']:.3f}, wrong-only={row['wrong_only_query_intensity']:.3f}</td></tr>
            <tr><th>True-only peaks</th><td>{html.escape(row['top_true_only_query_peaks'])}</td></tr>
            <tr><th>Wrong-only peaks</th><td>{html.escape(row['top_wrong_only_query_peaks'])}</td></tr>
            <tr><th>Rule evidence</th><td>{html.escape(row['rule_evidence_direction'])}; true={row['true_support_rule_count']}, wrong={row['wrong_support_rule_count']}</td></tr>
            <tr><th>True-support rules</th><td>{html.escape(row['top_true_support_rules'])}</td></tr>
            <tr><th>Wrong-support rules</th><td>{html.escape(row['top_wrong_support_rules'])}</td></tr>
            <tr><th>FG changes</th><td>{html.escape(row['functional_group_changes'])}</td></tr>
          </table>
          <p class="review"><b>Expert review:</b> edit class ______ &nbsp; key evidence ______ &nbsp; keep for training Y/N ______</p>
        </section>""")
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>P0 manual review</title>
    <style>
      body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:28px;color:#222}}
      .case{{page-break-after:always;border-top:4px solid #6b3fa0;padding-top:10px;margin-bottom:36px}}
      .structure{{width:48%;vertical-align:top}} .spectrum{{width:50%;vertical-align:top}}
      table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border:1px solid #aaa;padding:6px;text-align:left}} th{{width:150px;background:#eee}}
      .review{{padding:12px;border:2px dashed #6b3fa0}}
    </style></head><body><h1>P0 manual chemistry review: 30 representative DreaMS errors</h1>
    <p>Automated edit labels are candidates only. Review the structure pair, spectrum evidence, and rule evidence before assigning a final chemical category.</p>
    {''.join(cards)}</body></html>"""
    (output_dir / "manual_review_30.html").write_text(document, encoding="utf-8")


def decision_rows(pair_rows: List[Dict[str, Any]], review_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    n_pairs = len(pair_rows)
    local = [row for row in pair_rows if row["mces"] is not None and 0 <= row["mces"] <= 2]
    high_rule = [row for row in pair_rows if row["rule_jaccard_335_cached"] is not None and row["rule_jaccard_335_cached"] >= 0.75]
    peak_rescue = sum(row["peak_evidence_direction"] == "peak_evidence_may_rescue" for row in review_rows)
    rule_rescue = sum(row["rule_evidence_direction"] == "rule_evidence_may_rescue" for row in review_rows)
    data_conflicts = [row for row in pair_rows if row["edit_class_candidate"] == "connectivity_equivalent_label_conflict"]
    return [
        {
            "finding": "MCES 0-2 local identity confusion",
            "evidence": f"{len(local)}/{n_pairs} unique error pairs; enriched in P0 case-control analysis",
            "method_decision": "Use identity-labelled MCES-local hard negatives and ordinal local ranking",
            "priority": "P1",
            "gate": "Improve MCES 0-2 Top-1 error rate without lowering overall Recall@1 by >0.5 percentage point",
        },
        {
            "finding": "High cached Rule Jaccard among wrong identities",
            "evidence": f"{len(high_rule)}/{n_pairs} unique pairs have cached Jaccard >=0.75",
            "method_decision": "Use rule overlap to mine conflicts only; identity/MCES remains the label",
            "priority": "P1",
            "gate": "Conflict-set error rate decreases on held-out molecules",
        },
        {
            "finding": "Peak evidence can potentially rescue selected errors",
            "evidence": f"{peak_rescue}/{len(review_rows)} representative cases show stronger true-only peak intensity",
            "method_decision": "Test peak-aware explanation/consistency objective on verified cases",
            "priority": "P2",
            "gate": "Peak deletion intervention changes the decision in the predicted direction",
        },
        {
            "finding": "Rule evidence can potentially rescue selected errors",
            "evidence": f"{rule_rescue}/{len(review_rows)} representative cases have >=2 more true-support than wrong-support rules",
            "method_decision": "Use compact rule concept decoding only after expert validation",
            "priority": "P2",
            "gate": "Rule-decode AUPRC improves and retrieval drop is <=0.5 percentage point",
        },
        {
            "finding": "Possible label/representation conflicts",
            "evidence": f"{len(data_conflicts)}/{n_pairs} pairs are graph-equivalent; tautomer/protomer status is reserved for expert review",
            "method_decision": "Quarantine and manually resolve before training",
            "priority": "P0-data",
            "gate": "No unresolved graph-equivalent pair enters a positive/negative pool",
        },
        {
            "finding": "Acquisition-domain mismatch",
            "evidence": "P0 univariate audit: cross-instrument best positives have 23.75% error rate",
            "method_decision": "Build cross-instrument identity positives; then test controlled peak dropout",
            "priority": "P1",
            "gate": "Cross-instrument error improves without clean same-instrument degradation",
        },
    ]


def write_final_report(
    pair_rows: List[Dict[str, Any]], taxonomy: List[Dict[str, Any]],
    review_rows: List[Dict[str, Any]], decisions: List[Dict[str, Any]], output_dir: Path,
) -> None:
    class_lines = "\n".join(
        f"| {row['edit_class_candidate']} | {row['n_unique_error_pairs']} | {row['n_error_query_spectra']} | {row['fraction_error_queries']:.1%} | {row['median_mces']:.1f} |"
        for row in taxonomy
    )
    decision_lines = "\n".join(
        f"| {row['priority']} | {row['finding']} | {row['method_decision']} | {row['gate']} |"
        for row in decisions
    )
    peak_counts = Counter(row["peak_evidence_direction"] for row in review_rows)
    rule_counts = Counter(row["rule_evidence_direction"] for row in review_rows)
    report = f"""# P0 final: chemical failure taxonomy and training decision

## Scope

- 2,109 Top-1 error query spectra
- {len(pair_rows)} deduplicated unordered molecule pairs
- {len(review_rows)} balanced representative cases with structures, three spectra, peak evidence, and rules recomputed on the actual E0 spectra

Automated edit labels remain candidate categories until the manual-review columns are completed.

## Candidate structural taxonomy

| Candidate class | Unique pairs | Error queries | Error-query share | Median MCES |
|---|---:|---:|---:|---:|
{class_lines}

## Evidence audit in the representative set

- Peak evidence: {dict(peak_counts)}
- Rule evidence: {dict(rule_counts)}

These counts estimate whether the current evidence layer could rescue a model decision; they are not accuracy metrics and the 30 cases are intentionally stratified rather than random.

## Final method decisions

| Priority | Finding | Action | Pass gate |
|---|---|---|---|
{decision_lines}

## What is now determined

The next trainable baseline should be identity supervision with MCES-local hard negatives and cross-instrument positive pairs. Rule overlap remains a mining/diagnostic feature. Rule decoding and peak-level feedback enter only after experts validate the 30-case review pack and after peak-deletion faithfulness tests.

## Manual checkpoint

Open `manual_review_30.html`, review every case, and fill the final four columns in `manual_review_30.csv`. Training should not start from the automated edit labels alone.
"""
    (output_dir / "P0_FINAL_DECISION.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit_dir = Path(args.audit_dir)
    failures = read_csv(audit_dir / "e0_top1_failures.csv")
    with (audit_dir / "mces_cache.json").open(encoding="utf-8") as handle:
        mces_cache = json.load(handle)

    pair_rows, _ = build_pair_rows(failures, mces_cache)
    write_csv(audit_dir / "deduplicated_error_pairs.csv", pair_rows)
    taxonomy = taxonomy_summary(pair_rows)
    write_csv(audit_dir / "structural_taxonomy_summary.csv", taxonomy)
    selected = select_cases(pair_rows, args.manual_cases)
    review_rows = build_manual_review(
        selected, Path(args.hdf5), audit_dir, args.peak_tolerance)
    write_review_html(review_rows, audit_dir)
    decisions = decision_rows(pair_rows, review_rows)
    write_csv(audit_dir / "training_decision_table.csv", decisions)
    write_final_report(pair_rows, taxonomy, review_rows, decisions, audit_dir)

    summary = {
        "n_error_query_spectra": len(failures),
        "n_unique_error_pairs": len(pair_rows),
        "n_manual_cases": len(review_rows),
        "taxonomy": taxonomy,
        "manual_peak_evidence": dict(Counter(row["peak_evidence_direction"] for row in review_rows)),
        "manual_rule_evidence": dict(Counter(row["rule_evidence_direction"] for row in review_rows)),
        "outputs": [
            "deduplicated_error_pairs.csv", "structural_taxonomy_summary.csv",
            "manual_review_30.csv", "manual_review_30.html",
            "training_decision_table.csv", "P0_FINAL_DECISION.md",
        ],
    }
    with (audit_dir / "p0_completion_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps({
        "error_queries": len(failures), "unique_pairs": len(pair_rows),
        "manual_cases": len(review_rows), "output_dir": str(audit_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
