"""Small, training-free stress test for DreaMS failure modes and 3,486 rules.

The pilot answers three deliberately narrow questions:

1. Are the P0 error strata (MCES-local confusion, high-rule-overlap conflict,
   and cross-instrument mismatch) unusually fragile under controlled masking?
2. Does the masking scheme used in DreaMS pretraining preserve retrieval?
3. Do the 3,486 rule vectors preserve useful true-vs-wrong evidence after the
   same peaks become unavailable?

No labels are changed and no model is trained. Identity and the strict 10-ppm,
same-adduct candidate protocol remain the ground truth. Rules are diagnostic
features only.

Primary perturbation (``native_mask``) reproduces the checkpoint's pretraining
configuration: sample fragment peaks proportional to intensity, do not mask the
precursor, and replace selected m/z values by ``mask_val=-1``. The optional
``peak_dropout`` control removes the same selected tokens completely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1 pilot: DreaMS masking x 3,486-rule stress test")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "data/validation/e0_baseline")
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "data/validation/e0_failure_audit")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/rule_noise_pilot")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ppm-tol", type=float, default=10.0)
    parser.add_argument("--n-per-stratum", type=int, default=12)
    parser.add_argument("--n-controls", type=int, default=24)
    parser.add_argument("--mask-rates", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    parser.add_argument("--modes", nargs="+", choices=["native_mask", "peak_dropout"],
                        default=["native_mask", "peak_dropout"])
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--rules-only", action="store_true",
                        help="Validate the 3,486-rule intervention without costly DreaMS re-embedding")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def decode(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def stable_seed(*parts: Any) -> int:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def load_rule_engine_class() -> Any:
    module_path = ROOT / "dreams/models/chem_aware/chem_rules.py"
    spec = importlib.util.spec_from_file_location("pilot_chem_rules", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load rule engine from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ChemicalRuleEngine


def load_backbone(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, Any, Any]:
    from argparse import Namespace
    from dreams.models.dreams.dreams import DreaMS
    from dreams.utils.data import SpectrumPreprocessor
    from dreams.utils.dformats import DataFormatA
    from e1_checkpoint_io import torch_load_compat

    package = torch_load_compat(checkpoint, map_location="cpu")
    if "args" not in package or "state_dict" not in package:
        raise ValueError("This pilot expects the raw SSL checkpoint used by strict E0.")
    args = Namespace(**package["args"])
    args.dformat = DataFormatA()
    for name in [
        "max_mz", "max_peaks_n", "max_tbxic_stdev", "min_peaks_n", "min_charge",
        "max_charge", "max_prec_mz", "high_intensity_thld", "min_intensity_ampl", "max_ms_level",
    ]:
        if name in package["args"]:
            setattr(args.dformat, name, package["args"][name])
    args.d_graphormer_params = 0
    preprocessor = SpectrumPreprocessor(
        dformat=args.dformat,
        n_highest_peaks=int(package["args"].get("max_peaks_n", 128)),
    )
    model = DreaMS(args, preprocessor)
    model.load_state_dict(package["state_dict"], strict=False)
    model.to(device).eval()
    return model, preprocessor, args


def embed(model: torch.nn.Module, spectra: list[np.ndarray], device: torch.device) -> np.ndarray:
    batch = torch.as_tensor(np.stack(spectra), dtype=torch.float32, device=device)
    with torch.no_grad():
        value = model(batch, None)
        if value.ndim == 3:
            value = value[:, 0, :]
        value = F.normalize(value, p=2, dim=-1)
    return value.cpu().numpy()


def candidate_result(
    query_embedding: np.ndarray,
    query_index: int,
    embeddings: np.ndarray,
    manifest: list[dict[str, Any]],
    ppm_tol: float,
) -> dict[str, Any]:
    query = manifest[query_index]
    pmz = float(query["precursor_mz"])
    tolerance = ppm_tol * 1e-6 * pmz
    candidates = [
        i for i, row in enumerate(manifest)
        if i != query_index
        and row["adduct"] == query["adduct"]
        and abs(float(row["precursor_mz"]) - pmz) <= tolerance
        and row["peak_hash"] != query["peak_hash"]
    ]
    if not candidates:
        raise RuntimeError(f"No candidates for {query['spectrum_id']}")
    scores = embeddings[candidates] @ query_embedding
    per_molecule: dict[str, tuple[float, int]] = {}
    for candidate_index, score in zip(candidates, scores):
        ik = manifest[candidate_index]["inchikey_14"]
        previous = per_molecule.get(ik)
        if previous is None or float(score) > previous[0]:
            per_molecule[ik] = (float(score), candidate_index)
    ordered = sorted(per_molecule.items(), key=lambda item: (-item[1][0], item[0]))
    true_ik = query["inchikey_14"]
    true_rows = [(ik, value) for ik, value in ordered if ik == true_ik]
    wrong_rows = [(ik, value) for ik, value in ordered if ik != true_ik]
    if not true_rows or not wrong_rows:
        raise RuntimeError(f"Ineligible strict-E0 query {query['spectrum_id']}")
    rank = 1 + next(i for i, (ik, _) in enumerate(ordered) if ik == true_ik)
    true_score, true_index = true_rows[0][1]
    wrong_ik, (wrong_score, wrong_index) = wrong_rows[0]
    return {
        "top1_correct": int(rank == 1),
        "rank": rank,
        "true_score": true_score,
        "wrong_score": wrong_score,
        "margin": true_score - wrong_score,
        "true_spectrum_id": manifest[true_index]["spectrum_id"],
        "wrong_spectrum_id": manifest[wrong_index]["spectrum_id"],
        "wrong_ik14": wrong_ik,
        "candidate_molecules": len(per_molecule),
    }


def select_queries(args: argparse.Namespace, manifest_by_id: dict[str, int]) -> list[dict[str, Any]]:
    failures = read_csv(args.audit_dir / "e0_top1_failures.csv")
    case_control = read_csv(args.audit_dir / "mces_case_control.csv")
    mces_by_id = {row["query_spectrum_id"]: row for row in case_control}
    rng = np.random.default_rng(args.seed)

    strata: dict[str, list[dict[str, str]]] = {
        "local_mces_0_2": [
            row for row in failures
            if row["query_spectrum_id"] in mces_by_id
            and mces_by_id[row["query_spectrum_id"]].get("mces_bin") == "0-2"
        ],
        "high_rule_conflict": [
            row for row in failures if float(row.get("rule_jaccard_335") or 0) >= 0.75
        ],
        "cross_instrument": [
            row for row in failures if not as_bool(row.get("same_instrument_as_best_positive"))
        ],
    }
    selected: dict[str, dict[str, Any]] = {}
    for label, rows in strata.items():
        rows = [row for row in rows if row["query_spectrum_id"] in manifest_by_id]
        order = rng.permutation(len(rows))[: min(args.n_per_stratum, len(rows))]
        for position in order:
            row = rows[int(position)]
            sid = row["query_spectrum_id"]
            selected.setdefault(sid, {"spectrum_id": sid, "groups": set(), "source": "p0_error"})
            selected[sid]["groups"].add(label)

    controls = [
        row for row in case_control
        if not as_bool(row["is_top1_error"]) and row["query_spectrum_id"] in manifest_by_id
    ]
    # Controls are stratified across MCES bins so they are not dominated by easy remote negatives.
    control_bins: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in controls:
        control_bins[row.get("mces_bin", "unknown")].append(row)
    bins = ["0-2", "3-5", "6-10", ">10_or_bound"]
    quota = max(1, math.ceil(args.n_controls / len(bins)))
    control_ids: list[str] = []
    for label in bins:
        rows = control_bins.get(label, [])
        if rows:
            order = rng.permutation(len(rows))[: min(quota, len(rows))]
            control_ids.extend(rows[int(i)]["query_spectrum_id"] for i in order)
    if len(control_ids) < args.n_controls:
        remaining = [row["query_spectrum_id"] for row in controls if row["query_spectrum_id"] not in control_ids]
        order = rng.permutation(len(remaining))[: args.n_controls - len(control_ids)]
        control_ids.extend(remaining[int(i)] for i in order)
    for sid in control_ids[: args.n_controls]:
        selected.setdefault(sid, {"spectrum_id": sid, "groups": {"matched_correct_control"}, "source": "control"})

    output = []
    for row in selected.values():
        row["groups"] = sorted(row["groups"])
        row["manifest_index"] = manifest_by_id[row["spectrum_id"]]
        output.append(row)
    return sorted(output, key=lambda row: (row["source"], row["spectrum_id"]))


def hdf5_index(path: Path, wanted_ids: Iterable[str]) -> dict[str, int]:
    wanted = set(wanted_ids)
    found: dict[str, int] = {}
    with h5py.File(path, "r") as handle:
        # One bulk read is much faster than 231k scalar HDF5 reads on Windows.
        for index, value in enumerate(handle["IDENTIFIER"][:]):
            sid = decode(value)
            if sid in wanted:
                found[sid] = index
                if len(found) == len(wanted):
                    break
    missing = wanted - set(found)
    if missing:
        raise KeyError(f"Missing HDF5 spectra: {sorted(missing)[:5]}")
    return found


def preprocess_spectrum(handle: h5py.File, row_index: int, preprocessor: Any) -> np.ndarray:
    raw = np.asarray(handle["spectrum"][row_index], dtype=np.float32)
    precursor = float(handle["precursor_mz"][row_index])
    if preprocessor is None:
        valid = raw[0] > 0
        peaks = raw[:, valid].T.copy()
        if len(peaks) and peaks[:, 1].max() > 0:
            peaks[:, 1] /= peaks[:, 1].max()
        # A lightweight precursor token is sufficient for the rule-only path;
        # it is never sent to DreaMS.
        return np.concatenate([
            np.asarray([[precursor, 1.0]], dtype=np.float32), peaks.astype(np.float32)
        ], axis=0)
    return np.asarray(preprocessor(raw, prec_mz=precursor, high_form=False), dtype=np.float32)


def perturb(spec: np.ndarray, rate: float, mode: str, seed: int, mask_val: float) -> tuple[np.ndarray, np.ndarray]:
    result = spec.copy()
    # This reproduces DreaMS MaskedSpectraDataset: valid fragment tokens only,
    # precursor/base-peak tokens (intensity == 1) are protected.
    eligible = np.flatnonzero((spec[:, 0] > 0) & (spec[:, 1] > 0) & (spec[:, 1] < 1))
    selected = np.zeros(len(spec), dtype=bool)
    if len(eligible) <= 1 or rate <= 0:
        return result, selected
    n_mask = max(2, round(len(eligible) * rate))
    n_mask = min(n_mask, len(eligible) - 1)
    probabilities = spec[eligible, 1].astype(np.float64)
    probabilities = probabilities / probabilities.sum() if probabilities.sum() > 0 else None
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=n_mask, replace=False, p=probabilities)
    selected[chosen] = True
    if mode == "native_mask":
        result[chosen, 0] = mask_val
    elif mode == "peak_dropout":
        result[chosen, :] = 0.0
    else:
        raise ValueError(mode)
    return result, selected


class FastRuleMatcher:
    """Exact binary rule matcher without allocating rule x peak x peak tensors."""

    M_H = 1.0078250319

    def __init__(self, engine: Any, tolerance: float):
        self.rules = engine.rules
        self.tolerance = tolerance
        self.groups: dict[str, list[int]] = defaultdict(list)
        for index, rule in enumerate(self.rules):
            self.groups[rule.match_type].append(index)

    @staticmethod
    def _nearest_error(sorted_values: np.ndarray, targets: np.ndarray) -> np.ndarray:
        positions = np.searchsorted(sorted_values, targets)
        left = np.clip(positions - 1, 0, len(sorted_values) - 1)
        right = np.clip(positions, 0, len(sorted_values) - 1)
        return np.minimum(np.abs(targets - sorted_values[left]), np.abs(targets - sorted_values[right]))

    def __call__(self, spec: np.ndarray, precursor_mz: float) -> np.ndarray:
        # The rule engine cannot consume a BERT mask token. A masked peak is
        # treated as unavailable evidence; DreaMS itself still receives -1.
        fragments = spec[1:]
        mz = fragments[(fragments[:, 0] > 0) & (fragments[:, 1] > 0), 0].astype(np.float64)
        output = np.zeros(len(self.rules), dtype=bool)
        if not len(mz):
            return output
        differences = np.sort(np.abs(mz[:, None] - mz[None, :]).reshape(-1))
        sorted_mz = np.sort(mz)

        indices = self.groups.get("mass_diff", [])
        if indices:
            targets = np.asarray([float(self.rules[i].value) for i in indices])
            output[indices] = self._nearest_error(differences, targets) < self.tolerance

        indices = self.groups.get("peak_mz", [])
        if indices:
            targets = np.asarray([float(self.rules[i].value) for i in indices])
            output[indices] = self._nearest_error(sorted_mz, targets) < self.tolerance

        for index in self.groups.get("mass_range", []):
            lo, hi = self.rules[index].value
            position = np.searchsorted(differences, float(lo), side="left")
            output[index] = position < len(differences) and differences[position] <= float(hi)

        for index in self.groups.get("parity", []):
            precursor_parity = int(round(precursor_mz)) % 2
            output[index] = bool(np.any((np.rint(differences).astype(np.int64) % 2) == precursor_parity))

        for index in self.groups.get("mass_diff_range", []):
            lo, hi = self.rules[index].value
            output[index] = bool(np.any((differences > float(hi)) | (differences < float(lo))))

        for index in self.groups.get("hr_shift", []):
            n_h = float(self.rules[index].value)
            if n_h == 0:
                eligible = differences[differences >= 12.0]
                output[index] = bool(len(eligible) and np.any(np.abs(eligible - np.rint(eligible)) < self.tolerance))
            else:
                target = abs(n_h) * self.M_H
                output[index] = bool(self._nearest_error(differences, np.asarray([target]))[0] < self.tolerance)
        return output


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.logical_or(left, right).sum())
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def retention(clean: np.ndarray, noisy: np.ndarray) -> float:
    denominator = int(clean.sum())
    return float(np.logical_and(clean, noisy).sum() / denominator) if denominator else 1.0


def rule_metrics(query: np.ndarray, true: np.ndarray, wrong: np.ndarray) -> dict[str, float]:
    true_only = query & true & ~wrong
    wrong_only = query & wrong & ~true
    return {
        "rule_jaccard_true": jaccard(query, true),
        "rule_jaccard_wrong": jaccard(query, wrong),
        "rule_jaccard_margin": jaccard(query, true) - jaccard(query, wrong),
        "true_only_rule_count": int(true_only.sum()),
        "wrong_only_rule_count": int(wrong_only.sum()),
        "rule_evidence_margin": int(true_only.sum()) - int(wrong_only.sum()),
    }


def bootstrap_mean(values: list[float], seed: int, n_bootstrap: int = 2000) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_bootstrap, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def aggregate(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for group in row["groups"].split("|"):
            grouped[(group, row["mode"], float(row["mask_rate"]))].append(row)
    output = []
    metric_names = [
        "top1_correct", "rank", "margin", "embedding_cosine_to_clean",
        "rule_retention_3486", "rule_jaccard_margin_3486", "rule_evidence_margin_3486",
        "rule_retention_335", "rule_jaccard_margin_335",
    ]
    for (group, mode, rate), values in sorted(grouped.items()):
        item: dict[str, Any] = {"group": group, "mode": mode, "mask_rate": rate, "n_rows": len(values)}
        for metric in metric_names:
            estimate, lo, hi = bootstrap_mean([float(row[metric]) for row in values], stable_seed(seed, group, mode, rate, metric))
            item[metric] = estimate
            item[f"{metric}_ci_low"] = lo
            item[f"{metric}_ci_high"] = hi
        output.append(item)
    return output


def decision_summary(clean_rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    control = [row for row in aggregate_rows if row["group"] == "matched_correct_control"]
    native30 = [row for row in control if row["mode"] == "native_mask" and abs(row["mask_rate"] - 0.30) < 1e-8]
    dropout20 = [row for row in control if row["mode"] == "peak_dropout" and abs(row["mask_rate"] - 0.20) < 1e-8]
    clean_control = [row for row in clean_rows if row["source"] == "control"]
    clean_control_top1 = float(np.mean([row["top1_correct"] for row in clean_control])) if clean_control else float("nan")

    result: dict[str, Any] = {
        "clean_control_top1": clean_control_top1,
        "native_30_control_top1": native30[0]["top1_correct"] if native30 else None,
        "dropout_20_control_top1": dropout20[0]["top1_correct"] if dropout20 else None,
        "interpretation": [],
    }
    if native30:
        drop = clean_control_top1 - float(native30[0]["top1_correct"])
        result["native_30_top1_drop"] = drop
        result["native_mask_feasible"] = bool(drop <= 0.10 and native30[0]["embedding_cosine_to_clean"] >= 0.90)
        result["interpretation"].append(
            "DreaMS-native masking passes the pilot gate." if result["native_mask_feasible"]
            else "DreaMS-native masking is too destructive at 30%; use the lowest passing rate."
        )
    if dropout20:
        drop = clean_control_top1 - float(dropout20[0]["top1_correct"])
        result["dropout_20_top1_drop"] = drop
        result["peak_dropout_feasible"] = bool(drop <= 0.10 and dropout20[0]["embedding_cosine_to_clean"] >= 0.90)
        result["interpretation"].append(
            "Physical peak dropout passes the pilot gate." if result["peak_dropout_feasible"]
            else "Physical peak dropout is not yet safe at 20%; lower the deletion rate."
        )
    return result


def plot_results(aggregate_rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    groups = ["matched_correct_control", "local_mces_0_2", "high_rule_conflict", "cross_instrument"]
    labels = ["Correct controls", "MCES 0-2 errors", "High-rule conflicts", "Cross-instrument errors"]
    if not any(np.isfinite(float(row["embedding_cosine_to_clean"])) for row in aggregate_rows):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        colors = ["#4c78a8", "#59a14f", "#e45756", "#b279a2"]
        native = [row for row in aggregate_rows if row["mode"] == "native_mask"]
        for group, label, color in zip(groups, labels, colors):
            rows = sorted([row for row in native if row["group"] == group], key=lambda row: row["mask_rate"])
            rates = [row["mask_rate"] * 100 for row in rows]
            axes[0].plot(rates, [row["rule_retention_3486"] for row in rows], marker="o", label=label, color=color)
            axes[1].plot(rates, [row["rule_jaccard_margin_3486"] for row in rows], marker="o", label=label, color=color)
        axes[0].set_title("3,486-rule retention")
        axes[0].set_ylabel("Fraction of clean rules retained")
        axes[0].set_ylim(0, 1.02)
        axes[1].set_title("True minus wrong rule-Jaccard margin")
        axes[1].axhline(0, color="black", linewidth=1)
        rows30 = {row["group"]: row for row in native if abs(row["mask_rate"] - 0.30) < 1e-8}
        x = np.arange(len(groups))
        axes[2].bar(x - 0.17, [rows30[g]["rule_jaccard_margin_335"] for g in groups], 0.34,
                    color="#9ecae1", label="335 core")
        axes[2].bar(x + 0.17, [rows30[g]["rule_jaccard_margin_3486"] for g in groups], 0.34,
                    color="#f28e8b", label="3,486 total")
        axes[2].axhline(0, color="black", linewidth=1)
        axes[2].set_title("Core vs expanded rules at 30%")
        axes[2].set_xticks(x, labels, rotation=22, ha="right")
        axes[2].legend(frameon=False)
        for axis in axes[:2]:
            axis.set_xlabel("Masked fragment peaks (%)")
            axis.grid(alpha=0.2)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Rule evidence under intensity-proportional peak masking", fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = {"native_mask": "#4c78a8", "peak_dropout": "#e45756"}
    offsets = {"native_mask": -0.16, "peak_dropout": 0.16}
    for mode in sorted({row["mode"] for row in aggregate_rows}):
        rows30 = {
            row["group"]: row for row in aggregate_rows
            if row["mode"] == mode and abs(row["mask_rate"] - 0.30) < 1e-8
        }
        for axis, metric, title in [
            (axes[0], "top1_correct", "Top-1 after 30% masking"),
            (axes[1], "embedding_cosine_to_clean", "Embedding stability"),
            (axes[2], "rule_retention_3486", "3,486-rule retention"),
        ]:
            values = [rows30.get(group, {}).get(metric, np.nan) for group in groups]
            axis.bar(np.arange(len(groups)) + offsets[mode], values, width=0.30,
                     color=colors[mode], alpha=0.88, label=mode.replace("_", " "))
            axis.set_title(title)
            axis.set_xticks(np.arange(len(groups)), labels, rotation=22, ha="right")
            axis.set_ylim(0, 1.02)
            axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean over query × seed")
    axes[0].legend(frameon=False)
    fig.suptitle("DreaMS failure-mode stress test with controlled peak masking", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    decisions: dict[str, Any],
    n_rules: int,
) -> None:
    lines = [
        "# Rule × noise pilot",
        "",
        "## Scope",
        "",
        f"- Selected query spectra: {len(selected)}",
        f"- Rules evaluated: {n_rules}",
        f"- Mask rates: {', '.join(f'{rate:.0%}' for rate in args.mask_rates)}",
        f"- Repeats per condition: {args.n_seeds}",
        "- Retrieval protocol: same adduct, precursor mass within 10 ppm, molecule-level max aggregation",
        "- No training and no rule-derived labels",
        f"- Rules-only diagnostic: {args.rules_only}",
        "",
        "The primary perturbation exactly follows the raw DreaMS checkpoint metadata: intensity-proportional m/z masking, 30% mask fraction, precursor protected, mask token -1. Peak dropout is a matched missing-peak control.",
        "",
        "## Clean selected queries",
        "",
        "| Source | N | Top-1 | Mean margin |",
        "|---|---:|---:|---:|",
    ]
    for source in ["control", "p0_error"]:
        subset = [row for row in clean_rows if row["source"] == source]
        if subset:
            lines.append(
                f"| {source} | {len(subset)} | {np.mean([r['top1_correct'] for r in subset]):.3f} | "
                f"{np.mean([r['margin'] for r in subset]):.4f} |"
            )
    lines += [
        "",
        "## Aggregate perturbation results",
        "",
        "| Group | Mode | Rate | Top-1 | Embedding cosine | Rule retention | Rule margin |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['group']} | {row['mode']} | {row['mask_rate']:.0%} | "
            f"{row['top1_correct']:.3f} | {row['embedding_cosine_to_clean']:.3f} | "
            f"{row['rule_retention_3486']:.3f} | {row['rule_jaccard_margin_3486']:.3f} |"
        )
    lines += [
        "",
        "## Automatic pilot gates",
        "",
        f"```json\n{json.dumps(decisions, ensure_ascii=False, indent=2)}\n```",
        "",
        "## Interpretation rules",
        "",
        "1. A failure stratum is noise-sensitive only if its within-query margin/cosine deterioration exceeds the matched controls; selection on existing errors alone cannot prove causality.",
        "2. A rule signal is useful only when the true-minus-wrong rule margin is positive and stable across masks. High rule retention by itself is not evidence of correctness.",
        "3. If native masking passes but physical dropout fails, use masked-token consistency rather than deleting peaks during the first training pilot.",
        "4. If both fail, peak masking is not yet a safe fine-tuning augmentation; reduce the rate before changing the model.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch build has no CUDA support; use --device cpu.")
    if any(rate <= 0 or rate >= 1 for rate in args.mask_rates):
        raise ValueError("Mask rates must be between 0 and 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    manifest = json.loads((args.baseline_dir / "e0_manifest.json").read_text(encoding="utf-8"))
    embeddings = np.load(args.baseline_dir / "e0_embeddings.npy", mmap_mode="r")
    manifest_by_id = {row["spectrum_id"]: i for i, row in enumerate(manifest)}
    selected = select_queries(args, manifest_by_id)
    print(f"Selected {len(selected)} queries: " + json.dumps(
        {source: sum(row['source'] == source for row in selected) for source in ['p0_error', 'control']}), flush=True)

    # Lock clean candidate identities before applying noise.
    clean_rows: list[dict[str, Any]] = []
    for row in selected:
        index = int(row["manifest_index"])
        result = candidate_result(np.asarray(embeddings[index]), index, embeddings, manifest, args.ppm_tol)
        clean_rows.append({**row, **result})

    wanted_ids = {row["spectrum_id"] for row in clean_rows}
    wanted_ids.update(row["true_spectrum_id"] for row in clean_rows)
    wanted_ids.update(row["wrong_spectrum_id"] for row in clean_rows)
    h5_rows = hdf5_index(args.data, wanted_ids)

    if args.rules_only:
        from argparse import Namespace

        model = None
        model_args = Namespace(mask_val=-1.0, frac_masks=0.30, mask_intens_strategy="intens_p")
        preprocessor = None
    else:
        model, preprocessor, model_args = load_backbone(args.checkpoint, device)
    mask_val = float(getattr(model_args, "mask_val", -1.0))
    checkpoint_mask_rate = float(getattr(model_args, "frac_masks", 0.30))
    checkpoint_mask_strategy = str(getattr(model_args, "mask_intens_strategy", "intens_p"))
    if checkpoint_mask_strategy != "intens_p":
        raise ValueError(f"Unexpected checkpoint masking strategy: {checkpoint_mask_strategy}")
    print(f"Checkpoint masking: rate={checkpoint_mask_rate}, strategy={checkpoint_mask_strategy}, mask_val={mask_val}", flush=True)

    RuleEngine = load_rule_engine_class()
    engine = RuleEngine(tolerance=args.rule_tolerance, use_massbank=True)
    n_rules = len(engine.rules)
    if n_rules < 3000:
        raise RuntimeError(f"Expected the expanded rule library, got only {n_rules} rules")
    core_n = min(335, n_rules)
    rule_matcher = FastRuleMatcher(engine, args.rule_tolerance)
    print(f"Rule engine: {n_rules} rules ({core_n} core + {n_rules - core_n} expanded)", flush=True)

    spectra: dict[str, np.ndarray] = {}
    precursor_mz: dict[str, float] = {}
    with h5py.File(args.data, "r") as handle:
        for sid, index in h5_rows.items():
            spectra[sid] = preprocess_spectrum(handle, index, preprocessor)
            precursor_mz[sid] = float(handle["precursor_mz"][index])

    # Candidate rule vectors are fixed because only the query is perturbed.
    clean_rule_cache: dict[str, np.ndarray] = {}
    for sid in sorted(wanted_ids):
        clean_rule_cache[sid] = rule_matcher(spectra[sid], precursor_mz[sid])

    detail_rows: list[dict[str, Any]] = []
    for clean in clean_rows:
        sid = clean["spectrum_id"]
        clean_spec = spectra[sid]
        clean_embedding = np.asarray(embeddings[int(clean["manifest_index"])])
        clean_rules = clean_rule_cache[sid]
        true_rules = clean_rule_cache[clean["true_spectrum_id"]]
        wrong_rules = clean_rule_cache[clean["wrong_spectrum_id"]]
        for mode in args.modes:
            for rate in args.mask_rates:
                perturbed_batch = []
                masks = []
                for repeat in range(args.n_seeds):
                    noisy, selected_mask = perturb(
                        clean_spec, rate, mode,
                        stable_seed(args.seed, sid, rate, repeat), mask_val,
                    )
                    perturbed_batch.append(noisy)
                    masks.append(selected_mask)
                noisy_embeddings = (
                    np.repeat(clean_embedding[None, :], len(perturbed_batch), axis=0)
                    if args.rules_only else embed(model, perturbed_batch, device)
                )
                for repeat, (noisy_spec, noisy_embedding, selected_mask) in enumerate(
                    zip(perturbed_batch, noisy_embeddings, masks)
                ):
                    retrieval = (
                        {
                            "top1_correct": float("nan"), "rank": float("nan"),
                            "true_score": float("nan"), "wrong_score": float("nan"),
                            "margin": float("nan"),
                            **{key: clean[key] for key in [
                                "true_spectrum_id", "wrong_spectrum_id", "wrong_ik14", "candidate_molecules",
                            ]},
                        }
                        if args.rules_only else candidate_result(
                            noisy_embedding, int(clean["manifest_index"]), embeddings, manifest, args.ppm_tol)
                    )
                    noisy_rules = rule_matcher(noisy_spec, precursor_mz[sid])
                    metrics_full = rule_metrics(noisy_rules, true_rules, wrong_rules)
                    metrics_core = rule_metrics(noisy_rules[:core_n], true_rules[:core_n], wrong_rules[:core_n])
                    detail_rows.append({
                        "spectrum_id": sid,
                        "source": clean["source"],
                        "groups": "|".join(clean["groups"]),
                        "mode": mode,
                        "mask_rate": rate,
                        "repeat": repeat,
                        "n_masked": int(selected_mask.sum()),
                        "clean_top1_correct": clean["top1_correct"],
                        "clean_rank": clean["rank"],
                        "clean_margin": clean["margin"],
                        **retrieval,
                        "embedding_cosine_to_clean": (
                            float("nan") if args.rules_only else float(np.dot(noisy_embedding, clean_embedding))
                        ),
                        "rule_retention_3486": retention(clean_rules, noisy_rules),
                        "rule_jaccard_clean_noisy_3486": jaccard(clean_rules, noisy_rules),
                        **{f"{key}_3486": value for key, value in metrics_full.items()},
                        "rule_retention_335": retention(clean_rules[:core_n], noisy_rules[:core_n]),
                        "rule_jaccard_clean_noisy_335": jaccard(clean_rules[:core_n], noisy_rules[:core_n]),
                        **{f"{key}_335": value for key, value in metrics_core.items()},
                    })
        print(f"Completed {sid}", flush=True)

    aggregate_rows = aggregate(detail_rows, args.seed)
    decisions = (
        {"status": "rules_only", "interpretation": [
            "Rule stability was evaluated, but masking feasibility for DreaMS requires re-embedding."
        ]}
        if args.rules_only else decision_summary(clean_rows, aggregate_rows)
    )
    write_csv(args.output_dir / "selected_queries.csv", [
        {**row, "groups": "|".join(row["groups"])} for row in clean_rows
    ])
    write_csv(args.output_dir / "per_query_perturbation.csv", detail_rows)
    write_csv(args.output_dir / "aggregate_results.csv", aggregate_rows)
    (args.output_dir / "pilot_summary.json").write_text(
        json.dumps({
            "n_selected_queries": len(selected),
            "n_rules": n_rules,
            "checkpoint_mask_rate": checkpoint_mask_rate,
            "checkpoint_mask_strategy": checkpoint_mask_strategy,
            "conditions": {"rates": args.mask_rates, "modes": args.modes, "n_seeds": args.n_seeds},
            "rules_only": args.rules_only,
            "decision_gates": decisions,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not args.skip_plot:
        plot_results(aggregate_rows, args.output_dir / "rule_noise_pilot.png")
    write_report(args, selected, clean_rows, aggregate_rows, decisions, n_rules)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "selected_queries": len(selected),
        "detail_rows": len(detail_rows),
        "rules": n_rules,
        "decision_gates": decisions,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
