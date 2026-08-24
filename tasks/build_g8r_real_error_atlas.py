"""Build a P3-disjoint real-error atlas for official DreaMS and frozen P2b.

The atlas is a training-data diagnostic, not an evaluation result.  It keeps
complete strict-10ppm candidate groups and materializes three aligned tables:

* query_summary.csv.gz: one row per query and its DreaMS/P2b transition;
* candidate_edges.csv.gz: one edge per query/candidate molecule;
* spectrum_pair_edges.csv.gz: one edge per query/candidate spectrum pair.

Peak-rule overlap is computed only for the query and the two predicted top
candidate spectra.  Rule hits are observed mass motifs, never structure labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from build_spectrum_rule_label_cache import spectrum_rule_vector  # noqa: E402
from g8r_p2_rank_fusion_core import (  # noqa: E402
    fuse_one_query,
    fusion_configuration_from_mapping,
    grouped_max,
    normalize_pair_features,
    unique_top_index,
)


DEFAULT_CACHE = ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz"
DEFAULT_ARTIFACT = ROOT / "data/validation/g8r_p2b_rank_fusion.json"
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_P3 = ROOT / "data/validation/g8r_p3_test"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_real_error_atlas"
GRADE_NAME = {-2: "identity", -1: "unknown", 0: "near", 1: "mid", 2: "far"}
TRANSITIONS = ("protected_correct", "corrected", "introduced", "persistent_wrong")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--p3-dir", type=Path, default=DEFAULT_P3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rules", type=Path,
                        default=ROOT / "dreams/models/chem_aware/chem_rules_data.json")
    parser.add_argument("--skip-rule-audit", action="store_true")
    parser.add_argument(
        "--rule-control-cap", type=int, default=2000,
        help="Deterministic protected-correct controls; all error/changed queries are always kept.",
    )
    parser.add_argument("--allow-missing-p3", action="store_true",
                        help="Smoke tests only. Formal construction must fail closed.")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Smoke-only deterministic prefix; 0 means every query.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def decode(value) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def collect_ik14(value, output: set[str]) -> None:
    """Recursively collect identities from heterogeneous sealed manifests."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"ik14", "query_ik14", "inchikey_14"} and isinstance(item, str):
                output.add(item[:14])
            else:
                collect_ik14(item, output)
    elif isinstance(value, list):
        for item in value:
            collect_ik14(item, output)


def load_p3_identities(path: Path, allow_missing: bool = False) -> set[str]:
    if not path.is_dir():
        if allow_missing:
            return set()
        raise FileNotFoundError(f"sealed P3 directory is required: {path}")
    identities: set[str] = set()
    manifests = sorted(path.glob("p3_*manifest.json"))
    if not manifests and not allow_missing:
        raise RuntimeError(f"no sealed P3 manifests found in {path}")
    for manifest in manifests:
        body = json.loads(manifest.read_text(encoding="utf-8"))
        queries = body.get("queries")
        if not isinstance(queries, list):
            raise RuntimeError(f"sealed manifest has no query list: {manifest}")
        for query in queries:
            if not isinstance(query, dict) or not isinstance(query.get("ik14"), str):
                raise RuntimeError(f"malformed sealed query in {manifest}")
            identities.add(str(query["ik14"])[:14])
    if not identities and not allow_missing:
        raise RuntimeError("sealed P3 manifests contained no query identities")
    return identities


class Cache:
    REQUIRED = {
        "feature_names", "features", "pair_candidate_row", "query_ptr",
        "molecule_ptr", "molecule_label", "molecule_ik14", "molecule_formula",
        "molecule_mces_grade", "query_row", "query_ik14", "query_formula",
        "query_has_near",
    }

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=True) as body:
            missing = self.REQUIRED - set(body.files)
            if missing:
                raise RuntimeError(f"error-atlas cache missing arrays: {sorted(missing)}")
            for name in body.files:
                setattr(self, name, body[name])
        self.feature_names = list(map(str, self.feature_names))
        self.features = np.asarray(self.features, dtype=np.float64)
        self.pair_candidate_row = np.asarray(self.pair_candidate_row, dtype=np.int64)
        self.query_ptr = np.asarray(self.query_ptr, dtype=np.int64)
        self.molecule_ptr = np.asarray(self.molecule_ptr, dtype=np.int64)
        self.molecule_label = np.asarray(self.molecule_label, dtype=np.int8)
        self.molecule_mces_grade = np.asarray(self.molecule_mces_grade, dtype=np.int8)
        self.query_row = np.asarray(self.query_row, dtype=np.int64)
        self.query_ik14 = np.asarray(self.query_ik14, dtype=object)
        self.query_formula = np.asarray(self.query_formula, dtype=object)
        self.query_has_near = np.asarray(self.query_has_near, dtype=bool)
        self.n_queries = len(self.query_ptr) - 1
        if self.features.ndim != 2 or self.features.shape[1] != len(self.feature_names):
            raise RuntimeError("feature matrix/name mismatch")
        if self.query_ptr[0] != 0 or self.query_ptr[-1] != len(self.molecule_label):
            raise RuntimeError("query_ptr does not span candidate molecules")
        if self.molecule_ptr[0] != 0 or self.molecule_ptr[-1] != len(self.features):
            raise RuntimeError("molecule_ptr does not span spectrum pairs")
        if len(self.pair_candidate_row) != len(self.features):
            raise RuntimeError("candidate rows do not align to spectrum pairs")
        if len(self.query_row) != self.n_queries or len(self.query_ik14) != self.n_queries:
            raise RuntimeError("query metadata is not aligned")
        if np.any(np.diff(self.query_ptr) < 2) or np.any(np.diff(self.molecule_ptr) < 1):
            raise RuntimeError("every query needs >=2 molecules and every molecule >=1 pair")
        for left, right in zip(self.query_ptr[:-1], self.query_ptr[1:]):
            labels = self.molecule_label[left:right]
            if len(labels) < 2 or labels[0] != 1 or labels.sum() != 1:
                raise RuntimeError("positive molecule must be unique and first in every group")


def strict_candidate_ranks(scores: np.ndarray) -> np.ndarray:
    """Every tie counts against the candidate, matching the project protocol."""
    values = np.asarray(scores, dtype=np.float64)
    return np.asarray([
        1 + int(np.sum(np.delete(values, index) >= value))
        for index, value in enumerate(values)
    ], dtype=np.int32)


def transition_name(base_correct: bool, final_correct: bool) -> str:
    if base_correct and final_correct:
        return "protected_correct"
    if not base_correct and final_correct:
        return "corrected"
    if base_correct and not final_correct:
        return "introduced"
    return "persistent_wrong"


def read_rows(handle: h5py.File, rows: np.ndarray, names: list[str]) -> dict[int, dict]:
    unique = np.unique(np.asarray(rows, dtype=np.int64))
    result = {int(row): {} for row in unique}
    order = np.argsort(unique, kind="mergesort")
    sorted_rows = unique[order]
    for name in names:
        if name not in handle:
            continue
        values = handle[name][sorted_rows]
        for row, value in zip(sorted_rows, values):
            if np.isscalar(value) and isinstance(value, (float, np.floating)):
                result[int(row)][name] = float(value) if np.isfinite(value) else None
            else:
                result[int(row)][name] = decode(value)
    return result


@lru_cache(maxsize=None)
def scaffold(smiles: str) -> str:
    if not smiles or smiles == "nan":
        return ""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    except Exception:
        return ""


def scalar(meta: dict, name: str, default=""):
    value = meta.get(name, default)
    return default if value is None else value


def winner_index(scores: np.ndarray) -> int | None:
    return unique_top_index(np.asarray(scores, dtype=np.float64))


def candidate_grade_name(value: int) -> str:
    return GRADE_NAME.get(int(value), f"grade_{int(value)}")


def rule_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    # Two spectra with no observable rule hit do not provide evidence of
    # chemical agreement.  Calling that case Jaccard=1 would manufacture a
    # false "perfect match" cohort.
    return float(np.logical_and(left, right).sum() / union) if union else float("nan")


def rule_comparison(left: np.ndarray, right: np.ndarray, categories: np.ndarray) -> dict:
    output = {"rule_jaccard": rule_jaccard(left, right)}
    for category in ("CF", "NL", "ISO", "HR", "NR", "EE"):
        mask = categories == category
        output[f"rule_shared_{category.lower()}"] = int(np.logical_and(left[mask], right[mask]).sum())
        output[f"rule_query_{category.lower()}"] = int(left[mask].sum())
        output[f"rule_candidate_{category.lower()}"] = int(right[mask].sum())
    return output


def top_formula_counts(frame: pd.DataFrame, n: int = 30) -> list[dict]:
    if frame.empty:
        return []
    counts = frame["query_formula"].astype(str).value_counts().head(n)
    return [{"formula": str(key), "queries": int(value)} for key, value in counts.items()]


def main() -> None:
    args = parse_args()
    if args.max_queries < 0:
        raise ValueError("--max-queries must be non-negative")
    if args.rule_control_cap < 0:
        raise ValueError("--rule-control-cap must be non-negative")
    for required in (args.cache, args.artifact, args.data):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite atlas: {args.output_dir}")

    cache = Cache(args.cache)
    if args.max_queries:
        if args.max_queries > cache.n_queries:
            raise ValueError("--max-queries exceeds cache size")
        n_queries = args.max_queries
    else:
        n_queries = cache.n_queries
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    configuration = fusion_configuration_from_mapping(artifact["configuration"])
    selected_features = list(map(str, artifact["selected_features"]))
    if selected_features != [
        "dreams_similarity", "sqrt_cosine", "entropy_similarity",
        "neutral_loss_sqrt_cosine",
    ]:
        raise RuntimeError(f"unexpected P2b selected features: {selected_features}")
    selected_indices = [cache.feature_names.index(name) for name in selected_features]
    selected_pairs = cache.features[:, selected_indices]
    query_pair_ptr = cache.molecule_ptr[cache.query_ptr]
    normalized_pairs = normalize_pair_features(
        selected_pairs, query_pair_ptr, configuration.normalization,
    )
    fused_pair_all = normalized_pairs @ np.asarray(configuration.weights, dtype=np.float64)
    baseline_molecule_all = grouped_max(selected_pairs[:, 0], cache.molecule_ptr)
    fused_molecule_all = grouped_max(fused_pair_all, cache.molecule_ptr)

    p3_ik14 = load_p3_identities(args.p3_dir, args.allow_missing_p3)
    query_identities = {str(value) for value in cache.query_ik14[:n_queries]}
    p3_overlap = sorted(query_identities & p3_ik14)
    if p3_overlap:
        raise RuntimeError(f"P3 leakage: {len(p3_overlap)} identities; first={p3_overlap[0]}")

    # Read metadata only once for every row reachable by the selected query prefix.
    last_molecule = int(cache.query_ptr[n_queries])
    last_pair = int(cache.molecule_ptr[last_molecule])
    reachable_rows = np.concatenate((
        cache.query_row[:n_queries], cache.pair_candidate_row[:last_pair],
    ))
    with h5py.File(args.data, "r") as handle:
        metadata = read_rows(handle, reachable_rows, [
            "INCHIKEY", "FORMULA", "smiles", "adduct", "fold",
            "INSTRUMENT_TYPE", "COLLISION_ENERGY", "precursor_mz", "IDENTIFIER",
        ])

    query_rows: list[dict] = []
    candidate_rows: list[dict] = []
    pair_rows: list[dict] = []
    query_top_rows: list[tuple[int, int | None, int | None, int, int]] = []

    for query in range(n_queries):
        molecule_left = int(cache.query_ptr[query])
        molecule_right = int(cache.query_ptr[query + 1])
        pair_left = int(cache.molecule_ptr[molecule_left])
        pair_right = int(cache.molecule_ptr[molecule_right])
        base_block = baseline_molecule_all[molecule_left:molecule_right]
        fused_block = fused_molecule_all[molecule_left:molecule_right]
        base_top = winner_index(base_block)
        fused_top = winner_index(fused_block)
        local_ptr = cache.molecule_ptr[molecule_left:molecule_right + 1] - pair_left
        p2b_block, intervened, support = fuse_one_query(
            normalized_pairs[pair_left:pair_right],
            selected_pairs[pair_left:pair_right, 0],
            local_ptr,
            np.asarray(configuration.weights, dtype=np.float64),
            (1, 2, 3),
            configuration.min_support,
            configuration.min_advantage,
        )
        # Recompute the frozen gate margin only for audit/reporting.  The
        # decision above is made by the same shared helper used by sealed P3.
        if fused_top is None:
            advantage = -math.inf
        elif base_top is None:
            advantage = math.inf
        elif fused_top == base_top:
            advantage = 0.0
        else:
            advantage = float(fused_block[fused_top] - fused_block[base_top])
        p2b_top = winner_index(p2b_block)
        base_ranks = strict_candidate_ranks(base_block)
        p2b_ranks = strict_candidate_ranks(p2b_block)
        base_correct = bool(base_ranks[0] == 1)
        p2b_correct = bool(p2b_ranks[0] == 1)
        transition = transition_name(base_correct, p2b_correct)
        query_row = int(cache.query_row[query])
        qmeta = metadata[query_row]
        qsmiles = str(scalar(qmeta, "smiles"))

        def top_details(local_top: int | None, pair_score: np.ndarray):
            if local_top is None:
                return None, None, None, None
            molecule = molecule_left + local_top
            left = int(cache.molecule_ptr[molecule])
            right = int(cache.molecule_ptr[molecule + 1])
            local_pair = int(np.argmax(pair_score[left:right]))
            pair_index = left + local_pair
            return molecule, pair_index, int(cache.pair_candidate_row[pair_index]), local_top

        base_molecule, base_pair, base_row, _ = top_details(base_top, selected_pairs[:, 0])
        p2b_molecule, p2b_pair, p2b_row, _ = top_details(
            p2b_top, fused_pair_all if intervened else selected_pairs[:, 0],
        )
        positive_left = int(cache.molecule_ptr[molecule_left])
        positive_right = int(cache.molecule_ptr[molecule_left + 1])
        positive_base_pair = positive_left + int(np.argmax(selected_pairs[positive_left:positive_right, 0]))
        positive_p2b_pair = positive_left + int(np.argmax(
            (fused_pair_all if intervened else selected_pairs[:, 0])[positive_left:positive_right]
        ))
        positive_base_row = int(cache.pair_candidate_row[positive_base_pair])
        positive_p2b_row = int(cache.pair_candidate_row[positive_p2b_pair])
        query_top_rows.append((
            query_row, base_row, p2b_row, positive_base_row, positive_p2b_row,
        ))

        def top_identity(local_top: int | None) -> str:
            return "__TIE__" if local_top is None else str(cache.molecule_ik14[molecule_left + local_top])

        def top_formula(local_top: int | None) -> str:
            return "" if local_top is None else str(cache.molecule_formula[molecule_left + local_top])

        def top_grade(local_top: int | None) -> str:
            return "ambiguous" if local_top is None else candidate_grade_name(
                int(cache.molecule_mces_grade[molecule_left + local_top])
            )

        base_meta = {} if base_row is None else metadata[base_row]
        p2b_meta = {} if p2b_row is None else metadata[p2b_row]
        pos_base_meta = metadata[positive_base_row]
        pos_p2b_meta = metadata[positive_p2b_row]
        base_ce = scalar(base_meta, "COLLISION_ENERGY", None)
        p2b_ce = scalar(p2b_meta, "COLLISION_ENERGY", None)
        qce = scalar(qmeta, "COLLISION_ENERGY", None)
        pos_ce = scalar(pos_base_meta, "COLLISION_ENERGY", None)
        ce_delta = abs(float(qce) - float(pos_ce)) if qce is not None and pos_ce is not None else None
        positive_cross_instrument = (
            str(scalar(qmeta, "INSTRUMENT_TYPE")) != str(scalar(pos_base_meta, "INSTRUMENT_TYPE"))
        )
        query_rows.append({
            "query_index": query,
            "query_row": query_row,
            "query_ik14": str(cache.query_ik14[query]),
            "query_formula": str(cache.query_formula[query]),
            "query_smiles": qsmiles,
            "query_scaffold": scaffold(qsmiles),
            "query_adduct": scalar(qmeta, "adduct"),
            "query_fold": scalar(qmeta, "fold"),
            "query_instrument": scalar(qmeta, "INSTRUMENT_TYPE"),
            "query_collision_energy": qce,
            "query_precursor_mz": scalar(qmeta, "precursor_mz", None),
            "query_identifier": scalar(qmeta, "IDENTIFIER"),
            "n_candidate_molecules": molecule_right - molecule_left,
            "n_candidate_spectra": pair_right - pair_left,
            "has_near_candidate": bool(cache.query_has_near[query]),
            "dreams_correct": base_correct,
            "p2b_correct": p2b_correct,
            "transition": transition,
            "dreams_positive_rank": int(base_ranks[0]),
            "p2b_positive_rank": int(p2b_ranks[0]),
            "rank_delta_p2b_minus_dreams": int(p2b_ranks[0] - base_ranks[0]),
            "dreams_margin": float(base_block[0] - np.max(base_block[1:])),
            "p2b_margin": float(p2b_block[0] - np.max(p2b_block[1:])),
            "dreams_top_ik14": top_identity(base_top),
            "dreams_top_formula": top_formula(base_top),
            "dreams_top_grade": top_grade(base_top),
            "dreams_top_pair_row": base_row,
            "dreams_top_instrument": scalar(base_meta, "INSTRUMENT_TYPE"),
            "dreams_top_collision_energy": base_ce,
            "p2b_top_ik14": top_identity(p2b_top),
            "p2b_top_formula": top_formula(p2b_top),
            "p2b_top_grade": top_grade(p2b_top),
            "p2b_top_pair_row": p2b_row,
            "p2b_top_instrument": scalar(p2b_meta, "INSTRUMENT_TYPE"),
            "p2b_top_collision_energy": p2b_ce,
            "p2b_intervened": intervened,
            "p2b_vote_support": int(support),
            "p2b_advantage_over_dreams_top": advantage if np.isfinite(advantage) else None,
            "positive_dreams_pair_row": positive_base_row,
            "positive_p2b_pair_row": positive_p2b_row,
            "positive_cross_instrument": positive_cross_instrument,
            "positive_collision_energy_delta": ce_delta,
            "positive_p2b_instrument": scalar(pos_p2b_meta, "INSTRUMENT_TYPE"),
        })

        for local_molecule, molecule in enumerate(range(molecule_left, molecule_right)):
            left = int(cache.molecule_ptr[molecule])
            right = int(cache.molecule_ptr[molecule + 1])
            base_pair_index = left + int(np.argmax(selected_pairs[left:right, 0]))
            final_pair_values = fused_pair_all if intervened else selected_pairs[:, 0]
            p2b_pair_index = left + int(np.argmax(final_pair_values[left:right]))
            representative_row = int(cache.pair_candidate_row[base_pair_index])
            cmeta = metadata[representative_row]
            csmiles = str(scalar(cmeta, "smiles"))
            candidate = {
                "query_index": query,
                "query_row": query_row,
                "query_ik14": str(cache.query_ik14[query]),
                "candidate_local_index": local_molecule,
                "candidate_ik14": str(cache.molecule_ik14[molecule]),
                "candidate_formula": str(cache.molecule_formula[molecule]),
                "candidate_smiles": csmiles,
                "candidate_scaffold": scaffold(csmiles),
                "label": int(cache.molecule_label[molecule]),
                "mces_grade": candidate_grade_name(int(cache.molecule_mces_grade[molecule])),
                "n_candidate_spectra": right - left,
                "dreams_score": float(base_block[local_molecule]),
                "p2b_score": float(p2b_block[local_molecule]),
                "dreams_rank": int(base_ranks[local_molecule]),
                "p2b_rank": int(p2b_ranks[local_molecule]),
                "dreams_winning_pair_row": int(cache.pair_candidate_row[base_pair_index]),
                "p2b_winning_pair_row": int(cache.pair_candidate_row[p2b_pair_index]),
                "dreams_is_unique_top": base_top == local_molecule,
                "p2b_is_unique_top": p2b_top == local_molecule,
                "query_transition": transition,
                "p2b_intervened": intervened,
            }
            for prefix, pair_index in (("dreams_pair", base_pair_index), ("p2b_pair", p2b_pair_index)):
                for name, value in zip(cache.feature_names, cache.features[pair_index]):
                    candidate[f"{prefix}_{name}"] = float(value)
            candidate_rows.append(candidate)

            for pair_index in range(left, right):
                pair = {
                    "query_index": query,
                    "query_row": query_row,
                    "query_ik14": str(cache.query_ik14[query]),
                    "candidate_local_index": local_molecule,
                    "candidate_ik14": str(cache.molecule_ik14[molecule]),
                    "candidate_row": int(cache.pair_candidate_row[pair_index]),
                    "label": int(cache.molecule_label[molecule]),
                    "mces_grade": candidate_grade_name(int(cache.molecule_mces_grade[molecule])),
                    "query_transition": transition,
                    "p2b_intervened": intervened,
                    "p2b_pair_score_if_intervened": float(fused_pair_all[pair_index]),
                    "final_pair_score": float(
                        fused_pair_all[pair_index] if intervened else selected_pairs[pair_index, 0]
                    ),
                }
                for name, value in zip(cache.feature_names, cache.features[pair_index]):
                    pair[name] = float(value)
                pair_rows.append(pair)

        if (query + 1) % 500 == 0 or query + 1 == n_queries:
            print(
                f"[atlas] {query + 1:,}/{n_queries:,} queries; "
                f"{len(candidate_rows):,} molecule edges; {len(pair_rows):,} pair edges",
                flush=True,
            )

    query_frame = pd.DataFrame(query_rows)
    candidate_frame = pd.DataFrame(candidate_rows)
    pair_frame = pd.DataFrame(pair_rows)

    # Rule audit is intentionally restricted to observed query/top spectra.
    rule_report = {"enabled": False, "rows": 0}
    rule_rows = np.empty(0, dtype=np.int64)
    if not args.skip_rule_audit:
        mandatory = set(query_frame.index[query_frame["transition"] != "protected_correct"])
        protected = query_frame.index[query_frame["transition"] == "protected_correct"].to_numpy(int)
        if args.rule_control_cap and len(protected):
            protected = sorted(
                protected,
                key=lambda index: hashlib.blake2b(
                    str(int(query_frame.at[index, "query_row"])).encode(), digest_size=8,
                ).digest(),
            )[:args.rule_control_cap]
        selected_rule_queries = sorted(mandatory | set(map(int, protected)))
        top_rule_jobs: dict[int, None] = {}
        for index in selected_rule_queries:
            for row in query_top_rows[index]:
                if row is not None:
                    top_rule_jobs[int(row)] = None
        rule_rows = np.asarray(sorted(top_rule_jobs), dtype=np.int64)
        rule_body = json.loads(args.rules.read_text(encoding="utf-8"))
        rules = rule_body["rules"]
        categories = np.asarray([str(rule["category"]) for rule in rules], dtype=object)
        vectors: dict[int, np.ndarray] = {}
        with h5py.File(args.data, "r") as handle:
            for position, row in enumerate(rule_rows, start=1):
                spectrum = np.asarray(handle["spectrum"][int(row)])
                precursor = float(handle["precursor_mz"][int(row)])
                vectors[int(row)] = spectrum_rule_vector(spectrum[0], precursor, rules)
                if position % 1000 == 0 or position == len(rule_rows):
                    print(f"[rules] {position:,}/{len(rule_rows):,}", flush=True)
        np_labels = np.stack([vectors[int(row)] for row in rule_rows], axis=0)
        rule_report = {
            "enabled": True,
            "rows": int(len(rule_rows)),
            "queries": int(len(selected_rule_queries)),
            "all_non_protected_queries_included": True,
            "protected_correct_control_cap": int(args.rule_control_cap),
            "rules": int(len(rules)),
            "categories": Counter(map(str, categories)).copy(),
            "semantics": "observed spectrum-level mass motifs; not structure labels",
        }
        for index in selected_rule_queries:
            (
            query_row, base_row, p2b_row, positive_base_row, positive_p2b_row,
            ) = query_top_rows[index]
            query_vector = vectors[query_row]
            if base_row is not None:
                for key, value in rule_comparison(query_vector, vectors[base_row], categories).items():
                    query_frame.at[index, f"dreams_{key}"] = value
            if p2b_row is not None:
                for key, value in rule_comparison(query_vector, vectors[p2b_row], categories).items():
                    query_frame.at[index, f"p2b_{key}"] = value
            for key, value in rule_comparison(
                query_vector, vectors[positive_base_row], categories,
            ).items():
                query_frame.at[index, f"positive_dreams_{key}"] = value
            for key, value in rule_comparison(
                query_vector, vectors[positive_p2b_row], categories,
            ).items():
                query_frame.at[index, f"positive_p2b_{key}"] = value
    else:
        np_labels = np.empty((0, 0), dtype=np.uint8)
        categories = np.empty(0, dtype=object)

    transition_counts = query_frame["transition"].value_counts().to_dict()
    report = {
        "status": "g8r_real_error_atlas_complete",
        "purpose": "P3-disjoint training error mining; not model evaluation",
        "n_queries": int(len(query_frame)),
        "n_query_identities": int(query_frame["query_ik14"].nunique()),
        "n_query_formulas": int(query_frame["query_formula"].nunique()),
        "n_candidate_edges": int(len(candidate_frame)),
        "n_spectrum_pair_edges": int(len(pair_frame)),
        "transition_counts": {name: int(transition_counts.get(name, 0)) for name in TRANSITIONS},
        "dreams_recall1": float(query_frame["dreams_correct"].mean()),
        "p2b_recall1": float(query_frame["p2b_correct"].mean()),
        "p2b_intervention_rate": float(query_frame["p2b_intervened"].mean()),
        "identity_weighted": {
            "n_identities": int(query_frame["query_ik14"].nunique()),
            "dreams_recall1": float(
                query_frame.groupby("query_ik14", sort=False)["dreams_correct"].mean().mean()
            ),
            "p2b_recall1": float(
                query_frame.groupby("query_ik14", sort=False)["p2b_correct"].mean().mean()
            ),
        },
        "n_near_queries": int(query_frame["has_near_candidate"].sum()),
        "p3_query_identity_overlap": int(len(p3_overlap)),
        "top_formulas": {
            transition: top_formula_counts(query_frame.loc[query_frame["transition"] == transition])
            for transition in TRANSITIONS
        },
        "dreams_wrong_top_grade": {
            str(key): int(value) for key, value in
            query_frame.loc[~query_frame["dreams_correct"], "dreams_top_grade"].value_counts().items()
        },
        "p2b_wrong_top_grade": {
            str(key): int(value) for key, value in
            query_frame.loc[~query_frame["p2b_correct"], "p2b_top_grade"].value_counts().items()
        },
        "rule_audit": rule_report,
        "protocol": {
            "candidate_graph": "complete strict-10ppm same-adduct molecule groups from the input cache",
            "positive_label": "IK14 identity only",
            "fusion": artifact["configuration"],
            "selected_features": selected_features,
            "rank_ties": "every negative tie counts against the positive",
            "rule_role": "description/stratification only; never a label",
        },
        "provenance": {
            "cache": str(args.cache.resolve()),
            "cache_sha256": sha256_file(args.cache),
            "artifact": str(args.artifact.resolve()),
            "artifact_sha256": sha256_file(args.artifact),
            "hdf5": str(args.data.resolve()),
            "p3_dir": str(args.p3_dir.resolve()) if args.p3_dir.exists() else None,
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "This atlas localizes reproducible training errors and evidence conflicts. "
            "It does not estimate generalization or prove a fragmentation mechanism."
        ),
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{args.output_dir.name}.building-", dir=args.output_dir.parent,
    ))
    try:
        query_frame.to_csv(staging / "query_summary.csv.gz", index=False, compression="gzip")
        candidate_frame.to_csv(staging / "candidate_edges.csv.gz", index=False, compression="gzip")
        pair_frame.to_csv(staging / "spectrum_pair_edges.csv.gz", index=False, compression="gzip")
        if rule_report["enabled"]:
            np.savez_compressed(
                staging / "top_spectrum_rule_vectors.npz",
                hdf5_row=rule_rows,
                labels=np_labels,
                rule_name=np.asarray([rule["name"] for rule in rules], dtype=object),
                rule_category=categories,
            )
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        if args.output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(args.output_dir)
            backup = args.output_dir.with_name(args.output_dir.name + ".previous")
            if backup.exists():
                raise FileExistsError(f"refusing overwrite because backup exists: {backup}")
            args.output_dir.replace(backup)
        staging.replace(args.output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
