"""Build full-spectrum packed rule vectors for the ChemAware G0 candidate graph.

Formal mode covers every query and candidate spectrum reachable from the
P3-disjoint real-error candidate graph.  Both the 335-rule core library and
all 3,151 MassBank-derived rules are retained.  The cache records observed
mass motifs only; it is not a molecular-structure label cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from chemaware_g0_core import compile_rules, match_compiled_rules  # noqa: E402
from build_g8r_real_error_atlas import Cache  # noqa: E402


DEFAULT_GRAPH = ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz"
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_chemaware_g0_rule_cache.npz"
DEFAULT_CORE = ROOT / "dreams/models/chem_aware/chem_rules_data.json"
DEFAULT_MASSBANK = ROOT / "dreams/models/chem_aware/chem_rules_massbank.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--core-rules", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--massbank-rules", type=Path, default=DEFAULT_MASSBANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--max-spectra", type=int, default=0, help="Smoke only; 0 means full graph.")
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


def load_rules(core_path: Path, massbank_path: Path) -> tuple[list[dict], np.ndarray]:
    core = json.loads(core_path.read_text(encoding="utf-8"))["rules"]
    massbank = json.loads(massbank_path.read_text(encoding="utf-8"))["rules"]
    if len(core) != 335 or len(massbank) != 3151:
        raise RuntimeError(f"unexpected rule-library sizes: core={len(core)}, massbank={len(massbank)}")
    rules = list(core) + list(massbank)
    names = [str(rule.get("name", "")) for rule in rules]
    if len(set(names)) != len(names):
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        raise RuntimeError(f"duplicate rule names in combined library: {duplicates[:5]}")
    library = np.asarray(["core"] * len(core) + ["massbank"] * len(massbank), dtype=object)
    return rules, library


def main() -> None:
    args = parse_args()
    if args.tolerance <= 0 or args.max_spectra < 0:
        raise ValueError("invalid rule-cache parameters")
    for path in (args.graph, args.data, args.core_rules, args.massbank_rules):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")

    graph = Cache(args.graph)
    rows = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row))).astype(np.int64)
    formal_rows = len(rows)
    if args.max_spectra:
        rows = rows[:args.max_spectra]
    if len(rows) == 0:
        raise RuntimeError("candidate graph reaches no spectra")
    formal = args.max_spectra == 0
    if formal and (graph.n_queries < 15000 or formal_rows < 10000):
        raise RuntimeError("formal G0 rule cache refuses a small candidate graph")

    rules, library = load_rules(args.core_rules, args.massbank_rules)
    compiled = compile_rules(rules)
    n_rules = len(rules)
    n_bytes = (n_rules + 7) // 8
    packed = np.zeros((len(rows), n_bytes), dtype=np.uint8)
    spectrum_support = np.zeros(n_rules, dtype=np.int64)
    identity_accumulator: dict[str, np.ndarray] = {}

    with h5py.File(args.data, "r") as handle:
        if int(rows[-1]) >= len(handle["spectrum"]):
            raise RuntimeError("candidate graph contains an out-of-range HDF5 row")
        inchikey = np.asarray([decode(value) for value in handle["INCHIKEY"][rows]], dtype=object)
        precursor = np.asarray(handle["precursor_mz"][rows], dtype=np.float64)
        parent_mass = np.asarray(handle["PARENT_MASS"][rows], dtype=np.float64)
        started = time.time()
        for position, row in enumerate(rows):
            spectrum = np.asarray(handle["spectrum"][int(row)])
            labels = match_compiled_rules(
                spectrum[0], float(precursor[position]), compiled, args.tolerance,
                parent_mass=float(parent_mass[position]),
            )
            packed[position] = np.packbits(labels, bitorder="little")
            spectrum_support += labels
            identity = str(inchikey[position])[:14]
            existing = identity_accumulator.get(identity)
            if existing is None:
                identity_accumulator[identity] = labels.astype(bool)
            else:
                np.logical_or(existing, labels, out=existing)
            done = position + 1
            if done % 1000 == 0 or done == len(rows):
                elapsed = max(time.time() - started, 1e-9)
                print(
                    f"[rules] {done:,}/{len(rows):,} spectra; {done / elapsed:.1f} spectra/s",
                    flush=True,
                )

    identity_support = np.zeros(n_rules, dtype=np.int64)
    for labels in identity_accumulator.values():
        identity_support += labels
    prevalence = spectrum_support / float(len(rows))
    identity_prevalence = identity_support / float(len(identity_accumulator))
    categories = np.asarray([str(rule.get("category", "unknown")) for rule in rules], dtype=object)
    sources = np.asarray([str(rule.get("source", "unknown")) for rule in rules], dtype=object)
    names = np.asarray([str(rule.get("name", "")) for rule in rules], dtype=object)
    support_metadata = np.asarray([
        int(rule.get("support", 0) or 0) for rule in rules
    ], dtype=np.int64)
    semantics = np.asarray([
        (
            "precursor_exact_mass_offset"
            if rule.get("match_type") == "mass_diff"
            and str(rule.get("source", "")) == "MassBank record-derived"
            else "fragment_neutral_loss"
            if rule.get("match_type") == "mass_diff"
            else "fragment_mz"
            if rule.get("match_type") == "peak_mz"
            else "peak_pair_pattern"
        )
        for rule in rules
    ], dtype=object)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        hdf5_row=rows,
        packed_rule_hits=packed,
        n_rules=np.asarray([n_rules], dtype=np.int64),
        rule_name=names,
        rule_category=categories,
        rule_library=library,
        rule_source=sources,
        rule_semantics=semantics,
        rule_declared_support=support_metadata,
        spectrum_support=spectrum_support,
        identity_support=identity_support,
    )
    temporary.replace(args.output)
    report = {
        "status": "chemaware_g0_rule_cache_complete" if formal else "chemaware_g0_rule_cache_smoke",
        "formal": formal,
        "candidate_graph_queries": int(graph.n_queries),
        "reachable_spectra_in_full_graph": int(formal_rows),
        "cached_spectra": int(len(rows)),
        "cached_identities": int(len(identity_accumulator)),
        "rules": n_rules,
        "core_rules": int(np.sum(library == "core")),
        "massbank_rules": int(np.sum(library == "massbank")),
        "categories": {str(key): int(value) for key, value in Counter(categories).items()},
        "semantics_counts": {str(key): int(value) for key, value in Counter(semantics).items()},
        "coverage": {
            "median_spectrum_prevalence": float(np.median(prevalence)),
            "median_identity_prevalence": float(np.median(identity_prevalence)),
            "rules_zero_spectrum_support": int(np.sum(spectrum_support == 0)),
            "rules_lt_10_identity_support": int(np.sum(identity_support < 10)),
            "rules_gt_50pct_identity_coverage": int(np.sum(identity_prevalence > 0.5)),
        },
        "semantics": (
            "observed spectrum-level mass motifs; core NL uses precursor-fragment loss; "
            "MassBank NL is kept separately as precursor-exact-mass offset; never identity labels"
        ),
        "empty_union_policy": "pairwise Jaccard is missing when both spectra have no hit",
        "provenance": {
            "candidate_graph": str(args.graph.resolve()),
            "candidate_graph_sha256": sha256_file(args.graph),
            "hdf5": str(args.data.resolve()),
            "hdf5_sha256": sha256_file(args.data),
            "core_rules_sha256": sha256_file(args.core_rules),
            "massbank_rules_sha256": sha256_file(args.massbank_rules),
            "cache_sha256": sha256_file(args.output),
        },
        "parameters": {
            "tolerance": args.tolerance,
            "max_spectra": args.max_spectra,
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
