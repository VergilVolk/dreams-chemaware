"""Build and seal the leakage-resistant G8R P3 v3 evaluation set.

Primary queries and the reference library use experimental spectra only
(`SIMULATION_CHALLENGE=False`). Query identities exclude all identities consumed
by prior G8R/reranker development and audits. Missing MCES labels are recomputed
for reachable same-formula candidate pairs. Main, isomer, genuine MCES-near,
exposed, and sim-to-real views are sealed separately before P2 development.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_TRAIN = ROOT / "tasks/massspecgym_isomers/g8r_locked/train.json"
DEFAULT_VAL = ROOT / "tasks/massspecgym_isomers/g8r_locked/val.json"
DEFAULT_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache.npz"
DEFAULT_VAL_CACHE = ROOT / "data/validation/g8r_raw_reranker_cache_val.npz"
DEFAULT_TEST_DIR = ROOT / "data/validation/g8r_final_test"
DEFAULT_PAIRS = ROOT / "tasks/massspecgym_isomers/pairs.json"
DEFAULT_LARGE_DISCO = ROOT / "data/validation/large_observability_residual_audit/discovery_query_audit.csv"
DEFAULT_LARGE_CONFIRM = ROOT / "data/validation/large_observability_residual_audit/confirmation_query_audit.csv"
DEFAULT_OUT = ROOT / "data/validation/g8r_p3_test"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val", type=Path, default=DEFAULT_VAL)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--val-cache", type=Path, default=DEFAULT_VAL_CACHE)
    p.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    p.add_argument("--large-discovery", type=Path, default=DEFAULT_LARGE_DISCO)
    p.add_argument("--large-confirmation", type=Path, default=DEFAULT_LARGE_CONFIRM)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--main-target", type=int, default=3000)
    p.add_argument("--sim-target", type=int, default=2000)
    p.add_argument("--min-near-core", type=int, default=450)
    p.add_argument("--ppm-tol", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=20260824)
    p.add_argument("--query-folds", nargs="+", default=["train", "val"])
    p.add_argument("--mces-workers", type=int, default=8)
    p.add_argument("--mces-threshold", type=float, default=11.0)
    p.add_argument("--mces-time-limit", type=float, default=20.0)
    p.add_argument("--expected-hdf5-sha256", default="")
    p.add_argument("--expected-pairs-sha256", default="")
    p.add_argument("--smoke-query-rows", type=int, default=0)
    p.add_argument(
        "--allow-missing-exclusions-for-dry-run",
        action="store_true",
        help="local code audit only; forbidden for a formal lock",
    )
    p.add_argument("--dry-run", action="store_true", help="compute and validate, but write nothing")
    return p.parse_args()


def read_str(h, key: str) -> np.ndarray:
    raw = h[key][:]
    return np.asarray(
        [x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x) for x in raw],
        dtype=object,
    )


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_json(obj) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def mces_grade(distance: float | None) -> str:
    if distance is None or not np.isfinite(distance):
        return "unknown"
    if distance <= 2:
        return "near"
    if distance <= 5:
        return "mid"
    if distance <= 10:
        return "far"
    return "unrelated"


def _mces_worker(payload):
    key, smi_a, smi_b, threshold, time_limit = payload
    try:
        from myopic_mces import MCES

        _, distance, elapsed, mode = MCES(
            smi_a,
            smi_b,
            threshold=threshold,
            solver_options={"msg": False, "timeLimit": time_limit, "threads": 1},
            catch_errors=True,
        )
        distance = float(distance)
        if distance < 0 or not np.isfinite(distance):
            raise ValueError(f"invalid MCES distance: {distance}")
        return key, distance, float(elapsed), int(mode), "ok", None
    except Exception as exc:
        return key, None, 0.0, -1, "error", str(exc)


def load_exclusions(a: argparse.Namespace):
    sources: dict[str, Path] = {"g8r_train": a.train, "g8r_val": a.val}
    g8r_anchor: set[str] = set()
    g8r_neg: set[str] = set()
    for path in (a.train, a.val):
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data["entries"]:
            g8r_anchor.add(str(entry["ik14"]))
            g8r_neg.update(str(n["ik14"]) for n in entry.get("neg", []))

    base_excl = set(g8r_anchor) | set(g8r_neg)
    for label, path in (("raw_cache", a.cache), ("raw_cache_val", a.val_cache)):
        if not path.exists():
            if a.dry_run and a.allow_missing_exclusions_for_dry_run:
                print(f"[audit-warning] exclusion cache unavailable: {path}")
                continue
            raise FileNotFoundError(f"required exclusion cache missing: {path}")
        sources[label] = path
        data = np.load(path, allow_pickle=True)
        for key in ("query_ik14", "candidate_ik14"):
            if key not in data.files:
                raise KeyError(f"{path} lacks required identity field {key}")
            base_excl.update(str(value) for value in data[key])

    for name in ("test_a_manifest.json", "test_b_manifest.json", "test_c_manifest.json"):
        path = a.test_dir / name
        if not path.exists():
            raise FileNotFoundError(f"required consumed-test manifest missing: {path}")
        sources[name.removesuffix("_manifest.json")] = path
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for query in manifest["queries"]:
            base_excl.add(str(query["ik14"]))
            base_excl.update(str(candidate["ik14"]) for candidate in query.get("candidates", []))

    large_all: set[str] = set()
    for label, path in (
        ("large_discovery", a.large_discovery),
        ("large_confirmation", a.large_confirmation),
    ):
        if not path.exists():
            raise FileNotFoundError(f"required large-audit source missing: {path}")
        sources[label] = path
        with open(path, newline="", encoding="utf-8") as handle:
            large_all.update(str(row["ik14"]) for row in csv.DictReader(handle))

    large_excl = large_all - base_excl
    pristine_excl = base_excl | large_excl
    return g8r_anchor, base_excl, large_excl, pristine_excl, sources


def main() -> None:
    a = parse_args()
    if a.allow_missing_exclusions_for_dry_run and not a.dry_run:
        raise ValueError("missing exclusion sources may be tolerated only in --dry-run")
    if a.smoke_query_rows and not a.dry_run:
        raise ValueError("--smoke-query-rows is a dry-run diagnostic only")
    if min(a.main_target, a.mces_workers) <= 0 or min(a.sim_target, a.min_near_core) < 0:
        raise ValueError("invalid panel target or worker count")
    if not a.dry_run and a.min_near_core == 0:
        raise ValueError("a formal P3 lock requires --min-near-core > 0")
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from myopic_mces import MCES as _MCES  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"P3 requires RDKit and myopic_mces: {exc}")

    g8r_anchor, base_excl, large_excl, pristine_excl, exclusion_sources = load_exclusions(a)
    print(f"[exclude] base={len(base_excl)} large={len(large_excl)} pristine={len(pristine_excl)}")

    with h5py.File(a.data, "r") as h:
        required = {
            "fold", "INCHIKEY", "FORMULA", "adduct", "precursor_mz",
            "INSTRUMENT_TYPE", "COLLISION_ENERGY", "smiles", "SIMULATION_CHALLENGE",
        }
        missing = required - set(h.keys())
        if missing:
            raise KeyError(f"HDF5 missing fields: {sorted(missing)}")
        fold = read_str(h, "fold")
        ikf = read_str(h, "INCHIKEY")
        formula = read_str(h, "FORMULA")
        adduct = read_str(h, "adduct")
        pmz = np.asarray(h["precursor_mz"][:], dtype=float)
        inst = read_str(h, "INSTRUMENT_TYPE")
        ce = np.asarray(h["COLLISION_ENERGY"][:], dtype=float)
        smiles = read_str(h, "smiles")
        simulation = read_str(h, "SIMULATION_CHALLENGE")

    ik14 = np.asarray([str(key)[:14] for key in ikf], dtype=object)
    real_mask = simulation == "False"
    sim_mask = simulation == "True"
    if np.any(~(real_mask | sim_mask)):
        raise ValueError("SIMULATION_CHALLENGE contains values other than True/False")
    print(f"[data] all={len(ik14)} real={int(real_mask.sum())} simulated={int(sim_mask.sum())}")

    query_fold_mask = np.isin(fold, np.asarray(a.query_folds, dtype=object))
    unknown_folds = set(a.query_folds) - set(str(value) for value in np.unique(fold))
    if unknown_folds:
        raise ValueError(f"unknown query folds: {sorted(unknown_folds)}")
    pristine_real_mask = (
        query_fold_mask & real_mask
        & np.asarray([identity not in pristine_excl for identity in ik14], dtype=bool)
    )
    exposed_real_mask = (
        query_fold_mask & real_mask
        & np.asarray([identity in large_excl for identity in ik14], dtype=bool)
    )
    pristine_sim_mask = (
        query_fold_mask & sim_mask
        & np.asarray([identity not in pristine_excl for identity in ik14], dtype=bool)
    )

    if a.smoke_query_rows:
        def truncate_mask(mask):
            rows = np.where(mask)[0][: a.smoke_query_rows]
            limited = np.zeros_like(mask, dtype=bool)
            limited[rows] = True
            return limited

        pristine_real_mask = truncate_mask(pristine_real_mask)
        exposed_real_mask = truncate_mask(exposed_real_mask)
        pristine_sim_mask = truncate_mask(pristine_sim_mask)

    # Store one source spelling per IK14 now. Expensive charge/tautomer
    # standardisation is performed lazily only for MCES pairs actually missing.
    ik_to_smiles: dict[str, str] = {}
    uncharger = rdMolStandardize.Uncharger()
    for identity, smi in zip(ik14, smiles):
        identity, smi = str(identity), str(smi)
        if smi:
            ik_to_smiles.setdefault(identity, smi)

    standardized_smiles: dict[str, str] = {}

    def structure_for_mces(identity: str) -> str:
        if identity in standardized_smiles:
            return standardized_smiles[identity]
        mol = Chem.MolFromSmiles(ik_to_smiles.get(identity, ""))
        if mol is None:
            standardized_smiles[identity] = ""
        else:
            mol = uncharger.uncharge(mol)
            standardized_smiles[identity] = Chem.MolToSmiles(mol, isomericSmiles=False)
        return standardized_smiles[identity]

    scaffold_cache: dict[str, str | None] = {}

    def murcko(smi: str) -> str | None:
        if smi not in scaffold_cache:
            mol = Chem.MolFromSmiles(smi)
            scaffold = MurckoScaffold.GetScaffoldForMol(mol) if mol is not None else None
            scaffold_cache[smi] = Chem.MolToSmiles(scaffold) if scaffold is not None else None
        return scaffold_cache[smi]

    # Deployment reference library: experimental spectra only.
    real_rows = np.where(real_mask)[0]
    ad_to_idx: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ion in sorted(set(str(adduct[row]) for row in real_rows)):
        group = real_rows[adduct[real_rows] == ion]
        order = np.argsort(pmz[group], kind="stable")
        ad_to_idx[ion] = group[order], pmz[group][order]

    def candidate_rows(qrow: int) -> np.ndarray:
        ion = str(adduct[qrow])
        if ion not in ad_to_idx:
            return np.asarray([], dtype=np.int64)
        group, masses = ad_to_idx[ion]
        mass = float(pmz[qrow])
        delta = a.ppm_tol * 1e-6 * mass
        lo = np.searchsorted(masses, mass - delta, side="left")
        hi = np.searchsorted(masses, mass + delta, side="right")
        rows = group[lo:hi]
        return rows[rows != qrow]

    pair_payload = json.loads(a.pairs.read_text(encoding="utf-8"))
    grade_map: dict[tuple[str, str], str] = {}
    for grade in ("near", "mid", "far", "unrelated"):
        for entry in pair_payload.get(grade, []):
            key = canonical_pair(str(entry["ik_a"]), str(entry["ik_b"]))
            old = grade_map.setdefault(key, grade)
            if old != grade:
                raise ValueError(f"conflicting MCES grade for {key}: {old}/{grade}")

    eligible_rows = np.where(pristine_real_mask | exposed_real_mask | pristine_sim_mask)[0]
    missing_pairs: set[tuple[str, str]] = set()
    for index, qrow in enumerate(eligible_rows, start=1):
        qik, qformula = str(ik14[qrow]), str(formula[qrow])
        isomer_iks = {
            str(ik14[row]) for row in candidate_rows(int(qrow))
            if str(ik14[row]) != qik and str(formula[row]) == qformula
        }
        missing_pairs.update(
            key for key in (canonical_pair(qik, candidate) for candidate in isomer_iks)
            if key not in grade_map
        )
        if index % 20000 == 0:
            print(f"[mces-enumerate] {index}/{len(eligible_rows)}", flush=True)

    jobs, prefilter_failures = [], []
    for key in sorted(missing_pairs):
        smi_a, smi_b = structure_for_mces(key[0]), structure_for_mces(key[1])
        mol_a, mol_b = Chem.MolFromSmiles(smi_a), Chem.MolFromSmiles(smi_b)
        if mol_a is None or mol_b is None:
            prefilter_failures.append({"ik_a": key[0], "ik_b": key[1], "reason": "invalid_smiles"})
        elif mol_a.GetNumBonds() > 50 or mol_b.GetNumBonds() > 50:
            prefilter_failures.append({"ik_a": key[0], "ik_b": key[1], "reason": "gt50_bonds"})
        else:
            jobs.append((key, smi_a, smi_b, a.mces_threshold, a.mces_time_limit))
    print(
        f"[mces] existing={len(grade_map)} missing={len(missing_pairs)} "
        f"computable={len(jobs)} unknown={len(prefilter_failures)}",
        flush=True,
    )

    supplement = []
    if jobs:
        with ProcessPoolExecutor(max_workers=a.mces_workers) as pool:
            for index, result in enumerate(pool.map(_mces_worker, jobs, chunksize=16), start=1):
                key, distance, elapsed, mode, status, error = result
                grade = mces_grade(distance)
                if status == "ok":
                    grade_map[key] = grade
                supplement.append(
                    {"ik_a": key[0], "ik_b": key[1], "mces": distance, "grade": grade,
                     "elapsed": elapsed, "mode": mode, "status": status, "error": error}
                )
                if index % 1000 == 0:
                    print(f"[mces] {index}/{len(jobs)}", flush=True)

    def query_record(qrow: int):
        rows = candidate_rows(qrow)
        if not len(rows):
            return None
        qik, qformula = str(ik14[qrow]), str(formula[qrow])
        candidate_iks = {str(ik14[row]) for row in rows}
        if qik not in candidate_iks or not (candidate_iks - {qik}):
            return None
        isomer_iks = {
            str(ik14[row]) for row in rows
            if str(ik14[row]) != qik and str(formula[row]) == qformula
        }
        counts = Counter(grade_map.get(canonical_pair(qik, candidate), "unknown") for candidate in isomer_iks)
        positive_rows = rows[ik14[rows] == qik]
        dce = np.abs(ce[positive_rows] - ce[qrow])
        cross_condition = bool(
            ((inst[positive_rows] != inst[qrow]) | (np.isfinite(dce) & (dce >= 10.0))).any()
        )
        return {
            "row": int(qrow), "ik14": qik, "formula": qformula,
            "source_fold": str(fold[qrow]),
            "adduct": str(adduct[qrow]), "precursor_mz": float(pmz[qrow]),
            "instrument": str(inst[qrow]), "ce_finite": bool(np.isfinite(ce[qrow])),
            "simulation_challenge": str(simulation[qrow]), "scaffold": murcko(str(smiles[qrow])),
            "n_isomer_neg": len(isomer_iks), "n_near_candidate": int(counts["near"]),
            "n_mid_candidate": int(counts["mid"]),
            "n_nearmid_candidate": int(counts["near"] + counts["mid"]),
            "n_unknown_mces_candidate": int(counts["unknown"]),
            "pos_cross_condition": cross_condition,
            "candidate_rows": [int(row) for row in rows],
        }

    def analyze(mask: np.ndarray, tag: str):
        rows = np.where(mask)[0]
        valid = []
        for index, qrow in enumerate(rows, start=1):
            record = query_record(int(qrow))
            if record is not None:
                valid.append(record)
            if index % 20000 == 0:
                print(f"[{tag}] {index}/{len(rows)}", flush=True)
        isomer = [record for record in valid if record["n_isomer_neg"]]
        print(
            f"[{tag}] spectra={len(rows)} valid={len(valid)} "
            f"IK14={len({q['ik14'] for q in valid})} isomer={len({q['ik14'] for q in isomer})} "
            f"near={len({q['ik14'] for q in isomer if q['n_near_candidate']})}",
            flush=True,
        )
        return valid, isomer

    pristine_valid, pristine_isomer = analyze(pristine_real_mask, "real-pristine")
    _, exposed_isomer = analyze(exposed_real_mask, "real-exposed")
    sim_valid, _ = analyze(pristine_sim_mask, "sim-to-real")

    def one_per_ik(queries, score=None):
        grouped = defaultdict(list)
        for query in queries:
            grouped[query["ik14"]].append(query)
        selected = []
        for identity in sorted(grouped):
            if score is None:
                selected.append(min(grouped[identity], key=lambda query: query["row"]))
            else:
                selected.append(max(grouped[identity], key=score))
        return selected

    rng = np.random.default_rng(a.seed)
    def random_one_per_ik(queries):
        grouped = defaultdict(list)
        for query in queries:
            grouped[query["ik14"]].append(query)
        return [
            grouped[identity][int(rng.integers(0, len(grouped[identity])))]
            for identity in sorted(grouped)
        ]

    representatives = random_one_per_ik(pristine_valid)
    rng.shuffle(representatives)
    main_panel = representatives[: a.main_target]
    challenge_score = lambda q: (
        q["n_near_candidate"], q["n_mid_candidate"], q["n_isomer_neg"], -q["row"]
    )
    isomer_panel = one_per_ik(pristine_isomer, challenge_score)
    near_panel = [query for query in isomer_panel if query["n_near_candidate"]]
    nearmid_panel = [query for query in isomer_panel if query["n_nearmid_candidate"]]
    exposed_panel = one_per_ik(exposed_isomer, challenge_score)
    sim_panel = random_one_per_ik(sim_valid)
    rng.shuffle(sim_panel)
    sim_panel = sim_panel[: a.sim_target]

    def strata(queries):
        return {
            "n_queries": len(queries), "n_unique_ik14": len({q["ik14"] for q in queries}),
            "n_unique_formula": len({q["formula"] for q in queries}),
            "n_unique_scaffold": len({q["scaffold"] for q in queries if q["scaffold"]}),
            "instrument": dict(Counter(q["instrument"] for q in queries)),
            "source_fold": dict(Counter(q["source_fold"] for q in queries)),
            "ce_finite": sum(q["ce_finite"] for q in queries),
            "ce_missing": sum(not q["ce_finite"] for q in queries),
            "n_with_isomer": sum(q["n_isomer_neg"] > 0 for q in queries),
            "n_with_near": sum(q["n_near_candidate"] > 0 for q in queries),
            "n_with_nearmid": sum(q["n_nearmid_candidate"] > 0 for q in queries),
            "n_with_unknown_mces": sum(q["n_unknown_mces_candidate"] > 0 for q in queries),
            "pos_cross_condition": sum(q["pos_cross_condition"] for q in queries),
        }

    panels = {
        "P3-main-real-pristine": main_panel,
        "P3-isomer-real-pristine": isomer_panel,
        "P3-near-core-real-pristine": near_panel,
        "P3-nearmid-real-pristine": nearmid_panel,
        "P3-isomer-real-exposed-extension": exposed_panel,
        "P3-sim-to-real-secondary": sim_panel,
    }
    print(f"\n=== {'DRY-RUN' if a.dry_run else 'FORMAL LOCK'} ===")
    for name, queries in panels.items():
        print(f"{name}: {strata(queries)}")

    if len(main_panel) != a.main_target:
        raise RuntimeError(f"main has {len(main_panel)} queries; required {a.main_target}")
    if len(near_panel) < a.min_near_core:
        raise RuntimeError(f"near core has {len(near_panel)} queries; minimum {a.min_near_core}")
    for name, queries in panels.items():
        if len(queries) != len({query["ik14"] for query in queries}):
            raise AssertionError(f"{name} is not IK14-equal-weight")
    real_panels = [queries for name, queries in panels.items() if name != "P3-sim-to-real-secondary"]
    if any(query["simulation_challenge"] != "False" for queries in real_panels for query in queries):
        raise AssertionError("simulated query leaked into a real panel")
    if any(query["simulation_challenge"] != "True" for query in sim_panel):
        raise AssertionError("real query leaked into sim-to-real panel")

    if a.dry_run:
        print("[dry-run] PASS; no files written")
        return
    if a.output_dir.exists():
        raise SystemExit(f"{a.output_dir} already exists; refusing to overwrite sealed P3")

    data_sha, pairs_sha = sha256_of_file(a.data), sha256_of_file(a.pairs)
    if a.expected_hdf5_sha256 and data_sha.lower() != a.expected_hdf5_sha256.lower():
        raise RuntimeError(f"HDF5 SHA256 mismatch: {data_sha}")
    if a.expected_pairs_sha256 and pairs_sha.lower() != a.expected_pairs_sha256.lower():
        raise RuntimeError(f"pairs.json SHA256 mismatch: {pairs_sha}")

    protocol = {
        "version": "P3-v3", "query_unit": "one spectrum per IK14",
        "reference_library": "all SIMULATION_CHALLENGE=False HDF5 spectra",
        "candidate_filter": {"precursor_ppm": a.ppm_tol, "same_adduct": True, "exclude_self": True},
        "molecule_aggregation": "maximum score over spectra per candidate IK14",
        "positive": "candidate IK14 equals query IK14",
        "tie_rule": "rank = 1 + number of negative IK14 scores >= positive IK14 score",
        "metrics": ["Recall@1", "MRR", "macro query AUC"],
        "uncertainty": "paired bootstrap clustered by formula; McNemar for Top-1 transitions",
        "primary_panel": "P3-main-real-pristine",
        "secondary_panels": [name for name in panels if name != "P3-main-real-pristine"],
        "prohibition": "secondary panels must not be pooled into primary CI or SOTA claims",
    }
    library = {
        "description": "experimental MassSpecGym reference library",
        "simulation_challenge": "False only", "rows": [int(row) for row in real_rows],
        "ik14": [str(ik14[row]) for row in real_rows],
        "formula": [str(formula[row]) for row in real_rows],
        "adduct": [str(adduct[row]) for row in real_rows],
        "precursor_mz": [float(pmz[row]) for row in real_rows],
    }
    library_sha = sha256_of_json(library)

    all_p3_ik = {query["ik14"] for queries in panels.values() for query in queries}
    real_train_ik = {str(ik14[row]) for row in np.where((fold == "train") & real_mask)[0]}
    sim_train_ik = {str(ik14[row]) for row in np.where((fold == "train") & sim_mask)[0]}
    p2_real = sorted(real_train_ik - all_p3_ik)
    p2_sim = sorted(sim_train_ik - all_p3_ik)
    p2_real_rows = [int(row) for row in np.where(
        (fold == "train") & real_mask & np.isin(ik14, np.asarray(p2_real, dtype=object))
    )[0]]
    p2_sim_rows = [int(row) for row in np.where(
        (fold == "train") & sim_mask & np.isin(ik14, np.asarray(p2_sim, dtype=object))
    )[0]]
    if (set(p2_real) | set(p2_sim)) & all_p3_ik:
        raise AssertionError("P3 identity leaked into P2 allow-list")
    if not g8r_anchor.issubset(real_train_ik | sim_train_ik):
        raise AssertionError("a G8R anchor is absent from the HDF5 train fold")

    protocol_sha = sha256_of_json(protocol)

    def make_manifest(name, queries):
        ordered = sorted(queries, key=lambda query: query["row"])
        graph = [{"row": q["row"], "candidate_rows": q["candidate_rows"]} for q in ordered]
        body = {
            "panel": name, "primary": name == "P3-main-real-pristine",
            "protocol_sha256": protocol_sha, "candidate_library_sha256": library_sha,
            "strata": strata(ordered), "candidate_graph_sha256": sha256_of_json(graph),
            "queries": ordered,
        }
        body["query_manifest_sha256"] = sha256_of_json(body)
        return body

    manifests = {name: make_manifest(name, queries) for name, queries in panels.items()}
    supplement_body = {
        "source_pairs_json_sha256": pairs_sha, "reachable_missing_pairs": len(missing_pairs),
        "prefilter_failures": prefilter_failures, "computed": supplement,
    }
    a.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{a.output_dir.name}.building-", dir=a.output_dir.parent))
    (staging / "p3_evaluation_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (staging / "p3_reference_library_real.json").write_text(
        json.dumps(library, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    (staging / "p3_mces_supplement.json").write_text(
        json.dumps(supplement_body, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    filenames = {
        "P3-main-real-pristine": "p3_main_real_pristine_manifest.json",
        "P3-isomer-real-pristine": "p3_isomer_real_pristine_manifest.json",
        "P3-near-core-real-pristine": "p3_near_core_real_pristine_manifest.json",
        "P3-nearmid-real-pristine": "p3_nearmid_real_pristine_manifest.json",
        "P3-isomer-real-exposed-extension": "p3_isomer_real_exposed_extension_manifest.json",
        "P3-sim-to-real-secondary": "p3_sim_to_real_secondary_manifest.json",
    }
    for name, manifest in manifests.items():
        (staging / filenames[name]).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    p2_allow_body = {
        "real_train_primary": {"n": len(p2_real), "ik14": p2_real,
                               "n_rows": len(p2_real_rows), "rows": p2_real_rows},
        "simulation_train_optional": {"n": len(p2_sim), "ik14": p2_sim,
                                      "n_rows": len(p2_sim_rows), "rows": p2_sim_rows},
        "p3_query_overlap": 0,
        "rule": "P2 loaders must intersect rows with these identities; simulation is optional only",
    }
    (staging / "p3_p2_allowed_training_ik14.json").write_text(
        json.dumps(p2_allow_body, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    failed_mces = sum(row["status"] != "ok" for row in supplement) + len(prefilter_failures)
    summary = {
        "status": "g8r_p3_v3_sealed", "build_script_sha256": sha256_of_file(Path(__file__)),
        "hdf5_sha256": data_sha, "pairs_json_sha256": pairs_sha,
        "exclusion_source_sha256": {name: sha256_of_file(path) for name, path in exclusion_sources.items()},
        "evaluation_protocol_sha256": protocol_sha, "reference_library_sha256": library_sha,
        "mces_supplement_sha256": sha256_of_json(supplement_body), "seed": a.seed,
        "parameters": {"ppm_tol": a.ppm_tol, "main_target": a.main_target,
                       "sim_target": a.sim_target, "min_near_core": a.min_near_core,
                       "mces_threshold": a.mces_threshold, "mces_time_limit": a.mces_time_limit,
                       "query_folds": a.query_folds},
        "data_counts": {"all_spectra": len(ik14), "experimental_reference_spectra": int(real_mask.sum()),
                        "simulated_spectra": int(sim_mask.sum()), "mces_failed_or_unknown": int(failed_mces)},
        "excluded_counts": {"base": len(base_excl), "large_netnew": len(large_excl),
                            "pristine": len(pristine_excl)},
        "panels": {name: manifest["strata"] for name, manifest in manifests.items()},
        "panel_overlap": {
            "main_vs_isomer": len({q["ik14"] for q in main_panel} & {q["ik14"] for q in isomer_panel}),
            "main_vs_near_core": len({q["ik14"] for q in main_panel} & {q["ik14"] for q in near_panel}),
        },
        "p2_allow_lists": {"real_train": len(p2_real), "simulation_optional": len(p2_sim)},
        "p2_allow_list_sha256": sha256_of_json(p2_allow_body),
        "manifest_sha256": {name: manifest["query_manifest_sha256"] for name, manifest in manifests.items()},
    }
    (staging / "p3_lock_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if a.output_dir.exists():
        raise RuntimeError(f"output appeared while building: {a.output_dir}")
    staging.replace(a.output_dir)
    print(f"[sealed] {a.output_dir}")
    print(f"[sealed] HDF5={data_sha} pairs={pairs_sha} library={library_sha}")


if __name__ == "__main__":
    main()
