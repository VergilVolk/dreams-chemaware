"""GLM-authored CPU probe: naive spectral separability of held-retrieval rivals.

Purpose (pre-flight evidence for the T1/T3 route decision, no GPU):

1. For every held query, adjudicate the true molecule versus its
   selection-geometry top rival with three standard naive spectral scorers
   (sqrt-intensity greedy cosine, precursor-shifted modified cosine, and
   top-K diagnostic-fragment overlap), aggregated by molecule-max (the
   deployment fact T3 targets) and by mean.
2. On Stage-1 residual errors this measures the trainable headroom that the
   DreaMS embedding demonstrably misses; on Stage-1 correct queries it
   measures how often naive evidence would break an already-correct answer
   (fusion-referee corrected/introduced ledger with lambda-2 risk net).
3. Report rival/true measured-spectra pool sizes (T3 molecule-max
   feasibility) and molecule-level overlap between Stage-1 corrected and
   introduced errors (the see-saw exchange-rate mechanism test).

This is a read-only audit of frozen artifacts.  It trains nothing, encodes
nothing, and does not modify any existing file.  Ranks and transitions come
from the Stage-1 held ledger; the rival is re-derived here with exactly the
same molecule-max selection rule as the relation audit and cross-checked
against that audit's ledger.

Author: GLM-5.3 (DeepSeek Harness session).  All outputs are prefixed GLM.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

PROBE_VERSION = "GLM_noise_rival_separability_v2"
MATCH_TOL_DA = 0.01
TOPK = 20
SCORER_NAMES = ("cosine", "modified_cosine", "topk_overlap")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--held-per-query", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--self-test", action="store_true",
        help="run scorer unit checks on synthetic spectra and exit",
    )
    args = parser.parse_args()
    if not args.self_test:
        for name in ("graph", "held_per_query", "data", "output_dir"):
            if getattr(args, name) is None:
                raise SystemExit(f"argument --{name.replace('_', '-')} is required")
    return args


# --------------------------------------------------------------------------
# Pure scorer functions (deterministic, no randomness, no global state).
# --------------------------------------------------------------------------

def _greedy_match_sum(
    weights_q: np.ndarray,
    mz_q: np.ndarray,
    weights_c: np.ndarray,
    mz_c: np.ndarray,
    tol: float,
    shifts: tuple[float, ...] = (0.0,),
) -> float:
    """Greedy one-to-one peak matching maximising matched weight product.

    Query peak i may match candidate peak j when
    ``|mz_q[i] - (mz_c[j] + shift)| <= tol`` for some shift in ``shifts``.
    Candidate matches from all shifts are unioned (same pair under two shifts
    is one match), then accepted greedily by descending product, each peak at
    most once.  Returns the sum of accepted weight products.
    """
    if len(mz_q) == 0 or len(mz_c) == 0:
        return 0.0
    pairs_i: list[np.ndarray] = []
    pairs_j: list[np.ndarray] = []
    for shift in shifts:
        close = np.abs(mz_q[:, None] - (mz_c[None, :] + shift)) <= tol
        ii, jj = np.nonzero(close)
        if ii.size:
            pairs_i.append(ii)
            pairs_j.append(jj)
    if not pairs_i:
        return 0.0
    ii = np.concatenate(pairs_i)
    jj = np.concatenate(pairs_j)
    # Union duplicates (same i, j through different shifts) keep one entry.
    keys = ii.astype(np.int64) * (len(mz_c) + 1) + jj.astype(np.int64)
    _, unique_at = np.unique(keys, return_index=True)
    ii = ii[unique_at]
    jj = jj[unique_at]
    products = weights_q[ii] * weights_c[jj]
    order = np.argsort(-products, kind="stable")
    used_q = np.zeros(len(mz_q), dtype=bool)
    used_c = np.zeros(len(mz_c), dtype=bool)
    total = 0.0
    for position in order:
        i = int(ii[position])
        j = int(jj[position])
        if not used_q[i] and not used_c[j]:
            used_q[i] = True
            used_c[j] = True
            total += float(products[position])
    return total


def cosine_similarity(
    mz_q: np.ndarray,
    intensity_q: np.ndarray,
    mz_c: np.ndarray,
    intensity_c: np.ndarray,
    tol: float = MATCH_TOL_DA,
) -> float:
    """Standard sqrt-intensity greedy cosine between two peak lists."""
    weights_q = np.sqrt(np.asarray(intensity_q, dtype=np.float64))
    weights_c = np.sqrt(np.asarray(intensity_c, dtype=np.float64))
    denominator = float(np.linalg.norm(weights_q) * np.linalg.norm(weights_c))
    if denominator <= 0.0:
        return 0.0
    matched = _greedy_match_sum(weights_q, mz_q, weights_c, mz_c, tol)
    return matched / denominator


def modified_cosine_similarity(
    mz_q: np.ndarray,
    intensity_q: np.ndarray,
    mz_c: np.ndarray,
    intensity_c: np.ndarray,
    precursor_q: float,
    precursor_c: float,
    tol: float = MATCH_TOL_DA,
) -> float:
    """Cosine allowing direct and precursor-shifted peak alignment."""
    weights_q = np.sqrt(np.asarray(intensity_q, dtype=np.float64))
    weights_c = np.sqrt(np.asarray(intensity_c, dtype=np.float64))
    denominator = float(np.linalg.norm(weights_q) * np.linalg.norm(weights_c))
    if denominator <= 0.0:
        return 0.0
    shift = float(precursor_q) - float(precursor_c)
    shifts = (0.0,) if shift == 0.0 else (0.0, shift)
    matched = _greedy_match_sum(weights_q, mz_q, weights_c, mz_c, tol, shifts)
    return matched / denominator


def topk_overlap(
    mz_q: np.ndarray,
    intensity_q: np.ndarray,
    mz_c: np.ndarray,
    intensity_c: np.ndarray,
    tol: float = MATCH_TOL_DA,
    k: int = TOPK,
) -> float:
    """Fraction of the query's k most intense peaks present in the candidate."""
    if len(mz_q) == 0 or len(mz_c) == 0:
        return 0.0
    order = np.lexsort((mz_q, -intensity_q))
    selected = order[: min(k, len(mz_q))]
    mz_c = np.sort(np.asarray(mz_c, dtype=np.float64))
    lower = mz_c - tol
    upper = mz_c + tol
    matched = 0
    for index in selected:
        mz = mz_q[index]
        position = np.searchsorted(lower, mz, side="right") - 1
        if position >= 0 and mz <= upper[position]:
            matched += 1
    return matched / len(selected)


# --------------------------------------------------------------------------
# Self-test on synthetic spectra.
# --------------------------------------------------------------------------

def run_self_test() -> None:
    mz_a = np.array([100.0, 200.0, 300.0])
    in_a = np.array([0.5, 0.3, 0.2])

    # Identical spectra: every scorer saturates.
    assert abs(cosine_similarity(mz_a, in_a, mz_a, in_a) - 1.0) < 1e-12
    assert abs(modified_cosine_similarity(mz_a, in_a, mz_a, in_a, 500.0, 500.0) - 1.0) < 1e-12
    assert abs(topk_overlap(mz_a, in_a, mz_a, in_a) - 1.0) < 1e-12

    # Fully disjoint peaks: zero.
    mz_b = np.array([110.0, 210.0, 310.0])
    assert cosine_similarity(mz_a, in_a, mz_b, in_a) == 0.0
    assert topk_overlap(mz_a, in_a, mz_b, in_a) == 0.0

    # Hand-computed partial cosine: one matched pair.
    # weights_q = sqrt(0.5) on 100.0; weights_c = sqrt(0.4) on 100.005.
    mz_c = np.array([100.005, 400.0])
    in_c = np.array([0.4, 0.6])
    # Hand-computed: only the 100.0/100.005 pair matches, so the numerator is
    # sqrt(0.5)*sqrt(0.4); denominators are the full weight norms.
    denominator = np.sqrt(0.5 + 0.3 + 0.2) * np.sqrt(0.4 + 0.6)
    expected = (np.sqrt(0.5) * np.sqrt(0.4)) / denominator
    observed = cosine_similarity(mz_a, in_a, mz_c, in_c)
    assert abs(observed - expected) < 1e-12, (observed, expected)

    # Greedy one-to-one: two query peaks can both be near one candidate peak.
    mz_d = np.array([100.0, 100.004])
    in_d = np.array([0.9, 0.1])
    mz_e = np.array([100.002])
    in_e = np.array([1.0])
    # Only one match is possible; greedy takes the larger product (0.9).
    value = cosine_similarity(mz_d, in_d, mz_e, in_e)
    assert abs(value - np.sqrt(0.9) / np.sqrt(0.9 + 0.1)) < 1e-12, value

    # Modified cosine recovers a precursor-shifted alignment.
    mz_f = np.array([600.0, 700.0])
    in_f = np.array([0.5, 0.5])
    mz_g = np.array([500.0, 600.0])
    in_g = np.array([0.5, 0.5])
    # precursor shift 100.0 aligns 700->600 and 600->500: perfect match.
    value = modified_cosine_similarity(mz_f, in_f, mz_g, in_g, 800.0, 700.0)
    assert abs(value - 1.0) < 1e-12, value
    # Without the shift the direct match only aligns 600 with 600.
    direct = cosine_similarity(mz_f, in_f, mz_g, in_g)
    assert abs(direct - 0.5) < 1e-12, direct

    # Top-k selection uses intensity order; tolerance window is inclusive.
    mz_h = np.array([100.0, 200.0, 300.0, 400.0])
    in_h = np.array([0.1, 0.5, 0.3, 0.2])  # top-3: 200, 300, 400
    mz_i = np.array([200.009, 299.995, 401.0])
    in_i = np.ones(3)
    assert abs(topk_overlap(mz_h, in_h, mz_i, in_i, k=3) - 2.0 / 3.0) < 1e-12
    assert topk_overlap(mz_h, in_h, mz_i, in_i, k=1) == 1.0

    print("GLM probe self-test PASS")


# --------------------------------------------------------------------------
# Main probe.
# --------------------------------------------------------------------------

def load_audit_ledger(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("relation-audit ledger is empty")
    required = {
        "query_index", "query_row", "query_ik14", "query_formula",
        "official_rank", "stage1_rank", "transition",
        "selection_geometry_best_negative_ik14",
        "selection_geometry_best_negative_relation",
        "selection_geometry_best_negative_same_formula",
        "selection_geometry_best_negative_mces_name",
    }
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"relation-audit ledger lacks {sorted(missing)}")
    return rows


def summarize_pool_sizes(sizes: list[int]) -> dict[str, float | int]:
    if not sizes:
        return {"count": 0}
    array = np.asarray(sizes, dtype=np.int64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": int(array.min()),
        "max": int(array.max()),
        "fraction_at_least_2": float(np.mean(array >= 2)),
        "fraction_at_least_3": float(np.mean(array >= 3)),
    }


def referee_summary(
    wins: np.ndarray, errors: np.ndarray,
) -> dict[str, float | int]:
    """Corrected/introduced ledger of a naive referee against an error mask."""
    corrected = int(np.sum(wins & errors))
    introduced = int(np.sum(~wins & ~errors))
    return {
        "corrected": corrected,
        "introduced": introduced,
        "net": corrected - introduced,
        "net_pp": 100.0 * (corrected - introduced) / int(len(errors)),
        "risk_net_lambda2": corrected - 2 * introduced,
        "exchange_ratio_corrected_per_introduced": (
            float(corrected / introduced) if introduced else None
        ),
    }


def main() -> None:
    args = arguments()
    if args.self_test:
        run_self_test()
        return
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    for path in (args.graph, args.held_per_query, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)

    ledger = load_audit_ledger(args.held_per_query)
    with np.load(args.graph, allow_pickle=False) as body:
        graph = {name: body[name] for name in body.files}
    required_graph = {
        "features", "feature_names", "query_ptr", "molecule_ptr",
        "molecule_label", "molecule_ik14", "molecule_formula",
        "molecule_mces_grade", "query_row", "query_ik14", "query_formula",
        "pair_candidate_row",
    }
    missing = required_graph - set(graph)
    if missing:
        raise RuntimeError(f"candidate graph lacks {sorted(missing)}")
    feature_names = list(map(str, graph["feature_names"]))
    if "dreams_similarity" not in feature_names:
        raise RuntimeError("candidate graph lacks official DreaMS similarity")
    score_column = feature_names.index("dreams_similarity")

    query_ptr = np.asarray(graph["query_ptr"], dtype=np.int64)
    molecule_ptr = np.asarray(graph["molecule_ptr"], dtype=np.int64)
    molecule_label = np.asarray(graph["molecule_label"], dtype=np.int8)
    molecule_ik14 = np.asarray(graph["molecule_ik14"]).astype(str)
    query_row = np.asarray(graph["query_row"], dtype=np.int64)
    query_ik14 = np.asarray(graph["query_ik14"]).astype(str)
    pair_candidate_row = np.asarray(graph["pair_candidate_row"], dtype=np.int64)
    if np.any(np.diff(molecule_ptr) < 1):
        raise RuntimeError("graph contains an empty molecule block")
    pair_scores = np.asarray(graph["features"][:, score_column], dtype=np.float32)
    molecule_scores = np.maximum.reduceat(pair_scores, molecule_ptr[:-1])

    # Resolve, for every ledger query, the true-molecule pool (the query's own
    # row excluded) and the selection-geometry top-rival pool, using exactly
    # the audit's molecule-max rule.
    plans: list[dict[str, object]] = []
    needed_rows: set[int] = set()
    rival_disagreements = 0
    for source in ledger:
        query = int(source["query_index"])
        left, right = int(query_ptr[query]), int(query_ptr[query + 1])
        labels = molecule_label[left:right]
        if labels[0] != 1 or int(labels.sum()) != 1:
            raise RuntimeError(f"query {query} has a malformed positive molecule")
        scores = molecule_scores[left:right]
        negative_local = 1 + int(np.argmax(scores[1:]))
        rival_molecule = left + negative_local
        if labels[negative_local] == 1:
            raise RuntimeError(f"query {query} selected a labelled rival")
        positive_rows = pair_candidate_row[
            molecule_ptr[left]:molecule_ptr[left + 1]
        ]
        positive_rows = positive_rows[positive_rows != int(query_row[query])]
        rival_rows = pair_candidate_row[
            molecule_ptr[rival_molecule]:molecule_ptr[rival_molecule + 1]
        ]
        if str(molecule_ik14[rival_molecule]) != source[
            "selection_geometry_best_negative_ik14"
        ]:
            rival_disagreements += 1
        plans.append({
            "source": source,
            "query": query,
            "query_row_value": int(query_row[query]),
            "query_ik14_value": str(query_ik14[query]),
            "positive_rows": positive_rows.astype(np.int64),
            "rival_rows": rival_rows.astype(np.int64),
        })
        needed_rows.add(int(query_row[query]))
        needed_rows.update(int(row) for row in positive_rows)
        needed_rows.update(int(row) for row in rival_rows)

    sorted_rows = np.asarray(sorted(needed_rows), dtype=np.int64)
    row_position = {int(row): index for index, row in enumerate(sorted_rows)}

    # Bulk-load by reading the whole datasets sequentially (fast contiguous
    # IO) and selecting the needed rows in memory; random-row fancy indexing
    # on hundreds of thousands of rows is pathologically slow in h5py.
    print(f"[GLM probe] loading spectra for {len(sorted_rows)} rows", flush=True)
    with h5py.File(args.data, "r") as handle:
        all_spectra = np.asarray(handle["spectrum"][()], dtype=np.float32)
        all_precursors = np.asarray(handle["precursor_mz"][()], dtype=np.float64)
        all_inchikeys = np.asarray(handle["INCHIKEY"][()]).astype(str)
    if int(all_spectra.shape[0]) <= int(sorted_rows[-1]):
        raise RuntimeError("spectra file is smaller than the graph row space")
    spectra = all_spectra[sorted_rows]
    precursors = all_precursors[sorted_rows]
    inchikeys = all_inchikeys[sorted_rows]
    del all_spectra
    valid_counts = np.sum(
        (spectra[:, 0, :] > 0) & (spectra[:, 1, :] > 0), axis=1,
    ).astype(np.int64)
    print("[GLM probe] spectra loaded; verifying alignment", flush=True)

    identity_mismatches = 0
    precursor_violations = 0
    max_precursor_ppm = 0.0
    for number, plan in enumerate(plans):
        source = plan["source"]
        query_position = row_position[int(plan["query_row_value"])]
        expected_ik14 = str(plan["query_ik14_value"])
        if not inchikeys[query_position].startswith(expected_ik14):
            identity_mismatches += 1
        query_precursor = float(precursors[query_position])
        for rows, expect_ik14 in (
            (plan["positive_rows"], expected_ik14),
            (plan["rival_rows"], source["selection_geometry_best_negative_ik14"]),
        ):
            for row in rows:
                position = row_position[int(row)]
                if not inchikeys[position].startswith(expect_ik14):
                    identity_mismatches += 1
                ppm = abs(
                    float(precursors[position]) - query_precursor
                ) / query_precursor * 1e6
                max_precursor_ppm = max(max_precursor_ppm, ppm)
                if ppm > 10.0 + 1e-6:
                    precursor_violations += 1
        if (number + 1) % 3000 == 0:
            print(f"[GLM probe] aligned {number + 1}/{len(plans)} queries", flush=True)
    if identity_mismatches or precursor_violations:
        raise RuntimeError(
            "row alignment verification failed: "
            f"identity={identity_mismatches} precursor={precursor_violations}"
        )

    peaks_mz: list[np.ndarray] = []
    peaks_intensity: list[np.ndarray] = []
    for position in range(len(sorted_rows)):
        count = int(valid_counts[position])
        block = spectra[position, :, :count]
        peaks_mz.append(np.ascontiguousarray(block[0]))
        peaks_intensity.append(np.ascontiguousarray(block[1]))
    del spectra

    # Score every query: molecule-max and molecule-mean for each scorer.
    details: list[dict[str, object]] = []
    for number, plan in enumerate(plans):
        source = plan["source"]
        query_position = row_position[int(plan["query_row_value"])]
        precursor_q = float(precursors[query_position])
        positive_positions = [
            row_position[int(row)] for row in plan["positive_rows"]
        ]
        rival_positions = [row_position[int(row)] for row in plan["rival_rows"]]
        record: dict[str, object] = {
            "query_index": int(plan["query"]),
            "query_row": int(plan["query_row_value"]),
            "query_ik14": str(plan["query_ik14_value"]),
            "query_formula": source["query_formula"],
            "official_rank": int(source["official_rank"]),
            "stage1_rank": int(source["stage1_rank"]),
            "transition": source["transition"],
            "relation": source["selection_geometry_best_negative_relation"],
            "rival_ik14": source["selection_geometry_best_negative_ik14"],
            "rival_mces_name": source["selection_geometry_best_negative_mces_name"],
            "rival_same_formula": source[
                "selection_geometry_best_negative_same_formula"
            ] == "True",
            "true_pool_size": len(positive_positions),
            "rival_pool_size": len(rival_positions),
            "dreams_positive_score": source["selection_geometry_positive_score"],
            "dreams_rival_score": source["selection_geometry_best_negative_score"],
        }
        for scorer in SCORER_NAMES:
            true_scores: list[float] = []
            rival_scores: list[float] = []
            for positions, bucket in (
                (positive_positions, true_scores),
                (rival_positions, rival_scores),
            ):
                for position in positions:
                    if scorer == "cosine":
                        value = cosine_similarity(
                            peaks_mz[query_position], peaks_intensity[query_position],
                            peaks_mz[position], peaks_intensity[position],
                        )
                    elif scorer == "modified_cosine":
                        value = modified_cosine_similarity(
                            peaks_mz[query_position], peaks_intensity[query_position],
                            peaks_mz[position], peaks_intensity[position],
                            precursor_q, float(precursors[position]),
                        )
                    else:
                        value = topk_overlap(
                            peaks_mz[query_position], peaks_intensity[query_position],
                            peaks_mz[position], peaks_intensity[position],
                        )
                    bucket.append(float(value))
            record[f"{scorer}_true_max"] = max(true_scores) if true_scores else None
            record[f"{scorer}_rival_max"] = max(rival_scores) if rival_scores else None
            record[f"{scorer}_true_mean"] = (
                float(np.mean(true_scores)) if true_scores else None
            )
            record[f"{scorer}_rival_mean"] = (
                float(np.mean(rival_scores)) if rival_scores else None
            )
            # In-memory only (underscore keys never reach the CSV): the full
            # per-spectrum scores and source rows power the pool-size lottery
            # control in v2.
            record[f"_{scorer}_true_scores"] = true_scores
            record[f"_{scorer}_rival_scores"] = rival_scores
        record["_true_rows"] = [int(row) for row in plan["positive_rows"]]
        record["_rival_rows"] = [int(row) for row in plan["rival_rows"]]
        details.append(record)
        if (number + 1) % 1000 == 0:
            print(f"[GLM probe] scored {number + 1}/{len(plans)} queries", flush=True)

    # Win flags: strict (a tie is not a win), molecule-max aggregation.  A
    # query with an empty true pool cannot be won by naive evidence.
    # v2 semantics: "oracle_any" is a LABEL-INFORMED OR over the three
    # scorers -- when scorers disagree it credits whichever side any metric
    # prefers, so it has no anti-symmetry and cannot be a candidate score.
    # It is reported only as an information-presence oracle.  "majority_vote"
    # is the label-blind anti-symmetric reference rule.
    for record in details:
        wins: list[int] = []
        for scorer in SCORER_NAMES:
            true_max = record[f"{scorer}_true_max"]
            rival_max = record[f"{scorer}_rival_max"]
            record[f"{scorer}_win"] = bool(
                true_max is not None and rival_max is not None and true_max > rival_max
            )
            wins.append(int(record[f"{scorer}_win"]))
        record["majority_vote_win"] = sum(wins) >= 2
        record["oracle_any_win"] = sum(wins) >= 1
        # Pool-size lottery control: molecule-max gives the side with more
        # spectra an extremum-sampling advantage.  Cap BOTH pools to the
        # smaller size under a fixed deterministic rule (the N lowest HDF5
        # row ids) and re-evaluate the same max verdicts.
        true_rows = np.asarray(record["_true_rows"], dtype=np.int64)
        rival_rows = np.asarray(record["_rival_rows"], dtype=np.int64)
        cap = int(min(len(true_rows), len(rival_rows)))
        true_order = np.argsort(true_rows, kind="stable")[:cap]
        rival_order = np.argsort(rival_rows, kind="stable")[:cap]
        capped_wins: list[int] = []
        for scorer in SCORER_NAMES:
            true_scores = np.asarray(record[f"_{scorer}_true_scores"], dtype=np.float64)
            rival_scores = np.asarray(record[f"_{scorer}_rival_scores"], dtype=np.float64)
            record[f"{scorer}_capped_win"] = bool(
                cap > 0
                and float(np.max(true_scores[true_order]))
                > float(np.max(rival_scores[rival_order]))
            )
            capped_wins.append(int(record[f"{scorer}_capped_win"]))
        record["oracle_capped_win"] = sum(capped_wins) >= 1
        record["capped_pool_size"] = cap

    official_errors = np.asarray(
        [record["official_rank"] > 1 for record in details], dtype=bool,
    )
    stage1_errors = np.asarray(
        [record["stage1_rank"] > 1 for record in details], dtype=bool,
    )
    n_queries = len(details)

    # Sanity anchor: the DreaMS selection scores must reproduce the held
    # official correctness up to the documented frozen-geometry proxy.
    dreams_wins = np.asarray(
        [
            float(record["dreams_positive_score"])
            > float(record["dreams_rival_score"])
            for record in details
        ], dtype=bool,
    )
    dreams_vs_official = referee_summary(dreams_wins, official_errors)

    fusion_referee: dict[str, dict[str, object]] = {}
    scorer_keys = list(SCORER_NAMES) + ["majority_vote", "oracle_any"]
    for key in scorer_keys:
        wins = np.asarray(
            [bool(record[f"{key}_win"]) for record in details], dtype=bool,
        )
        fusion_referee[key] = {
            # Rival always comes from the frozen OFFICIAL-selection geometry.
            # Pairing it with the official verdict is geometrically consistent;
            # pairing it with the Stage-1 verdict is a PROXY -- Stage-1's own
            # top rival may be a different molecule and is not available
            # without re-encoding on the cluster.
            "vs_official": referee_summary(wins, official_errors),
            "vs_stage1_verdict_official_geometry_rival_proxy": referee_summary(
                wins, stage1_errors,
            ),
            "win_rate_on_stage1_errors": float(np.mean(wins[stage1_errors])),
            "win_rate_on_stage1_correct": float(np.mean(wins[~stage1_errors])),
            "win_rate_overall": float(np.mean(wins)),
        }

    # Label-informed OR decomposition: which scorer patterns produce the
    # oracle's wins on Stage-1 residual errors (proxy pairing).
    pattern_counter: Counter = Counter()
    for record in details:
        if record["stage1_rank"] > 1:
            pattern = (
                f"c={int(record['cosine_win'])}"
                f"m={int(record['modified_cosine_win'])}"
                f"t={int(record['topk_overlap_win'])}"
            )
            if record["oracle_any_win"]:
                pattern_counter[pattern] += 1
    oracle_decomposition = dict(sorted(
        pattern_counter.items(), key=lambda item: (-item[1], item[0]),
    ))

    separability_strata: dict[str, dict[str, object]] = {}
    strata_values = sorted({str(record["relation"]) for record in details})
    for scorer_key in scorer_keys:
        wins = np.asarray(
            [bool(record[f"{scorer_key}_win"]) for record in details], dtype=bool,
        )
        by_stratum: dict[str, dict[str, float | int]] = {}
        for stratum in strata_values:
            mask = np.asarray(
                [str(record["relation"]) == stratum for record in details], dtype=bool,
            )
            for population_name, population in (
                ("stage1_errors", stage1_errors & mask),
                ("official_errors", official_errors & mask),
            ):
                total = int(np.sum(population))
                by_stratum[f"{population_name}|{stratum}"] = {
                    "queries": total,
                    "naive_true_wins": int(np.sum(wins & population)),
                    "win_rate": float(np.mean(wins[population])) if total else 0.0,
                }
        separability_strata[scorer_key] = by_stratum

    true_pool_sizes = [int(record["true_pool_size"]) for record in details]
    rival_pool_sizes = [int(record["rival_pool_size"]) for record in details]
    pool_by_stratum: dict[str, dict[str, object]] = {}
    for stratum in strata_values:
        mask = [str(record["relation"]) == stratum for record in details]
        pool_by_stratum[stratum] = {
            "queries": int(sum(mask)),
            "true_pool": summarize_pool_sizes(
                [size for size, keep in zip(true_pool_sizes, mask) if keep]
            ),
            "rival_pool": summarize_pool_sizes(
                [size for size, keep in zip(rival_pool_sizes, mask) if keep]
            ),
        }

    # Pool-size lottery control: capped-pool win rates versus the uncapped
    # verdicts, plus win rate by true/rival pool-size ratio bucket.  The gap
    # between uncapped and capped oracle rates measures how much of the
    # apparent separability is extremum-sampling advantage from library
    # density rather than chemistry.
    def _rate(flag: str, mask: np.ndarray) -> float:
        values = np.asarray(
            [bool(record[flag]) for record in details], dtype=bool,
        )
        return float(np.mean(values[mask])) if int(np.sum(mask)) else 0.0

    pool_ratio = np.asarray(
        [
            (record["true_pool_size"] / record["rival_pool_size"])
            if record["rival_pool_size"] else float("inf")
            for record in details
        ], dtype=np.float64,
    )
    ratio_edges = np.quantile(
        pool_ratio[stage1_errors], [0.25, 0.5, 0.75],
    )
    ratio_buckets: list[dict[str, float | int]] = []
    for bucket_index in range(4):
        if bucket_index == 0:
            mask = pool_ratio <= ratio_edges[0]
        elif bucket_index == 3:
            mask = pool_ratio > ratio_edges[2]
        else:
            mask = (
                (pool_ratio > ratio_edges[bucket_index - 1])
                & (pool_ratio <= ratio_edges[bucket_index])
            )
        errors_in_bucket = stage1_errors & mask
        ratio_buckets.append({
            "queries": int(np.sum(mask)),
            "stage1_errors": int(np.sum(errors_in_bucket)),
            "oracle_win_rate_on_stage1_errors": _rate(
                "oracle_any_win", errors_in_bucket,
            ),
            "oracle_win_rate_overall": _rate("oracle_any_win", mask),
            "cosine_win_rate_on_stage1_errors": _rate(
                "cosine_win", errors_in_bucket,
            ),
        })
    pool_size_control = {
        "uncapped_oracle_win_rate_on_stage1_errors": _rate(
            "oracle_any_win", stage1_errors,
        ),
        "capped_oracle_win_rate_on_stage1_errors": _rate(
            "oracle_capped_win", stage1_errors,
        ),
        "uncapped_cosine_win_rate_on_stage1_errors": _rate(
            "cosine_win", stage1_errors,
        ),
        "capped_cosine_win_rate_on_stage1_errors": _rate(
            "cosine_capped_win", stage1_errors,
        ),
        "mean_capped_pool_size": float(np.mean([
            record["capped_pool_size"] for record in details
        ])),
        "pool_ratio_quartile_edges": [float(edge) for edge in ratio_edges],
        "ratio_buckets_low_to_high": ratio_buckets,
        "interpretation": (
            "The capped pools remove the extremum-sampling advantage of the "
            "larger library side; the drop from uncapped to capped rates is "
            "the pool-size lottery share of apparent separability.  The "
            "bucket gradient shows the same confound directionally."
        ),
    }

    corrected_records = [
        record for record in details if record["transition"] == "corrected"
    ]
    introduced_records = [
        record for record in details if record["transition"] == "introduced"
    ]
    corrected_true_ik14s = {record["query_ik14"] for record in corrected_records}
    corrected_rival_ik14s = {record["rival_ik14"] for record in corrected_records}
    # See-saw signatures with a molecule-frequency background null.  High
    # frequency molecules appear in relation sets by chance, so the excess
    # over the background rate -- not the raw overlap -- is the coupling
    # evidence.  v1 double counted the two signatures; v2 reports the union.
    signature_a_records = [
        record for record in introduced_records
        if record["rival_ik14"] in corrected_true_ik14s
    ]
    signature_b_records = [
        record for record in introduced_records
        if record["rival_ik14"] in corrected_rival_ik14s
    ]
    signature_a_queries = {record["query_index"] for record in signature_a_records}
    signature_b_queries = {record["query_index"] for record in signature_b_records}
    union_queries = signature_a_queries | signature_b_queries
    background_records = [
        record for record in details
        if record["transition"] == "unchanged" and record["stage1_rank"] == 1
    ]
    background_union = sum(
        1 for record in background_records
        if record["rival_ik14"] in corrected_true_ik14s
        or record["rival_ik14"] in corrected_rival_ik14s
    )
    background_rate = (
        background_union / len(background_records) if background_records else 0.0
    )
    observed_rate = len(union_queries) / len(introduced_records)
    excess_rate = observed_rate - background_rate
    see_saw = {
        "introduced_queries": len(introduced_records),
        "signature_a_rival_is_corrected_true": len(signature_a_queries),
        "signature_b_rival_is_corrected_rival": len(signature_b_queries),
        "signature_overlap_queries": len(signature_a_queries & signature_b_queries),
        "union_queries": len(union_queries),
        "union_rate": observed_rate,
        "background_queries": len(background_records),
        "background_union": background_union,
        "background_union_rate": background_rate,
        "excess_rate_over_background": excess_rate,
        "excess_queries_estimate": excess_rate * len(introduced_records),
        "distinct_corrected_true_molecules": len(corrected_true_ik14s),
        "distinct_corrected_rival_molecules": len(corrected_rival_ik14s),
        "interpretation": (
            "Only the excess over the molecule-frequency background is "
            "evidence of causal see-saw coupling; the raw union mostly "
            "reflects how often popular molecules appear in any relation set."
        ),
    }

    report = {
        "status": "GLM_NOISE_RIVAL_SEPARABILITY_PROBE_COMPLETE",
        "probe_version": PROBE_VERSION,
        "author": "GLM-5.3 via DeepSeek Harness",
        "inputs": {
            "graph": str(args.graph),
            "held_per_query": str(args.held_per_query),
            "data": str(args.data),
            "graph_sha256": sha256_file(args.graph),
            "held_per_query_sha256": sha256_file(args.held_per_query),
            "data_sha256": sha256_file(args.data),
        },
        "settings": {
            "match_tolerance_da": MATCH_TOL_DA,
            "topk": TOPK,
            "aggregation": "molecule max (deployment) with mean reported",
            "strictness": "a tie is not a win",
        },
        "population": {
            "queries": n_queries,
            "official_errors": int(np.sum(official_errors)),
            "stage1_errors": int(np.sum(stage1_errors)),
            "rival_selection_disagreements_vs_relation_audit": rival_disagreements,
            "identity_mismatches": identity_mismatches,
            "precursor_violations_over_10ppm": precursor_violations,
            "max_precursor_ppm": float(max_precursor_ppm),
            "empty_true_pool_queries": int(np.sum(
                [record["true_pool_size"] == 0 for record in details]
            )),
        },
        "sanity_dreams_selection_referee_vs_official": dreams_vs_official,
        "fusion_referee": fusion_referee,
        "oracle_support_decomposition_on_stage1_errors": oracle_decomposition,
        "separability_by_stratum": separability_strata,
        "pool_size_distribution": {
            "overall": {
                "true_pool": summarize_pool_sizes(true_pool_sizes),
                "rival_pool": summarize_pool_sizes(rival_pool_sizes),
            },
            "by_relation": pool_by_stratum,
        },
        "pool_size_control": pool_size_control,
        "see_saw_molecule_pairing": see_saw,
        "claim_limit": (
            "ORACLE SEMANTICS: oracle_any is a label-informed OR over three "
            "scorers.  When scorers disagree it credits whichever side any "
            "metric prefers; it is not anti-symmetric, cannot produce a "
            "candidate ranking, and its net numbers are information-presence "
            "bounds, never achievable improvements.  An independent "
            "anti-symmetric logistic fusion with formula-group cross-validation "
            "(reviewer recomputation, 2026-09-29) nets +0.14 to +0.33pp, not "
            "the oracle figure.  GEOMETRY PROXY: every rival is the frozen "
            "official-selection top rival; verdicts paired with Stage-1 are "
            "proxies because Stage-1's own top rival may differ and requires "
            "cluster re-encoding.  PAIRWISE ONLY: single-rival adjudication is "
            "not full re-ranking, so net counts are not Recall@1 deltas; a "
            "third candidate may dominate both compared molecules.  POOL-SIZE "
            "CONFOUND: molecule-max mixes chemistry with library-density "
            "extremum advantage; see pool_size_control for the capped and "
            "ratio-bucket quantification.  NO TRAINING USE: every statistic "
            "here is selected on held labels and held errors; using these "
            "query sets to mine or weight training examples would contaminate "
            "the only frozen evaluation fold.  Relation mining for training "
            "must be rebuilt from outer-train formulas."
        ),
    }

    args.output_dir.mkdir(parents=True)
    fieldnames = [
        name for name in details[0] if not name.startswith("_")
    ]
    with gzip.open(
        args.output_dir / "per_query.csv.gz", "wt", encoding="utf-8", newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(details)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        key: report[key] for key in (
            "status", "population", "sanity_dreams_selection_referee_vs_official",
            "see_saw_molecule_pairing",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
