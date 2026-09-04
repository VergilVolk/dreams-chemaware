"""Mine the complete A4-B1 error space before designing a peak-token expert.

This diagnostic joins locked retrieval geometry, B0 teacher outcomes, B1 OOF
student outcomes, raw candidate-pair evidence, the complete 3,486-rule cache,
and acquisition metadata.  It trains only formula-group OOF diagnostic probes;
it does not update DreaMS or select a sealed-test model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from audit_noise_v3_a4_exact_peak_scan import load_embeddings, query_candidate_block, strict_detail
from build_g8r_real_error_atlas import Cache
from diagnose_noise_v3_a4b_positive_evidence import normalized_mean


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--rule-cache", type=Path, default=ROOT / "data/validation/g8r_chemaware_g0_rule_cache.npz")
    parser.add_argument("--b0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_a4b_positive_evidence")
    parser.add_argument("--b1-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_a4b_rescue_adapter")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_a4b_b1_error_space")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-queries", type=int, default=0, help="Smoke only")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def decode(value) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def finite_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def jaccard(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    left = a[mask]
    right = b[mask]
    union = int(np.sum(left | right))
    return float(np.sum(left & right) / union) if union else np.nan


def displacement_alignment(clean: np.ndarray, target: np.ndarray, student: np.ndarray) -> tuple[float, float, float]:
    teacher_delta = target.astype(np.float64) - clean.astype(np.float64)
    student_delta = student.astype(np.float64) - clean.astype(np.float64)
    teacher_norm = float(np.linalg.norm(teacher_delta))
    student_norm = float(np.linalg.norm(student_delta))
    if teacher_norm < 1e-12 or student_norm < 1e-12:
        alignment = np.nan
    else:
        alignment = float(np.dot(teacher_delta, student_delta) / (teacher_norm * student_norm))
    return teacher_norm, student_norm, alignment


def diagnostic_oof(frame: pd.DataFrame, feature_columns: list[str], label: np.ndarray,
                   groups: np.ndarray, folds: int, seed: int) -> dict:
    usable = np.isfinite(label.astype(float))
    y = label[usable].astype(int)
    local = frame.loc[usable, feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    local_groups = groups[usable]
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 10:
        return {"status": "insufficient_classes", "n": int(len(y)), "positives": int(y.sum())}
    n_splits = min(folds, len(np.unique(local_groups)))
    splitter = GroupKFold(n_splits=n_splits)
    prediction = np.full(len(y), np.nan, dtype=float)
    for fold, (train, test) in enumerate(splitter.split(local, y, local_groups)):
        train_frame = local.iloc[train]
        medians = train_frame.median().fillna(0.0)
        x_train = train_frame.fillna(medians).to_numpy(np.float32)
        x_test = local.iloc[test].fillna(medians).to_numpy(np.float32)
        model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=1.0,
            random_state=seed + fold,
        )
        model.fit(x_train, y[train])
        prediction[test] = model.predict_proba(x_test)[:, 1]
    if np.any(~np.isfinite(prediction)):
        raise RuntimeError("diagnostic OOF prediction is incomplete")
    return {
        "status": "formula_group_oof", "n": int(len(y)), "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, prediction)),
        "average_precision": float(average_precision_score(y, prediction)),
        "prevalence": float(y.mean()), "folds": int(n_splits),
    }


def main() -> None:
    args = parse_args()
    required = [
        args.graph, args.embeddings, args.data, args.rule_cache,
        args.b0_dir / "decision.json", args.b0_dir / "paired_results.csv.gz",
        args.b1_dir / "decision.json", args.b1_dir / "oof_queries.csv.gz",
        args.b1_dir / "oof_embeddings.npz",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    graph = Cache(args.graph)
    score_column = graph.feature_names.index("dreams_similarity")
    embedding_rows, official_embedding, embedding_index = load_embeddings(args.embeddings)
    frame = pd.read_csv(args.b1_dir / "oof_queries.csv.gz").sort_values("query_index").reset_index(drop=True)
    with np.load(args.b1_dir / "oof_embeddings.npz") as archive:
        clean = np.asarray(archive["clean"], dtype=np.float32)
        linear = np.asarray(archive["linear"], dtype=np.float32)
        nonlinear = np.asarray(archive["nonlinear"], dtype=np.float32)
    if args.max_queries:
        frame = frame.head(args.max_queries).copy()
        clean, linear, nonlinear = clean[:len(frame)], linear[:len(frame)], nonlinear[:len(frame)]
    formal = args.max_queries == 0
    if not (len(frame) == len(clean) == len(linear) == len(nonlinear)):
        raise RuntimeError("B1 frame/embedding alignment failed")
    if formal and len(frame) != 4998:
        raise RuntimeError("formal C0 requires all 4,998 B1 queries")

    with np.load(args.rule_cache, allow_pickle=True) as rules:
        rule_rows = np.asarray(rules["hdf5_row"], dtype=np.int64)
        packed = np.asarray(rules["packed_rule_hits"], dtype=np.uint8)
        n_rules = int(rules["n_rules"][0])
        category = np.asarray(rules["rule_category"], dtype=str)
        library = np.asarray(rules["rule_library"], dtype=str)
        semantics = np.asarray(rules["rule_semantics"], dtype=str)
    rule_index = {int(row): pos for pos, row in enumerate(rule_rows)}
    rule_masks = {
        "core": library == "core", "massbank": library == "massbank",
        "CF": category == "CF", "NL": category == "NL", "ISO": category == "ISO",
        "fragment_mz": semantics == "fragment_mz",
        "fragment_neutral_loss": semantics == "fragment_neutral_loss",
    }

    with h5py.File(args.data, "r") as handle:
        ik14 = np.asarray([decode(value)[:14] for value in handle["INCHIKEY"][embedding_rows]], dtype=object)
        adduct = np.asarray([decode(value) for value in handle["adduct"][embedding_rows]], dtype=object)
        instrument_ds = handle.get("INSTRUMENT_TYPE")
        ce_ds = handle.get("COLLISION_ENERGY")
        identity_groups: dict[tuple[str, str], list[int]] = {}
        for position, key in enumerate(zip(ik14, adduct)):
            identity_groups.setdefault((str(key[0]), str(key[1])), []).append(position)

        records = []
        for done, row in enumerate(frame.itertuples(index=False), start=1):
            query = int(row.query_index)
            query_row = int(row.query_row)
            scores, candidate_rows, ptr, molecule_left = query_candidate_block(graph, query, score_column)
            detail = strict_detail(scores, candidate_rows, ptr)
            if int(detail["rank"]) != int(row.baseline_rank):
                raise RuntimeError(f"baseline mismatch at query {query}")
            pos_left, pos_right = map(int, ptr[:2])
            pos_pair = pos_left + int(np.argmax(scores[pos_left:pos_right]))
            neg_row = int(detail["adversarial_pair_row"])
            pos_row = int(candidate_rows[pos_pair])
            query_position = embedding_index[query_row]

            key = (str(row.query_ik14)[:14], str(adduct[query_position]))
            support = [value for value in identity_groups.get(key, []) if int(embedding_rows[value]) != query_row]
            support = sorted(support, key=lambda value: int(embedding_rows[value]))[:12]
            prototype = normalized_mean(official_embedding[support])
            teacher = 0.75 * clean[done - 1].astype(np.float64) + 0.25 * prototype.astype(np.float64)
            teacher = (teacher / np.linalg.norm(teacher)).astype(np.float32)
            teacher_norm, student_norm, alignment = displacement_alignment(
                clean[done - 1], teacher, nonlinear[done - 1]
            )

            molecule_features = np.maximum.reduceat(
                graph.features[int(graph.molecule_ptr[molecule_left]):int(graph.molecule_ptr[molecule_left + len(ptr) - 1])],
                ptr[:-1], axis=0,
            )
            raw = {}
            negative_local = int(detail["adversarial_molecule_local"])
            for feature_index, feature_name in enumerate(graph.feature_names):
                raw[f"gap_{feature_name}"] = float(
                    molecule_features[0, feature_index] - molecule_features[negative_local, feature_index]
                )

            needed = (query_row, pos_row, neg_row)
            if any(value not in rule_index for value in needed):
                raise RuntimeError(f"rule cache misses a reachable row at query {query}")
            hit = []
            for value in needed:
                hit.append(np.unpackbits(packed[rule_index[value]], bitorder="little")[:n_rules].astype(bool))
            q_rule, p_rule, n_rule = hit
            rule_values = {"query_rule_count": int(q_rule.sum())}
            for name, mask in rule_masks.items():
                qp = jaccard(q_rule, p_rule, mask)
                qn = jaccard(q_rule, n_rule, mask)
                rule_values[f"rule_pos_{name}"] = qp
                rule_values[f"rule_neg_{name}"] = qn
                rule_values[f"rule_gap_{name}"] = qp - qn if np.isfinite(qp) and np.isfinite(qn) else np.nan

            query_instrument = decode(instrument_ds[query_row]) if instrument_ds is not None else "unknown"
            pos_instrument = decode(instrument_ds[pos_row]) if instrument_ds is not None else "unknown"
            neg_instrument = decode(instrument_ds[neg_row]) if instrument_ds is not None else "unknown"
            query_ce = finite_float(ce_ds[query_row]) if ce_ds is not None else np.nan
            pos_ce = finite_float(ce_ds[pos_row]) if ce_ds is not None else np.nan

            baseline_correct = int(row.baseline_rank) == 1
            student_correct = int(row.nonlinear_rank) == 1
            teacher_rescue = bool(row.corrected)
            if teacher_rescue and student_correct:
                outcome = "teacher_rescue_student_recovered"
            elif teacher_rescue:
                outcome = "teacher_rescue_student_missed"
            elif (not baseline_correct) and student_correct:
                outcome = "student_independent_correction"
            elif baseline_correct and not student_correct:
                outcome = "student_introduced_error"
            elif not baseline_correct:
                outcome = "persistent_official_error"
            else:
                outcome = "protected_correct"
            records.append({
                **row._asdict(), "outcome": outcome,
                "teacher_displacement_norm": teacher_norm,
                "student_displacement_norm": student_norm,
                "student_teacher_displacement_alignment": alignment,
                "student_margin_delta": float(row.nonlinear_margin - row.baseline_margin),
                "positive_row": pos_row, "adversarial_row": neg_row,
                "adversarial_mces_grade": int(graph.molecule_mces_grade[molecule_left + negative_local]),
                "candidate_molecules": int(len(ptr) - 1),
                "query_instrument": query_instrument,
                "positive_instrument": pos_instrument,
                "negative_instrument": neg_instrument,
                "positive_cross_instrument": bool(query_instrument != pos_instrument),
                "positive_ce_difference": abs(query_ce - pos_ce) if np.isfinite(query_ce) and np.isfinite(pos_ce) else np.nan,
                **raw, **rule_values,
            })
            if done % 500 == 0 or done == len(frame):
                print(f"[B1-C0] {done:,}/{len(frame):,} queries", flush=True)

    atlas = pd.DataFrame(records)
    outcome_counts = {str(k): int(v) for k, v in atlas["outcome"].value_counts().items()}
    deployable_features = [
        "baseline_margin", "candidate_molecules", "support_spectra", "has_near",
        "positive_deficit", "negative_excess", "positive_cross_instrument",
        "positive_ce_difference", "query_rule_count",
    ] + [column for column in atlas if column.startswith("gap_") or column.startswith("rule_gap_")]
    post_adapter_features = deployable_features + [
        "student_displacement_norm", "student_margin_delta",
    ]
    groups = atlas["query_formula"].astype(str).to_numpy()
    error = atlas["baseline_rank"].to_numpy(int) > 1
    teacher_label = atlas["corrected"].astype(bool).to_numpy()
    changed = (
        (atlas["nonlinear_rank"].to_numpy(int) == 1)
        != (atlas["baseline_rank"].to_numpy(int) == 1)
    )
    beneficial = np.full(len(atlas), np.nan)
    beneficial[changed] = (
        (atlas.loc[changed, "baseline_rank"].to_numpy(int) > 1)
        & (atlas.loc[changed, "nonlinear_rank"].to_numpy(int) == 1)
    ).astype(float)
    probes = {
        "teacher_rescue_among_official_errors": diagnostic_oof(
            atlas.loc[error].reset_index(drop=True), deployable_features,
            teacher_label[error], groups[error], args.folds, args.seed,
        ),
        "beneficial_intervention_among_rank_changes": diagnostic_oof(
            atlas, post_adapter_features, beneficial, groups, args.folds, args.seed + 100,
        ),
    }

    summary_by_outcome = {}
    numeric_summary = [
        "baseline_margin", "target_margin", "nonlinear_margin",
        "teacher_displacement_norm", "student_displacement_norm",
        "student_teacher_displacement_alignment", "student_margin_delta",
        "support_spectra", "candidate_molecules", "rule_gap_core", "rule_gap_massbank",
        "gap_sqrt_cosine", "gap_entropy_similarity", "gap_neutral_loss_sqrt_cosine",
    ]
    for outcome, local in atlas.groupby("outcome", sort=False):
        summary_by_outcome[str(outcome)] = {
            "queries": int(len(local)), "identities": int(local["query_ik14"].nunique()),
            "formulas": int(local["query_formula"].nunique()),
            "near_fraction": float(local["has_near"].mean()),
            "positive_deficit_fraction": float(local["positive_deficit"].mean()),
            "metrics": {
                column: {
                    "median": float(local[column].median()),
                    "p10": float(local[column].quantile(0.10)),
                    "p90": float(local[column].quantile(0.90)),
                } for column in numeric_summary if column in local
            },
        }

    decision = {
        "status": "noise_v3_a4b_b1_error_space_complete", "formal": formal,
        "queries": int(len(atlas)), "outcome_counts": outcome_counts,
        "teacher_rescue_recovery_rate": float(
            outcome_counts.get("teacher_rescue_student_recovered", 0)
            / max(int(teacher_label.sum()), 1)
        ),
        "summary_by_outcome": summary_by_outcome,
        "formula_group_oof_diagnostic_probes": probes,
        "interpretation_rule": (
            "Teacher labels and displacement statistics are diagnostic-only. Only raw/rule/metadata "
            "features enter the deployable rescue probe; no probe result is sealed-test performance."
        ),
        "provenance": {str(path): sha256_file(path) for path in required},
    }
    staging = Path(tempfile.mkdtemp(prefix="a4b_b1_error_", dir=args.output_dir.parent))
    try:
        atlas.to_csv(staging / "query_error_space.csv.gz", index=False, compression="gzip")
        atlas.loc[atlas["outcome"].eq("teacher_rescue_student_missed")].sort_values(
            ["target_margin", "teacher_displacement_norm"], ascending=[False, False]
        ).to_csv(staging / "priority_teacher_rescue_missed.csv.gz", index=False, compression="gzip")
        atlas.loc[atlas["outcome"].eq("student_introduced_error")].sort_values(
            "student_margin_delta"
        ).to_csv(staging / "introduced_errors.csv.gz", index=False, compression="gzip")
        (staging / "decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(decision, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
