"""Compute the unique local-pair MCES table with a resumable SQLite cache."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
from myopic_mces import MCES


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "data/e2/mces_local_rank"


SCHEMA = """
CREATE TABLE IF NOT EXISTS mces_pair (
    pair_key TEXT PRIMARY KEY,
    split TEXT NOT NULL,
    ik14_a TEXT NOT NULL,
    ik14_b TEXT NOT NULL,
    smiles_a TEXT NOT NULL,
    smiles_b TEXT NOT NULL,
    distance REAL,
    mode INTEGER,
    distance_kind TEXT,
    usable_exact_local INTEGER,
    usable_proven_far INTEGER,
    elapsed_seconds REAL,
    status TEXT NOT NULL,
    error TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


EXACT_MODE = 1
PROVEN_FAR_MODES = {2, 4, 20, 21, 22}


def compute_pair(payload: tuple[str, str, str, str, str, float, float]) -> tuple:
    """Compute one pair in a worker process; keep every SQLite write in the parent."""
    key, ik14_a, ik14_b, smiles_a, smiles_b, threshold, time_limit = payload
    try:
        _, distance, elapsed, mode = MCES(
            smiles_a,
            smiles_b,
            threshold=threshold,
            solver_options={"msg": False, "timeLimit": time_limit, "threads": 1},
            catch_errors=True,
        )
        status = "ok" if float(distance) >= 0 else "failed"
        error = None if status == "ok" else "myopic_mces returned a negative distance"
    except Exception as exc:
        distance, elapsed, mode = None, 0.0, -1
        status, error = "error", str(exc)
    kind, usable_local, usable_far = distance_semantics(
        None if distance is None else float(distance), int(mode), threshold,
    )
    return (
        key, ik14_a, ik14_b, smiles_a, smiles_b, distance, int(mode), kind,
        usable_local, usable_far, float(elapsed), status, error,
    )


def distance_semantics(distance: float | None, mode: int, threshold: float) -> tuple[str, int, int]:
    """Separate exact local distances from threshold/bound-only far pairs."""
    if distance is None or distance < 0:
        return "invalid", 0, 0
    if mode == EXACT_MODE:
        return "exact", int(distance <= threshold), 0
    if mode == 2:
        return "proven_above_threshold", 0, 1
    if mode in {4, 20, 21, 22}:
        return "lower_bound_above_threshold", 0, 1
    if mode == 5:
        return "timeout_unproven_solution", 0, 0
    if mode == 6:
        return "timeout_bound", 0, 0
    return "unknown", 0, 0


def migrate_schema(connection: sqlite3.Connection) -> None:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(mces_pair)")}
    for name, definition in (
        ("distance_kind", "TEXT"),
        ("usable_exact_local", "INTEGER"),
        ("usable_proven_far", "INTEGER"),
    ):
        if name not in columns:
            connection.execute(f"ALTER TABLE mces_pair ADD COLUMN {name} {definition}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_DIR / "mces_cache.sqlite")
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["val", "train"])
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--time-limit", type=float, default=3.0)
    parser.add_argument("--commit-every", type=int, default=25)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--export-every", type=int, default=250)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Independent MCES worker processes. SQLite remains single-writer in the parent.",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Retry cache rows whose previous status was not ok.",
    )
    args = parser.parse_args()
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.cache)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(SCHEMA)
    migrate_schema(connection)
    connection.commit()

    total_new = total_ok = total_error = 0
    started = time.time()
    for split in args.splits:
        path = args.manifest_dir / f"{split}_unique_molecule_pairs.csv"
        frame = pd.read_csv(path)
        keys = frame["ik14_a"].astype(str) + "|" + frame["ik14_b"].astype(str)
        done_query = (
            "SELECT pair_key FROM mces_pair WHERE split = ? AND status = 'ok'"
            if args.retry_failed else
            "SELECT pair_key FROM mces_pair WHERE split = ?"
        )
        done = {row[0] for row in connection.execute(done_query, (split,))}
        pending = frame.loc[~keys.isin(done)].copy()
        if args.max_pairs:
            pending = pending.head(args.max_pairs)
        print(
            f"{split}: manifest={len(frame):,}, cached={len(done):,}, pending={len(pending):,}",
            flush=True,
        )
        payloads = (
            (
                f"{row.ik14_a}|{row.ik14_b}", str(row.ik14_a), str(row.ik14_b),
                str(row.smiles_a), str(row.smiles_b), args.threshold, args.time_limit,
            )
            for row in pending.itertuples(index=False)
        )
        executor = None
        if args.workers > 1:
            executor = ProcessPoolExecutor(max_workers=args.workers)
            results = executor.map(compute_pair, payloads, chunksize=1)
        else:
            results = map(compute_pair, payloads)
        try:
            for position, result in enumerate(results, 1):
                (
                    key, ik14_a, ik14_b, smiles_a, smiles_b, distance, mode, kind,
                    usable_local, usable_far, elapsed, status, error,
                ) = result
                connection.execute(
                    """INSERT OR REPLACE INTO mces_pair
                       (pair_key, split, ik14_a, ik14_b, smiles_a, smiles_b,
                        distance, mode, distance_kind, usable_exact_local,
                        usable_proven_far, elapsed_seconds, status, error)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (key, split, ik14_a, ik14_b, smiles_a, smiles_b,
                     distance, mode, kind, usable_local, usable_far,
                     elapsed, status, error),
                )
                total_new += 1
                total_ok += status == "ok"
                total_error += status != "ok"
                if position % args.commit_every == 0:
                    connection.commit()
                if position % args.export_every == 0 or position == len(pending):
                    connection.commit()
                    rate = total_new / max(time.time() - started, 1e-9)
                    remaining = len(pending) - position
                    print(
                        f"  {split} {position:,}/{len(pending):,} | ok={total_ok:,} "
                        f"failed={total_error:,} | {rate:.2f} pairs/s | "
                        f"split ETA={remaining / max(rate, 1e-9) / 60:.1f} min",
                        flush=True,
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    connection.commit()
    # Backfill semantic columns for caches made by older versions of this script.
    for pair_key, distance, mode in connection.execute(
        "SELECT pair_key, distance, mode FROM mces_pair WHERE distance_kind IS NULL"
    ):
        kind, usable_local, usable_far = distance_semantics(distance, int(mode), args.threshold)
        connection.execute(
            "UPDATE mces_pair SET distance_kind=?, usable_exact_local=?, "
            "usable_proven_far=? WHERE pair_key=?",
            (kind, usable_local, usable_far, pair_key),
        )
    connection.commit()
    rows = pd.read_sql_query("SELECT * FROM mces_pair ORDER BY split, pair_key", connection)
    rows.to_csv(args.manifest_dir / "mces_pair_cache.csv", index=False)
    summary = {
        "status": "mces_cache_updated",
        "cache": str(args.cache.resolve()),
        "rows": int(len(rows)),
        "status_counts": {str(k): int(v) for k, v in rows["status"].value_counts().items()},
        "split_counts": {str(k): int(v) for k, v in rows["split"].value_counts().items()},
        "mode_counts": {str(k): int(v) for k, v in rows["mode"].value_counts(dropna=False).items()},
        "distance_kind_counts": {
            str(k): int(v) for k, v in rows["distance_kind"].value_counts(dropna=False).items()
        },
        "usable_exact_local": int(rows["usable_exact_local"].fillna(0).sum()),
        "usable_proven_far": int(rows["usable_proven_far"].fillna(0).sum()),
        "new_rows_this_run": total_new,
        "elapsed_seconds": time.time() - started,
    }
    (args.manifest_dir / "mces_cache_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    connection.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
