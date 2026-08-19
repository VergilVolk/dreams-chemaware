"""A1 -- Parallel, resumable .raw -> .mzML conversion (ThermoRawFileParser).

Converts every .raw under ``--raw-root`` (default ``data/msv100574/raw``) into a
``.mzML`` that mirrors the source tree into ``data/msv100574/``::

    raw/Metabolomics/pos/HF_1.raw   ->   Metabolomics/pos/HF_1.mzML

Uses ``-f=1`` (mzML) with ThermoRawFileParser's defaults, which already include
peak picking + zlib compression (the smallest mzML the tool produces; the same
settings that produced the existing Met/neg files). Skips files whose .mzML
already exists, so an interrupted run resumes instead of restarting.

Usage (conda dreams_env, from repo root):
    python tasks/convert_raw_parallel.py --jobs 6            # convert all remaining
    python tasks/convert_raw_parallel.py --dry-run           # just count, no convert
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "data/tools/ThermoRawFileParser/ThermoRawFileParser.exe"


def collect_jobs(raw_root: Path, out_root: Path) -> list[tuple[Path, Path]]:
    """Return [(raw, out_mzml)] for every .raw whose .mzML does not yet exist."""
    jobs: list[tuple[Path, Path]] = []
    for raw in sorted(raw_root.rglob("*.raw")):
        rel = raw.relative_to(raw_root)          # Metabolomics/pos/HF_1.raw
        out_dir = out_root / rel.parent          # data/msv100574/Metabolomics/pos
        out_mzml = out_dir / (raw.stem + ".mzML")
        if not out_mzml.exists():
            jobs.append((raw, out_dir))
    return jobs


def convert_one(job: tuple[Path, Path], exe: Path) -> str:
    raw, out_dir = job
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(exe), f"-i={raw}", f"-o={out_dir}", "-f=1"],
                   check=True, capture_output=True)
    return f"{raw.name} -> {out_dir / (raw.stem + '.mzML')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=ROOT / "data/msv100574/raw")
    ap.add_argument("--jobs", type=int, default=6,
                    help="concurrent ThermoRawFileParser processes (18-core box: 6 is safe)")
    ap.add_argument("--exe", type=Path, default=EXE)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be converted, then exit")
    args = ap.parse_args()

    if not args.exe.exists():
        print(f"[convert] ThermoRawFileParser not found: {args.exe}", file=sys.stderr)
        sys.exit(1)

    out_root = args.raw_root.parent  # data/msv100574
    jobs = collect_jobs(args.raw_root, out_root)
    print(f"[convert] {len(jobs)} .raw files to convert "
          f"(output -> {out_root})", flush=True)
    if args.dry_run:
        for raw, out_dir in jobs[:10]:
            print(f"  {raw.relative_to(args.raw_root)} -> {out_dir.relative_to(out_root)}/")
        if len(jobs) > 10:
            print(f"  ... and {len(jobs) - 10} more")
        sys.exit(0)

    if not jobs:
        print("[convert] nothing to do", flush=True)
        sys.exit(0)

    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(convert_one, j, args.exe): j for j in jobs}
        for fut in futures:
            try:
                msg = fut.result()
            except subprocess.CalledProcessError as e:
                raw = futures[fut][0]
                print(f"[convert] FAILED {raw.name}: {e}", file=sys.stderr, flush=True)
                continue
            done += 1
            print(f"[convert] {done}/{len(jobs)}  {msg}", flush=True)

    print(f"[convert] finished: {done}/{len(jobs)} converted", flush=True)


if __name__ == "__main__":
    main()
