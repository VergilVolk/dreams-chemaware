#!/usr/bin/env python
"""Download the frozen MSV000096214 mzXML set (truth-blind contract stage 3.1).

Files, expected sizes and ordering come from the frozen readiness manifest.
Byte sizes are verified per file; sha256 is recorded. QC files are fetched
first so genotype-blind noise calibration can start early. Resume-safe via
.part temporary files. This script never reads the genotype column.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

DATASET = "MSV000096214"
BASE = "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?"
UA = "DreaMS-biology/1.0"


def fetch_stream(remote: str, target: Path, expected_bytes: int, state: dict) -> None:
    """Download one remote file to target with resume and verification."""
    url = BASE + urllib.parse.urlencode({"file": f"f.{DATASET}/{remote}"})
    part = target.with_suffix(target.suffix + ".part")
    if target.exists() and target.stat().st_size == expected_bytes:
        state["skipped_complete"] += 1
        return
    for attempt in range(8):
        headers = {"User-Agent": UA}
        offset = part.stat().st_size if part.exists() else 0
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=600) as response, open(part, "ab" if offset else "wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    state["bytes_downloaded"] += len(chunk)
            size = part.stat().st_size
            if size != expected_bytes:
                # Range may be unsupported or mismatched; restart once from zero.
                if offset:
                    part.unlink(missing_ok=True)
                    state["restarted"] += 1
                    continue
                raise IOError(f"size mismatch: expected {expected_bytes} got {size} for {remote}")
            digest = hashlib.sha256()
            with open(part, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(chunk)
            part.replace(target)
            state["completed"] += 1
            state["sha256"][target.name] = digest.hexdigest()
            return
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < 7:
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(30.0, 5.0 * (attempt + 1))
                time.sleep(delay)
                continue
            state["failed"][remote] = f"HTTP {error.code}"
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < 7:
                time.sleep(min(60.0, 10.0 * (attempt + 1)))
                continue
            state["failed"][remote] = repr(exc)
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/validation/msv000096214_readiness_v2/sample_manifest.csv.gz"))
    parser.add_argument("--root", type=Path, default=Path("data/external/msv000096214"))
    parser.add_argument("--status-every", type=int, default=5)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    todo = frame.loc[frame["deposited"]].copy()
    todo["order_key"] = [(0 if role == "pooled_qc" else 1, int(tube) if str(tube).isdigit() else 0)
                         for role, tube in zip(todo["role"], todo["tube"])]
    todo = todo.sort_values(["order_key", "polarity", "filename"])

    args.root.mkdir(parents=True, exist_ok=True)
    state = {"completed": 0, "skipped_complete": 0, "restarted": 0,
             "bytes_downloaded": 0, "failed": {}, "sha256": {}}
    progress_path = args.root / "download_progress.json"
    rows: list[dict[str, object]] = []
    for i, row in enumerate(todo.itertuples(index=False), start=1):
        dest_dir = args.root / "mzxml" / row.polarity
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / row.filename
        fetch_stream(str(row.remote_path), target, int(row.bytes), state)
        rows.append({"filename": row.filename, "polarity": row.polarity, "role": row.role,
                     "tube": row.tube, "expected_bytes": int(row.bytes),
                     "ok": target.exists() and target.stat().st_size == int(row.bytes),
                     "sha256": state["sha256"].get(row.filename, "")})
        if i % args.status_every == 0 or i == len(todo):
            progress_path.write_text(json.dumps(
                {"status": "running", "done": i, "total": len(todo), **state,
                 "failed": state["failed"]}, indent=2), encoding="utf-8")
            print(f"[{i}/{len(todo)}] ok={state['completed'] + state['skipped_complete']} "
                  f"failed={len(state['failed'])} gb={state['bytes_downloaded'] / 1e9:.1f}", flush=True)

    pd.DataFrame(rows).to_csv(args.root / "download_manifest.csv", index=False)
    summary = {
        "status": "MSV000096214_MZXML_DOWNLOAD_COMPLETE" if not state["failed"] else "COMPLETE_WITH_FAILURES",
        "dataset": DATASET,
        "data_role": "development/consumed",
        "files_expected": int(len(todo)),
        "files_ok": int(sum(r["ok"] for r in rows)),
        "bytes_downloaded": state["bytes_downloaded"],
        "failed": state["failed"],
        "contract": "docs/GLM_MSV000096214_STRUCTURE_HIDDEN_CONTRACT_20261012.md",
        "genotype_column_read": False,
    }
    (args.root / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
