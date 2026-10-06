"""Match MoNA MetFrag snapshot (.mb) spectra against the sealed polarity
transfer panels by normalized-spectrum hash.

The sealed panels (chemaware_mona_polarity_transfer_local_preflight) reference
spectra by content hash (sha256 of the normalized 2xN array + precursor token,
as computed by chemaware_mona_transfer_core.normalized_spectrum_signature).
The server-side MoNA full MGFs are not local; this script tests whether the
public Zenodo MetFrag MoNA snapshot (2024-10-14) contains the same spectra.

Output: coverage report per polarity (query/candidate hash coverage) + a
hash -> (mz, intensities, precursor, ionmode) npz for downstream scoring.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(ROOT))
from chemaware_mona_transfer_core import normalized_spectrum_signature  # noqa: E402

MB = ROOT / "third_party/public_models/MoNA-LC-MS-MS-20241014.mb"
PREFLIGHT = ROOT / "data/validation/chemaware_mona_polarity_transfer_local_preflight"
OUT = ROOT / "data/validation/GLM_mona_metfrag_hash_match"


def iter_mb_records(path: Path):
    """MetFrag flat-format iterator: '# key = value' headers + 'mz int' lines."""
    peaks: list[tuple[float, float]] = []
    precursor = None
    ionmode = None

    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("#"):
                if "SampleName" in line and peaks and precursor is not None:
                    yield peaks, precursor, ionmode
                    peaks, precursor, ionmode = [], None, None
                if "IsPositiveIonMode" in line:
                    ionmode = ("positive" if "true" in line.lower() else
                               "negative")
                elif "IonizedPrecursorMass" in line:
                    try:
                        precursor = float(line.split("=")[1])
                    except (ValueError, IndexError):
                        precursor = None
                continue
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    peaks.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
    if peaks and precursor is not None:
        yield peaks, precursor, ionmode


def main() -> None:
    wanted: dict[str, set[str]] = {"positive": set(), "negative": set()}
    for pol in ("positive", "negative"):
        z = np.load(PREFLIGHT / pol / "panel.npz", allow_pickle=True)
        wanted[pol].update(map(str, z["query_spectrum_hash"]))
        wanted[pol].update(map(str, z["candidate_spectrum_hash"]))
    all_wanted = wanted["positive"] | wanted["negative"]
    print(f"wanted hashes: pos {len(wanted['positive']):,}, "
          f"neg {len(wanted['negative']):,}, union {len(all_wanted):,}")

    found: dict[str, dict] = {}
    n_records = 0
    for peaks, precursor, ionmode in iter_mb_records(MB):
        n_records += 1
        arr = np.asarray(peaks, dtype=np.float32).T  # (2, n)
        h = normalized_spectrum_signature(arr, float(precursor))
        if h in all_wanted:
            found.setdefault(h, {"mz": arr[0], "inten": arr[1],
                                 "precursor": float(precursor),
                                 "ionmode": ionmode or "unknown"})
        if n_records % 20000 == 0:
            print(f"  scanned {n_records:,} records, matched "
                  f"{len(found):,}/{len(all_wanted):,}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    cov = {}
    for pol in ("positive", "negative"):
        q = wanted[pol]
        hit = sum(1 for h in q if h in found)
        cov[pol] = {"wanted": len(q), "matched": hit,
                    "coverage_pct": round(100 * hit / max(1, len(q)), 2)}
    report = {"status": "GLM_MONA_METFRAG_HASH_MATCH",
              "mb_records_scanned": n_records,
              "coverage": cov,
              "note": ("Hash match against Zenodo MetFrag MoNA snapshot "
                       "2024-10-14; spectra equal only if peaks+precursor "
                       "identical after normalization.")}
    (OUT / "hash_match_report.json").write_text(json.dumps(report, indent=2),
                                                encoding="utf-8")
    if found:
        keys = sorted(found)
        mz_cat, in_cat, off = [], [], [0]
        meta = []
        for k in keys:
            v = found[k]
            mz_cat.append(v["mz"]); in_cat.append(v["inten"])
            off.append(off[-1] + len(v["mz"]))
            meta.append((v["precursor"], v["ionmode"]))
        np.savez_compressed(
            OUT / "matched_spectra.npz",
            hashes=np.asarray(keys),
            mz=np.concatenate(mz_cat).astype(np.float32),
            inten=np.concatenate(in_cat).astype(np.float32),
            offsets=np.asarray(off, dtype=np.int64),
            precursor=np.asarray([m[0] for m in meta], dtype=np.float64),
            ionmode=np.asarray([m[1] for m in meta]))
    print(json.dumps(cov, indent=2))
    print(f"records scanned: {n_records:,}; written {OUT / 'hash_match_report.json'}")


if __name__ == "__main__":
    main()
