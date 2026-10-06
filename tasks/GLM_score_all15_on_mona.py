"""Score the 15-method bench on the SEALED MoNA polarity transfer panels.

Panels reference spectra by normalized-content hash; spectra come from the
server-side MoNA full MGFs (mona_pos_full.mgf / mona_neg_full.mgf) that the
panels were built from, so hash coverage must be exactly 100%.

Stage-1 methods (this file, CPU, pinned backends):
  cosine_greedy modified_cosine weighted_spectral_entropy
Stage-2 (separate submission): entropy_raw_public, ms2deepscore_2x_public,
  spec2vec_gnps_public, spec2vec_2026_retrained (embedding machinery) and
  our encoder variants (server checkpoints).

Outputs scores_{polarity}_{method}.npy aligned to each panel's
candidate_spectrum_hash pair order, + a meta json with md5 anchors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(ROOT))

from chemaware_mona_transfer_core import iter_mgf_records  # noqa: E402
from noise_gnps_article_spectral_scores import (  # noqa: E402
    cosine_greedy, modified_cosine, weighted_entropy_similarity,
)

POLARITIES = ("positive", "negative")


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_hash_map(mgf: Path, wanted: set[str]) -> dict[str, tuple]:
    got: dict[str, tuple] = {}
    for rec in iter_mgf_records(mgf, include_peaks=True):
        h = str(rec.get("spectrum_hash", ""))
        if h in wanted and h not in got:
            got[h] = (np.asarray(rec["peaks"][0], dtype=np.float64),
                      np.asarray(rec["peaks"][1], dtype=np.float64),
                      float(rec["precursor_mz"]))
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", type=Path, required=True)
    ap.add_argument("--positive-mgf", type=Path, required=True)
    ap.add_argument("--negative-mgf", type=Path, required=True)
    ap.add_argument("--staging", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", required=True,
                    choices=("cosine_greedy", "modified_cosine",
                             "weighted_spectral_entropy"))
    args = ap.parse_args()
    args.staging.mkdir(parents=True, exist_ok=True)
    mgfs = {"positive": args.positive_mgf, "negative": args.negative_mgf}

    for pol in POLARITIES:
        with np.load(args.panels / pol / "panel.npz",
                     allow_pickle=True) as z:
            panel = {k: np.asarray(z[k]) for k in z.files}
        q_hash = panel["query_spectrum_hash"]
        wanted = set(map(str, q_hash)) | set(
            map(str, panel["candidate_spectrum_hash"]))
        spectra = build_hash_map(mgfs[pol], wanted)
        missing = wanted - set(spectra)
        if missing:
            raise RuntimeError(
                f"{pol}: {len(missing)}/{len(wanted)} panel hashes missing "
                f"from {mgfs[pol]} -- sealed panels must match exactly")
        print(f"[{pol}] spectra resolved: {len(spectra):,}/{len(wanted):,}",
              flush=True)

        # pair -> query hash: molecules carry pairs; queries carry molecules
        q_of_molecule = np.repeat(np.arange(len(q_hash)),
                                  np.diff(panel["query_ptr"]))
        pair_query = np.repeat(q_hash[q_of_molecule],
                               np.diff(panel["molecule_ptr"]))
        assert len(pair_query) == len(panel["candidate_spectrum_hash"])
        pair_cand = panel["candidate_spectrum_hash"]

        for name in args.methods:
            out = np.zeros(len(pair_cand), dtype=np.float32)
            for i, (qh, ch) in enumerate(zip(pair_query, pair_cand)):
                a, b = spectra[str(qh)], spectra[str(ch)]
                sa = np.vstack((a[0], a[1]))
                sb = np.vstack((b[0], b[1]))
                if name == "modified_cosine":
                    out[i] = float(modified_cosine(sa, a[2], sb, b[2]))
                elif name == "cosine_greedy":
                    out[i] = float(cosine_greedy(sa, sb))
                else:
                    out[i] = float(weighted_entropy_similarity(sa, sb))
            np.save(args.staging / f"scores_{pol}_{name}.npy", out)
            print(f"[{pol}] {name} staged ({len(out):,} pairs)", flush=True)

    meta = {"status": "GLM_MONA_STAGE1_CLASSICAL",
            "panels": str(args.panels),
            "positive_mgf_md5": md5_of(args.positive_mgf),
            "negative_mgf_md5": md5_of(args.negative_mgf),
            "methods": args.methods,
            "pair_alignment": "candidate_spectrum_hash order per panel"}
    (args.staging / "mona_scores_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print("meta written:", args.staging / "mona_scores_meta.json")


if __name__ == "__main__":
    main()
