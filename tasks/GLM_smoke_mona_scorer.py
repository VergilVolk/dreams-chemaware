"""Smoke test for GLM_score_all15_on_mona.py with synthetic spectra.

Builds two tiny MGFs + panel npzs that mimic the sealed MoNA preflight
structure (query/molecule/pair pointers, hash-referenced spectra), runs the
stage-1 scorer end-to-end, and validates outputs.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(ROOT))
from chemaware_mona_transfer_core import (  # noqa: E402
    iter_mgf_records, normalized_spectrum_signature,
)

SMOKE = ROOT / "data/validation/GLM_mona_scorer_smoke"
if SMOKE.exists():
    shutil.rmtree(SMOKE)
(SMOKE / "panels" / "positive").mkdir(parents=True)
(SMOKE / "panels" / "negative").mkdir(parents=True)
(SMOKE / "staging").mkdir(parents=True)

rng = np.random.default_rng(7)


def make_spectra(n, seed_shift):
    out = []
    for i in range(n):
        mz = np.sort(rng.uniform(50, 500, 12 + i % 5)) + seed_shift
        inten = rng.uniform(1, 100, len(mz))
        out.append((mz, inten))
    return out


def write_mgf(path, spectra, precursors):
    with path.open("w", encoding="utf-8") as f:
        for idx, ((mz, inten), prec) in enumerate(zip(spectra, precursors)):
            f.write("BEGIN IONS\n")
            f.write(f"PEPMASS={prec}\n")
            f.write(f"SMILES=C{idx}H{idx}N{idx}O{idx}\n")
            f.write(f"INCHIKEY=AAAAAAAAAAAAAA-{idx:02d}\n")
            for m, s in zip(mz, inten):
                f.write(f"{m:.6f} {s:.6f}\n")
            f.write("END IONS\n")


def build_panel(spectra, precursors, n_queries):
    hashes = [normalized_spectrum_signature(
        np.vstack((mz, inten)).astype(np.float32), float(prec))
        for (mz, inten), prec in zip(spectra, precursors)]
    # queries 0..n-1; each query gets 2 molecules; each molecule 1-2 pairs
    query_ptr = [0]
    molecule_ptr = [0]
    molecule_label = []
    pair_cand = []
    q_hashes, mol_per_q = [], []
    pair_q_list = []
    for q in range(n_queries):
        mols = [q * 2, q * 2 + 1]
        for k, mol_i in enumerate(mols):
            pair_q_list.append(hashes[q])
            n_pairs = 1 + (mol_i % 2)
            for p in range(n_pairs):
                pair_cand.append(hashes[(mol_i + p) % len(hashes)])
            molecule_ptr.append(molecule_ptr[-1] + n_pairs)
            molecule_label.append(1 if k == 0 else 0)
        query_ptr.append(query_ptr[-1] + len(mols))
        q_hashes.append(hashes[q])
    return (np.asarray(q_hashes), np.asarray(query_ptr),
            np.asarray(molecule_ptr), np.asarray(molecule_label),
            np.asarray(pair_cand), np.asarray(pair_q_list))


for pol, seed in (("positive", 0.0), ("negative", 3.0)):
    spectra = make_spectra(12, seed)
    precursors = [200.0 + 10 * i + seed for i in range(12)]
    mgf = SMOKE / f"mona_{pol}_full.mgf"
    write_mgf(mgf, spectra, precursors)
    # re-parse so panel hashes are computed from the MGF text round-trip,
    # exactly as the production panels were built from the server MGFs
    parsed = list(iter_mgf_records(mgf, include_peaks=True))
    assert len(parsed) == 12
    p_spectra = [(np.asarray(r["peaks"][0], dtype=np.float64),
                  np.asarray(r["peaks"][1], dtype=np.float64))
                 for r in parsed]
    p_prec = [float(r["precursor_mz"]) for r in parsed]
    q_hashes, query_ptr, molecule_ptr, label, pair_cand, pair_q = (
        build_panel(p_spectra, p_prec, n_queries=4))
    np.savez(SMOKE / "panels" / pol / "panel.npz",
             query_spectrum_hash=q_hashes,
             query_ptr=query_ptr,
             molecule_ptr=molecule_ptr,
             molecule_label=label.astype(np.int8),
             candidate_spectrum_hash=pair_cand)

# (placeholder cleaned)
proc = subprocess.run(
    [sys.executable, "-X", "utf8", str(ROOT / "tasks/GLM_score_all15_on_mona.py"),
     "--panels", str(SMOKE / "panels"),
     "--positive-mgf", str(SMOKE / "mona_positive_full.mgf"),
     "--negative-mgf", str(SMOKE / "mona_negative_full.mgf"),
     "--staging", str(SMOKE / "staging"),
     "--methods", "cosine_greedy", "modified_cosine",
     "weighted_spectral_entropy"],
    capture_output=True, text=True)
print(proc.stdout)
print(proc.stderr)
assert proc.returncode == 0, "scorer failed"
for pol in ("positive", "negative"):
    with np.load(SMOKE / "panels" / pol / "panel.npz") as z:
        n_pairs = len(z["candidate_spectrum_hash"])
    for m in ("cosine_greedy", "modified_cosine",
              "weighted_spectral_entropy"):
        arr = np.load(SMOKE / "staging" / f"scores_{pol}_{m}.npy")
        assert arr.shape == (n_pairs,), (pol, m, arr.shape)
        assert np.all(np.isfinite(arr))
print("SMOKE PASS: scorer resolves hashes, aligns pairs, scores finitely")
