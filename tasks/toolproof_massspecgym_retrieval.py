"""
DreaMS retrieval benchmark on MassSpecGym (read-only, CPU).

IMPORTANT: this measures TWO different, honestly-separated tasks. Do not conflate them.

  A) Library matching  ("molecule IS in the library")
       query spectrum -> retrieve -> is the correct InChIKey14 in top-k?
       DreaMS's home turf. Compare vs modified-cosine / DreaMS paper's own numbers.

  B) Reference-free   ("molecule NOT in the library" = the 950 novel test molecules)
       query spectrum -> retrieve -> how STRUCTURALLY SIMILAR (Morgan Tanimoto) is
       the top-k retrieved SMILES vs the TRUE SMILES?
       This is the "dark matter" discovery case. It is NOT de novo generation
       (MetGenX 21.7% is generation; we only retrieve the nearest library structure).

Why the old 92% top-1 was nonsense: it sampled only query spectra whose molecule was
already in the (tiny) library, so it measured trivial same-molecule self-retrieval.

Usage (conda, CPU):
    # quick corrected sanity check (stratified query: ~200 retrievable + ~200 novel)
    python tasks/toolproof_massspecgym_retrieval.py --n_lib 5000 --n_retrievable 200 --n_novel 200

    # full benchmark (all val=22592 library vs all test=22593 query; ~hours on CPU)
    python tasks/toolproof_massspecgym_retrieval.py --full
"""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

# Make the repo-root `dreams` package importable when run as a script (matches other tasks/*.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from dreams.api import dreams_embeddings
from dreams.utils.data import MSData


def iter_records(path):
    """Yield one MS/MS record (BEGIN IONS ... END IONS) at a time."""
    cur = []
    with open(path) as f:
        for ln in f:
            cur.append(ln)
            if ln.strip() == "END IONS":
                yield "".join(cur)
                cur = []


def header(rec, key):
    m = re.search(rf"^{key}=(\S+)\s*$", rec, re.M)
    return m.group(1) if m else None


def load_records(records):
    """Load a list of MGF records as an in-memory MSData via a temp file."""
    if not records:
        raise ValueError("no records")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".mgf", delete=False, encoding="utf-8")
    tmp.write("".join(records))
    tmp.close()
    try:
        return MSData.from_mgf(tmp.name)
    finally:
        os.unlink(tmp.name)


def compute_topk(embs_q, embs_lib, k=10, chunk=512):
    """Chunked cosine top-k over the library; never materializes the full sim matrix."""
    nq = embs_q.shape[0]
    topk_idx = np.zeros((nq, k), dtype=np.int64)
    topk_sim = np.full((nq, k), -np.inf)
    for i in range(0, nq, chunk):
        sims = cosine_similarity(embs_q[i:i + chunk], embs_lib)
        idx = np.argsort(-sims, axis=1)[:, :k]
        val = np.take_along_axis(sims, idx, axis=1)
        merged = np.concatenate([topk_idx[i:i + chunk], idx], axis=1)
        merged_sim = np.concatenate([topk_sim[i:i + chunk], val], axis=1)
        keep = np.argsort(-merged_sim, axis=1)[:, :k]
        topk_idx[i:i + chunk] = np.take_along_axis(merged, keep, axis=1)
        topk_sim[i:i + chunk] = np.take_along_axis(merged_sim, keep, axis=1)
    return topk_idx


def morgan_tanimoto(smiles_a, smiles_b):
    """Morgan(2, 2048) Tanimoto; 0.0 on any parse failure."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        ma = Chem.MolFromSmiles(smiles_a)
        mb = Chem.MolFromSmiles(smiles_b)
        if ma is None or mb is None:
            return 0.0
        fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
        fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fa, fb)
    except Exception:
        return 0.0


def report(topk_idx, q_keys, q_smiles, lib_keys, lib_smiles, label):
    """Two separated, honestly-labeled metrics: library matching vs reference-free."""
    n = len(q_keys)
    lib_set = set(lib_keys)
    retrievable = [i for i in range(n) if q_keys[i] in lib_set]
    novel = [i for i in range(n) if q_keys[i] not in lib_set]

    print(f"\n=== {label} ===")
    print(f"queries: {n}  |  molecule-in-library: {len(retrievable)}  |  novel (not in library): {len(novel)}")

    # A) library matching
    if retrievable:
        print("\n[A] Library matching (molecule IS in library) — top-k exact InChIKey14:")
        for k in (1, 5, 10):
            hits = sum(q_keys[i] in set(lib_keys[topk_idx[i, :k]]) for i in retrievable)
            print(f"    top-{k:>2}: {hits}/{len(retrievable)} = {100*hits/len(retrievable):.2f}%")
    else:
        print("\n[A] (no retrievable queries in this sample)")

    # B) reference-free / novel
    if novel:
        print("\n[B] Reference-free (novel, NOT in library) — structural similarity of top-k hits:")
        for k in (1, 5, 10):
            tans = []
            for i in novel:
                hits = topk_idx[i, :k]
                tans.append(max(morgan_tanimoto(q_smiles[i], lib_smiles[j]) for j in hits))
            tans = np.array(tans)
            print(f"    top-{k:>2}: mean max-Tanimoto = {tans.mean():.3f}  |  >=0.6 (useful hit): "
                  f"{(tans >= 0.6).sum()}/{len(novel)} = {100*(tans >= 0.6).mean():.1f}%")
        print("    (note: retrieval != de novo generation; MetGenX 21.7% DB-free is generation, not this)")
    else:
        print("\n[B] (no novel queries in this sample)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="data/massspecgym/val.mgf")
    ap.add_argument("--query", default="data/massspecgym/test.mgf")
    ap.add_argument("--n_lib", type=int, default=5000)
    ap.add_argument("--n_retrievable", type=int, default=200)
    ap.add_argument("--n_novel", type=int, default=200)
    ap.add_argument("--full", action="store_true", help="use full val vs full test")
    args = ap.parse_args()

    if args.full:
        lib_recs = list(iter_records(args.lib))
        q_recs = list(iter_records(args.query))
    else:
        print("Reading library headers to build InChIKey set...", flush=True)
        lib_all = list(iter_records(args.lib))
        lib_keys_all = {header(r, "INCHIKEY") for r in lib_all}
        # sample library records
        rng = np.random.default_rng(0)
        lib_idx = rng.choice(len(lib_all), size=min(args.n_lib, len(lib_all)), replace=False)
        lib_recs = [lib_all[i] for i in lib_idx]

        print("Classifying query records (retrievable vs novel)...", flush=True)
        q_all = list(iter_records(args.query))
        q_retr = [r for r in q_all if header(r, "INCHIKEY") in lib_keys_all]
        q_novel = [r for r in q_all if header(r, "INCHIKEY") not in lib_keys_all]
        n_r = min(args.n_retrievable, len(q_retr))
        n_n = min(args.n_novel, len(q_novel))
        q_recs = q_retr[:n_r] + q_novel[:n_n]
        print(f"    library: {len(lib_recs)} records | query: {n_r} retrievable + {n_n} novel")

    print("Loading library MSData...", flush=True)
    lib = load_records(lib_recs)
    print("Loading query MSData...", flush=True)
    q = load_records(q_recs)

    print("Computing library embeddings...", flush=True)
    embs_lib = dreams_embeddings(lib)
    print("Computing query embeddings...", flush=True)
    embs_q = dreams_embeddings(q)

    print("Computing chunked cosine top-k...", flush=True)
    topk_idx = compute_topk(embs_q, embs_lib)

    lib_keys = np.array(lib.get_values("INCHIKEY"))
    lib_smiles = np.array(lib.get_values("smiles"))
    q_keys = np.array(q.get_values("INCHIKEY"))
    q_smiles = np.array(q.get_values("smiles"))
    report(topk_idx, q_keys, q_smiles, lib_keys, lib_smiles, label="DreaMS retrieval vs MassSpecGym")


if __name__ == "__main__":
    main()
