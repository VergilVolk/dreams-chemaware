"""Full de novo pipeline end-to-end test (embedding -> fingerprint -> SMILES).

Pipeline under test::

    library/query MS2 -> DreaMS 1024-d embedding   [already embedded]
        -> linear head -> 2048-bit ECFP4            [annotation.molforge_head, trained here]
        -> MolForge -> SMILES                       [annotation.molforge_decoder, pretrained]

This script (a) trains the emb->fp head on the reference library (each spectrum
has a known structure, so (embedding, fingerprint) supervision is free), and
(b) runs the *full* pipeline on a held-out subset of library spectra and scores
the de novo SMILES against ground truth.

Three numbers are reported, in descending order of importance for honesty:
  1. ``de_novo_smiles_tanimoto``   -- full pipeline: embedding -> head -> fp -> SMILES.
  2. ``head_fp_tanimoto``          -- head alone: embedding -> predicted fp vs true fp
                                      (isolates head quality from decoder quality).
  3. ``decoder_reconstruct_tanimoto`` -- ceiling: true fp -> MolForge -> SMILES.
                                      The de novo number can never beat this.

The de novo claim is only meaningful for *out-of-library* chemistry; here we hold
out by molecule (InChIKey connectivity), so it is an honest in-distribution
estimate, not a dark-matter guarantee.

Usage (CPU):
    python tasks/molforge_full_pipeline.py \
        --library-dir data/models/mona_neg_dreams_emb \
        --checkpoint third_party/MolForge/saved_models/ECFP4_selfies_checkpoint.pth \
        --out-dir data/models/molforge_head \
        --decode-n 120
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from annotation.molforge_decoder import MolForgeDecoder  # noqa: E402
from annotation.molforge_head import (  # noqa: E402
    bits_from_probs,
    ecfp4_bit_vector,
    load_head,
    predict_fp,
    save_head,
    train_head,
)
from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import rdMolDescriptors  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DEFAULT_LIB = REPO / "data" / "models" / "mona_neg_dreams_emb"
DEFAULT_CKPT = REPO / "third_party" / "MolForge" / "saved_models" / "ECFP4_selfies_checkpoint.pth"
DEFAULT_OUT = REPO / "data" / "models" / "molforge_head"


def _tanimoto(sa: str, sb: str) -> float:
    ma, mb = Chem.MolFromSmiles(sa), Chem.MolFromSmiles(sb)
    if ma is None or mb is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(
        rdMolDescriptors.GetMorganFingerprintAsBitVect(ma, 2, 2048),
        rdMolDescriptors.GetMorganFingerprintAsBitVect(mb, 2, 2048),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library-dir", type=Path, default=DEFAULT_LIB)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--decode-n", type=int, default=120,
                    help="held-out spectra to run through the *full* decoder (~1.8s each)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=None,
                    help="MLP hidden width (option 1); omit for the linear probe")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"[pipeline] MolForge checkpoint missing: {args.checkpoint}", file=sys.stderr)
        return 1

    # -- 1) library -> (embedding, fingerprint, inchikey) supervision ---------
    emb = np.load(args.library_dir / "embeddings.npy")
    manifest = pd.read_csv(args.library_dir / "manifest.csv")
    assert len(emb) == len(manifest), "embedding/manifest length mismatch"
    print(f"library: {emb.shape[0]} spectra x {emb.shape[1]} dims", flush=True)

    valid_idx, fps, ikeys, smis = [], [], [], []
    for i, s in enumerate(manifest["smiles"].tolist()):
        if not isinstance(s, str) or not s:
            continue
        try:
            fp = ecfp4_bit_vector(s)
        except ValueError:
            continue
        valid_idx.append(i)
        fps.append(fp)
        ikeys.append(str(manifest["inchikey"].iloc[i]))
        smis.append(s)
    fps = np.asarray(fps, dtype=np.float32)
    emb_valid = emb[np.asarray(valid_idx, dtype=np.int64)]
    print(f"valid structures: {len(fps)} / {len(manifest)}", flush=True)

    # -- 2) train the head ----------------------------------------------------
    tag = f"mlp{args.hidden}" if args.hidden else "linear"
    print(f"training emb->fp head ({tag}) ...", flush=True)
    res = train_head(emb_valid, fps, ikeys, seed=args.seed, epochs=args.epochs,
                     cpu_threads=args.cpu_threads, hidden=args.hidden)
    head = res["model"]
    save_head(args.out_dir / f"emb_fp_head_{tag}.pt", head,
              res["embedding_mean"], res["embedding_std"], res["topk"])
    print(f"topk={res['topk']} (tuned on val)  head fp-Tanimoto "
          f"val={res['val_fp_tanimoto']:.3f} test={res['test_fp_tanimoto']:.3f}", flush=True)

    # -- 3) full de novo decode on a held-out subset --------------------------
    dec = MolForgeDecoder(model_type="selfies")
    dec.load_checkpoint(args.checkpoint)

    rng = np.random.default_rng(args.seed + 7)
    test_idx = res["test_idx"]
    decode_idx = np.sort(rng.choice(test_idx, size=min(args.decode_n, len(test_idx)), replace=False))

    rows = []
    for i in decode_idx:
        truth = smis[i]
        # head -> predicted fp (top-k bits)
        pred_bits = predict_fp(head, emb_valid[i:i + 1], res["embedding_mean"],
                               res["embedding_std"], res["topk"])[0]
        cand = dec.decode_bits_to_smiles(pred_bits)
        # ceiling: decode the TRUE fingerprint through the same decoder
        truth_bits = sorted(int(b) for b in np.flatnonzero(fps[i]))
        recon = dec.decode_bits_to_smiles(truth_bits)
        rows.append({
            "ik14": ikeys[i],
            "true_smiles": truth,
            "de_novo_smiles": cand,
            "decoder_reconstruct_smiles": recon,
            "de_novo_tanimoto": round(_tanimoto(cand, truth), 3),
            "reconstruct_tanimoto": round(_tanimoto(recon, truth), 3),
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_dir / f"full_pipeline_test_{tag}.csv", index=False)

    de_novo = out["de_novo_tanimoto"].mean()
    recon = out["reconstruct_tanimoto"].mean()
    exact = int((out["de_novo_smiles"] == out["true_smiles"]).sum())
    report = {
        "status": "molforge_full_pipeline_complete",
        "head_type": tag,
        "head_hidden": args.hidden,
        "library_spectra": int(len(manifest)),
        "valid_structures": int(len(fps)),
        "n_train": res["n_train"], "n_val": res["n_val"], "n_test": res["n_test"],
        "head_topk": res["topk"],
        "head_val_fp_tanimoto": float(res["val_fp_tanimoto"]),
        "head_test_fp_tanimoto": float(res["test_fp_tanimoto"]),
        "head_test_macro_auprc": float(res["test_macro_auprc"]),
        "decode_n": int(len(out)),
        "de_novo_smiles_tanimoto": float(de_novo),
        "de_novo_exact": exact,
        "decoder_reconstruct_tanimoto": float(recon),
        "de_novo_over_ceiling": float(de_novo / recon) if recon else None,
        "claim_limit": (
            "Linear emb->fp readout, in-distribution (molecule-disjoint) estimate. "
            "de novo <= decoder reconstruction ceiling; out-of-library dark matter "
            "expected substantially worse."
        ),
    }
    (args.out_dir / f"report_{tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    print("\n=== sample de novo decodes ===", flush=True)
    for _, r in out.head(8).iterrows():
        print(f"[{r['ik14']}] T={r['de_novo_tanimoto']:.2f}", flush=True)
        print(f"    true   {r['true_smiles']}", flush=True)
        print(f"    de novo {r['de_novo_smiles']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
