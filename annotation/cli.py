"""M9 -- Command-line entrypoint for the annotation platform.

End-to-end example (CPU):

    python -m annotation.cli embed --kind query \
        --hdf5 data/msv100574/Metabolomics/neg/PF_1.hdf5 \
        --hdf5 data/msv100574/Metabolomics/neg/HF_1.hdf5 \
        --out data/msv100574/embeddings/met_neg

    python -m annotation.cli annotate \
        --query-dir data/msv100574/embeddings/met_neg \
        --library-dir data/models/mona_neg_dreams_emb \
        --out data/msv100574/annotation/met_neg \
        --fdr --library-mgf data/models/mona_neg_full.mgf

Every stage can be run separately; each writes its own artifacts so the pipeline
is auditable and ablatable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .params import Params, DEFAULT


def parse_mgf(path: Path) -> list[dict]:
    """Parse a standard MGF (SMILES / INCHIKEY / PEPMASS / peaks) into records."""
    records: list[dict] = []
    cur: dict | None = None
    peaks: list[tuple[float, float]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if line == "BEGIN IONS":
                cur = {}
                peaks = []
            elif line == "END IONS":
                if cur is not None and peaks and cur.get("precursor_mz"):
                    arr = np.asarray(peaks, dtype=np.float32)
                    cur["peaks"] = arr.T
                    records.append(cur)
                cur = None
            elif cur is not None and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip()
                if k == "PEPMASS":
                    cur["precursor_mz"] = float(v.split()[0])
                elif k == "SMILES":
                    cur["smiles"] = v
                elif k == "INCHIKEY":
                    cur["inchikey"] = v
                elif k == "NAME":
                    cur["name"] = v
            elif cur is not None:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
    return records


def cmd_embed(args) -> int:
    from . import embed as E

    device = "cpu"
    model, weight, bias = E.load_embedder(device)
    if args.kind == "query":
        embs, manifests = [], []
        for hdf5 in args.hdf5:
            e, m = E.embed_hdf5(Path(hdf5), model, weight, bias, device)
            embs.append(e)
            manifests.append(m)
        emb = np.concatenate(embs)
        manifest = pd.concat(manifests, ignore_index=True)
    else:
        records = []
        for mgf in args.mgf:
            records.extend(parse_mgf(Path(mgf)))
        emb = E.embed_records(records, model, weight, bias, device)
        manifest = pd.DataFrame({
            "smiles": [r.get("smiles", "") for r in records],
            "inchikey": [r.get("inchikey", "") for r in records],
            "name": [r.get("name", "") for r in records],
            "precursor_mz": [r.get("precursor_mz", float("nan")) for r in records],
        })
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", emb.astype(np.float32))
    manifest.to_csv(out / "manifest.csv", index=False)
    print(f"wrote {emb.shape} embeddings + manifest to {out}")
    return 0


def cmd_annotate(args) -> int:
    from . import retrieve, confidence, fdr, calibrate, ablation
    from .params import DEFAULT, Params
    import dataclasses

    params = DEFAULT
    query_dir, library_dir, out = Path(args.query_dir), Path(args.library_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    hits, report = retrieve.retrieve(query_dir, library_dir, params, group_by=args.group_col)
    print(json.dumps(report, indent=2))

    if args.fdr:
        if not args.library_mgf:
            print("[warn] --fdr requires --library-mgf to build decoys; skipping FDR", file=sys.stderr)
        else:
            from . import embed as E
            device = "cpu"
            model, weight, bias = E.load_embedder(device)
            records = parse_mgf(Path(args.library_mgf))
            decoys = fdr.make_shuffle_decoys(records, seed=0)
            decoy_emb = E.embed_records(decoys, model, weight, bias, device, batch_size=64)
            query, _ = retrieve.load_embedding_set(query_dir)
            library, _ = retrieve.load_embedding_set(library_dir)
            target_scores = fdr.top1_scores(query, library)
            decoy_scores = fdr.top1_scores(query, decoy_emb)
            hits = fdr.annotate_fdr(hits, target_scores, decoy_scores, params)
            n_fdr_pass = int(hits[hits["rank"] == 1]["fdr_pass"].sum())
            print(f"[fdr] {n_fdr_pass} top-1 hits pass q-value <= {params.qvalue_threshold}")

    rules_ev = None
    if args.rules:
        from . import rule_evidence
        cache = query_dir / "rule_hits.npy"
        if not cache.exists():
            print(f"[rules] no rule-hit cache at {cache}; run tasks/explore_rule_evidence.py first",
                  file=sys.stderr)
        else:
            V = np.load(cache)
            rules_ev = rule_evidence.diagnostic_evidence(V)
            print(f"[rules] diagnostic evidence in {int(rules_ev.sum())}/{len(rules_ev)} spectra")
    hits = confidence.assign_schymanski(hits, params, rules_evidence=rules_ev)
    if rules_ev is not None:
        hits["diagnostic_rule_evidence"] = np.asarray(rules_ev)[hits["query_idx"].to_numpy()]

    if args.calibrate:
        query, _ = retrieve.load_embedding_set(query_dir)
        library, l_manifest = retrieve.load_embedding_set(library_dir)
        scores, labels = calibrate.library_self_scores(library, l_manifest)
        cal = calibrate.fit_calibrator(scores, labels, params.calibration_method)
        hits = calibrate.apply_calibrator(hits, cal, params)
        print(f"[calibrate] fitted {params.calibration_method} on {len(scores)} labelled examples")

    retrieve.save(hits, report, out)
    print(ablation.report(ablation.run_ablation(hits, n_query=report["n_query_spectra"])))
    print(f"\nannotations -> {out / 'annotations.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="annotation", description="DreaMS LC-MS/MS annotation platform")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="embed query hdf5 or library mgf")
    e.add_argument("--kind", choices=["query", "library"], required=True)
    e.add_argument("--hdf5", nargs="*", default=[])
    e.add_argument("--mgf", nargs="*", default=[])
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_embed)

    a = sub.add_parser("annotate", help="retrieve + confidence (+ FDR / calibration)")
    a.add_argument("--query-dir", required=True)
    a.add_argument("--library-dir", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--group-col", default=None, help="query manifest column holding group labels")
    a.add_argument("--fdr", action="store_true")
    a.add_argument("--library-mgf", default=None, help="original library MGF for decoy generation")
    a.add_argument("--calibrate", action="store_true")
    a.add_argument("--rules", action="store_true",
                   help="inject diagnostic-rule evidence (reads rule_hits.npy cache)")
    a.set_defaults(func=cmd_annotate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
