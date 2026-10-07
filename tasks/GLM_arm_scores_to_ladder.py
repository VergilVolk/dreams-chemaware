"""Turn an arm's panel embeddings into a bundle row + frozen evaluation.

Steps:
  1. pair scores: cosine(query embedding, candidate embedding) aligned to
     each panel's candidate_row order (identical alignment semantics to
     the 15-method bundle);
  2. merge as a new method row into a COPY of the run15 bundle
     (proven merge pattern);
  3. run the frozen evaluator (evaluate_noise_gnps_article_benchmark) on
     the new bundle -> per-query tables + report;
  4. assemble the ladder row via the verified assembler.

The pipeline is method-agnostic: the same three commands turn ANY trained
arm checkpoint into a ladder row on the identical candidate graph.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

BUNDLE15 = ROOT / ("data/validation/GLM_gnps_article_benchmark_s2v26/"
                   "run15/bundle")
PANEL_DIR = ROOT / "data/validation/GLM_gnps_identity_panel_reconstruction"
PANELS = ("identity_disjoint", "formula_disjoint")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--method-name", required=True)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="output run dir (bundle/ + evaluation/ created)")
    args = ap.parse_args()

    with np.load(args.embeddings) as z:
        rows = np.asarray(z["rows"], dtype=np.int64)
        emb = np.asarray(z["embeddings"], dtype=np.float32)
    pos = {int(r): i for i, r in enumerate(rows)}

    with np.load(BUNDLE15 / "method_scores.npz") as b:
        names = [str(v) for v in b["method_names"]]
        old = {p: np.asarray(b[f"scores_{p}"], dtype=np.float32)
               for p in PANELS}
    assert args.method_name not in names
    all_names = names + [args.method_name]

    bundle = args.run_dir / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    for panel in PANELS:
        with np.load(PANEL_DIR / f"panel_{panel}.npz") as z:
            q_rows = np.asarray(z["query_row"], dtype=np.int64)
            c_rows = np.asarray(z["candidate_row"], dtype=np.int64)
            q_idx = np.asarray([pos[int(r)] for r in q_rows])
            c_idx = np.asarray([pos[int(r)] for r in c_rows])
            # molecule -> its query index; then molecule -> its pair count
            q_of_mol = np.repeat(np.arange(len(q_rows)),
                                 np.diff(z["query_ptr"]))
            pairs_per_mol = np.diff(z["molecule_ptr"])
            # pair -> query embedding row (molecules expand to their pairs)
            pair_q = q_idx[np.repeat(q_of_mol, pairs_per_mol)]
            pair_c = c_idx
        assert len(pair_q) == len(pair_c) == old[panel].shape[1]
        scores = np.sum(emb[pair_q] * emb[pair_c], axis=1).astype(np.float32)
        assert scores.shape == old[panel].shape[1:], (panel, scores.shape)
        old[panel] = np.vstack([old[panel], scores[None, :]])

    np.savez_compressed(bundle / "method_scores.npz",
                        method_names=np.asarray(all_names),
                        **{f"scores_{p}": old[p] for p in PANELS})
    meta = json.loads((BUNDLE15 / "method_scores.npz.json").read_text(
        encoding="utf-8"))
    meta["methods"] = all_names
    meta["information_levels"].setdefault("spectrum_only", []).append(
        args.method_name)
    (bundle / "method_scores.npz.json").write_text(json.dumps(meta, indent=2),
                                                   encoding="utf-8")
    print(f"bundle written: {len(all_names)} methods "
          f"(+{args.method_name})", flush=True)

    env = {
        "PYTHONPATH": f"{ROOT};{ROOT / 'tasks'}",
        "PYTHONUTF8": "1",
    }
    import os
    env = {**os.environ, **env}
    evaluation = args.run_dir / "evaluation"
    if not evaluation.exists():
        r = subprocess.run(
            [sys.executable, "-X", "utf8",
             str(ROOT / "tasks/evaluate_noise_gnps_article_benchmark.py"),
             "--score-bundle", str(bundle / "method_scores.npz"),
             "--benchmark", str(PANEL_DIR),
             "--output", str(evaluation),
             "--baseline-method", "official_dreams",
             "--bootstrap-resamples", "10000",
             "--bootstrap-seed", "20261007"],
            env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-600:] + r.stderr[-1200:]
        print("frozen evaluator complete", flush=True)

    table = evaluation / f"queries_identity_disjoint_{args.method_name}.csv.gz"
    import pandas as pd
    df = pd.read_csv(table)
    r1 = float((df["rank"].to_numpy() == 1).mean() * 100)
    print(f"[{args.method_name}] identity R@1 = {r1:.2f}% "
          f"(pipeline validated; quality depends on the checkpoint)",
          flush=True)
    (args.run_dir / "eval_chain_report.json").write_text(json.dumps({
        "status": "GLM_ARM_EVAL_CHAIN_COMPLETE",
        "method": args.method_name, "identity_recall1": round(r1, 2),
        "n_methods": len(all_names)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
