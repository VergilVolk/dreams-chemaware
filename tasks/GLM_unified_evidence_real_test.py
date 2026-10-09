"""Development-only smoke: unified evidence model on 15-method GNPS data.

Feeds the existing 15-method bundle through the evidence pipeline
(rank-logit standardization → reliability-weighted fusion → listwise
evaluation) on the identity-disjoint panel. Reports R@1 vs WSE and
per-module learned weights. GNPS Gold/Silver has been consumed during project
development; this script must never be cited as an independent external test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(ROOT))

from grand_unified_evidence_core import (  # noqa: E402
    AllModuleEvidenceModel, listwise_loss, rank_logit_evidence,
    strict_ranks,
)
from evaluate_gnps_gold_silver_10ppm_embeddings import graph_from_panel  # noqa: E402

PANEL_DIR = ROOT / "data/validation/GLM_gnps_identity_panel_reconstruction"
BUNDLE = ROOT / ("data/validation/GLM_gnps_article_benchmark_s2v26/"
                 "run15/bundle/method_scores.npz")


def main() -> None:
    with np.load(BUNDLE) as z:
        names = [str(v) for v in z["method_names"]]
        scores = np.asarray(z["scores_identity_disjoint"], dtype=np.float32)
    g = graph_from_panel(PANEL_DIR / "panel_identity_disjoint.npz")

    # molecule-level scores (frozen semantics: max per molecule block)
    mol = np.maximum.reduceat(scores, g.molecule_ptr[:-1], axis=1)
    M, n_q = len(names), g.n_queries

    # flatten candidates per query (variable-length → padded)
    max_c = int(np.diff(g.query_ptr).max())
    flat_scores = np.zeros((M, n_q * max_c), dtype=np.float32)
    labels_flat = np.zeros(n_q * max_c, dtype=np.int8)
    query_ptr = np.zeros(n_q + 1, dtype=np.int64)
    avail = np.ones((M, len(g.molecule_label)), dtype=np.float64)
    for q in range(n_q):
        lo, hi = int(g.query_ptr[q]), int(g.query_ptr[q + 1])
        k = hi - lo
        flat_scores[:, q * max_c:q * max_c + k] = mol[:, lo:hi]
        labels_flat[q * max_c] = 1  # positive is block-first
        query_ptr[q + 1] = query_ptr[q] + k
    # compact: rebuild with actual counts
    flat = np.zeros((M, query_ptr[-1]), dtype=np.float32)
    lab = np.zeros(query_ptr[-1], dtype=np.int8)
    pos = 0
    for q in range(n_q):
        lo, hi = int(g.query_ptr[q]), int(g.query_ptr[q + 1])
        k = hi - lo
        flat[:, pos:pos + k] = mol[:, lo:hi]
        lab[pos] = 1
        pos += k

    evidence, summaries = rank_logit_evidence(flat, query_ptr, avail)
    print(f"modules={M}, queries={n_q:,}, candidates={query_ptr[-1]:,}")

    # train on 70% / evaluate on 30% (formula-cluster split for honesty)
    formulas = g.query_formula.astype(str)
    uniq = np.array(sorted(set(formulas)))
    rng = np.random.default_rng(20261007)
    train_f = set(uniq[rng.permutation(len(uniq))[:int(len(uniq) * 0.7)]])
    train_mask = np.array([f in train_f for f in formulas])
    test_mask = ~train_mask

    # build train subset
    tr_rows = []
    for q in range(n_q):
        if train_mask[q]:
            tr_rows.extend(range(query_ptr[q], query_ptr[q + 1]))
    tr_ptr = [0]
    for q in range(n_q):
        if train_mask[q]:
            tr_ptr.append(tr_ptr[-1] + (query_ptr[q + 1] - query_ptr[q]))
    tr_ptr = np.asarray(tr_ptr)
    tr_evidence = evidence[:, tr_rows]
    tr_labels = lab[tr_rows]
    tr_avail = avail[:, tr_rows]
    tr_summaries = summaries[train_mask]

    model = AllModuleEvidenceModel(tuple(names))
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

    ev_t = torch.tensor(tr_evidence, dtype=torch.float32)
    ptr_t = torch.tensor(tr_ptr, dtype=torch.long)
    sum_t = torch.tensor(tr_summaries, dtype=torch.float32)
    av_t = torch.tensor(tr_avail, dtype=torch.float32)
    lab_t = torch.tensor(tr_labels, dtype=torch.long)

    for epoch in range(30):
        out = model(ev_t, ptr_t, sum_t, av_t)
        loss = listwise_loss(out.logits, ptr_t, lab_t,
                             out.module_weights, av_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            ranks = strict_ranks(out.logits.detach().numpy(), tr_ptr, tr_labels)
            print(f"  epoch {epoch}: loss={float(loss):.4f} "
                  f"R@1={np.mean(ranks == 1) * 100:.2f}%")

    # evaluate on test
    te_rows = []
    for q in range(n_q):
        if test_mask[q]:
            te_rows.extend(range(query_ptr[q], query_ptr[q + 1]))
    te_ptr = [0]
    for q in range(n_q):
        if test_mask[q]:
            te_ptr.append(te_ptr[-1] + (query_ptr[q + 1] - query_ptr[q]))
    te_ptr = np.asarray(te_ptr)
    te_evidence = evidence[:, te_rows]
    te_labels = lab[te_rows]
    te_avail = avail[:, te_rows]
    te_summaries = summaries[test_mask]

    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(te_evidence), torch.tensor(te_ptr),
                    torch.tensor(te_summaries), torch.tensor(te_avail))
        te_ranks = strict_ranks(out.logits.numpy(), te_ptr, te_labels)
        unified_r1 = float(np.mean(te_ranks == 1) * 100)

    # WSE baseline on same test queries
    wse_i = names.index("weighted_spectral_entropy")
    wse_r1 = 0.0
    n_test = 0
    for q in range(n_q):
        if not test_mask[q]:
            continue
        lo, hi = int(g.query_ptr[q]), int(g.query_ptr[q + 1])
        block = mol[wse_i, lo:hi]
        wse_r1 += int(np.argmax(block) == 0)
        n_test += 1
    wse_r1 = wse_r1 / max(1, n_test) * 100

    # learned module weights (mean over test queries)
    w = out.module_weights.numpy().mean(axis=1)
    print(f"\n=== DEVELOPMENT HOLDOUT ONLY (n={n_test:,}; NOT EXTERNAL) ===")
    print(f"Unified: {unified_r1:.2f}%  |  WSE: {wse_r1:.2f}%  |  "
          f"delta: {unified_r1 - wse_r1:+.2f}pp")
    print("\nModule weights (mean):")
    for i, name in enumerate(names):
        print(f"  {name:30s} {w[i]:.4f}")
    print(f"\nloss final: {float(loss):.4f}")


if __name__ == "__main__":
    main()
