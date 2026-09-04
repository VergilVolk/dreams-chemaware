"""M-PART3 -- the ``embedding -> ECFP4 fingerprint`` head (the missing de novo leg).

Full de novo pipeline::

    query MS2 -> DreaMS 1024-d embedding   [annotation.embed]
    -> ECFP4 fingerprint (this head)       [annotation.molforge_head]
    -> SMILES (MolForge)                   [annotation.molforge_decoder]

This is the ONLY untrained piece. We train a **linear probe** (1024 -> 2048,
multi-label BCE over the 2048 hashed-Morgan bits, r=2) on the reference library,
where every spectrum has a known structure, so ``(embedding, fingerprint)``
supervision is free.

Honest scope: a linear readout only recovers structure *linearly* encoded in the
DreaMS embedding. The embedding is optimised for spectrum-spectrum similarity
(Bushuiev et al., Nat Biotechnol 2025, DOI 10.1038/s41587-025-02663-3), not for
linear fingerprint reconstruction, so out-of-library de novo chemistry is
expected to be poor. The full-pipeline test (``tasks/molforge_full_pipeline.py``)
quantifies exactly how poor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    _HAVE_RDKIT = True
except Exception:  # pragma: no cover - rdkit is a hard dep of the annotation stack
    _HAVE_RDKIT = False

N_BITS = 2048
DIM_EMB = 1024


# --------------------------------------------------------------------------- #
# fingerprint helpers
# --------------------------------------------------------------------------- #
def ecfp4_bit_vector(smiles: str, nbits: int = N_BITS) -> np.ndarray:
    """Dense 0/1 ECFP4 bit vector (hashed Morgan, r=2), matching MolForge's ECFP4.

    Bit positions are identical to ``GetHashedMorganFingerprint(..., nBits).``
    ``GetNonzeroElements().keys()`` (verified), so the head's output feeds
    ``molforge_decoder.decode_bits`` directly.
    """
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nbits)
    return np.frombuffer(bv.ToBitString().encode("ascii"), dtype=np.uint8) - ord("0")


def bits_from_probs(probs: np.ndarray, topk: int) -> list[list[int]]:
    """Threshold a [N, 2048] probability matrix to per-row ON-bit indices (top-k)."""
    idx = np.argsort(-probs, axis=1)[:, :topk]
    return [sorted(int(b) for b in row) for row in idx]


# --------------------------------------------------------------------------- #
# the head
# --------------------------------------------------------------------------- #
class EcFpHead(nn.Module):
    """embedding -> 2048-bit ECFP4 logits.

    ``hidden`` is None for a linear probe, or the width of a single GELU hidden
    layer for the MLP variant (option 1: test whether nonlinearity helps read the
    fingerprint out of the DreaMS embedding).
    """

    def __init__(self, dim_emb: int = DIM_EMB, n_bits: int = N_BITS,
                 hidden: int | None = None):
        super().__init__()
        self.hidden = hidden
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(dim_emb, hidden),
                nn.GELU(),
                nn.Linear(hidden, n_bits),
            )
        else:
            self.net = nn.Linear(dim_emb, n_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def split_molecules(ik14: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Molecule-disjoint 70/15/15 split on InChIKey connectivity layer."""
    molecules = np.unique(ik14)
    rng = np.random.default_rng(seed)
    molecules = molecules[rng.permutation(len(molecules))]
    te = int(round(0.70 * len(molecules)))
    ve = int(round(0.85 * len(molecules)))
    return molecules[:te], molecules[te:ve], molecules[ve:]


def _members(ik14: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    s = set(allowed.tolist())
    return np.asarray(sorted(i for i, m in enumerate(ik14.tolist()) if m in s), dtype=np.int64)


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    pos = int(y_true.sum())
    if pos == 0 or pos == y_true.size:
        return float("nan")
    ranked = y_true[np.argsort(-score, kind="stable")].astype(np.float64)
    prec = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float((prec * ranked).sum() / pos)


def fp_tanimoto(pred_bits: list[list[int]], true_vec: np.ndarray) -> float:
    """Mean Jaccard/Tanimoto between top-k predicted ON-bits and the true bit vector."""
    t = 0.0
    n = 0
    for bits, tv in zip(pred_bits, true_vec):
        s = set(bits)
        inter = int(sum(1 for b in s if tv[b]))
        union = len(s) + int(tv.sum()) - inter
        t += inter / union if union else 0.0
        n += 1
    return t / max(n, 1)


def train_head(
    embeddings: np.ndarray,
    fingerprints: np.ndarray,
    inchikeys: list[str],
    seed: int = 20260820,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 2e-3,
    topk_grid: tuple[int, ...] = (30, 40, 50, 60),
    cpu_threads: int = 8,
    hidden: int | None = None,
) -> dict:
    """Train a linear emb->fp head and return (model, stats, metrics) as a dict.

    ``embeddings`` is [N, 1024] L2-normalised; ``fingerprints`` is [N, 2048] 0/1;
    ``inchikeys`` aligns row-for-row (molecule-disjoint split on the 14-char
    connectivity layer prevents leakage).
    """
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, cpu_threads))
    ik14 = np.asarray([str(k)[:14] for k in inchikeys], dtype=object)
    x_all = np.asarray(embeddings, dtype=np.float32)
    y_all = np.asarray(fingerprints, dtype=np.float32)

    tr_m, va_m, te_m = split_molecules(ik14, seed)
    tr_idx, va_idx, te_idx = _members(ik14, tr_m), _members(ik14, va_m), _members(ik14, te_m)

    mean = x_all[tr_idx].mean(axis=0, keepdims=True)
    std = x_all[tr_idx].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    def prep(idx):
        return ((x_all[idx] - mean) / std).astype(np.float32), y_all[idx]

    x_tr, y_tr = prep(tr_idx)
    x_va, y_va = prep(va_idx)
    x_te, y_te = prep(te_idx)

    pos = y_tr.sum(axis=0)
    neg = len(y_tr) - pos
    pos_weight = np.clip(neg / np.maximum(pos, 1.0), 1.0, 20.0).astype(np.float32)

    model = EcFpHead(hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    def macro_ap(x, y):
        model.eval()
        with torch.inference_mode():
            s = torch.sigmoid(model(torch.from_numpy(x))).numpy()
        return float(np.nanmean([average_precision(y[:, i], s[:, i]) for i in range(y.shape[1])]))

    best_ap, best_state, best_ep = -np.inf, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        va = macro_ap(x_va, y_va)
        print(f"epoch={ep:02d} train={np.mean(losses):.5f} val_macro_auprc={va:.5f}", flush=True)
        if va > best_ap:
            best_ap, best_ep = va, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        te_probs = torch.sigmoid(model(torch.from_numpy(x_te))).numpy()

    # tune top-k on validation by fingerprint Tanimoto
    with torch.inference_mode():
        va_probs = torch.sigmoid(model(torch.from_numpy(x_va))).numpy()
    best_k, best_k_t = None, -1.0
    for k in topk_grid:
        t = fp_tanimoto(bits_from_probs(va_probs, k), y_va)
        if t > best_k_t:
            best_k, best_k_t = k, t
    te_fp_t = fp_tanimoto(bits_from_probs(te_probs, best_k), y_te)

    return {
        "model": model,
        "embedding_mean": mean.squeeze(0),
        "embedding_std": std.squeeze(0),
        "topk": best_k,
        "val_fp_tanimoto": best_k_t,
        "test_fp_tanimoto": te_fp_t,
        "test_macro_auprc": macro_ap(x_te, y_te),
        "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx),
        "test_idx": te_idx, "test_probs": te_probs,
        "best_epoch": best_ep,
    }


# --------------------------------------------------------------------------- #
# predict / save / load
# --------------------------------------------------------------------------- #
def predict_fp(model: EcFpHead, embeddings: np.ndarray, mean: np.ndarray, std: np.ndarray,
               topk: int) -> list[list[int]]:
    """embedding [N,1024] -> top-k ON-bit indices for each row."""
    x = (np.asarray(embeddings, dtype=np.float32) - mean[None]) / std[None]
    model.eval()
    with torch.inference_mode():
        probs = torch.sigmoid(model(torch.from_numpy(x))).numpy()
    return bits_from_probs(probs, topk)


def save_head(path: Path, head: EcFpHead, mean: np.ndarray, std: np.ndarray, topk: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "molforge_emb_fp_head_v1",
        "state_dict": head.state_dict(),
        "embedding_mean": torch.from_numpy(np.asarray(mean, dtype=np.float32)),
        "embedding_std": torch.from_numpy(np.asarray(std, dtype=np.float32)),
        "topk": int(topk),
        "n_bits": N_BITS,
        "dim_emb": DIM_EMB,
        "hidden": head.hidden,
    }, path)


def load_head(path: Path, device: str | torch.device = "cpu") -> tuple[EcFpHead, np.ndarray, np.ndarray, int]:
    ck = torch.load(str(path), map_location=device)
    head = EcFpHead(ck["dim_emb"], ck["n_bits"], hidden=ck.get("hidden"))
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head, ck["embedding_mean"].numpy(), ck["embedding_std"].numpy(), int(ck["topk"])
