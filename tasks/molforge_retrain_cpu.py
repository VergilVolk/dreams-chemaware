"""
CPU-only retraining of the MolForge ECFP4 -> SMILES decoder.

The pretrained checkpoint lives on Google Drive, which is unreachable from this
machine. This script trains the *same* architecture on the bundled MolForge
training file (``third_party/MolForge/data/fingerprints/ECFP4.smiles.test``,
10,000 ground-truth ``SMILES-tokens \\t bit-indices`` pairs) so the decoder
module has a runnable checkpoint locally.

Output format matches upstream ``save_checkpoint``::
    {"model_state_dict": ..., "loss": <valid loss>}

so the result is drop-in compatible with ``MolForgeDecoder.load_checkpoint``.

NOTE: this is a *demo* checkpoint. The paper trained on ~2.8M compounds; a
10k-compound CPU retrain learns SMILES grammar but not publication-grade
fp->SMILES reconstruction. For real quality, drop the pretrained
``ECFP4_smiles_checkpoint.pth`` into ``third_party/MolForge/saved_models/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from annotation.molforge_decoder import (  # noqa: E402
    EOS_ID,
    PAD_ID,
    SOS_ID,
    TRG_SEQ_LEN,
    Transformer,
    WordTokenizer,
    _pad_or_truncate,
    _make_src_mask,
    greedy_search,
    bits_to_str,
    ecfp4_bits,
    detokenize,
)

MOLFORGE_DIR = REPO / "third_party" / "MolForge"
DEFAULT_TRAIN = MOLFORGE_DIR / "data" / "fingerprints" / "ECFP4.smiles.test"


def load_pairs(path: Path):
    """Yield (src_bits_str, trg_tokens_str) lines as in the MolForge test files."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            trg_tok, src_bits = line.rstrip("\n").split("\t")
            pairs.append((src_bits.strip(), trg_tok.strip()))
    return pairs


def build_dataset(src_sp, trg_sp, pairs, src_seq_len):
    src_list, trg_in_list, trg_out_list = [], [], []
    for src_text, trg_text in pairs:
        src_ids = _pad_or_truncate(src_sp.encode(src_text) + [EOS_ID], src_seq_len)
        trg_ids = trg_sp.encode(trg_text)
        trg_in = _pad_or_truncate([SOS_ID] + trg_ids, TRG_SEQ_LEN)
        trg_out = _pad_or_truncate(trg_ids + [EOS_ID], TRG_SEQ_LEN)
        src_list.append(src_ids)
        trg_in_list.append(trg_in)
        trg_out_list.append(trg_out)
    return (
        torch.tensor(src_list, dtype=torch.long),
        torch.tensor(trg_in_list, dtype=torch.long),
        torch.tensor(trg_out_list, dtype=torch.long),
    )


def make_trg_mask(trg_input):
    d_mask = (trg_input != PAD_ID).unsqueeze(1)
    nopeak = torch.ones(1, TRG_SEQ_LEN, TRG_SEQ_LEN, dtype=torch.bool)
    nopeak = torch.tril(nopeak)
    return d_mask & nopeak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", default=str(DEFAULT_TRAIN))
    ap.add_argument("--train-size", type=int, default=8000, help="number of pairs to train on")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--out", default=str(MOLFORGE_DIR / "saved_models" / "ECFP4_smiles_checkpoint.pth"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--log-interval", type=int, default=25)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    else:
        torch.set_num_threads(max(1, (torch.get_num_threads() or 1)))

    device = torch.device("cpu")

    sp_dir = MOLFORGE_DIR / "data" / "sp"
    src_sp = WordTokenizer(sp_dir / "ECFP4_vocab_sp.vocab")
    trg_sp = WordTokenizer(sp_dir / "smiles_vocab_sp.vocab")

    print("Loading pairs...", flush=True)
    pairs = load_pairs(Path(args.train_file))
    pairs = pairs[: args.train_size]
    n_val = int(len(pairs) * args.val_split)
    train_pairs, valid_pairs = pairs[n_val:], pairs[:n_val]
    print(f"train={len(train_pairs)} valid={len(valid_pairs)}", flush=True)

    src_seq_len = 104  # ECFP4
    print("Tokenizing...", flush=True)
    train_src, train_trg_in, train_trg_out = build_dataset(src_sp, trg_sp, train_pairs, src_seq_len)
    valid_src, valid_trg_in, valid_trg_out = build_dataset(src_sp, trg_sp, valid_pairs, src_seq_len)

    model = Transformer(2052, 109, src_seq_len, device).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    crit = nn.NLLLoss()

    n_train = train_src.shape[0]
    steps_per_epoch = (n_train + args.batch_size - 1) // args.batch_size

    def run_epoch(data, train=True):
        total = 0.0
        n = 0
        model.train() if train else model.eval()
        perm = torch.randperm(data[0].shape[0]) if train else torch.arange(data[0].shape[0])
        with torch.set_grad_enabled(train):
            for i in range(0, data[0].shape[0], args.batch_size):
                idx = perm[i : i + args.batch_size]
                src = data[0][idx]
                trg_in = data[1][idx]
                trg_out = data[2][idx]
                e_mask = _make_src_mask(src)
                d_mask = make_trg_mask(trg_in)
                out, _ = model(src, trg_in, e_mask, d_mask)
                loss = crit(out.reshape(-1, 109), trg_out.reshape(-1))
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                total += loss.item() * idx.shape[0]
                n += idx.shape[0]
        return total / max(n, 1)

    print(f"Training {args.epochs} epochs x {steps_per_epoch} steps/epoch (CPU)...", flush=True)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tloss = run_epoch((train_src, train_trg_in, train_trg_out), train=True)
        vloss = run_epoch((valid_src, valid_trg_in, valid_trg_out), train=False)
        dt = time.time() - t0
        print(f"epoch {ep}/{args.epochs}  train_loss={tloss:.4f}  valid_loss={vloss:.4f}  "
              f"{dt:.0f}s elapsed", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "loss": vloss}, out)
    print(f"saved -> {out}", flush=True)

    # quick smoke decode on 3 training fingerprints
    model.eval()
    print("\nSmoke decode (greedy):")
    for src_text, trg_tok in train_pairs[:3]:
        with torch.no_grad():
            src_ids = torch.tensor(_pad_or_truncate(src_sp.encode(src_text), src_seq_len),
                                   dtype=torch.long).unsqueeze(0)
            e_mask = _make_src_mask(src_ids)
            src_emb = model.src_embedding(src_ids)
            src_pos = model.src_positional_encoder(src_emb)
            e_out = model.encoder(src_pos, e_mask)
            pred = greedy_search(model, e_out, e_mask, trg_sp, device)
        print(f"  target: {trg_tok}")
        print(f"  pred  : {pred}")


if __name__ == "__main__":
    main()
