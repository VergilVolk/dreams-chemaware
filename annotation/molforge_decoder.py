"""
Fingerprint -> SMILES decoder (Part 3), self-contained MolForge reimplementation.

Reference
    Ucak UV, Ashyrmamatov I, Lee J (2023). "Reconstruction of lossless molecular
    representations from fingerprints." J Cheminformatics 15:26.
    https://doi.org/10.1186/s13321-023-00693-0
    Upstream code: https://github.com/knu-lcbc/MolForge  (CC BY-NC-SA 4.0)

Why self-contained
    The upstream package imports ``sentencepiece`` and ``selfies`` at module
    import time. Neither is needed for the ECFP4 -> SMILES path, and
    ``sentencepiece`` has no wheel for the Python in this environment. This
    module therefore (a) re-implements the exact MolForge Transformer in pure
    torch, and (b) re-implements the sentencepiece *word* tokenizer directly
    over the bundled ``*.vocab`` files (ID = line index, token = first field).

    The pretrained checkpoint (``saved_models/ECFP4_smiles_checkpoint.pth``)
    is architecture-compatible with this module: its ``model_state_dict`` keys
    match the :class:`Transformer` defined below exactly (no ``module.``
    prefix after stripping).

Usage
    >>> dec = MolForgeDecoder()            # resolves third_party/MolForge
    >>> dec.load_checkpoint("saved_models/ECFP4_smiles_checkpoint.pth")
    >>> bits = ecfp4_bits("CCOC1=CC(=CC=C1)C(C)(C)N")
    >>> smiles = dec.decode_bits(bits)     # greedy
"""

from __future__ import annotations

import heapq
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors

# --------------------------------------------------------------------------- #
# Parameters (mirror third_party/MolForge/MolForge/parameters.py)
# --------------------------------------------------------------------------- #
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

NUM_HEADS = 8
NUM_LAYERS = 6
DIM_MODEL = 512
DIM_FF = 2048
DIM_K = DIM_MODEL // NUM_HEADS
DROPOUT_RATE = 0.1

TRG_SEQ_LEN = 130
BEAM_SIZE = 10

# fingerprint -> (src_vocab_size, src_seq_len)
FP_VOCAB_SIZES = {"ECFP4": 2052}
FP_SEQ_LENS = {"ECFP4": 104}
# molecular representation -> trg_vocab_size
TRG_VOCAB_SIZES = {"smiles": 109, "selfies": 205}

_WSP = "▁"  # sentencepiece whitespace marker used in the vocab files


# --------------------------------------------------------------------------- #
# Pure-python sentencepiece "word" tokenizer over a .vocab file
# --------------------------------------------------------------------------- #
class WordTokenizer:
    """Minimal replacement for sentencepiece's word-model EncodeAsIds/decode_ids.

    The MolForge vocabs are `sp_model_type='word'` models: whitespace-separated
    tokens are looked up verbatim (with a leading ``▁`` marker), and the
    vocab file lists tokens in ID order (line index == id).
    """

    def __init__(self, vocab_path: Path):
        self.id2token: Dict[int, str] = {}
        self.token2id: Dict[str, int] = {}
        with open(vocab_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                tok = line.strip().split("\t")[0]
                self.id2token[i] = tok
                self.token2id[tok] = i

    def encode(self, text: str) -> List[int]:
        """Whitespace-split then map each token (with ``▁`` prefix) to id."""
        ids: List[int] = []
        for w in text.split():
            ids.append(self.token2id.get(_WSP + w, UNK_ID))
        return ids

    def decode_ids(self, ids: Sequence[int]) -> str:
        """Map ids back to tokens, strip the ``▁`` marker, join with space."""
        out: List[str] = []
        for i in ids:
            if i in (PAD_ID, SOS_ID, EOS_ID, UNK_ID):
                continue  # control tokens decode to nothing in this flow
            tok = self.id2token.get(i, "")
            out.append(tok.lstrip(_WSP))
        return " ".join(out)


def detokenize(smiles_tokens: str) -> str:
    """'C C O C 1 = C ...' -> 'CCOC1=C(...)'."""
    return "".join(smiles_tokens.split())


def tokenize(smiles: str) -> str:
    """Canonical SMILES -> space-separated token form used by MolForge targets."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    can = Chem.MolToSmiles(mol)
    return " ".join(_smiles_tokens(can))


_SMILES_RE = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def selfies_to_smiles(selfies_str: str) -> str:
    """MolForge SELFIES (old selfies-1.x token format, joined) -> SMILES.

    MolForge's SELFIES vocab/data use the selfies **1.x** token format
    (e.g. ``[Branch2_1]``, ``[Cexpl]``). Decoding therefore requires the
    ``selfies`` 1.x package (``sf.decoder``), *not* selfies 2.x, which
    rejects the ``_<bond-order>`` suffix.
    """
    import selfies as sf
    return sf.decoder(selfies_str)


def _smiles_tokens(smiles: str) -> List[str]:
    return _SMILES_RE.findall(smiles)


# --------------------------------------------------------------------------- #
# Transformer (verbatim architecture copy; state_dict keys must match upstream)
# --------------------------------------------------------------------------- #
class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, trg_vocab_size: int,
                 src_seq_len: int, device: torch.device):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        self.src_embedding = nn.Embedding(src_vocab_size, DIM_MODEL)
        self.trg_embedding = nn.Embedding(trg_vocab_size, DIM_MODEL)
        self.src_positional_encoder = PositionalEncoder(src_seq_len, device)
        self.trg_positional_encoder = PositionalEncoder(TRG_SEQ_LEN, device)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output_linear = nn.Linear(DIM_MODEL, trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src_input, trg_input, e_mask=None, d_mask=None):
        src_input = self.src_embedding(src_input)
        trg_input = self.trg_embedding(trg_input)
        src_input = self.src_positional_encoder(src_input)
        trg_input = self.trg_positional_encoder(trg_input)
        e_output = self.encoder(src_input, e_mask)
        d_output, attn_weight = self.decoder(trg_input, e_output, e_mask, d_mask)
        output = self.softmax(self.output_linear(d_output))
        return output, attn_weight


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(NUM_LAYERS)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, e_mask):
        for layer in self.layers:
            x = layer(x, e_mask)
        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(NUM_LAYERS)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, e_output, e_mask, d_mask):
        for layer in self.layers:
            x, attn_weight = layer(x, e_output, e_mask, d_mask)
        return self.layer_norm(x), attn_weight


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_1 = LayerNormalization()
        self.multihead_attention = MultiheadAttention()
        self.drop_out_1 = nn.Dropout(DROPOUT_RATE)
        self.layer_norm_2 = LayerNormalization()
        self.feed_forward = FeedFowardLayer()
        self.drop_out_2 = nn.Dropout(DROPOUT_RATE)

    def forward(self, x, e_mask):
        x_1 = self.layer_norm_1(x)
        x = x + self.drop_out_1(self.multihead_attention(x_1, x_1, x_1, mask=e_mask)[0])
        x_2 = self.layer_norm_2(x)
        x = x + self.drop_out_2(self.feed_forward(x_2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_1 = LayerNormalization()
        self.masked_multihead_attention = MultiheadAttention()
        self.drop_out_1 = nn.Dropout(DROPOUT_RATE)
        self.layer_norm_2 = LayerNormalization()
        self.multihead_attention = MultiheadAttention()
        self.drop_out_2 = nn.Dropout(DROPOUT_RATE)
        self.layer_norm_3 = LayerNormalization()
        self.feed_forward = FeedFowardLayer()
        self.drop_out_3 = nn.Dropout(DROPOUT_RATE)

    def forward(self, x, e_output, e_mask, d_mask):
        x_1 = self.layer_norm_1(x)
        x = x + self.drop_out_1(self.masked_multihead_attention(x_1, x_1, x_1, mask=d_mask)[0])
        x_2 = self.layer_norm_2(x)
        attn_output, attn_weight = self.multihead_attention(x_2, e_output, e_output, mask=e_mask)
        x = x + self.drop_out_2(attn_output)
        x_3 = self.layer_norm_3(x)
        x = x + self.drop_out_3(self.feed_forward(x_3))
        return x, attn_weight


class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.inf = 1e9
        self.w_q = nn.Linear(DIM_MODEL, DIM_MODEL)
        self.w_k = nn.Linear(DIM_MODEL, DIM_MODEL)
        self.w_v = nn.Linear(DIM_MODEL, DIM_MODEL)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.attn_softmax = nn.Softmax(dim=-1)
        self.w_0 = nn.Linear(DIM_MODEL, DIM_MODEL)

    def forward(self, q, k, v, mask=None):
        input_shape = q.shape
        q = self.w_q(q).view(input_shape[0], -1, NUM_HEADS, DIM_K)
        k = self.w_k(k).view(input_shape[0], -1, NUM_HEADS, DIM_K)
        v = self.w_v(v).view(input_shape[0], -1, NUM_HEADS, DIM_K)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_values, attn_weights = self.self_attention(q, k, v, mask=mask)
        concat_output = attn_values.transpose(1, 2).contiguous().view(input_shape[0], -1, DIM_MODEL)
        return self.w_0(concat_output), attn_weights

    def self_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(DIM_K)
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)
        attn_distribs = self.attn_softmax(attn_scores)
        attn_weights = self.attn_softmax(attn_scores)
        attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v)
        return attn_values, attn_weights


class FeedFowardLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(DIM_MODEL, DIM_FF, bias=True)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(DIM_FF, DIM_MODEL, bias=True)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.dropout(x)
        return self.linear_2(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.layer = nn.LayerNorm([DIM_MODEL], elementwise_affine=True, eps=self.eps)

    def forward(self, x):
        return self.layer(x)


class PositionalEncoder(nn.Module):
    def __init__(self, seq_len: int, device: torch.device):
        super().__init__()
        pe = torch.zeros(seq_len, DIM_MODEL)
        for pos in range(seq_len):
            for i in range(DIM_MODEL):
                if i % 2 == 0:
                    pe[pos, i] = math.sin(pos / (10000 ** (2 * i / DIM_MODEL)))
                else:
                    pe[pos, i] = math.cos(pos / (10000 ** (2 * i / DIM_MODEL)))
        # plain attribute (not a buffer/param) so it stays out of state_dict,
        # matching upstream's `.to(args.rank).requires_grad_(False)`.
        self.register_buffer("positional_encoding", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        x = x * math.sqrt(DIM_MODEL)
        return x + self.positional_encoding


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def _pad_or_truncate(tokenized: List[int], seq_len: int) -> List[int]:
    if len(tokenized) < seq_len:
        return tokenized + [PAD_ID] * (seq_len - len(tokenized))
    return tokenized[:seq_len]


def _make_src_mask(src_input: torch.Tensor) -> torch.Tensor:
    return (src_input != PAD_ID).unsqueeze(1)  # (B, 1, L)


def greedy_search(model: Transformer, e_output, e_mask, trg_sp: WordTokenizer,
                  device: torch.device) -> str:
    last_words = torch.full((TRG_SEQ_LEN,), PAD_ID, dtype=torch.long, device=device)
    last_words[0] = SOS_ID
    cur_len = 1

    model.eval()
    with torch.no_grad():
        for i in range(TRG_SEQ_LEN):
            d_mask = (last_words.unsqueeze(0) != PAD_ID).unsqueeze(1)
            nopeak = torch.ones(1, TRG_SEQ_LEN, TRG_SEQ_LEN, dtype=torch.bool, device=device)
            nopeak = torch.tril(nopeak)
            d_mask = d_mask & nopeak

            trg_embedded = model.trg_embedding(last_words.unsqueeze(0))
            trg_pos = model.trg_positional_encoder(trg_embedded)
            decoder_output, _ = model.decoder(trg_pos, e_output, e_mask, d_mask)
            output = model.softmax(model.output_linear(decoder_output))  # (1, L, V)
            last_word_id = int(torch.argmax(output[0][i]).item())

            if i < TRG_SEQ_LEN - 1:
                last_words[i + 1] = last_word_id
                cur_len += 1
            if last_word_id == EOS_ID:
                break

    if last_words[-1].item() == PAD_ID:
        ids = last_words[1:cur_len].tolist()
    else:
        ids = last_words[1:].tolist()
    return trg_sp.decode_ids(ids)


def beam_search(model: Transformer, e_output, e_mask, trg_sp: WordTokenizer,
                device: torch.device, beam_size: int = BEAM_SIZE) -> List[str]:
    cur_queue = _PriorityQueue()
    cur_queue.put(_BeamNode(SOS_ID, 0.0, [SOS_ID]))

    model.eval()
    with torch.no_grad():
        for pos in range(TRG_SEQ_LEN):
            new_queue = _PriorityQueue()
            for k in range(beam_size):
                if pos == 0 and k > 0:
                    continue
                node = cur_queue.get()
                if node.is_finished:
                    new_queue.put(node)
                    continue
                trg_input = torch.tensor(
                    node.decoded + [PAD_ID] * (TRG_SEQ_LEN - len(node.decoded)),
                    dtype=torch.long, device=device,
                ).unsqueeze(0)
                d_mask = (trg_input != PAD_ID).unsqueeze(1)
                nopeak = torch.ones(1, TRG_SEQ_LEN, TRG_SEQ_LEN, dtype=torch.bool, device=device)
                nopeak = torch.tril(nopeak)
                d_mask = d_mask & nopeak

                trg_embedded = model.trg_embedding(trg_input)
                trg_pos = model.trg_positional_encoder(trg_embedded)
                decoder_output, _ = model.decoder(trg_pos, e_output, e_mask, d_mask)
                output = model.softmax(model.output_linear(decoder_output))
                top = torch.topk(output[0][pos], dim=-1, k=beam_size)
                for idx, prob in zip(top.indices.tolist(), top.values.tolist()):
                    new_node = _BeamNode(idx, node.prob + prob, node.decoded + [idx])
                    if idx == EOS_ID:
                        new_node.is_finished = True
                    new_queue.put(new_node)
            cur_queue = new_queue

    results: List[str] = []
    for _ in range(beam_size):
        node = cur_queue.get()
        ids = node.decoded
        if ids[-1] == EOS_ID:
            ids = ids[1:-1]
        else:
            ids = ids[1:]
        results.append(trg_sp.decode_ids(ids))
    return results


class _BeamNode:
    __slots__ = ("idx", "prob", "decoded", "is_finished")

    def __init__(self, idx, prob, decoded):
        self.idx = idx
        self.prob = prob
        self.decoded = decoded
        self.is_finished = False

    def __gt__(self, other): return self.prob > other.prob
    def __ge__(self, other): return self.prob >= other.prob
    def __lt__(self, other): return self.prob < other.prob
    def __le__(self, other): return self.prob <= other.prob


class _PriorityQueue:
    def __init__(self):
        self.queue: List = []

    def put(self, obj):
        heapq.heappush(self.queue, (obj.prob, obj))

    def get(self):
        return heapq.heappop(self.queue)[1]


# --------------------------------------------------------------------------- #
# Fingerprint helpers
# --------------------------------------------------------------------------- #
def ecfp4_bits(smiles: str, radius: int = 2, nbits: int = 2048) -> List[int]:
    """Sorted ON-bit indices of the ECFP4 (hashed Morgan, r=2, 2048-bit) fingerprint.

    Matches MolForge's ``ECFP4(mol, return_bits=True)`` which returns
    ``list(GetHashedMorganFingerprint(mol, r, nBits).GetNonzeroElements())``.
    """
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=radius, nBits=nbits)
    return sorted(fp.GetNonzeroElements().keys())


def bits_to_str(bits: Sequence[int]) -> str:
    """Bit indices -> the space-separated input string MolForge expects."""
    return " ".join(str(int(b)) for b in bits)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _default_molforge_dir() -> Path:
    here = Path(__file__).resolve().parent  # annotation/
    return here.parent / "third_party" / "MolForge"


class MolForgeDecoder:
    """Fingerprint (ECFP4) -> SMILES decoder backed by a MolForge checkpoint."""

    def __init__(self, molforge_dir: Optional[os.PathLike] = None,
                 model_type: str = "smiles", device: Optional[torch.device] = None):
        if model_type not in TRG_VOCAB_SIZES:
            raise ValueError(f"unknown model_type: {model_type!r} (choose {list(TRG_VOCAB_SIZES)})")
        self.molforge_dir = Path(molforge_dir) if molforge_dir else _default_molforge_dir()
        self.model_type = model_type
        self.fp = "ECFP4"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sp_dir = self.molforge_dir / "data" / "sp"
        self.src_sp = WordTokenizer(sp_dir / f"{self.fp}_vocab_sp.vocab")
        self.trg_sp = WordTokenizer(sp_dir / f"{model_type}_vocab_sp.vocab")

        self.src_vocab_size = FP_VOCAB_SIZES[self.fp]
        self.trg_vocab_size = TRG_VOCAB_SIZES[model_type]
        self.src_seq_len = FP_SEQ_LENS[self.fp]

        self.model = Transformer(
            self.src_vocab_size, self.trg_vocab_size, self.src_seq_len, self.device
        ).to(self.device)
        self.model.eval()

    # -- checkpoint --
    def load_checkpoint(self, checkpoint_path: os.PathLike) -> None:
        ckpt = torch.load(str(checkpoint_path), map_location=self.device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        updated = {}
        for key, val in state.items():
            updated[key.replace("module.", "")] = val
        self.model.load_state_dict(updated)
        self.model.to(self.device)
        self.model.eval()

    # -- decode --
    def decode_bits(self, bits: Sequence[int], method: str = "greedy",
                    return_tokens: bool = False) -> str:
        """Decode ON-bit indices to SMILES (or space-separated tokens if asked)."""
        text = bits_to_str(bits)
        tokenized = self.src_sp.encode(text)
        src_data = torch.tensor(
            _pad_or_truncate(tokenized, self.src_seq_len), dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        e_mask = _make_src_mask(src_data)

        self.model.eval()
        with torch.no_grad():
            src_embedded = self.model.src_embedding(src_data)
            src_pos = self.model.src_positional_encoder(src_embedded)
            e_output = self.model.encoder(src_pos, e_mask)

        if method == "greedy":
            tokens = greedy_search(self.model, e_output, e_mask, self.trg_sp, self.device)
        elif method == "beam":
            tokens = beam_search(self.model, e_output, e_mask, self.trg_sp, self.device)[0]
        else:
            raise ValueError(f"unknown decode method: {method!r}")

        return tokens if return_tokens else detokenize(tokens)

    def decode_bits_to_smiles(self, bits: Sequence[int], method: str = "greedy") -> str:
        """Decode ON-bit indices to a SMILES string.

        For ``model_type='smiles'`` this is a direct decode; for
        ``model_type='selfies'`` the decoded SELFIES is converted to SMILES via
        the ``selfies`` package (``sf.decoder``, guaranteed-valid output).
        """
        out = self.decode_bits(bits, method=method)
        if self.model_type == "selfies":
            return selfies_to_smiles(out)
        return out

    def decode_smiles(self, smiles: str, method: str = "greedy") -> str:
        """Convenience: SMILES -> ECFP4 bits -> decoded SMILES (round-trip test)."""
        return self.decode_bits_to_smiles(ecfp4_bits(smiles), method=method)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="MolForge ECFP4 fingerprint -> SMILES decoder (self-contained)")
    ap.add_argument("--checkpoint", required=True,
                    help="path to a MolForge checkpoint (e.g. ECFP4_selfies_checkpoint.pth)")
    ap.add_argument("--smiles", help="decode this SMILES' ECFP4 fingerprint (round-trip)")
    ap.add_argument("--bits", help="space-separated ON-bit indices, e.g. '1 80 94 ...'")
    ap.add_argument("--method", default="greedy", choices=["greedy", "beam"])
    ap.add_argument("--model-type", default="selfies", choices=["smiles", "selfies"],
                    help="target vocabulary of the checkpoint; 'selfies' for "
                         "ECFP4_selfies_checkpoint.pth (converted back to SMILES)")
    ap.add_argument("--molforge-dir", default=None)
    args = ap.parse_args()

    dec = MolForgeDecoder(molforge_dir=args.molforge_dir, model_type=args.model_type)
    dec.load_checkpoint(args.checkpoint)
    if args.bits:
        bits = [int(b) for b in args.bits.split()]
        print(dec.decode_bits_to_smiles(bits, method=args.method))
    elif args.smiles:
        print(dec.decode_smiles(args.smiles, method=args.method))
    else:
        ap.error("provide --smiles or --bits")
