"""M2b -- Rule-based diagnostic evidence for Schymanski Level 3.

Rule hits answer a narrow question: *does this spectrum exhibit a known,
chemically-interpretable mass pattern* (a neutral loss, a characteristic
fragment m/z, an isotope spacing, a hydrogen-rearrangement spacing)?

Rules come from the 335-rule main library
(``dreams/models/chem_aware/chem_rules_data.json``, categories NL/CF/ISO/NR/EE/HR).
The MassBank 3,151-rule extension is deliberately EXCLUDED: the rule engine
itself labels those ``tier=extended, evidence=medium`` "noise rules"
(see ``dreams/models/chem_aware/chem_rules.py``).

The matcher is the canonical one (``spectrum_rule_vector``), bit-for-bit equal
to ``tasks/build_spectrum_rule_label_cache.py`` and to the P1 code path
``pilot_rule_noise_stress.FastRuleMatcher``.

HONEST LIMITS -- measured on Met/neg (13,770 spectra), not assumed:

  * 100% of spectra hit >=1 rule; NL (neutral-loss) rules average 18.6
    hits/spectrum, so "hits any rule" is NOT discriminating.
  * CF (characteristic-fragment) hits are sparse (1.29/spectrum) but do NOT
    predict the confident-annotation rate (CF>=4 spectra are *less* often
    confidently annotated than CF=0).
  * Consequently rule evidence is a Schymanski *semantic* upgrade
    (exact-mass -> tentative-candidate: "we can explain some fragmentation"),
    NOT a confidence/accuracy upgrade. This mirrors the rule engine's own
    ``claim_limit``: "a matched mass pattern is not a unique fragment structure
    or bond-breaking mechanism."
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
_RULES_JSON = ROOT / "dreams/models/chem_aware/chem_rules_data.json"
_M_H = 1.00782503223
_TOL = 0.02  # Da, matches the rule engine default

# Sparse, structurally specific categories used as diagnostic evidence.
# NL is excluded (too common: 18.6 hits/spectrum); NR/EE are near-universal.
DIAGNOSTIC_CATEGORIES = ("CF", "ISO")


def load_main_rules() -> list[dict]:
    """Load the 335-rule main library (MassBank extension excluded)."""
    return json.loads(_RULES_JSON.read_text(encoding="utf-8"))["rules"]


def _target_hit(sv: np.ndarray, t: float, tol: float) -> bool:
    if sv.size == 0:
        return False
    p = int(np.searchsorted(sv, t))
    if p < sv.size and abs(float(sv[p]) - t) < tol:
        return True
    return p > 0 and abs(float(sv[p - 1]) - t) < tol


def _range_hit(sv: np.ndarray, lo: float, hi: float) -> bool:
    if sv.size == 0:
        return False
    p = int(np.searchsorted(sv, lo, side="left"))
    return p < sv.size and float(sv[p]) <= hi


def spectrum_rule_vector(mz_padded: np.ndarray, precursor: float, rules: list[dict]) -> np.ndarray:
    """Binary rule-hit vector (n_rules,) for one spectrum.

    ``mz_padded`` is the m/z row (zero padding allowed) of a (2, n_peaks)
    spectrum; zero padding is excluded. Canonical matcher (see module docstring).
    """
    mz = np.sort(mz_padded[np.isfinite(mz_padded) & (mz_padded > 0)].astype(np.float64))
    diffs = np.sort(np.abs(mz[:, None] - mz[None, :]).reshape(-1)) if mz.size else np.empty(0)
    labels = np.zeros(len(rules), dtype=np.uint8)
    for i, r in enumerate(rules):
        k = r["match_type"]
        v = r["value"]
        if k == "mass_diff":
            labels[i] = _target_hit(diffs, float(v), _TOL)
        elif k == "peak_mz":
            labels[i] = _target_hit(mz, float(v), _TOL)
        elif k == "mass_range":
            labels[i] = _range_hit(diffs, float(v[0]), float(v[1]))
        elif k == "hr_shift":
            nh = float(v)
            if nh == 0:
                e = diffs[diffs >= 12.0]
                labels[i] = bool(e.size and np.any(np.abs(e - np.round(e)) < _TOL))
            else:
                labels[i] = _target_hit(diffs, abs(nh) * _M_H, _TOL)
        elif k == "parity":
            labels[i] = bool(diffs.size and np.any((np.round(diffs).astype(np.int64) % 2) == (round(precursor) % 2)))
        elif k == "mass_diff_range":
            lo, hi = map(float, v)
            labels[i] = bool(diffs.size and np.any((diffs > hi) | (diffs < lo)))
    return labels


def compute_rule_hits(manifest: pd.DataFrame, neg_dir: Path) -> np.ndarray:
    """Rule-hit matrix (n_query, n_rules) for the query spectra referenced by
    ``manifest`` (columns ``file_name`` and ``row_in_file``). ``neg_dir`` holds
    the ``<file_name>.hdf5`` files."""
    rules = load_main_rules()
    vecs = []
    for fname, grp in manifest.groupby("file_name"):
        import h5py
        with h5py.File(neg_dir / f"{fname}.hdf5", "r") as h:
            spec = np.asarray(h["spectrum"][:], dtype=np.float32)
            prec = np.asarray(h["precursor_mz"][:], dtype=np.float32)
        for r in grp["row_in_file"].to_numpy():
            vecs.append(spectrum_rule_vector(spec[r][0], float(prec[r]), rules))
    return np.stack(vecs).astype(np.uint8)


def rule_meta(rules: list[dict] | None = None) -> dict:
    rules = rules if rules is not None else load_main_rules()
    return {
        "rule_name": [r["name"] for r in rules],
        "rule_category": [r["category"] for r in rules],
        "rule_match_type": [r["match_type"] for r in rules],
    }


def diagnostic_evidence(
    rule_hits: np.ndarray,
    categories: tuple[str, ...] = DIAGNOSTIC_CATEGORIES,
) -> np.ndarray:
    """Boolean vector (n_query,) -- True if the spectrum hits >=1 sparse
    diagnostic rule (CF/ISO by default). This is a *semantic* Schymanski signal,
    not a correctness prediction (see module docstring)."""
    rules = load_main_rules()
    idx = [i for i, r in enumerate(rules) if r["category"] in categories]
    if not idx:
        return np.zeros(rule_hits.shape[0], dtype=bool)
    return rule_hits[:, idx].sum(axis=1) >= 1


def load_cached_rule_hits(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (rule_hits, diagnostic_flags) from a cached ``rule_hits.npy``."""
    V = np.load(cache_dir / "rule_hits.npy")
    return V, diagnostic_evidence(V)
