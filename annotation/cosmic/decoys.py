"""Spectrum-space decoy generators for the COSMIC confidence layer.

The pluggable :class:`DecoyGenerator` protocol produces decoys of a single
spectrum in the ``{"peaks": (2, n) ndarray, "precursor_mz": float}`` schema that
``annotation.embed.embed_records`` consumes. Spectrum-space decoys permute the
peak table directly:

  * :class:`ShuffleIntensityDecoy` -- permute intensities, keep the m/z axis
    (canonical Elias-Gygi / passatutto decoy).
  * :class:`ShuffleMZDecoy`      -- permute m/z values, destroying the mass
    pattern (neutral losses / characteristic fragments / isotope spacings).
  * :class:`StructureSpaceDecoy` -- RESERVED full COSMIC replication (PubChem
    structure decoys scored in-silico by CSI:FingerID), not implemented.

``peaks`` must be padding-free (real peaks only, m/z > 0). Each call with a
distinct ``seed`` is deterministic; the ``n`` decoys draw consecutive permutations
from that per-spectrum RNG, so a caller should pass ``seed = base_seed + spectrum_index``
to keep decoys independent across spectra.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DecoyGenerator(ABC):
    """Protocol for generating decoys of a single spectrum."""

    name: str = "abstract"

    @abstractmethod
    def generate(
        self, peaks: np.ndarray, precursor_mz: float, n: int = 1, seed: int = 0
    ) -> list[dict]:
        """Return ``n`` decoy records for one spectrum ``peaks`` (2, n_peaks)."""


class ShuffleIntensityDecoy(DecoyGenerator):
    """Permute intensities, keep the m/z axis + precursor.

    Canonical MS target-decoy decoy (Elias & Gygi 2007, DOI 10.1038/nmeth1013;
    Scheubert et al. passatutto 2017, DOI 10.1038/s41467-017-01318-5). It keeps the
    source compound's fragment m/z pattern, so for the m/z-dominated DreaMS
    embedding it is a *weak* decoy (measured: shuffle keeps cosine ~0.92).
    """

    name = "shuffle_intensity"

    def generate(self, peaks, precursor_mz, n=1, seed=0):
        rng = np.random.default_rng(seed)
        base = np.asarray(peaks, dtype=np.float32)
        decoys = []
        for _ in range(n):
            permuted = base.copy()
            permuted[1] = permuted[1][rng.permutation(permuted.shape[1])]
            decoys.append({"peaks": permuted, "precursor_mz": float(precursor_mz)})
        return decoys


class ShuffleMZDecoy(DecoyGenerator):
    """Permute m/z values, keep intensity positions + precursor.

    Destroys the mass-pattern structure while preserving peak count and intensity
    distribution, so it is a stronger decoy for a rule-matching-based score.
    """

    name = "shuffle_mz"

    def generate(self, peaks, precursor_mz, n=1, seed=0):
        rng = np.random.default_rng(seed)
        base = np.asarray(peaks, dtype=np.float32)
        decoys = []
        for _ in range(n):
            permuted = base.copy()
            permuted[0] = permuted[0][rng.permutation(permuted.shape[1])]
            decoys.append({"peaks": permuted, "precursor_mz": float(precursor_mz)})
        return decoys


class StructureSpaceDecoy(DecoyGenerator):
    """RESERVED -- full COSMIC replication.

    COSMIC decoys are *structure-space*: random PubChem structures scored in-silico
    (CSI:FingerID), not spectrum permutations. Implementing this generator (a
    structure database -> in-silico spectrum bridge) drops into the same E-value /
    calibration code without further change.
    """

    name = "structure_space"

    def generate(self, peaks, precursor_mz, n=1, seed=0):
        raise NotImplementedError(
            "Structure-space decoy is the reserved COSMIC-replication interface; "
            "not implemented in Step-1 (spectrum-space decoy + truth calibration)."
        )


DECOYS: dict[str, DecoyGenerator] = {
    decoy.name: decoy
    for decoy in (ShuffleIntensityDecoy(), ShuffleMZDecoy(), StructureSpaceDecoy())
}
