"""Model-aligned MassSpecGym retrieval with an optional frozen P2b preview.

The central safety invariant is strict: a query embedding may only be compared
with a reference index produced by the same checkpoint fingerprint. P2b is a
downstream spectrum-pair fusion and is reported separately because its sealed
near-isomer result was negative.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd

from deploy.p2b_rank_fusion import FROZEN_CONFIG, fuse_one_query, normalize_pair_features
from spectral_features import p2b_pair_features


@lru_cache(maxsize=8)
def _sha256_file(path_text: str) -> str:
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


def _index_environment(model_type: str) -> tuple[Path | None, Path | None]:
    prefix = {
        "official_dreams": "DREAMS_OFFICIAL",
        "e4a_shared": "DREAMS_E4",
        "e8_shared": "DREAMS_E8",
    }.get(model_type)
    if prefix is None:
        return None, None
    value = os.getenv(f"{prefix}_EMBEDDING_INDEX")
    index = Path(value) if value else None
    manifest_value = os.getenv(f"{prefix}_EMBEDDING_INDEX_MANIFEST")
    manifest = Path(manifest_value) if manifest_value else (index.with_suffix(".json") if index else None)
    return index, manifest


class MoleculeDatabase:
    """Mass-window retrieval over a model-aligned memory-mapped index."""

    def __init__(
        self,
        hdf5_path: Optional[str] = None,
        *,
        model_type: str = "official_dreams",
        model_fingerprint: str | None = None,
    ):
        configured = hdf5_path or os.getenv("DREAMS_MOLECULE_DB")
        self.hdf5_path = Path(configured) if configured else None
        self.model_type = model_type
        self.model_fingerprint = model_fingerprint
        self.h5file: h5py.File | None = None
        self.embeddings: np.ndarray | None = None
        self.error: str | None = None
        self.index_manifest: dict[str, Any] = {}
        self.reference_scope = "unknown_reference"
        self._mass_order: np.ndarray | None = None
        self._sorted_mass: np.ndarray | None = None
        if self.hdf5_path is None or not self.hdf5_path.is_file():
            self.error = "未配置可用的 MassSpecGym HDF5"
            return
        try:
            self._load()
        except Exception as exc:
            self.close()
            self.error = f"候选索引加载失败: {type(exc).__name__}: {exc}"

    def _load(self) -> None:
        self.h5file = h5py.File(str(self.hdf5_path), "r")
        source_status = _decode(self.h5file.attrs.get("status", "production_reference"))
        if source_status == "showspace_real_spectrum_smoke_reference":
            self.reference_scope = "engineering_smoke_real_spectra"
        else:
            self.reference_scope = source_status or "production_reference"
        required = {"precursor_mz", "spectrum", "INCHIKEY", "FORMULA", "smiles"}
        missing = required - set(self.h5file.keys())
        if missing:
            raise ValueError(f"HDF5 缺少字段 {sorted(missing)}")
        index_path, manifest_path = _index_environment(self.model_type)
        if index_path is None or not index_path.is_file():
            raise FileNotFoundError(f"{self.model_type} 缺少配套 reference embedding index")
        if manifest_path is None or not manifest_path.is_file():
            raise FileNotFoundError(f"{self.model_type} 缺少 reference index manifest")
        self.index_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.index_manifest.get("status") != "showspace_model_aligned_embedding_index":
            raise ValueError("reference index manifest status 不合法")
        if self.index_manifest.get("complete") is not True:
            raise ValueError("reference embedding index 不是完整谱库索引")
        expected = str(self.index_manifest.get("model_fingerprint", ""))
        if self.model_fingerprint and expected != self.model_fingerprint:
            raise ValueError("query checkpoint 与 reference index fingerprint 不一致")
        if self.index_manifest.get("model_type") != self.model_type:
            raise ValueError("reference index model_type 不匹配")
        source_hash = str(self.index_manifest.get("source_hdf5_sha256", ""))
        if not source_hash or _sha256_file(str(self.hdf5_path.resolve())) != source_hash:
            raise ValueError("reference index 与当前 HDF5 谱库指纹不匹配")
        self.embeddings = np.load(index_path, mmap_mode="r")
        n_rows = len(self.h5file["precursor_mz"])
        if self.embeddings.shape != (n_rows, 1024):
            raise ValueError(f"reference embedding index shape={self.embeddings.shape}, expected={(n_rows, 1024)}")
        masses = np.asarray(self.h5file["precursor_mz"][:], dtype=np.float64)
        self._mass_order = np.argsort(masses, kind="stable")
        self._sorted_mass = masses[self._mass_order]

    def close(self) -> None:
        if self.h5file is not None:
            self.h5file.close()
        self.h5file = None
        self.embeddings = None

    def __del__(self):
        self.close()

    @property
    def ready(self) -> bool:
        return self.error is None and self.h5file is not None and self.embeddings is not None

    def _mass_rows(self, precursor_mz: float, ppm: float) -> np.ndarray:
        if self._sorted_mass is None or self._mass_order is None:
            return np.empty(0, dtype=np.int64)
        tolerance = float(precursor_mz) * float(ppm) * 1e-6
        left = np.searchsorted(self._sorted_mass, precursor_mz - tolerance, side="left")
        right = np.searchsorted(self._sorted_mass, precursor_mz + tolerance, side="right")
        return self._mass_order[left:right]

    def _values(self, key: str, rows: np.ndarray) -> list[str]:
        assert self.h5file is not None
        dataset = self.h5file.get(key)
        if dataset is None:
            return [""] * len(rows)
        return [_decode(dataset[int(row)]) for row in rows]

    def retrieve(
        self,
        query_embedding: np.ndarray,
        query_spectrum: np.ndarray,
        precursor_mz: float,
        *,
        adduct: str = "",
        ppm: float = 10.0,
        max_results: int = 10,
        p2b_preview: bool = True,
        peak_tolerance: float = 0.02,
        maximum_pair_evaluations: int = 5000,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if not self.ready:
            raise RuntimeError(self.error or "candidate database is not ready")
        assert self.h5file is not None and self.embeddings is not None
        rows = self._mass_rows(float(precursor_mz), float(ppm))
        if adduct and "adduct" in self.h5file:
            library_adduct = self._values("adduct", rows)
            rows = rows[np.asarray([value == adduct for value in library_adduct], dtype=bool)]
        if len(rows) == 0:
            return pd.DataFrame(), {
                "candidate_spectra": 0,
                "candidate_molecules": 0,
                "p2b_state": "no_candidates",
                "reference_scope": self.reference_scope,
            }

        query = np.asarray(query_embedding, dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        reference = np.asarray(self.embeddings[rows], dtype=np.float32)
        reference /= np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
        similarities = reference @ query
        identities = self._values("INCHIKEY", rows)
        groups: dict[str, list[int]] = {}
        for local, identity in enumerate(identities):
            groups.setdefault(identity or f"row:{int(rows[local])}", []).append(local)
        ordered_groups = sorted(groups.items(), key=lambda item: (-float(np.max(similarities[item[1]])), item[0]))

        p2b_scores: dict[str, float] = {}
        p2b_used = False
        p2b_state = "disabled"
        if p2b_preview:
            if not adduct:
                p2b_state = "requires_query_adduct"
            elif len(rows) > maximum_pair_evaluations:
                p2b_state = "candidate_group_too_large"
            else:
                pair_features: list[np.ndarray] = []
                molecule_ptr = [0]
                for _, local_rows in ordered_groups:
                    for local in sorted(local_rows, key=lambda i: -float(similarities[i])):
                        pair_features.append(p2b_pair_features(
                            query_spectrum, precursor_mz,
                            np.asarray(self.h5file["spectrum"][int(rows[local])]),
                            float(self.h5file["precursor_mz"][int(rows[local])]),
                            float(similarities[local]), peak_tolerance,
                        ))
                    molecule_ptr.append(len(pair_features))
                features = np.stack(pair_features)
                normalized = normalize_pair_features(features, np.asarray([0, len(features)]), "absolute")
                scores, p2b_used, _ = fuse_one_query(
                    normalized, features[:, 0], np.asarray(molecule_ptr),
                    np.asarray(FROZEN_CONFIG["weights"], dtype=np.float64), (1, 2, 3),
                    int(FROZEN_CONFIG["min_support"]), float(FROZEN_CONFIG["min_advantage"]),
                )
                p2b_scores = {identity: float(score) for (identity, _), score in zip(ordered_groups, scores)}
                p2b_state = "preview_computed"

        output: list[dict[str, Any]] = []
        for identity, local_rows in ordered_groups:
            best_local = max(local_rows, key=lambda i: float(similarities[i]))
            row = int(rows[best_local])
            output.append({
                "Rank": 0,
                "Embedding rank": 0,
                "P2b preview rank": None,
                "Precursor m/z": float(self.h5file["precursor_mz"][row]),
                "Delta ppm": 1e6 * (float(self.h5file["precursor_mz"][row]) - precursor_mz) / precursor_mz,
                "Ref. name": _decode(self.h5file["IDENTIFIER"][row]) if "IDENTIFIER" in self.h5file else identity,
                "Ref. ID": identity,
                "Formula": _decode(self.h5file["FORMULA"][row]),
                "SMILES": _decode(self.h5file["smiles"][row]),
                "Embedding similarity": float(similarities[best_local]),
                "P2b preview score": p2b_scores.get(identity),
                "Reference spectra": len(local_rows),
            })
        output.sort(key=lambda item: (-item["Embedding similarity"], item["Ref. ID"]))
        for rank, item in enumerate(output, 1):
            item["Rank"] = rank
            item["Embedding rank"] = rank
        if p2b_scores:
            for rank, item in enumerate(sorted(output, key=lambda x: (-float(x["P2b preview score"]), x["Ref. ID"])), 1):
                item["P2b preview rank"] = rank
        report = {
            "candidate_spectra": int(len(rows)), "candidate_molecules": int(len(output)),
            "model_type": self.model_type, "model_fingerprint": self.model_fingerprint,
            "ppm": float(ppm), "same_adduct": bool(adduct), "p2b_state": p2b_state,
            "reference_scope": self.reference_scope,
            "reference_spectra": int(len(self.h5file["precursor_mz"])),
            "p2b_gate_used": bool(p2b_used),
            "p2b_claim_limit": "preview only; sealed near-core performance was negative",
        }
        return pd.DataFrame(output[:max_results]), report

    def get_candidate_molecules(
        self, query_embedding: np.ndarray, precursor_mz: float,
        tolerance_da: float = 0.05, max_results: int = 10,
    ) -> pd.DataFrame:
        ppm = tolerance_da / max(float(precursor_mz), 1e-12) * 1e6
        empty = np.asarray([[1.0, 1.0], [2.0, 0.5]], dtype=np.float32)
        frame, _ = self.retrieve(query_embedding, empty, precursor_mz, ppm=ppm, max_results=max_results, p2b_preview=False)
        return frame
