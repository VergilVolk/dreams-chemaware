"""Fail-closed data and evaluation utilities for ChemAware shared embedding v2.

The v2 cache is intentionally distinct from the historical noise cache.  In
particular it stores precursor m/z (needed for neutral-loss features) and the
official normalized embedding produced in the same encoding pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

try:  # Script execution and package import are both supported.
    from noise_final_core import CandidateGraph, sha256_file, strict_metrics, strict_rank
except ModuleNotFoundError:  # pragma: no cover - exercised by pytest package import
    from .noise_final_core import CandidateGraph, sha256_file, strict_metrics, strict_rank


TOKEN_STATUS = "chemaware_shared_v2_token_cache_complete"
MOLFORMER_STATUS = "chemaware_shared_v2_frozen_molformer_cache_complete"
MORGAN_STATUS = "chemaware_shared_v3_frozen_morgan_cache_complete"


class ChemAwareTokenStore:
    """Memory-mapped, provenance-checked contextual-token cache."""

    REQUIRED_FILES = (
        "report.json", "rows.npy", "tokens_f16.npy", "mz_f32.npy",
        "intensity_f32.npy", "valid.npy", "precursor_mz_f32.npy",
        "official_embeddings_f32.npy",
    )

    def __init__(
        self,
        directory: Path,
        graph_path: Path,
        official_checkpoint: Path,
        require_formal: bool = True,
    ) -> None:
        self.directory = Path(directory)
        missing = [name for name in self.REQUIRED_FILES if not (self.directory / name).exists()]
        if missing:
            raise FileNotFoundError(f"ChemAware token cache missing files: {missing}")
        self.report = json.loads((self.directory / "report.json").read_text(encoding="utf-8"))
        if self.report.get("status") != TOKEN_STATUS:
            raise RuntimeError("wrong token-cache status; historical noise caches are forbidden")
        if require_formal and self.report.get("formal") is not True:
            raise RuntimeError("formal training requires a formal, complete token cache")
        provenance = self.report.get("provenance", {})
        expected = {
            "graph_sha256": sha256_file(Path(graph_path)),
            "official_checkpoint_sha256": sha256_file(Path(official_checkpoint)),
        }
        mismatches = {
            key: (provenance.get(key), value)
            for key, value in expected.items() if provenance.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"ChemAware token-cache provenance mismatch: {mismatches}")

        self.rows = np.load(self.directory / "rows.npy")
        self.tokens = np.load(self.directory / "tokens_f16.npy", mmap_mode="r")
        self.mz = np.load(self.directory / "mz_f32.npy", mmap_mode="r")
        self.intensity = np.load(self.directory / "intensity_f32.npy", mmap_mode="r")
        self.valid = np.load(self.directory / "valid.npy", mmap_mode="r")
        self.precursor_mz = np.load(self.directory / "precursor_mz_f32.npy", mmap_mode="r")
        self.official_embeddings = np.load(
            self.directory / "official_embeddings_f32.npy", mmap_mode="r"
        )
        self._validate_arrays()
        self.position = {int(row): index for index, row in enumerate(self.rows)}
        self.dimension = int(self.official_embeddings.shape[1])

    def _validate_arrays(self) -> None:
        self.rows = np.asarray(self.rows, dtype=np.int64)
        n = len(self.rows)
        if self.rows.ndim != 1 or len(np.unique(self.rows)) != n:
            raise RuntimeError("token-cache rows must be a unique vector")
        if self.tokens.ndim != 3 or self.tokens.shape[0] != n:
            raise RuntimeError("malformed contextual token array")
        peak_shape = self.tokens.shape[:2]
        if any(array.shape != peak_shape for array in (self.mz, self.intensity, self.valid)):
            raise RuntimeError("peak token/measurement array shape mismatch")
        if self.precursor_mz.shape != (n,):
            raise RuntimeError("precursor m/z array shape mismatch")
        if self.official_embeddings.ndim != 2 or self.official_embeddings.shape[0] != n:
            raise RuntimeError("official embedding array shape mismatch")
        if self.tokens.shape[2] != self.official_embeddings.shape[1]:
            raise RuntimeError("token and embedding dimensions differ")
        if not np.all(np.asarray(self.valid).any(axis=1)):
            raise RuntimeError("every cached spectrum must contain a valid fragment")
        if not np.all(np.isfinite(np.asarray(self.precursor_mz))) or np.any(self.precursor_mz <= 0):
            raise RuntimeError("invalid precursor m/z values")
        sample = np.asarray(self.official_embeddings, dtype=np.float32)
        if not np.all(np.isfinite(sample)):
            raise RuntimeError("non-finite official embeddings")
        norm_error = float(np.max(np.abs(np.linalg.norm(sample, axis=1) - 1.0)))
        if norm_error > 2e-4:
            raise RuntimeError(f"official embeddings are not normalized (max error {norm_error})")

    def require_graph_coverage(self, graph: CandidateGraph) -> None:
        reachable = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row)))
        if not np.array_equal(np.sort(self.rows), reachable):
            missing = np.setdiff1d(reachable, self.rows)
            extra = np.setdiff1d(self.rows, reachable)
            raise RuntimeError(
                f"token cache does not exactly cover graph: missing={len(missing)}, extra={len(extra)}"
            )

    def verify_official_graph_scores(
        self,
        graph: CandidateGraph,
        tolerance: float = 5e-4,
    ) -> float:
        """Reconcile cached official embeddings against every graph pair."""
        query_position = np.asarray([self.position[int(row)] for row in graph.query_row])
        candidate_position = np.asarray([self.position[int(row)] for row in graph.pair_candidate_row])
        molecule_query = np.repeat(np.arange(graph.n_queries), np.diff(graph.query_ptr))
        pair_query = np.repeat(molecule_query, np.diff(graph.molecule_ptr))
        query = np.asarray(self.official_embeddings[query_position[pair_query]], dtype=np.float32)
        candidate = np.asarray(self.official_embeddings[candidate_position], dtype=np.float32)
        observed = np.einsum("ij,ij->i", query, candidate)
        expected = graph.features[:, graph.dreams_column]
        error = float(np.max(np.abs(observed - expected)))
        if error > tolerance:
            raise RuntimeError(
                f"official embedding cache disagrees with graph (max abs error {error:.6g})"
            )
        return error

    def tensors(
        self,
        rows: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = np.asarray(rows, dtype=np.int64)
        try:
            positions = np.asarray([self.position[int(row)] for row in rows], dtype=np.int64)
        except KeyError as error:
            raise RuntimeError(f"spectrum row absent from ChemAware token cache: {error}") from error
        return (
            torch.from_numpy(np.asarray(self.official_embeddings[positions])).to(device=device, dtype=torch.float32),
            torch.from_numpy(np.asarray(self.tokens[positions])).to(device=device, dtype=torch.float32),
            torch.from_numpy(np.asarray(self.mz[positions])).to(device=device, dtype=torch.float32),
            torch.from_numpy(np.asarray(self.intensity[positions])).to(device=device, dtype=torch.float32),
            torch.from_numpy(np.asarray(self.precursor_mz[positions])).to(device=device, dtype=torch.float32),
            torch.from_numpy(np.asarray(self.valid[positions])).to(device=device, dtype=torch.bool),
        )

    def adapt(self, adapter, rows: np.ndarray, device: torch.device):
        official, tokens, mz, intensity, precursor, valid = self.tensors(rows, device)
        adapted, delta, support, conflict = adapter(
            official, tokens, mz, intensity, precursor, valid
        )
        return official, adapted, delta, support, conflict


class MoleculeTeacherStore:
    """Frozen, graph-aligned molecule teacher with auditable controls."""

    def __init__(
        self,
        directory: Path,
        graph_path: Path,
        graph: CandidateGraph,
        require_formal: bool = True,
    ) -> None:
        self.directory = Path(directory)
        required = ("report.json", "molecule_ik14.npy", "molecule_formula.npy", "embeddings_f32.npy")
        missing = [name for name in required if not (self.directory / name).exists()]
        if missing:
            raise FileNotFoundError(f"molecule teacher cache missing files: {missing}")
        self.report = json.loads((self.directory / "report.json").read_text(encoding="utf-8"))
        if self.report.get("status") not in {MOLFORMER_STATUS, MORGAN_STATUS}:
            raise RuntimeError("invalid molecule teacher cache status")
        if require_formal and self.report.get("formal") is not True:
            raise RuntimeError("formal G2 requires a formal molecule teacher cache")
        if self.report.get("training_only") is not True:
            raise RuntimeError("molecule cache does not declare training-only use")
        expected_graph = sha256_file(Path(graph_path))
        if self.report.get("provenance", {}).get("graph_sha256") != expected_graph:
            raise RuntimeError("molecule teacher graph provenance mismatch")
        self.ik14 = np.load(self.directory / "molecule_ik14.npy").astype(str)
        self.formula = np.load(self.directory / "molecule_formula.npy").astype(str)
        self.embeddings = np.load(self.directory / "embeddings_f32.npy", mmap_mode="r")
        if (
            self.ik14.ndim != 1 or self.formula.shape != self.ik14.shape
            or self.embeddings.ndim != 2 or len(self.embeddings) != len(self.ik14)
            or len(np.unique(self.ik14)) != len(self.ik14)
        ):
            raise RuntimeError("malformed molecule teacher arrays")
        values = np.asarray(self.embeddings, dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("non-finite molecule teacher embeddings")
        norm_error = float(np.max(np.abs(np.linalg.norm(values, axis=1) - 1.0)))
        if norm_error > 2e-4:
            raise RuntimeError(f"molecule teacher embeddings are not normalized: {norm_error}")
        expected_ik14 = np.unique(graph.molecule_ik14.astype(str))
        if not np.array_equal(np.sort(self.ik14), expected_ik14):
            raise RuntimeError("molecule teacher does not exactly cover graph identities")
        self.index = {value: index for index, value in enumerate(self.ik14)}
        self.graph_index = np.asarray([self.index[value] for value in graph.molecule_ik14.astype(str)])
        if not np.array_equal(self.formula[self.graph_index], graph.molecule_formula.astype(str)):
            raise RuntimeError("molecule teacher formula ledger disagrees with graph")
        self.dimension = int(self.embeddings.shape[1])

    def identity_targets(
        self,
        graph: CandidateGraph,
        allowed_molecule: np.ndarray,
        control: str,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Return deterministic control targets aligned to ``self.ik14``.

        Controls operate on unique identities, so a molecule keeps the same
        pseudo-target wherever it recurs in the candidate graph.
        """
        allowed_molecule = np.asarray(allowed_molecule, dtype=bool)
        allowed_identity = np.unique(graph.molecule_ik14[allowed_molecule].astype(str))
        if len(allowed_identity) < 3:
            raise RuntimeError("teacher control needs at least three training identities")
        rng = np.random.default_rng(seed)
        source_index = np.asarray([self.index[value] for value in allowed_identity])
        source = np.asarray(self.embeddings[source_index], dtype=np.float32)
        availability = {value: True for value in allowed_identity}
        if control == "correct":
            mapping = {value: self.embeddings[self.index[value]] for value in allowed_identity}
            audit = {"control": control, "fixed_points": len(allowed_identity)}
        elif control == "identity_permuted":
            shift = int(rng.integers(1, len(allowed_identity)))
            target = np.roll(source, shift=shift, axis=0)
            mapping = {value: target[index] for index, value in enumerate(allowed_identity)}
            audit = {"control": control, "fixed_points": 0, "cyclic_shift": shift}
        elif control == "random_marginal":
            # Independently scramble coordinates per identity.  Norm and each
            # vector's empirical coordinate marginal are preserved, while the
            # cross-identity chemical geometry is destroyed.
            target = np.stack([values[rng.permutation(self.dimension)] for values in source])
            target /= np.clip(np.linalg.norm(target, axis=1, keepdims=True), 1e-12, None)
            mapping = {value: target[index] for index, value in enumerate(allowed_identity)}
            audit = {"control": control, "fixed_points": 0, "coordinate_marginal_preserved": True}
        elif control in {"correct_same_formula_scope", "same_formula_mismatched"}:
            identity_formula = {
                value: self.formula[self.index[value]] for value in allowed_identity
            }
            groups: dict[str, list[str]] = {}
            for value, formula in identity_formula.items():
                groups.setdefault(str(formula), []).append(value)
            mapping = {}
            for values in groups.values():
                values.sort()
                if len(values) < 2:
                    availability[values[0]] = False
                    continue
                shift = int(rng.integers(1, len(values)))
                for index, value in enumerate(values):
                    target_identity = (
                        value if control == "correct_same_formula_scope"
                        else values[(index + shift) % len(values)]
                    )
                    mapping[value] = self.embeddings[self.index[target_identity]]
            audit = {
                "control": control,
                "fixed_points": (
                    int(sum(availability.values()))
                    if control == "correct_same_formula_scope" else 0
                ),
                "same_formula_only": True,
                "singleton_formula_identities_unobservable": int(sum(
                    not value for value in availability.values()
                )),
            }
        else:
            raise ValueError(f"unsupported molecule teacher control: {control}")
        output = np.zeros((len(self.ik14), self.dimension), dtype=np.float32)
        assigned = np.zeros(len(output), dtype=bool)
        for identity in allowed_identity:
            if availability[identity]:
                index = self.index[identity]
                output[index] = mapping[identity]
                assigned[index] = True
        audit.update({
            "allowed_unique_identities": int(len(allowed_identity)),
            "assigned_unique_identities": int(np.sum(assigned)),
            "teacher_dimension": self.dimension,
        })
        return output, assigned, audit

    def graph_embeddings(
        self,
        graph: CandidateGraph,
        allowed_molecule: np.ndarray,
        control: str,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Return one deterministic teacher vector per allowed graph molecule."""

        allowed_molecule = np.asarray(allowed_molecule, dtype=bool)
        identity_values, identity_assigned, audit = self.identity_targets(
            graph, allowed_molecule, control, seed
        )
        output = np.asarray(identity_values[self.graph_index], dtype=np.float32).copy()
        assigned = identity_assigned[self.graph_index] & allowed_molecule
        output[~assigned] = 0.0
        audit["assigned_graph_molecules"] = int(np.sum(assigned))
        return output, assigned, audit


def identity_reference_centroids(
    graph: CandidateGraph,
    store: ChemAwareTokenStore,
    teacher: MoleculeTeacherStore,
) -> np.ndarray:
    """Official normalized reference-spectrum centroid for each teacher identity."""

    rows_by_identity: dict[str, set[int]] = {
        str(value): set() for value in teacher.ik14
    }
    for molecule_index, identity in enumerate(graph.molecule_ik14.astype(str)):
        left = int(graph.molecule_ptr[molecule_index])
        right = int(graph.molecule_ptr[molecule_index + 1])
        rows_by_identity[identity].update(map(int, graph.pair_candidate_row[left:right]))
    centroids = np.empty((len(teacher.ik14), store.dimension), dtype=np.float32)
    for index, identity in enumerate(teacher.ik14.astype(str)):
        rows = sorted(rows_by_identity[identity])
        if not rows:
            raise RuntimeError(f"teacher identity lacks reference spectra: {identity}")
        positions = [store.position[row] for row in rows]
        value = np.mean(np.asarray(store.official_embeddings[positions]), axis=0)
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"invalid official centroid for identity {identity}")
        centroids[index] = value / norm
    return centroids


def formula_folds(values: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Assign one stable fold per molecular formula without Python hash drift."""
    try:
        from noise_final_core import stable_fold
    except ModuleNotFoundError:  # pragma: no cover - package import path
        from .noise_final_core import stable_fold

    if folds < 3:
        raise ValueError("at least three formula folds are required")
    return np.asarray([stable_fold(str(value), folds, seed) for value in values], dtype=np.int8)


def split_allowed_molecules(
    graph: CandidateGraph,
    outer_fold: int,
    inner_fold: int,
    folds: int,
    fold_seed: int,
) -> np.ndarray:
    """Exclude validation/test-formula molecules from all training groups."""
    molecule_fold = formula_folds(graph.molecule_formula, folds, fold_seed)
    return (molecule_fold != outer_fold) & (molecule_fold != inner_fold)


def encode_all(adapter, store: ChemAwareTokenStore, device: torch.device, batch_size: int) -> np.ndarray:
    adapter.eval()
    output = np.empty((len(store.rows), store.dimension), dtype=np.float32)
    with torch.no_grad():
        for left in range(0, len(store.rows), batch_size):
            right = min(left + batch_size, len(store.rows))
            _, adapted, _, _, _ = store.adapt(adapter, store.rows[left:right], device)
            output[left:right] = adapted.cpu().numpy()
    return output


def ranks_and_margins_for_queries(
    encoded: np.ndarray,
    store: ChemAwareTokenStore,
    graph: CandidateGraph,
    query_subset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Strict ranks and positive-vs-best-negative molecule margins."""
    query_position = np.asarray([store.position[int(row)] for row in graph.query_row])
    candidate_position = np.asarray([store.position[int(row)] for row in graph.pair_candidate_row])
    molecule_query = np.repeat(np.arange(graph.n_queries), np.diff(graph.query_ptr))
    pair_query = np.repeat(molecule_query, np.diff(graph.molecule_ptr))
    pair_scores = np.einsum(
        "ij,ij->i", encoded[query_position[pair_query]], encoded[candidate_position]
    )
    molecule_scores = np.maximum.reduceat(pair_scores, graph.molecule_ptr[:-1])
    ranks = []
    margins = []
    for query in np.asarray(query_subset, dtype=np.int64):
        left, right = map(int, graph.query_ptr[query:query + 2])
        scores = molecule_scores[left:right]
        ranks.append(strict_rank(scores))
        margins.append(float(scores[0] - np.max(scores[1:])))
    return np.asarray(ranks, dtype=np.int32), np.asarray(margins, dtype=np.float32)


def ranks_for_queries(
    encoded: np.ndarray,
    store: ChemAwareTokenStore,
    graph: CandidateGraph,
    query_subset: np.ndarray,
) -> np.ndarray:
    """Strict molecule ranks using the exact deployment max-over-spectra rule."""

    return ranks_and_margins_for_queries(
        encoded, store, graph, query_subset
    )[0]


def paired_evaluation(
    encoded: np.ndarray,
    official_encoded: np.ndarray,
    store: ChemAwareTokenStore,
    graph: CandidateGraph,
    query_subset: np.ndarray,
) -> dict:
    query_subset = np.asarray(query_subset, dtype=np.int64)
    old_rank, old_margin = ranks_and_margins_for_queries(
        official_encoded, store, graph, query_subset
    )
    new_rank, new_margin = ranks_and_margins_for_queries(
        encoded, store, graph, query_subset
    )
    old = strict_metrics(old_rank, graph.query_has_near[query_subset])
    new = strict_metrics(new_rank, graph.query_has_near[query_subset])
    old_correct, new_correct = old_rank == 1, new_rank == 1
    preservation = np.einsum("ij,ij->i", official_encoded, encoded)
    return {
        "query": query_subset,
        "old_rank": old_rank,
        "new_rank": new_rank,
        "old_margin": old_margin,
        "new_margin": new_margin,
        "summary": {
            "n_queries": int(len(query_subset)),
            "baseline_recall1": old["recall1"],
            "recall1": new["recall1"],
            "delta_recall1": float(new["recall1"] - old["recall1"]),
            "baseline_mrr": old["mrr"],
            "mrr": new["mrr"],
            "delta_mrr": float(new["mrr"] - old["mrr"]),
            "corrected": int(np.sum(~old_correct & new_correct)),
            "introduced": int(np.sum(old_correct & ~new_correct)),
            "rank_changed_queries": int(np.sum(old_rank != new_rank)),
            "mean_rank_delta": float(np.mean(new_rank.astype(float) - old_rank)),
            "baseline_mean_margin": float(np.mean(old_margin)),
            "mean_margin": float(np.mean(new_margin)),
            "delta_mean_margin": float(np.mean(new_margin - old_margin)),
            "baseline_median_margin": float(np.median(old_margin)),
            "median_margin": float(np.median(new_margin)),
            "near_n": int(np.sum(graph.query_has_near[query_subset])),
            "baseline_near_recall1": old.get("near_recall1"),
            "near_recall1": new.get("near_recall1"),
            "delta_near_recall1": (
                float(new["near_recall1"] - old["near_recall1"])
                if "near_recall1" in old else None
            ),
            "preservation_mean": float(np.mean(preservation)),
            "preservation_min": float(np.min(preservation)),
            "preservation_q01": float(np.quantile(preservation, 0.01)),
            "preservation_q05": float(np.quantile(preservation, 0.05)),
        },
    }
