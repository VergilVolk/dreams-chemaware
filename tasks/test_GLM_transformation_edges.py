"""Contract tests for GLM transformation-edge module.

Pure-logic tests on synthetic entities and effects; no real data, no GPU.
"""
from __future__ import annotations

import numpy as np

import GLM_transformation_edges as te


def _entity(fid: int, mass: float, peaks: list) -> te.SpectralEntity:
    return te.SpectralEntity(
        family_id=fid, neutral_mass=mass, precursor_mz=mass + 1.00794,
        ms2_peaks=tuple(sorted(peaks)),
    )


def _peaks(n: int = 10, seed: int = 0, base: float = 100.0) -> list:
    r = np.random.default_rng(seed)
    return [(float(base + 20 * i + r.uniform(0, 2)), float(r.uniform(20, 100)))
            for i in range(n)]


def test_pmd_table_has_expected_transformations() -> None:
    table = te.TRANSFORMATION_PMD_TABLE
    assert any("glucuronidation" in k for k in table)
    assert any("sulfation" in k for k in table)
    assert any("methylation" in k for k in table)
    assert any("hydroxylation" in k for k in table)
    assert any("dehydrogenation" in k for k in table)
    assert all(abs(v) > 0.5 for v in table.values()), "PMDs must be > 0.5 Da"


def test_edge_construction_basic() -> None:
    a = _entity(1, 200.0000, _peaks(seed=1))
    b = _entity(2, 214.0150, _peaks(seed=1))  # +CH2 = 14.0157
    c = _entity(3, 300.0000, _peaks(seed=2))
    edges = te.edge_candidates([a, b, c], ppm_tolerance=50)
    pmds = {e.pmd_name for e in edges}
    assert any("methylation" in p for p in pmds), f"expected methylation, got {pmds}"
    assert all(e.u != e.v for e in edges)


def test_edge_no_false_positive_at_tight_ppm() -> None:
    a = _entity(1, 200.0000, _peaks(seed=1))
    b = _entity(2, 214.5000, _peaks(seed=1))  # 14.5 Da off, no match
    edges = te.edge_candidates([a, b], ppm_tolerance=20)
    assert len(edges) == 0, f"expected no edges for unmatched mass diff, got {len(edges)}"


def test_fragment_support_identical_spectra() -> None:
    peaks = _peaks(seed=42, n=5)
    a = _entity(1, 200.0, peaks)
    b = _entity(2, 214.0157, peaks)  # same peaks (structurally related)
    assert te.fragment_support(a, b) is True


def test_fragment_support_unrelated_spectra() -> None:
    a = _entity(1, 200.0, [(100.0, 50.0), (150.0, 80.0), (200.0, 90.0)])
    b = _entity(2, 214.0157, [(50.0, 60.0), (75.0, 70.0), (300.0, 90.0)])
    assert te.fragment_support(a, b) is False


def test_edge_differentials_basic() -> None:
    effects = np.array([[1.0, 2.0], [0.5, 1.5], [-1.0, 0.0]])
    family_ids = [1, 2]
    edges = [te.TransformationEdge(
        u=1, v=2, pmd_name="test", pmd_expected=14.0157,
        pmd_observed=14.0157, ppm_error=0.0, fragment_shared=True)]
    diffs = te.edge_differentials(effects, family_ids, edges)
    g = diffs[(1, 2)]
    assert len(g) == 3
    assert g[0] == 1.0  # 2.0 - 1.0
    assert g[1] == 1.0  # 1.5 - 0.5
    assert g[2] == 1.0  # 0.0 - (-1.0)


def test_patient_consistency_all_same_direction() -> None:
    g = np.array([0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.1, 0.9, 0.3])
    result = te.patient_consistency(g, alternative="greater")
    assert result["n"] == 10
    assert result["n_positive"] == 10
    assert result["p_value"] < 0.01


def test_patient_consistency_random_directions() -> None:
    rng = np.random.default_rng(42)
    g = rng.choice([-0.5, 0.5], size=20)
    result = te.patient_consistency(g, alternative="greater")
    assert result["p_value"] > 0.05


def test_patient_consistency_too_few_patients() -> None:
    result = te.patient_consistency(np.array([0.5, -0.3]))
    assert result["p_value"] == 1.0


def test_transformation_edge_summary() -> None:
    a = _entity(1, 200.0, _peaks(seed=1))
    b = _entity(2, 214.0157, _peaks(seed=1))
    edges = te.edge_candidates([a, b], ppm_tolerance=50)
    assert len(edges) >= 1
    effects = np.array([[1.0, 2.0]] * 10)  # 10 patients, consistent +1
    family_ids = [1, 2]
    diffs = te.edge_differentials(effects, family_ids, edges)
    summary = te.transformation_edge_summary(edges, diffs)
    assert summary["n_edges"] >= 1
    assert summary["edges"][0]["sign_test_p"] <= 0.05


def test_glucuronidation_edge() -> None:
    a = _entity(1, 300.0000, _peaks(seed=3))
    b = _entity(2, 476.0321, _peaks(seed=3))  # +176.0321
    edges = te.edge_candidates([a, b], ppm_tolerance=20)
    pmds = {e.pmd_name for e in edges}
    assert any("glucuronidation" in p for p in pmds), f"expected glucuronidation, got {pmds}"


def test_sulfation_edge() -> None:
    a = _entity(1, 250.0000, _peaks(seed=4))
    b = _entity(2, 329.9568, _peaks(seed=4))  # +79.9568
    edges = te.edge_candidates([a, b], ppm_tolerance=20)
    pmds = {e.pmd_name for e in edges}
    assert any("sulfation" in p for p in pmds), f"expected sulfation, got {pmds}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("GLM transformation-edge contracts passed")
