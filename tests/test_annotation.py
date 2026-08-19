"""A2 -- Unit tests for the annotation platform (annotation/ package).

Covers the deterministic, import-light core: params loading (7.9), retrieval
top-k (incl. GPU backend equivalence, 7.7), Schymanski confidence assignment,
target-decoy FDR q-values, and sample-level differential analysis. Nothing here
touches the network or a trained model, so the suite runs on CPU in seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `annotation` importable however pytest is invoked (rootdir vs python -m).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from annotation.confidence import assign_schymanski  # noqa: E402
from annotation.diff import confident_top1, differential, sample_counts  # noqa: E402
from annotation.fdr import (  # noqa: E402
    annotate_fdr,
    make_shuffle_decoys,
    target_decoy_qvalues,
)
from annotation.params import DEFAULT, load_params  # noqa: E402
from annotation.retrieve import (  # noqa: E402
    _normalize,
    chunked_topk,
    chunked_topk_torch,
    dppm,
)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------- #
# params.load_params (interface 7.9)
# --------------------------------------------------------------------------- #
def test_load_params_none_returns_default():
    assert load_params(None) is DEFAULT


def test_load_params_overrides_and_keeps_defaults(tmp_path):
    p = tmp_path / "p.json"
    p.write_text('{"cosine_confident": 0.75, "ppm_tolerance": 15.0}')
    out = load_params(p)
    assert out.cosine_confident == 0.75
    assert out.ppm_tolerance == 15.0
    assert out.topk == DEFAULT.topk == 10  # missing key keeps DEFAULT


def test_load_params_unknown_key_fails_fast(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"cosine_confidentt": 0.9}')
    with pytest.raises(ValueError, match="cosine_confident"):
        load_params(p)


def test_load_params_non_object_rejected(tmp_path):
    p = tmp_path / "list.json"
    p.write_text('[1, 2, 3]')
    with pytest.raises(ValueError, match="object"):
        load_params(p)


# --------------------------------------------------------------------------- #
# retrieve: dppm + top-k (incl. GPU backend 7.7)
# --------------------------------------------------------------------------- #
def test_dppm_formula():
    # |q - l| / |l| * 1e6
    got = dppm(np.array([100.0]), np.array([100.1]))[0]
    assert got == pytest.approx(0.1 / 100.1 * 1e6, rel=1e-6)


def test_chunked_topk_finds_self_and_is_descending():
    rng = np.random.default_rng(0)
    lib = _normalize(rng.standard_normal((50, 16)).astype(np.float32))
    query = lib[[3, 17]]  # already normalized
    vals, idx = chunked_topk(query, lib, k=3)
    assert idx[0, 0] == 3 and idx[1, 0] == 17
    assert vals[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert np.all(np.diff(vals, axis=1) <= 0)  # descending


def test_chunked_topk_chunk_invariance():
    rng = np.random.default_rng(1)
    q = _normalize(rng.standard_normal((37, 32)).astype(np.float32))
    lib = _normalize(rng.standard_normal((100, 32)).astype(np.float32))
    v1, i1 = chunked_topk(q, lib, 5, chunk=1)
    v2, i2 = chunked_topk(q, lib, 5, chunk=100)
    # Values are numerically equal but not bit-identical: different chunk shapes
    # dispatch to different BLAS kernels, drifting the last float32 bit (~1e-7).
    assert np.allclose(v1, v2, atol=1e-6)
    assert np.array_equal(i1, i2)


def test_gpu_backend_matches_cpu():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    q = _normalize(rng.standard_normal((50, 24)).astype(np.float32))
    lib = _normalize(rng.standard_normal((80, 24)).astype(np.float32))
    vn, in_ = chunked_topk(q, lib, 10)
    vt, it = chunked_topk_torch(q, lib, 10, device="cpu")
    assert np.allclose(vn, vt, atol=1e-4)
    assert np.array_equal(in_, it)


# --------------------------------------------------------------------------- #
# confidence.assign_schymanski
# --------------------------------------------------------------------------- #
def test_schymanski_levels():
    hits = pd.DataFrame({
        "rank": [1, 1, 1, 1, 2],
        "query_idx": [0, 1, 2, 3, 0],
        "cosine": [0.8, 0.6, 0.4, 0.4, 0.99],
        "mz_pass": [True, True, True, True, True],
    })
    rules = np.array([False, False, False, True])  # query_idx 3 has rule evidence
    out = assign_schymanski(hits, DEFAULT, rules_evidence=rules)
    # row0 L2a; row1 L3 (cos>=dark 0.5); row2 L5; row3 L3 (rule evidence); row4 rank!=1 -> L5
    assert out["schymanski_level"].tolist() == [2, 3, 5, 3, 5]
    assert set(out["schymanski_level"]) <= {2, 3, 5}  # never emit Level 1 or 4


# --------------------------------------------------------------------------- #
# fdr: decoys + q-values
# --------------------------------------------------------------------------- #
def test_shuffle_decoys_preserve_structure():
    records = [
        {"peaks": np.vstack([np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])]).astype(np.float32),
         "precursor_mz": 123.4},
        {"peaks": np.vstack([np.array([4.0, 5.0]), np.array([40.0, 50.0])]).astype(np.float32),
         "precursor_mz": 567.8},
    ]
    decoys = make_shuffle_decoys(records, n_decoys_per_target=2, seed=0)
    assert len(decoys) == 4
    for i, d in enumerate(decoys):
        src = records[i % 2]
        assert d["precursor_mz"] == src["precursor_mz"]
        assert np.array_equal(d["peaks"][0], src["peaks"][0])  # m/z axis intact
        assert sorted(d["peaks"][1]) == sorted(src["peaks"][1])  # intensities permuted


def test_qvalue_known_case():
    target = np.array([0.9, 0.5])
    decoy = np.array([0.8, 0.4, 0.3])
    q = target_decoy_qvalues(target, decoy)
    # highest 0.9: N_decoy>=0.9 = 0, N_target>=0.9 = 2 -> (0+1)/(2+1) = 1/3
    # lowest  0.5: N_decoy>=0.5 = 1, N_target>=0.5 = 1 -> (1+1)/(1+1) = 1 -> min(1/3, 1) = 1/3
    assert q[0] == pytest.approx(1 / 3)
    assert q[1] == pytest.approx(1 / 3)


def test_qvalue_bounded_and_monotonic():
    rng = np.random.default_rng(3)
    target = rng.uniform(0.3, 1.0, 200).astype(np.float32)
    decoy = rng.uniform(0.0, 0.7, 400).astype(np.float32)
    q = target_decoy_qvalues(target, decoy)
    assert q.shape == target.shape
    assert np.all(q >= 0) and np.all(q <= 1)
    # higher target score -> lower/equal q
    assert np.all(np.diff(q[np.argsort(target)]) <= 0)


def test_annotate_fdr_attaches_columns():
    hits = pd.DataFrame({"query_idx": [0, 0, 1], "rank": [1, 2, 1]})
    target = np.array([0.9, 0.5], dtype=np.float32)
    decoy = np.array([0.4, 0.4], dtype=np.float32)
    out = annotate_fdr(hits, target, decoy, DEFAULT)
    assert {"qvalue", "fdr_pass"} <= set(out.columns)
    assert out["fdr_pass"].dtype == bool
    assert out["qvalue"].isna().sum() == 0


# --------------------------------------------------------------------------- #
# diff: sample-level presence/absence + Fisher
# --------------------------------------------------------------------------- #
def test_confident_top1_filters():
    hits = pd.DataFrame({
        "rank": [1, 1, 1, 2, 1],
        "cosine": [0.8, 0.6, 0.9, 0.95, 0.75],
        "mz_pass": [True, True, False, True, True],
        "fdr_pass": [True, True, True, True, False],
    })
    assert len(confident_top1(hits, DEFAULT)) == 2  # rows 0, 4
    assert len(confident_top1(hits, DEFAULT, fdr_pass=True)) == 1  # row 0 only


def test_sample_counts_presence_absence():
    conf = pd.DataFrame({
        "lib_inchikey": ["X", "X", "X"],
        "_group": ["A", "A", "B"],
        "query_file": ["A1", "A1", "B1"],  # A1 seen twice -> one sample
    })
    sc = sample_counts(conf, "_group")
    x = sc[sc["lib_inchikey"] == "X"]
    assert x.loc[x["_group"] == "A", "n_samples"].iloc[0] == 1  # not 2
    assert x.loc[x["_group"] == "B", "n_samples"].iloc[0] == 1


def test_differential_sample_level():
    conf = pd.DataFrame({
        "lib_inchikey": ["X", "X", "X", "X", "Y", "Y", "Y"],
        "lib_name": ["x"] * 7,
        "_group": ["A", "A", "A", "B", "A", "A", "A"],
        "query_file": ["A1", "A2", "A3", "B1", "A1", "A2", "A3"],
        "rank": [1] * 7,
        "cosine": [0.9] * 7,
        "mz_pass": [True] * 7,
    })
    res = differential(conf, "_group", "A", "B", DEFAULT, total_a=4, total_b=4)
    x = res[res["lib_inchikey"] == "X"].iloc[0]
    assert x["n_samples_A"] == 3 and x["n_samples_B"] == 1
    y = res[res["lib_inchikey"] == "Y"].iloc[0]
    assert y["n_samples_A"] == 3 and y["n_samples_B"] == 0
    assert res["q_value"].between(0, 1).all()
    assert res["q_value"].is_monotonic_increasing  # sorted by q ascending
