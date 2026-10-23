import numpy as np
import pandas as pd
from scipy import sparse

from run_contrast_atlas_sparse_local_alignment import freeze_sparse_graphs


def test_sparse_graphs_keep_identity_worlds_and_thresholds_separate() -> None:
    studies = ["A", "B"]
    universes = {
        "A": pd.DataFrame({
            "entity_id": ["A::known", "A::unknown"],
            "identity_status": ["library_supported", "unresolved_entity"],
        }),
        "B": pd.DataFrame({
            "entity_id": ["B::known", "B::unknown"],
            "identity_status": ["library_supported", "unresolved_entity"],
        }),
    }
    spectra = {
        "A": (
            np.asarray(["A::known", "A::unknown"]),
            sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ),
        "B": (
            np.asarray(["B::known", "B::unknown"]),
            sparse.csr_matrix([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32),
        ),
    }
    graphs, edges = freeze_sparse_graphs(
        studies, universes, spectra, chunk_rows=1, minimum_cosine=0.70
    )

    known = graphs[("A", "B", "known")]
    unresolved = graphs[("A", "B", "unresolved")]
    all_world = graphs[("A", "B", "all")]
    assert known["left_to_right"] == {"A::known": ("B::known", 1.0)}
    assert known["right_to_left"] == {"B::known": ("A::known", 1.0)}
    assert unresolved["left_to_right"] == {}
    assert unresolved["right_to_left"] == {}
    # A known query may select a chemically related unresolved target only in all.
    assert all_world["left_to_right"]["A::known"][0] == "B::known"
    assert set(edges["evidence_world"]) == {"all", "known"}


def test_sparse_graph_tie_breaking_is_entity_order_stable() -> None:
    universes = {
        "A": pd.DataFrame({"entity_id": ["A::1"], "identity_status": ["unresolved_entity"]}),
        "B": pd.DataFrame({"entity_id": ["B::1", "B::2"], "identity_status": ["unresolved_entity", "unresolved_entity"]}),
    }
    spectra = {
        "A": (np.asarray(["A::1"]), sparse.csr_matrix([[1.0, 0.0]], dtype=np.float32)),
        "B": (np.asarray(["B::1", "B::2"]), sparse.csr_matrix([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)),
    }
    graphs, _ = freeze_sparse_graphs(["A", "B"], universes, spectra, 1, 0.70)
    assert graphs[("A", "B", "all")]["left_to_right"]["A::1"][0] == "B::1"
