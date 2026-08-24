import numpy as np

from annotation.retrieve import chunked_mz_constrained_topk


def test_mass_candidate_is_ranked_before_global_cosine_distractors():
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    # First three spectra are globally more similar but outside the mass window.
    library = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.8, 0.6]], dtype=np.float32)
    library /= np.linalg.norm(library, axis=1, keepdims=True)
    values, indices, valid = chunked_mz_constrained_topk(
        query,
        np.asarray([100.0]),
        library,
        np.asarray([200.0, 300.0, 400.0, 100.0005]),
        k=2,
        ppm_tolerance=10.0,
    )
    assert valid[0, 0]
    assert indices[0, 0] == 3
    assert np.isfinite(values[0, 0])
    assert not valid[0, 1]
    assert indices[0, 1] == -1

