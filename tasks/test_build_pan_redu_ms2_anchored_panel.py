from __future__ import annotations

import numpy as np

from build_pan_redu_ms2_anchored_panel import cluster_anchors


def test_anchor_support_counts_biological_units_not_replicate_files() -> None:
    observations = {
        "mz": np.asarray([300.0000, 300.0010, 300.0015, 300.0018, 450.0]),
        "rt": np.asarray([100.0, 102.0, 103.0, 104.0, 500.0]),
        "tic": np.asarray([1.0, 2.0, 10.0, 3.0, 5.0]),
        "file": np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
        # Files 2 and 3 are technical/repeated acquisitions of the same unit.
        "unit": np.asarray([0, 1, 2, 2, 3], dtype=np.int32),
        "local": np.asarray([0, 0, 0, 0, 0], dtype=np.int32),
    }
    anchors = cluster_anchors(observations, minimum_units=3, precursor_ppm=10.0, rt_seconds=8.0)
    assert len(anchors) == 1
    anchor = anchors.iloc[0]
    assert anchor["supporting_biological_units"] == 3
    assert anchor["supporting_files"] == 4
    assert anchor["representative_file_index"] == 2
    assert anchor["cluster_mz_span_ppm"] <= 10.0
    assert anchor["cluster_rt_span_seconds"] <= 8.0

