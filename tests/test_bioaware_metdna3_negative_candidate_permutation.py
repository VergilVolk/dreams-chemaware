import numpy as np
import pandas as pd

from tasks.audit_bioaware_metdna3_negative_candidate_permutation import (
    permute_candidate_feature_blocks,
)


def test_joint_candidate_permutation_preserves_each_query_multiset() -> None:
    frame = pd.DataFrame({
        "query_id": ["q1", "q1", "q1", "q2", "q2"],
        "a": [1, 2, 3, 8, 9], "b": [10, 20, 30, 80, 90],
    })
    result = permute_candidate_feature_blocks(frame, ["a", "b"], np.random.default_rng(3))
    for query in ("q1", "q2"):
        before = sorted(map(tuple, frame.loc[frame.query_id.eq(query), ["a", "b"]].to_numpy()))
        after = sorted(map(tuple, result.loc[result.query_id.eq(query), ["a", "b"]].to_numpy()))
        assert before == after
