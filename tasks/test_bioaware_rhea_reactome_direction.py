from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation.bioaware_relations import _direction_label  # noqa: E402
from tasks.build_bioaware_rhea_reactome_direction_cache import consensus_semantics  # noqa: E402


def main() -> None:
    assert consensus_semantics(pd.Series(["LR", "LR"])) == "reactome_consensus_lr"
    assert consensus_semantics(pd.Series(["RL"])) == "reactome_consensus_rl"
    assert consensus_semantics(pd.Series(["LR", "RL"])) == "reactome_consensus_bidirectional"
    assert consensus_semantics(pd.Series(["LR", "UN"])) == "reaction_direction_unknown"
    assert consensus_semantics(pd.Series(["UN"])) == "reaction_direction_unknown"
    assert _direction_label("reactome_consensus_lr", "left", "right") == "reaction_forward"
    assert _direction_label("reactome_consensus_rl", "left", "right") == "reaction_reverse"
    assert _direction_label("reactome_consensus_bidirectional", "left", "right") == "reaction_bidirectional"
    assert _direction_label("reaction_direction_unknown", "left", "right") == "reaction_direction_unknown"
    assert _direction_label("canonical_lr_not_physiological", "left", "right") == "reaction_direction_unknown"
    print("[test_bioaware_rhea_reactome_direction] PASS")


if __name__ == "__main__":
    main()
