from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from build_contrast_atlas_control_worlds import filter_world, write_world
from test_contrast_atlas_core import fixtures


def test_pretruth_stack_runs_without_truth_and_preserves_worlds(tmp_path: Path) -> None:
    effects, loadings, links = fixtures()
    control_root = tmp_path / "controls"
    all_keys = set(zip(effects["study_id"], effects["contrast_id"], effects["entity_id"]))
    known_rows = effects.loc[effects["identity_status"].ne("unresolved_entity")]
    known_keys = set(zip(known_rows["study_id"], known_rows["contrast_id"], known_rows["entity_id"]))
    matched_rows = known_rows.groupby(["study_id", "contrast_id"], sort=False).head(1)
    matched_keys = set(zip(matched_rows["study_id"], matched_rows["contrast_id"], matched_rows["entity_id"]))
    worlds = {
        "all": filter_world(effects, loadings, links, all_keys),
        "known_only": filter_world(effects, loadings, links, known_keys),
        "measurement_matched_known": filter_world(effects, loadings, links, matched_keys),
        "unresolved_shuffled_ms2": filter_world(effects, loadings, links, all_keys),
    }
    for name, tables in worlds.items():
        write_world(control_root / name, *tables)
    (control_root / "report.json").write_text('{"status":"synthetic"}\n', encoding="utf-8")
    universe_path = tmp_path / "universe.csv"
    registry_path = tmp_path / "registry.csv"
    effects.to_csv(universe_path, index=False)
    pd.DataFrame([
        {
            "query_id": "q", "query_study_id": "Q", "query_contrast_id": "q",
            "cluster_id": "cq", "source_holdout": "sq", "instrument_holdout": "iq",
            "data_role": "synthetic",
        },
        {
            "query_id": "l", "query_study_id": "L", "query_contrast_id": "l",
            "cluster_id": "cl", "source_holdout": "sl", "instrument_holdout": "il",
            "data_role": "synthetic",
        },
    ]).to_csv(registry_path, index=False)
    output = tmp_path / "pretruth"
    script = Path(__file__).with_name("run_contrast_atlas_pretruth_stack.py")
    subprocess.run([
        sys.executable, str(script),
        "--control-root", str(control_root),
        "--universe-contrast-records", str(universe_path),
        "--query-registry", str(registry_path),
        "--output-dir", str(output),
        "--min-patterns", "2",
        "--null-draws", "99",
    ], check=True, capture_output=True, text=True)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    candidates = pd.read_csv(output / "calibrated_candidates.csv.gz")
    assert report["truth_opened"] is False
    assert report["candidate_universe_invariant"] is True
    assert set(candidates["world"]) == set(worlds)
    assert candidates.groupby(["query_id", "world"]).size().nunique() == 1
