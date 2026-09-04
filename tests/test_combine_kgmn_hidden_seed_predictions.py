from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tasks.combine_kgmn_hidden_seed_predictions import main


def write_shards(root: Path, repeats: int = 2) -> None:
    root.mkdir()
    for repeat in range(repeats):
        for polarity in ("positive", "negative"):
            pd.DataFrame(
                {
                    "repeat": [repeat],
                    "truth_inchikey1": ["ABCDEFGHIJKLMN"],
                    "candidate_inchikey1": ["ABCDEFGHIJKLMN"],
                    "candidate_score": [2.0],
                    "propagation_depth": [1],
                    "polarity": [polarity],
                    "peak_name": [f"p{repeat}_{polarity}"],
                    "adduct": ["[M+H]+"],
                }
            ).to_csv(root / f"repeat_{repeat:02d}_{polarity}.csv", index=False)


def test_combines_exact_shard_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shards = tmp_path / "shards"
    write_shards(shards)
    output = tmp_path / "combined.csv.gz"
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        ["combine", "--shard-dir", str(shards), "--output", str(output), "--report", str(report), "--repeats", "2"],
    )
    main()
    frame = pd.read_csv(output)
    assert len(frame) == 4
    assert report.is_file()


def test_missing_shard_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shards = tmp_path / "shards"
    write_shards(shards)
    (shards / "repeat_01_negative.csv").unlink()
    monkeypatch.setattr(
        "sys.argv",
        ["combine", "--shard-dir", str(shards), "--output", str(tmp_path / "out.gz"), "--report", str(tmp_path / "r.json"), "--repeats", "2"],
    )
    with pytest.raises(FileNotFoundError):
        main()
