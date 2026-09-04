import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT = Path("tasks/freeze_kgmn_metdna2_baseline.py")


def build_fixture(root: Path, complete: bool = True) -> None:
    pd.DataFrame({"name": ["a", "b"], "mz": [100.0, 101.0], "rt": [1.0, 2.0]}).to_csv(
        root / "data.csv", index=False
    )
    (root / "spectra.msp").write_text("NAME: a\nNum Peaks: 1\n50 1\n", encoding="utf-8")
    pd.DataFrame({"sample.name": ["s1"], "group": ["g"]}).to_csv(
        root / "sample.info.csv", index=False
    )
    pd.DataFrame(
        {"name": ["a", "b"], "mz": [100.0, 101.0], "rt": [1.0, 2.0], "adduct": ["[M-H]-", "[M-H]-"]}
    ).to_csv(root / "peak_table_annotated_200STD_neg_200805.csv", index=False)
    pd.DataFrame(
        {"name": ["a", "b", "b"], "id": ["S1", "S2", "S3"], "adduct": ["[M-H]-"] * 3}
    ).to_csv(root / "annotation_initial.csv", index=False)
    (root / "frozen_parameters.R").write_text("list(version='1.2.10')\n", encoding="utf-8")
    (root / "source_commit.txt").write_text("5685ab219269c2f35cd5087655b0470b2da4d93c\n", encoding="utf-8")
    genform = root / "_runtime_genform" / "GenForm"
    genform.parent.mkdir(parents=True)
    genform.write_bytes(b"\x7fELFfixture")
    (root / "run.log.txt").write_text(
        "Initial seed annotation\n"
        "Metabolic reaction network based metabolite annotation\n"
        "Annotaion Credential\n"
        "Merge and export result tables\n",
        encoding="utf-8",
    )
    seed = root / "01_result_initial_seed_annotation" / "ms2_match_annotation_result.csv"
    seed.parent.mkdir(parents=True)
    pd.DataFrame({"name": ["a"]}).to_csv(seed, index=False)
    credential = root / "03_annotation_credential" / "annontation_credential_long.csv"
    credential.parent.mkdir(parents=True)
    pd.DataFrame({"name": ["a"], "id": ["S1"]}).to_csv(credential, index=False)
    if complete:
        table = root / "00_annotation_table"
        table.mkdir(parents=True)
        pd.DataFrame({"name": ["a"]}).to_csv(table / "table1_identification.csv", index=False)
        pd.DataFrame(
            {
                "peak_name": ["a", "a_b", "b_a", "b_b"],
                "id_zhulab": ["S1", "S9", "S2", "S3"],
                "total_score": [0.9, 0.2, 0.8, 0.7],
            }
        ).to_csv(table / "table3_identification_pair.csv", index=False)


def test_freezes_complete_author_run(tmp_path: Path) -> None:
    build_fixture(tmp_path)
    subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((tmp_path / "frozen_author_baseline.json").read_text(encoding="utf-8"))
    assert report["status"] == "kgmn_metdna2_200std_author_baseline_frozen"
    assert report["contracts"]["author_algorithm_modified"] is False
    assert report["counts"]["input_features"] == 2
    assert report["author_200std_truth_evaluation"]["recall1"] == 1.0
    assert report["author_200std_truth_evaluation"]["recall5"] == 1.0


def test_fails_closed_on_incomplete_run(tmp_path: Path) -> None:
    build_fixture(tmp_path, complete=False)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "incomplete author baseline" in result.stderr


def test_truth_evaluation_counts_top_score_identity_ties_against_top1(tmp_path: Path) -> None:
    build_fixture(tmp_path)
    final_path = tmp_path / "00_annotation_table" / "table3_identification_pair.csv"
    final = pd.read_csv(final_path)
    final.loc[final["peak_name"].isin(["b_a", "b_b"]), "total_score"] = 0.8
    final.to_csv(final_path, index=False)
    subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((tmp_path / "frozen_author_baseline.json").read_text(encoding="utf-8"))
    metrics = report["author_200std_truth_evaluation"]
    assert metrics["ambiguous_top_ties"] == 1
    assert metrics["recall1"] == 0.5
    assert metrics["recall5"] == 1.0
