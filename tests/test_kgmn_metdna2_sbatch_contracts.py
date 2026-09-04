from __future__ import annotations

from pathlib import Path


def test_author_baseline_captures_log_before_freezing() -> None:
    script = Path("tasks/run_kgmn_metdna2_200std_baseline.sbatch").read_text(encoding="utf-8")
    run_position = script.index("Rscript tasks/run_kgmn_metdna2_200std_baseline.R")
    log_position = script.index('mv "$TEMPORARY_LOG" "$OUT/run.log.txt"')
    freeze_position = script.index("python -u tasks/freeze_kgmn_metdna2_baseline.py")
    assert run_position < log_position < freeze_position
    assert "2>&1 | tee" in script[run_position:log_position]
    assert "set -euo pipefail" in script


def test_every_formal_kgmn_sbatch_requests_one_gpu_and_no_memory_override() -> None:
    scripts = sorted(Path("tasks").glob("run_kgmn*.sbatch"))
    assert scripts
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert "#SBATCH --gpus=1" in text, path
        assert "#SBATCH --mem" not in text, path


def test_oep_xcms_stage_is_outcome_blind_and_fail_closed() -> None:
    script = Path("tasks/run_kgmn_oep003284_xcms.sbatch").read_text(encoding="utf-8")
    assert "--require-local" in script
    assert "preprocess_kgmn_oep003284_xcms.R" in script
    assert "positive negative" in script
    assert "--allow-author-mapped-mzxml" in script
    assert "MTBLS13729" not in script
    r_script = Path("tasks/preprocess_kgmn_oep003284_xcms.R").read_text(encoding="utf-8")
    assert "CentWaveParam" in r_script
    assert "ObiwarpParam" in r_script
    assert "PeakDensityParam" in r_script
    assert "hidden_seed_split_used = FALSE" in r_script
    assert "phenotype_blind = TRUE" in r_script


def test_formal_oep_inputs_use_published_author_peak_tables_not_xcms() -> None:
    script = Path("tasks/run_kgmn_oep003284_author_inputs.sbatch").read_text(encoding="utf-8")
    assert "prepare_kgmn_oep003284_author_inputs.py" in script
    assert "freeze_kgmn_external_validation_contract.py" in script
    assert "preprocess_kgmn_oep003284_xcms.R" not in script
    assert "--require-local" in script
    assert "--require-dreams-edge" in script


def test_propagation_arms_fail_closed_on_edge_calibration_gate() -> None:
    script = Path("tasks/run_kgmn_metdna2_dreams_arms.sbatch").read_text(encoding="utf-8")
    gate = "official_dreams_eligible_for_dynamic_propagation_test"
    assert gate in script
    assert script.index(gate) < script.index("run_arm noop_author")


def test_external_hidden_seed_runs_author_and_both_frozen_edge_arms() -> None:
    script = Path("tasks/run_kgmn_oep003284_hidden_seed.sbatch").read_text(encoding="utf-8")
    assert "--require-dreams-edge" in script
    assert "--allow-author-mapped-mzxml" in script
    assert "export_kgmn_metdna2_mapped_ms2.R" in script
    assert '--spectra "$mapped_msp"' in script
    assert '--ms1-table "$initial_dir/data.csv"' in script
    assert "official_dreams_eligible_for_dynamic_propagation_test" in script
    assert "run_author 0" in script
    assert "run_edge_arm noop_author 0" in script
    assert "audit_kgmn_hidden_seed_noop.py" in script
    assert '"$ROOT/noop_audit_$polarity.json"' in script
    assert "run_edge_arm official_dreams" in script
    assert "run_edge_arm author_official_intersection" in script
    assert script.index("audit_kgmn_hidden_seed_noop.py") < script.index("for repeat in $(seq 0 9)")
    assert "MTBLS13729" not in script


def test_author_mapped_ms2_export_is_identifier_preserving_and_label_free() -> None:
    text = Path("tasks/export_kgmn_metdna2_mapped_ms2.R").read_text(encoding="utf-8")
    assert 'loaded, "ms2"' in text
    assert 'extract_info_value(entry$info, "NAME")' in text
    assert "mapped-MS2 list name/info NAME mismatch" in text
    assert "mapped-MS2 names absent from MS1 table" in text
    assert 'writeLines(paste0("NAME: ", feature_names[[index]])' in text
    assert "identity_labels_used = FALSE" in text
    assert "phenotype_used = FALSE" in text
