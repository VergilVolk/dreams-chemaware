from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tasks" / "prepare_kgmn_hidden_seed_state.R"


def test_hidden_seed_state_filters_both_identity_bearing_objects() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "keep_csv <- seed_ik14 %in% selected_polarity" in text
    assert 'methods::slot(object, "annotation_result") <- frame' in text
    assert "keep <- ids %in% selected_polarity" in text
    assert "hidden identity leaked into initial state" in text
    assert "non-whitelisted identities remain in initial state" in text


def test_hidden_seed_state_does_not_copy_identity_library_caches() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'for (name in c("ms2", "ms1_data"))' in text
    assert "lib_meta" not in text
    assert "lib_spec" not in text
    assert "hidden_identities_remain_available_in_mrn = TRUE" in text
    assert "hidden identity appears in label-free execution cache" in text
    assert "hidden_identity_labels_in_execution_caches = FALSE" in text


def test_hidden_seed_state_preserves_all_public_homogeneous_ms2_inputs() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "length(supported) < 1" in text
    assert "length(supported_extensions) != 1" in text
    assert "file.symlink" in text
    assert "all_homogeneous_ms2_inputs_preserved = TRUE" in text
    assert "expected exactly one supported MS2 input" not in text


def test_server_preflight_requests_one_gpu_and_no_explicit_memory() -> None:
    text = (ROOT / "tasks" / "run_kgmn_oep003284_preflight.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH --mem" not in text


def test_author_runner_continues_from_filtered_state_without_seed_rematch() -> None:
    text = (ROOT / "tasks" / "run_kgmn_metdna2_46std_hidden_seed_author.R").read_text(encoding="utf-8")
    assert "is_anno_initial_seed = FALSE" in text
    assert 'test_evaluation = "46STD"' in text
    assert 'scoring_approach_recursive = "dp"' in text
    assert 'is_credential = TRUE' in text
    assert 'is_cred_pg_filter = TRUE' in text
    assert 'is_cred_formula_filter = TRUE' in text
    assert "hidden_seed_state_audit.json" in text
    assert "hidden_identity_leakage" in text
    assert "dreams_edge_used = FALSE" in text
    assert "length(ms2_files) < 1" in text
    assert "length(extensions) != 1" in text


def test_edge_arm_changes_only_dynamic_ms2_edge_reliability() -> None:
    text = (ROOT / "tasks" / "run_kgmn_metdna2_46std_hidden_seed_arm.R").read_text(
        encoding="utf-8"
    )
    assert "is_anno_initial_seed = FALSE" in text
    assert 'test_evaluation = "46STD"' in text
    assert 'candidate_generation = "author"' in text
    assert 'reaction_network = "author"' in text
    assert 'propagation_depth = "author"' in text
    assert 'is_credential = TRUE' in text
    assert 'is_cred_formula_filter = TRUE' in text
    assert "MetDNA2.recursive_edge_score_hook" in text
    assert "hidden_identity_leakage" in text
    assert "P2b" not in text
    assert "length(ms2_files) < 1" in text
    assert "length(ms2_extensions) != 1" in text


def test_prediction_export_reconciles_final_score_with_internal_round() -> None:
    text = (ROOT / "tasks" / "export_kgmn_hidden_seed_predictions.R").read_text(
        encoding="utf-8"
    )
    assert '"list_identification"' in text
    assert '"table_identification"' in text
    assert "candidate_score = scores$candidate_score" in text
    assert "propagation_depth = as.integer(scores$propagation_depth)" in text
    assert "failed to reconcile final score with propagation depth" in text
