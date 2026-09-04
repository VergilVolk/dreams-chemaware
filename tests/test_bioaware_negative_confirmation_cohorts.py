from tasks.audit_bioaware_negative_confirmation_cohorts import assess_candidate


def complete_candidate(**updates):
    facts = {
        "cohort": "synthetic",
        "negative_ms2_locally_available": True,
        "independent_structure_truth": True,
        "truth_identities": 20,
        "candidate_search_reconstructable": True,
        "sample_network_context_available": True,
        "used_for_v2_development": False,
    }
    facts.update(updates)
    return assess_candidate(facts)


def test_confirmation_requires_every_gate() -> None:
    assert complete_candidate()["ready_for_frozen_performance_confirmation"]
    for field, value in [
        ("negative_ms2_locally_available", False),
        ("independent_structure_truth", False),
        ("truth_identities", 19),
        ("candidate_search_reconstructable", False),
        ("sample_network_context_available", False),
        ("used_for_v2_development", True),
    ]:
        assert not complete_candidate(**{field: value})["ready_for_frozen_performance_confirmation"]


def test_opened_development_data_cannot_pass() -> None:
    result = complete_candidate(used_for_v2_development=True, truth_identities=1000)
    assert not result["gates"]["not_used_for_v2_development"]
    assert not result["ready_for_frozen_performance_confirmation"]
