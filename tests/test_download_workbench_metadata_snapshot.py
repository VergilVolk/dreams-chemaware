from tasks.download_workbench_metadata_snapshot import ENDPOINTS


def test_required_endpoints_are_unique():
    assert len(ENDPOINTS) == len(set(ENDPOINTS))
    assert {"summary", "analysis", "mwtab", "files"}.issubset(ENDPOINTS)
