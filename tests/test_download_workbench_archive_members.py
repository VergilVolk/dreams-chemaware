import json
from pathlib import Path

from tasks.download_workbench_archive_members import inventory


def test_inventory_flattens_archives(tmp_path):
    path = tmp_path / "files.json"
    path.write_text(
        json.dumps({"compressed_file_content": {"a.zip": [{"name": "x.raw", "size": 9}]}}),
        encoding="utf-8",
    )
    assert inventory(path) == {"x.raw": 9}


def test_ledger_merge_contract_is_present():
    source = Path("tasks/download_workbench_archive_members.py").read_text(
        encoding="utf-8"
    )
    assert "download ledger conflict" in source
    assert "for key in sorted(existing_records)" in source
    assert 'suffix + ".partial"' in source
