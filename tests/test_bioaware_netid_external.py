from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "tasks" / "fetch_bioaware_netid_external.py"
SPEC = importlib.util.spec_from_file_location("fetch_bioaware_netid_external", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_safe_member_rejects_traversal_and_windows_paths() -> None:
    for value in ["../escape", "/absolute", "C:/drive", "folder\\file"]:
        with pytest.raises(RuntimeError):
            MODULE._safe_member_path(value)
    assert MODULE._safe_member_path("NetID/FDR_example/raw_data.csv").as_posix() == (
        "NetID/FDR_example/raw_data.csv"
    )


def test_required_suffixes_must_be_unique(tmp_path: Path) -> None:
    root = tmp_path / "release"
    for suffix in MODULE.REQUIRED_SUFFIXES:
        path = root / "wrapper" / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(suffix, encoding="utf-8")
    located = MODULE._locate_required(root)
    assert set(located) == set(MODULE.REQUIRED_SUFFIXES)
    duplicate = root / "another" / "LICENSE"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text("duplicate", encoding="utf-8")
    with pytest.raises(RuntimeError):
        MODULE._locate_required(root)


def test_zip_member_symlink_mode_is_detectable(tmp_path: Path) -> None:
    archive = tmp_path / "test.zip"
    info = zipfile.ZipInfo("link")
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(info, "target")
    with zipfile.ZipFile(archive) as bundle:
        member = bundle.infolist()[0]
        assert ((member.external_attr >> 16) & 0o170000) == 0o120000
