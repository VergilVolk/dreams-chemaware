from pathlib import Path

from tasks.extract_timsbench_minimal_groundtruth import MEMBER_SUFFIXES, file_sha256


def test_member_contract_is_small_and_specific():
    assert len(MEMBER_SUFFIXES) == 8
    assert len(set(MEMBER_SUFFIXES)) == len(MEMBER_SUFFIXES)
    assert all(
        "groundtruth_dataset/" in value or "library_spectra/" in value
        for value in MEMBER_SUFFIXES
    )
    assert not any("raw/" in value for value in MEMBER_SUFFIXES)


def test_sha256(tmp_path: Path):
    target = tmp_path / "x.bin"
    target.write_bytes(b"abc")
    assert file_sha256(target) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )
