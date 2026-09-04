import base64
import zlib
from pathlib import Path

import numpy as np

from tasks.metdna3_mzml import iter_ms2_spectra, iter_spectrum_metadata


def test_stream_metadata_normalises_minutes_to_seconds(tmp_path: Path) -> None:
    source = tmp_path / "tiny.mzML"
    source.write_text(
        '''<?xml version="1.0" encoding="UTF-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml"><run><spectrumList count="1">
<spectrum id="scan=7" defaultArrayLength="12">
<cvParam accession="MS:1000511" value="2"/>
<scanList count="1"><scan><cvParam accession="MS:1000016" value="1.5" unitAccession="UO:0000031"/></scan></scanList>
<precursorList count="1"><precursor><selectedIonList count="1"><selectedIon>
<cvParam accession="MS:1000744" value="321.123"/><cvParam accession="MS:1000041" value="1"/>
</selectedIon></selectedIonList><activation><cvParam accession="MS:1000045" value="30"/></activation></precursor></precursorList>
</spectrum></spectrumList></run></mzML>''',
        encoding="utf-8",
    )
    rows = list(iter_spectrum_metadata(source))
    assert rows == [{
        "spectrum_id": "scan=7", "ms_level": 2, "rt_sec": 90.0,
        "precursor_mz": 321.123, "charge": 1.0, "collision_energy": 30.0,
        "n_peaks": 12,
    }]


def test_decode_zlib_float32_ms2(tmp_path: Path) -> None:
    mz = np.asarray([50.0, 75.25], dtype="<f4")
    intensity = np.asarray([10.0, 2.5], dtype="<f4")
    encode = lambda value: base64.b64encode(zlib.compress(value.tobytes())).decode()
    source = tmp_path / "binary.mzML"
    source.write_text(
        f'''<?xml version="1.0"?><mzML xmlns="http://psi.hupo.org/ms/mzml"><run><spectrumList count="1">
<spectrum id="scan=1" defaultArrayLength="2"><cvParam accession="MS:1000511" value="2"/>
<scanList><scan><cvParam accession="MS:1000016" value="2" unitAccession="UO:0000010"/></scan></scanList>
<precursorList><precursor><selectedIonList><selectedIon><cvParam accession="MS:1000744" value="100"/></selectedIon></selectedIonList></precursor></precursorList>
<binaryDataArrayList count="2"><binaryDataArray><cvParam accession="MS:1000521"/><cvParam accession="MS:1000574"/><cvParam accession="MS:1000514"/><binary>{encode(mz)}</binary></binaryDataArray>
<binaryDataArray><cvParam accession="MS:1000521"/><cvParam accession="MS:1000574"/><cvParam accession="MS:1000515"/><binary>{encode(intensity)}</binary></binaryDataArray></binaryDataArrayList>
</spectrum></spectrumList></run></mzML>''', encoding="utf-8")
    rows = list(iter_ms2_spectra(source))
    assert len(rows) == 1
    np.testing.assert_allclose(rows[0]["mz"], mz)
    np.testing.assert_allclose(rows[0]["intensity"], intensity)
    assert rows[0]["rt_sec"] == 2.0
