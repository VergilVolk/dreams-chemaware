from __future__ import annotations

import io

from audit_lcnec_hsst3n_acquisition import classify, stream_mzml


def test_classify_injection_ledger() -> None:
    assert classify("Study sample") == "study"
    assert classify("QC sample") == "pooled_qc"
    assert classify("Method blank") == "blank"
    assert classify("Serial dilution of QC sample (1/4)") == "qc_dilution"


def test_stream_mzml_reads_level_precursor_collision_and_minutes() -> None:
    document = b'''<?xml version="1.0" encoding="UTF-8"?>
    <mzML xmlns="http://psi.hupo.org/ms/mzml">
      <run><spectrumList count="2">
        <spectrum id="scan=1">
          <cvParam accession="MS:1000511" value="1"/>
          <scanList><scan><cvParam accession="MS:1000016" value="1.5" unitName="minute"/></scan></scanList>
        </spectrum>
        <spectrum id="scan=2">
          <cvParam accession="MS:1000511" value="2"/>
          <scanList><scan><cvParam accession="MS:1000016" value="2.0" unitName="minute"/></scan></scanList>
          <precursorList><precursor><selectedIonList><selectedIon>
            <cvParam accession="MS:1000744" value="321.12345"/>
          </selectedIon></selectedIonList><activation>
            <cvParam accession="MS:1000045" value="25"/>
          </activation></precursor></precursorList>
        </spectrum>
      </spectrumList></run>
    </mzML>'''
    result = stream_mzml(io.BytesIO(document))
    assert result["ms1"] == 1
    assert result["ms2"] == 1
    assert result["ms2_with_precursor"] == 1
    assert result["unique_precursors_4dp"] == 1
    assert result["precursor_min"] == 321.12345
    assert result["collision_energy_median"] == 25.0
    assert result["rt_max_sec"] == 120.0
