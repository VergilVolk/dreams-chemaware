#!/usr/bin/env python
"""Synthetic-table checks for the B47 gate remediation audit."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tasks" / "audit_bioaware_b47_gate_remediation.py"


def synthetic_dir(root: Path, *, dominated: bool) -> Path:
    u3 = root / ("u3_dominated" if dominated else "u3_diverse")
    u3.mkdir()
    rng = np.random.default_rng(11)
    n_pool = 400
    features = pd.DataFrame({
        "query_id": np.repeat(np.arange(n_pool), 2).astype(str),
        "candidate_id": [f"C{i:04d}" for i in range(n_pool * 2)],
        "reference_spectra": rng.integers(1, 40, n_pool * 2),
        "candidate_catalogue_degree": rng.integers(0, 200, n_pool * 2),
    })
    n_actions = 1600 if not dominated else 1600
    if dominated:
        candidates = np.array(["HUB"] * 900 + [f"D{i:03d}" for i in range(700)])
        formulas = np.array(["FHUB"] * 1200 + [f"FD{i:03d}" for i in range(400)])
        studies = np.array(["ST001122"] * 1500 + ["ST003356"] * 100)
        advantage = np.linspace(1.0, 0.01, n_actions)
    else:
        candidates = np.array([f"D{i:03d}" for i in range(80)]).repeat(20)
        formulas = np.array([f"FD{i:03d}" for i in range(80)]).repeat(20)
        studies = np.tile(["ST001122", "ST003356"], n_actions // 2)
        advantage = np.linspace(1.0, 0.5, n_actions)
    opportunities = pd.DataFrame({
        "query_id": [f"q{i:05d}" for i in range(n_actions)],
        "study": studies,
        "event_top_candidate": candidates,
        "event_top_formula": formulas,
        "event_advantage": advantage,
        "candidate_specific_intervention_opportunity": True,
        "event_top_reference_spectra": rng.integers(1, 15, n_actions),
        "event_top_catalogue_degree": rng.integers(0, 90, n_actions),
    })
    decoys = pd.DataFrame({
        "query_id": [f"x{i:05d}" for i in range(200)],
        "study": "ST001122",
        "event_top_candidate": "NONE",
        "event_top_formula": "NONE",
        "event_advantage": 0.0,
        "candidate_specific_intervention_opportunity": False,
        "event_top_reference_spectra": 5,
        "event_top_catalogue_degree": 5,
    })
    pd.concat([opportunities, decoys], ignore_index=True).to_csv(
        u3 / "query_event_opportunities.csv.gz", index=False, compression="gzip"
    )
    features.to_csv(u3 / "candidate_event_features.csv.gz", index=False, compression="gzip")
    (u3 / "report.json").write_text(json.dumps({
        "protocol_version": "U3-v3-reaction-signature-20260922",
        "nulls_passed": True,
    }), encoding="utf-8")
    return u3


def run(u3: Path, out: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-u", str(SCRIPT), "--u3-dir", str(u3), "--output", str(out)],
        capture_output=True, text=True, check=True,
    )
    return json.loads((out / "report.json").read_text(encoding="utf-8"))


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)

        diverse = run(synthetic_dir(root, dominated=False), root / "out_diverse")
        assert diverse["authorization_after_repair"] is True, diverse
        assert diverse["raw_intervention_actions"] == 1600
        gates = diverse["gates"]
        assert gates["R1_concentration"]["pass"] is True
        assert gates["R3_headroom"]["required_actions"] == 1560
        assert gates["R5_null_supremacy_inherited"] is True

        dominated = run(synthetic_dir(root, dominated=True), root / "out_dominated")
        assert dominated["authorization_after_repair"] is False
        assert dominated["gates"]["R1_concentration"]["pass"] is False
        # The 25% candidate cap must cut the 900-action HUB down.
        ledger = pd.read_csv(root / "out_dominated" / "repaired_action_ledger.csv.gz")
        assert int((ledger["event_top_candidate"] == "HUB").sum()) <= 400
        assert dominated["gates"]["R4_study_materiality"]["pass"] is False

        # Provenance mismatch must fail closed.
        bad = root / "u3_diverse"
        report = json.loads((bad / "report.json").read_text(encoding="utf-8"))
        report["protocol_version"] = "U3-v2-invalid"
        (bad / "report.json").write_text(json.dumps(report), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-u", str(SCRIPT), "--u3-dir", str(bad),
             "--output", str(root / "out_bad")],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0 and "provenance mismatch" in proc.stderr

    print("[test_audit_bioaware_b47_gate_remediation] PASS", flush=True)


if __name__ == "__main__":
    main()
