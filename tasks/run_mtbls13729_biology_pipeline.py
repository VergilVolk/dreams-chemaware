#!/usr/bin/env python
"""Checkpointed orchestrator for the MTBLS13729 biology analysis line."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / "tasks" / script), *args]
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-requantification", action="store_true")
    parser.add_argument("--skip-candidate-audit", action="store_true")
    parser.add_argument("--skip-ms2-link", action="store_true")
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    args = parser.parse_args()
    panels = ["--panels", *args.panels]

    if not args.skip_extraction:
        run("extract_mtbls13729_ms1_features.py", *panels)
    run("build_mtbls13729_ms1_consensus.py", *panels)
    if not args.skip_requantification:
        run("requantify_mtbls13729_targeted_eic.py", *panels)
    run("analyze_mtbls13729_paired_ms1.py", *panels)
    if not args.skip_candidate_audit:
        if "neg_rp" in args.panels:
            run("audit_mtbls13729_discovery_candidates.py", "--panel", "neg_rp")
            run("audit_mtbls13729_candidate_isotopes.py", "--panel", "neg_rp")
        if "pos_rp" in args.panels:
            run(
                "audit_mtbls13729_discovery_candidates.py",
                "--panel",
                "pos_rp",
                "--priority-table",
                "data/mtbls13729/ms1_paired_analysis/pos_rp__discovery_priority_features.csv",
            )
            run(
                "audit_mtbls13729_candidate_isotopes.py",
                "--panel",
                "pos_rp",
                "--priority-table",
                "data/mtbls13729/ms1_paired_analysis/pos_rp__discovery_priority_features.csv",
            )
            run("group_mtbls13729_ion_families.py", "--panel", "pos_rp")
        run("prioritize_mtbls13729_biology_candidates.py", *panels)
        run("find_mtbls13729_candidate_ms2.py", *panels)
    if not args.skip_ms2_link:
        run("link_mtbls13729_ms2_annotations_to_ms1.py", *panels)
        run("analyze_mtbls13729_reaction_pairs.py", *panels)


if __name__ == "__main__":
    main()
