#!/usr/bin/env python
"""Freeze BioAware V6 by adding a scale-free identifiability gate to V4.

No outcome is used to fit a number.  The only new rule is mechanistic: a
proposed replacement must be the strict unique winner in at least one typed
biological mechanism (reaction network or structure network).  Opened panels
are replayed only as retrospective audits and are permanently excluded from
confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from bioaware_identifiable_router import apply_identifiable_router, weights_from_artifact


OPENED = ("BV2cell__hilic", "BV2cell__rplc", "Mouse_brain__hilic")
CONFIRMATORY = (
    "Mouse_brain__rplc", "Mouse_liver__hilic", "Mouse_liver__rplc",
    "NIST_plasma__hilic", "NIST_plasma__rplc",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replay(path: Path, v4: dict) -> dict:
    frame = pd.read_csv(path)
    gate = v4["router"]["gate"]
    result, _ = apply_identifiable_router(
        frame, weights_from_artifact(v4),
        maximum_spectral_margin=float(gate["maximum_spectral_margin"]),
        minimum_fusion_advantage=float(gate["minimum_fusion_advantage"]),
        minimum_support_families=int(gate["minimum_support_families"]),
    )
    return {
        "queries": int(len(result)),
        "baseline_recall1": float(result.baseline_correct.mean()),
        "recall1": float(result.final_correct.mean()),
        "delta_recall1": float(result.delta.mean()),
        "corrected": int(result.corrected.sum()),
        "introduced": int(result.introduced.sum()),
        "interventions": int(result.intervene.sum()),
        "biologically_identifiable_proposals": int(result.biologically_identifiable.sum()),
        "ledger_sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-artifact", type=Path, required=True)
    parser.add_argument("--audit-ledger", action="append", default=[],
                        help="NAME=PATH; retrospective only, never selection")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.v4_artifact.exists():
        raise FileNotFoundError(args.v4_artifact)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output: {args.output_dir}")
    v4 = json.loads(args.v4_artifact.read_text(encoding="utf-8"))
    if v4.get("status") != "bioaware_v4_high_precision_router_artifact_frozen":
        raise RuntimeError("unexpected V4 artifact")
    audits: dict[str, dict] = {}
    for item in args.audit_ledger:
        if "=" not in item:
            raise ValueError("--audit-ledger must be NAME=PATH")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        audits[name] = replay(path, v4)
    artifact = {
        "status": "bioaware_v6_identifiable_router_artifact_frozen",
        "formal": True,
        "router": {
            **v4["router"],
            "type": "rank_consensus_with_typed_biological_identifiability_abstention",
            "minimum_unique_biological_mechanisms": 1,
            "biological_mechanisms": {
                "reaction_network": ["family_known_reaction", "family_predicted_reaction"],
                "structure_network": ["family_structure_network"],
            },
            "identifiability_rule": (
                "proposed candidate must be the strict unique nonzero winner in at least "
                "one typed biological mechanism; RT, rules and decoder cannot activate"
            ),
            "fallback": "exact official DreaMS order",
        },
        "selection": {
            "numeric_refit": False,
            "new_numeric_thresholds": False,
            "rule_source": "mechanism identifiability failure observed before freezing V6",
            "retrospective_audits": audits,
            "audit_outcomes_used_for_selection": False,
        },
        "confirmatory_external_panels": {
            "excluded": list(OPENED), "required": list(CONFIRMATORY),
            "rule": (
                "zero refit; pooled formula-cluster CI lower bound >0; corrected > introduced; "
                "lambda=2 net >0; no panel degradation; real graph beats degree-preserving decoys"
            ),
        },
        "contracts": {
            "fit_performed": False, "threshold_tuning": False,
            "opened_panels_permanently_excluded": True, "P2b": "forbidden",
            "phenotype": "forbidden", "RT_cannot_activate": True,
            "rules_cannot_activate": True, "decoder_cannot_activate": True,
            "ties": "count against the positive exactly as the frozen baseline builder",
        },
        "provenance": {"v4_artifact_sha256": sha256(args.v4_artifact)},
        "claim_limit": (
            "V6 is frozen before five-panel confirmation. Retrospective audits are descriptive; "
            "only the five named untouched panels and graph-decoy tests can support a claim."
        ),
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "artifact.json"
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
