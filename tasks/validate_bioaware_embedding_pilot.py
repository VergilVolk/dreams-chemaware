#!/usr/bin/env python
"""Decision gate before the five-fold BioAware embedding run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-rank-gradient", type=float, default=1e-7)
    parser.add_argument("--minimum-relation-gradient", type=float, default=1e-7)
    args = parser.parse_args()
    body = json.loads(args.report.read_text(encoding="utf-8"))
    history = body["training"]["history"]
    relation_epochs = [row for row in history if row.get("grad_relation", 0.0) > 0]
    gates = {
        "rank_gradient_reaches_adapter": max(row.get("grad_rank", 0.0) for row in history) >= args.minimum_rank_gradient,
        "relation_gradient_reaches_adapter": bool(relation_epochs) and max(row["grad_relation"] for row in relation_epochs) >= args.minimum_relation_gradient,
        "relation_gradient_capped": all(0.0 < row.get("relation_multiplier", 0.0) <= 1.0 for row in history),
        "embedding_preserved": body["heldout"]["preservation_mean"] >= 0.995,
        "heldout_recall_nondegrading": body["heldout"]["delta_recall1"] >= 0,
        "heldout_near_nondegrading": body["heldout"]["delta_near_recall1"] >= 0,
        "corrected_ge_introduced": body["heldout"]["corrected"] >= body["heldout"]["introduced"],
    }
    result = {
        "status": "bioaware_embedding_pilot_decision",
        "report": str(args.report),
        "gates": gates,
        "pass_to_five_fold": all(gates.values()),
        "claim_limit": "A passing pilot only authorizes five-fold training; it is not an embedding improvement claim.",
    }
    print(json.dumps(result, indent=2), flush=True)
    if not result["pass_to_five_fold"]:
        raise RuntimeError(f"BioAware embedding pilot failed: {gates}")


if __name__ == "__main__":
    main()

