#!/usr/bin/env python
"""Static fail-closed checks for the BioAware B7 direct pipeline."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(path: Path, tokens: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    for token in tokens:
        if token not in text:
            raise AssertionError(f"{path.name} lacks contract token: {token}")


def string_constants_in_function(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(functions) != 1:
        raise AssertionError(f"expected one {function_name} in {path.name}")
    return {
        node.value for node in ast.walk(functions[0])
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def main() -> None:
    builder = ROOT / "tasks/build_bioaware_b7_graph_action_router.py"
    audit = ROOT / "tasks/audit_bioaware_b7_direct_gradient.py"
    trainer = ROOT / "tasks/train_bioaware_b4_direct_shared_embedding.py"
    require(builder, (
        'ARM = "graph_prior_direct"',
        '"action_routes.csv.gz"',
        '"graph_score_is_not_student_target": True',
        '"student_target_is_true_molecular_identity": True',
        '"inner_truth_identity_and_formula_purged": True',
        '"P2b_used": False',
    ))
    require(audit, (
        '"action_routes.csv.gz"',
        '"graph_score_or_probability_in_loss": False',
        '"true_identity_directly_supervises_ranking": True',
        '"direct_truth_objective_decreased"',
        'gates = {name: bool(value) for name, value in gates.items()}',
        'serialized = json.dumps(report, indent=2)',
        '"shared_query_reference_encoder": True',
        '"P2b_used": False',
    ))
    require(trainer, (
        '"graph_prior_direct": (None, None)',
        '"action_routes.csv.gz"',
        'forbids candidate-score distillation',
        'load_direct_action_routes(',
        'args.supervision_mode != "direct_onehot"',
        '**route_provenance',
    ))
    direct_sources = [path.read_text(encoding="utf-8") for path in (builder, audit)]
    if any('"teacher_actions.csv.gz"' in text for text in direct_sources):
        raise AssertionError("B7 direct producer/audit reverted to legacy teacher filename")
    direct_loader_strings = string_constants_in_function(
        trainer, "load_direct_action_routes"
    )
    legacy_loader_strings = string_constants_in_function(
        trainer, "load_candidate_teacher"
    )
    if "action_routes.csv.gz" not in direct_loader_strings:
        raise AssertionError("direct action loader is not bound to action_routes.csv.gz")
    if "teacher_actions.csv.gz" in direct_loader_strings:
        raise AssertionError("direct action loader is bound to the legacy teacher file")
    if "teacher_actions.csv.gz" not in legacy_loader_strings:
        raise AssertionError("legacy soft-teacher loader filename changed unexpectedly")
    # Exercise the exact class that caused the first server failure whenever
    # the scientific environment is available.  The lightweight local Python
    # may lack NumPy, so source checks above remain runnable during sync QA.
    try:
        import numpy as np
    except ModuleNotFoundError:
        pass
    else:
        payload = {"gates": {"example": bool(np.bool_(True))}}
        if json.loads(json.dumps(payload)) != {"gates": {"example": True}}:
            raise AssertionError("NumPy gate normalization is not JSON safe")
    print("[BioAware B7 direct pipeline checks] PASS")


if __name__ == "__main__":
    main()
