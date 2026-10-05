"""Freeze the ChemAware MassBank-layered corpus, evidence and exact code."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = (
    "tasks/build_chemaware_massbank_annotated_fragment_corpus.py",
    "tasks/build_chemaware_massbank_candidate_source_ledger.py",
    "tasks/qualify_chemaware_candidate_source_ledger.py",
    "tasks/build_chemaware_dynamic_reference_native_triplets.py",
    "tasks/build_chemaware_layered_10k_native_pool.py",
    "tasks/build_chemaware_multisource_native_triplets.py",
    "tasks/build_chemaware_sirius_native_triplets.py",
    "tasks/build_chemaware_max_boundary_native_triplets.py",
    "tasks/build_chemaware_dreams_native_triplets.py",
    "tasks/build_chemaware_layered_rule_source_ledger.py",
    "tasks/build_chemaware_motifdb_candidate_source_ledger.py",
    "tasks/import_chemaware_msfinder_rule_tables.py",
    "tasks/train_chemaware_dreams_native.py",
    "tasks/evaluate_chemaware_v2_direct_triplet.py",
    "tasks/select_chemaware_residual_checkpoint.py",
    "tasks/run_chemaware_massbank_layered_native.sbatch",
    "tasks/test_chemaware_massbank_annotated_source.py",
    "tasks/test_chemaware_dynamic_reference_native_triplets.py",
    "tasks/test_chemaware_massbank_layered_pipeline_contracts.py",
    "tasks/test_chemaware_massbank_layered_runtime.py",
    "tasks/freeze_chemaware_massbank_layered_release.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layered", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def checked_report(directory: Path, expected: str) -> dict:
    path = directory / "report.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("status") != expected:
        raise RuntimeError(f"unexpected report status in {path}: {body.get('status')}")
    return body


def checked_qualification(directory: Path) -> dict:
    path = directory / "qualification_report.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("status") != "CHEMAWARE_SOURCE_SPECIFICITY_QUALIFICATION_COMPLETE":
        raise RuntimeError(f"unexpected qualification status in {path}: {body.get('status')}")
    return body


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    layered = checked_report(args.layered, "CHEMAWARE_LAYERED_10K_NATIVE_CORPUS_COMPLETE")
    corpus = checked_report(args.corpus, "CHEMAWARE_MASSBANK_ANNOTATED_FRAGMENT_CORPUS_COMPLETE")
    development = checked_qualification(args.development)
    confirmation = checked_qualification(args.confirmation)
    dynamic = checked_report(args.dynamic, "CHEMAWARE_DYNAMIC_REFERENCE_NATIVE_TRIPLETS_COMPLETE")
    if not development.get("admitted_families") or not confirmation.get("admitted_families"):
        raise RuntimeError("MassBank source families were not independently confirmed")
    if not all(layered["gates"].values()) or not all(dynamic["gates"].values()):
        raise RuntimeError("frozen triplet gates did not all pass")

    core_files = {
        "layered_train_pool": args.layered / "train_pool.npz",
        "layered_validation_pool": args.layered / "val_pool.npz",
        "layered_event_manifest": args.layered / "corpus_event_manifest.tsv",
        "chemical_event_ledger": args.layered / "chemical_source_event_ledger.tsv",
        "layered_report": args.layered / "report.json",
        "massbank_rule_corpus": args.corpus / "massbank_fragment_corpus.json",
        "massbank_rule_report": args.corpus / "report.json",
        "development_qualified_ledger_report": args.development / "report.json",
        "development_qualification": args.development / "qualification_report.json",
        "confirmation_qualified_ledger_report": args.confirmation / "report.json",
        "confirmation_qualification": args.confirmation / "qualification_report.json",
        "qualified_candidate_scores": args.confirmation / "candidate_scores.tsv",
        "dynamic_triplet_report": args.dynamic / "report.json",
    }
    missing = [str(path) for path in core_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"release inputs are missing: {missing}")
    code_paths = {name: ROOT / name for name in CODE_FILES}
    missing_code = [name for name, path in code_paths.items() if not path.is_file()]
    if missing_code:
        raise FileNotFoundError(f"release code is missing: {missing_code}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_massbank_release_", dir=args.output.parent))
    try:
        code_dir = temporary / "code_snapshot"
        evidence_dir = temporary / "evidence_snapshot"
        code_dir.mkdir()
        evidence_dir.mkdir()
        for name, path in code_paths.items():
            target = code_dir / Path(name).name
            shutil.copy2(path, target)
        shutil.copy2(args.corpus / "report.json", evidence_dir / "massbank_corpus_report.json")
        shutil.copy2(
            args.development / "qualification_report.json",
            evidence_dir / "development_qualification_report.json",
        )
        shutil.copy2(
            args.confirmation / "qualification_report.json",
            evidence_dir / "confirmation_qualification_report.json",
        )
        shutil.copy2(args.dynamic / "report.json", evidence_dir / "dynamic_triplet_report.json")
        shutil.copy2(args.layered / "report.json", evidence_dir / "layered_corpus_report.json")
        manifest = {
            "status": "CHEMAWARE_MASSBANK_LAYERED_RELEASE_FROZEN",
            "scientific_claim": (
                "16,272 recurrent MassBank rules yielded two formula-disjoint confirmed source "
                "families and 1,658 active chemical native events inside a 12,000-event DreaMS pool"
            ),
            "performance_claim": "none until role-2 selection and untouched role-3 confirmation",
            "phasea_initialization": "protected +2.1266 pp embedding checkpoint",
            "counts": {
                "rules": corpus["records"],
                "development_admitted_families": development["admitted_families"],
                "confirmation_admitted_families": confirmation["admitted_families"],
                "dynamic_chemical_events": dynamic["dynamic_chemical_events_added"],
                "dynamic_chemical_queries": dynamic["dynamic_chemical_queries"],
                "native_triplet_capacity": dynamic["sampled_native_triplet_capacity"],
                "layered_events": layered["events"],
                "layered_strata": layered["strata"],
            },
            "core_files": {
                name: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for name, path in core_files.items()
            },
            "code_snapshot": {
                path.name: {"source": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                for name, path in code_paths.items()
            },
        }
        (temporary / "release_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": manifest["status"],
        "output": str(args.output),
        "core_files": len(core_files),
        "code_files": len(code_paths),
        "release_manifest_sha256": sha256(args.output / "release_manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
