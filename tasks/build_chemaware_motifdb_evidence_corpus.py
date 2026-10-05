"""Build a candidate-conditioned ChemAware corpus from pinned MS2LDA MotifDB.

MotifDB contains fragment/neutral-loss co-occurrence motifs and, for a subset,
several automatically proposed structures.  This builder keeps only motifs
supported by at least three distinct valid structures and derives a consensus
set of unhashed Morgan environments.  The consensus is a candidate predicate,
not an identity label: downstream scoring also suppresses exact source-identity
matches and must beat matched controls on formula-disjoint queries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motifdb",
        type=Path,
        default=(
            ROOT
            / "data/validation/chemaware_public_fragmentation_sources_v1_20260928"
            / "ms2lda_motifdb/motifDB.json"
        ),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=(
            ROOT
            / "data/validation/chemaware_public_fragmentation_sources_v1_20260928"
            / "source_manifest.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-source-structures", type=int, default=3)
    parser.add_argument("--consensus-fraction", type=float, default=0.8)
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--minimum-consensus-bits", type=int, default=3)
    parser.add_argument("--maximum-ms2-accuracy", type=float, default=0.01)
    parser.add_argument("--maximum-fragments", type=int, default=32)
    parser.add_argument("--maximum-losses", type=int, default=24)
    parser.add_argument("--minimum-feature-intensity", type=float, default=0.005)
    return parser.parse_args()


def finite_features(
    masses: Any, intensities: Any, minimum_intensity: float, maximum: int,
) -> list[dict[str, float]]:
    pairs = []
    for mass, intensity in zip(masses or [], intensities or []):
        try:
            mass_value = float(mass)
            intensity_value = float(intensity)
        except (TypeError, ValueError):
            continue
        if (
            math.isfinite(mass_value)
            and math.isfinite(intensity_value)
            and mass_value > 0
            and intensity_value >= minimum_intensity
        ):
            pairs.append((mass_value, intensity_value))
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return [
        {"mz": round(mass, 6), "weight": float(intensity)}
        for mass, intensity in pairs[:maximum]
    ]


def canonical_structures(values: Any) -> list[tuple[str, str, Chem.Mol]]:
    output: dict[str, tuple[str, str, Chem.Mol]] = {}
    for value in values or []:
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            continue
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        try:
            ik14 = Chem.MolToInchiKey(molecule).split("-", 1)[0]
        except Exception:
            continue
        output[ik14] = (canonical, ik14, molecule)
    return [output[key] for key in sorted(output)]


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if (
        args.minimum_source_structures < 3
        or not 0.5 <= args.consensus_fraction <= 1
        or args.minimum_consensus_bits < 1
        or args.maximum_ms2_accuracy > 0.02
    ):
        raise ValueError("MotifDB evidence thresholds were weakened")
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    expected_hash = source_manifest["sources"]["ms2lda_motifdb"]["sha256"]
    observed_hash = sha256_file(args.motifdb)
    if observed_hash != expected_hash:
        raise RuntimeError("MotifDB differs from the acquired pinned source")
    body = json.loads(args.motifdb.read_text(encoding="utf-8"))
    motifs = body.get("ms2")
    if not isinstance(motifs, list):
        raise RuntimeError("MotifDB ms2 table is missing")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=args.morgan_radius)
    records = []
    excluded = Counter()
    motifset_counts = Counter()
    for source_index, motif in enumerate(motifs):
        accuracy = float(motif.get("ms2accuracy") or math.inf)
        if not math.isfinite(accuracy) or accuracy > args.maximum_ms2_accuracy:
            excluded["low_resolution"] += 1
            continue
        structures = canonical_structures(motif.get("auto_annotation"))
        if len(structures) < args.minimum_source_structures:
            excluded["insufficient_structure_support"] += 1
            continue
        annotation = str(motif.get("annotation") or "").strip()
        short_annotation = str(motif.get("short_annotation") or "").strip()
        if annotation in {"", "No annotation available"}:
            excluded["missing_annotation"] += 1
            continue
        fragments = finite_features(
            motif.get("frag_mz"), motif.get("frag_intens"),
            args.minimum_feature_intensity, args.maximum_fragments,
        )
        losses = finite_features(
            motif.get("loss_mz"), motif.get("loss_intens"),
            args.minimum_feature_intensity, args.maximum_losses,
        )
        if len(fragments) + len(losses) < 2:
            excluded["insufficient_spectral_features"] += 1
            continue
        bit_counts: Counter[int] = Counter()
        for _, _, molecule in structures:
            bit_counts.update(int(bit) for bit in generator.GetSparseFingerprint(molecule).GetOnBits())
        threshold = math.ceil(args.consensus_fraction * len(structures))
        consensus_bits = sorted(bit for bit, count in bit_counts.items() if count >= threshold)
        if len(consensus_bits) < args.minimum_consensus_bits:
            excluded["insufficient_consensus_structure"] += 1
            continue
        motifset = str(motif.get("motifset") or "unknown")
        motifset_counts[motifset] += 1
        records.append({
            "record_id": f"MS2LDA:{motifset}:{motif.get('motif_id', source_index)}",
            "source_index": source_index,
            "source_motif_id": str(motif.get("motif_id") or source_index),
            "motifset": motifset,
            "annotation": annotation,
            "short_annotation": short_annotation,
            "charge": int(motif.get("charge") or 0),
            "ms2accuracy": accuracy,
            "fragments": fragments,
            "neutral_losses": losses,
            "source_structure_count": len(structures),
            "source_structure_ik14": [ik14 for _, ik14, _ in structures],
            "consensus_morgan_bits": consensus_bits,
            "consensus_support_required": threshold,
            "confidence_tier": "C_calibration_required",
            "training_policy": "formula_disjoint_matched_control_qualification_only",
        })
    if not records:
        raise RuntimeError("no MotifDB record passed the frozen quality filters")

    corpus = {
        "schema": "chemaware.ms2lda-motif-evidence.v1",
        "purpose": "truth-blind candidate selection for native DreaMS triplets",
        "source": source_manifest["sources"]["ms2lda_motifdb"],
        "fingerprint": {
            "kind": "RDKit sparse Morgan bit identifiers",
            "radius": args.morgan_radius,
            "consensus_fraction": args.consensus_fraction,
            "minimum_source_structures": args.minimum_source_structures,
            "minimum_consensus_bits": args.minimum_consensus_bits,
        },
        "spectral_filter": {
            "maximum_ms2_accuracy": args.maximum_ms2_accuracy,
            "maximum_fragments": args.maximum_fragments,
            "maximum_losses": args.maximum_losses,
            "minimum_feature_intensity": args.minimum_feature_intensity,
        },
        "records": records,
        "contracts": {
            "source_identity_match_is_suppressed_downstream": True,
            "candidate_truth_not_used": True,
            "matched_controls_required": True,
            "confidence_is_admission_priority_not_loss_weight": True,
            "native_dreams_loss_unchanged": True,
        },
    }
    report = {
        "status": "CHEMAWARE_MOTIFDB_EVIDENCE_CORPUS_COMPLETE",
        "source_motifs": len(motifs),
        "admitted_motifs": len(records),
        "excluded": dict(excluded),
        "motifsets": dict(motifset_counts),
        "positive_motifs": sum(record["charge"] > 0 for record in records),
        "negative_motifs": sum(record["charge"] < 0 for record in records),
        "source_sha256": observed_hash,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_motifdb_corpus_", dir=args.output.parent))
    try:
        (temporary / "motif_corpus.json").write_text(
            json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report["corpus_sha256"] = sha256_file(temporary / "motif_corpus.json")
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
