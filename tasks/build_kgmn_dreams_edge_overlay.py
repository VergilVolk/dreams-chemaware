#!/usr/bin/env python3
"""Build a provenance-locked MetDNA2 source overlay with one recursive-edge hook.

The untouched author source remains the only source used for the author baseline.
This builder copies that source and inserts a single optional hook immediately
after the author feature--feature MS2 score is calculated.  With the option
unset, the overlay is numerically identical to the author implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


EXPECTED_COMMIT = "5685ab219269c2f35cd5087655b0470b2da4d93c"
EXPECTED_RECURSIVE_SHA256 = "44c3495564fdfd0b2176b9610077f6edf82c5f197ba38f2e2ebecd48ec292621"
RELATIVE_TARGET = Path("R/RecursiveAnnotationMRN.R")

SCORE_SENTINEL = """             ms2_result_score <- sapply(ms2_result, function(x){x[1]})
             ms2_result_n_frag <- sapply(ms2_result, function(x){x[2]})
"""

SCORE_PATCHED = SCORE_SENTINEL + """

             # Optional experiment-only interface.  The untouched author
             # baseline never exports this object.  annotateMRN explicitly
             # exports the hook to every PSOCK worker; relying on a parent R
             # option here would silently disable it in parallel execution.
             if (exists("recursive_edge_score_hook", inherits = TRUE) &&
                 !is.null(recursive_edge_score_hook)) {
               if (!is.function(recursive_edge_score_hook)) {
                 stop("MetDNA2.recursive_edge_score_hook must be a function")
               }
               edge_context <- tibble::tibble(
                 seed_peak_name = rep(as.character(metabolite_name), length(ms2_result_score)),
                 neighbor_peak_name = as.character(neighbor_result$peak_name),
                 author_score = as.numeric(ms2_result_score),
                 matched_fragments = as.numeric(ms2_result_n_frag)
               )
               hooked_score <- recursive_edge_score_hook(edge_context)
               if (!is.numeric(hooked_score) || length(hooked_score) != nrow(edge_context)) {
                 stop("recursive edge hook must return one numeric score per dynamic edge")
               }
               if (any(!is.finite(hooked_score)) || any(hooked_score < 0 | hooked_score > 1)) {
                 stop("recursive edge hook scores must be finite and lie in [0, 1]")
               }
               ms2_result_score <- as.numeric(hooked_score)
             }
"""

CLUSTER_SENTINEL = """               cl <- parallel::makeCluster(threads)
               parallel::clusterExport(cl, c('tempFun',
                                             "showTags2",
"""

CLUSTER_PATCHED = """               cl <- parallel::makeCluster(threads)
               # PSOCK workers do not inherit getOption() state.  Materialise
               # and export the optional hook explicitly so an experiment can
               # never appear to run while silently falling back to author DP.
               recursive_edge_score_hook <- getOption("MetDNA2.recursive_edge_score_hook", NULL)
               parallel::clusterExport(cl, c('tempFun',
                                             'recursive_edge_score_hook',
                                             "showTags2",
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(source: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("third_party/MetDNA2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    target = source / RELATIVE_TARGET
    if not source.is_dir() or not target.is_file():
        raise FileNotFoundError(f"incomplete MetDNA2 source: {source}")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite overlay path: {output}")

    commit = git_commit(source)
    target_hash = sha256(target)
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"unexpected MetDNA2 commit: {commit}")
    if target_hash != EXPECTED_RECURSIVE_SHA256:
        raise RuntimeError(f"unexpected RecursiveAnnotationMRN.R sha256: {target_hash}")

    original = target.read_text(encoding="utf-8")
    if original.count(SCORE_SENTINEL) != 1:
        raise RuntimeError("recursive-edge score patch sentinel is not unique")
    if original.count(CLUSTER_SENTINEL) != 1:
        raise RuntimeError("recursive-edge cluster patch sentinel is not unique")

    shutil.copytree(source, output, ignore=shutil.ignore_patterns(".git"))
    overlay_target = output / RELATIVE_TARGET
    patched = original.replace(SCORE_SENTINEL, SCORE_PATCHED)
    patched = patched.replace(CLUSTER_SENTINEL, CLUSTER_PATCHED)
    overlay_target.write_text(patched, encoding="utf-8")

    manifest = {
        "status": "kgmn_metdna2_dreams_edge_overlay_built",
        "formal": True,
        "author_source": str(source),
        "overlay_source": str(output),
        "author_commit": commit,
        "target": RELATIVE_TARGET.as_posix(),
        "author_target_sha256": target_hash,
        "overlay_target_sha256": sha256(overlay_target),
        "hook_option": "MetDNA2.recursive_edge_score_hook",
        "contracts": {
            "author_baseline_uses_overlay": False,
            "hook_unset_preserves_author_scores": True,
            "hook_scope": "dynamic recursive feature-feature MS2 edges only",
            "hook_must_cover_every_dynamic_edge": True,
            "hook_is_explicitly_exported_to_psock_workers": True,
            "hook_score_range": [0.0, 1.0],
            "candidate_generation_modified": False,
            "reaction_network_modified": False,
            "propagation_logic_modified": False,
            "redundancy_removal_modified": False,
        },
        "claim_limit": (
            "This creates a controlled source overlay. It contains no calibrated DreaMS scores "
            "and establishes no annotation-performance improvement."
        ),
    }
    manifest_path = output / "DREAMS_EDGE_OVERLAY_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
