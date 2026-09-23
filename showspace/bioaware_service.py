"""Optional, phenotype-blind BioAware evidence preview for Showspace.

BioAware never changes the shared embedding.  The public UI reports its
support and abstention state beside the spectral ranking; an uploaded seed
table is required so a single spectrum cannot manufacture its own context.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from annotation.bioaware import (
    BioAwareConfig,
    aggregate_query_support,
    build_one_hop_evidence,
    fuse_candidates,
)


FORBIDDEN_SEED_TOKENS = ("phenotype", "case", "control", "tumor", "normal", "qvalue", "pvalue", "fold_change")


class BioAwareService:
    def __init__(self, participants_path: str | None = None):
        configured = participants_path or os.getenv("BIOAWARE_RHEA_PARTICIPANTS")
        self.path = Path(configured) if configured else None
        self.participants: pd.DataFrame | None = None
        self.error: str | None = None
        if self.path is None or not self.path.is_file():
            self.error = "未配置 Rhea participant cache"
            return
        try:
            self.participants = pd.read_csv(self.path)
        except Exception as exc:
            self.error = f"Rhea cache 加载失败: {type(exc).__name__}"

    @property
    def ready(self) -> bool:
        return self.participants is not None and self.error is None

    @staticmethod
    def _read_seeds(path: str) -> pd.DataFrame:
        seeds = pd.read_csv(path)
        suspicious = [column for column in seeds.columns if any(token in column.lower() for token in FORBIDDEN_SEED_TOKENS)]
        if suspicious:
            raise ValueError(f"BioAware seeds 禁止 phenotype/statistics 字段: {suspicious}")
        required = {"seed_compound_id", "seed_score"}
        missing = required - set(seeds.columns)
        if missing:
            raise ValueError(f"BioAware seeds 缺少字段: {sorted(missing)}")
        if "seed_query_id" not in seeds:
            seeds["seed_query_id"] = [f"context_{index}" for index in range(len(seeds))]
        return seeds

    def preview(self, candidates: list[dict[str, Any]], seed_path: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not seed_path:
            return candidates, {"state": "abstained_no_context", "applied": False}
        if not self.ready:
            return candidates, {"state": "unavailable", "error": self.error, "applied": False}
        assert self.participants is not None
        seeds = self._read_seeds(seed_path)
        spectral = pd.DataFrame({
            "query_id": "showspace_query",
            "candidate_id": [str(row.get("Ref. ID", "")) for row in candidates],
            "spectral_score": [float(row.get("Embedding similarity", 0.0)) for row in candidates],
        })
        paths = build_one_hop_evidence(self.participants, seeds, BioAwareConfig())
        supported, explanations = aggregate_query_support(spectral, paths, exclude_same_query=True)
        scored, decisions = fuse_candidates(supported, BioAwareConfig())
        by_id = scored.set_index("candidate_id")
        output = []
        for row in candidates:
            item = dict(row)
            candidate_id = str(item.get("Ref. ID", ""))
            if candidate_id in by_id.index:
                evidence = by_id.loc[candidate_id]
                item["BioAware support"] = float(evidence["network_support"])
                item["BioAware paths"] = int(evidence["network_path_count"])
                item["BioAware state"] = str(evidence["evidence_state"])
            output.append(item)
        decision = decisions.iloc[0].to_dict() if not decisions.empty else {}
        report = {
            "state": decision.get("evidence_state", "no_network_evidence"),
            "applied": bool(decision.get("bioaware_applied", False)),
            "baseline_top": decision.get("baseline_top_candidate"),
            "network_top": decision.get("network_top_candidate"),
            "preview_top": decision.get("final_top_candidate"),
            "evidence_paths": int(len(explanations)),
            "claim_limit": "contextual evidence preview; no identity, flux, or enzyme claim",
        }
        return output, report
