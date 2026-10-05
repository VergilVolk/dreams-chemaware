#!/usr/bin/env python
import argparse, json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser(); p.add_argument("output_dir",type=Path); a=p.parse_args()
r=json.loads((a.output_dir/"report.json").read_text(encoding="utf-8")); t=pd.read_csv(a.output_dir/"nested_domain_loso_transitions.csv.gz")
assert r["status"]=="bioaware_b25_branch_disagreement_action_complete" and len(t)==860 and t.query_id.nunique()==860
assert not any(r["frozen_B17_comparator"]["querywise_replay_mismatches"].values())
print("[validate_bioaware_b25_branch_disagreement_action] PASS",{k:r["nested_oof"][k] for k in ("delta_recall1","corrected","introduced","risk_net_lambda2")})
