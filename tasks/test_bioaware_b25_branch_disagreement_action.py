#!/usr/bin/env python
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]
from audit_bioaware_b25_branch_disagreement_action import apply_arbitration

base = pd.DataFrame([{"query_id":"q","source":"d","truth_candidate_id":"b","truth_formula":"F","baseline_candidate_id":"x","baseline_correct":False,"b12_final_candidate_id":"a","b16_final_candidate_id":"b","final_candidate_id":"a","intervene":True,"final_correct":False,"corrected":False,"introduced":False,"delta":0}])
candidates = pd.DataFrame([{"query_id":"q","candidate_id":"a","known_log_degree":1.0},{"query_id":"q","candidate_id":"b","known_log_degree":2.0},{"query_id":"q","candidate_id":"x","known_log_degree":0.0}])
r = apply_arbitration(base, candidates, True).iloc[0]
assert r["final_candidate_id"] == "b" and bool(r["corrected"])
print("[test_bioaware_b25_branch_disagreement_action] PASS")
