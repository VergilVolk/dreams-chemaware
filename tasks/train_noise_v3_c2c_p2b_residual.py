"""Fixed-recipe formula-OOF residual training on top of frozen P2b."""
from __future__ import annotations
import argparse,json
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
from g8r_p2_listwise_core import deterministic_formula_fold
from train_g8r_p2_listwise import (ListwiseCache,Configuration,combine_oof,formula_cluster_ci,
                                   fit_standardizer,train_fixed_epochs,evaluate)

ROOT=Path(__file__).resolve().parent.parent

def paired_ci(records_a,records_b,n,seed):
 a={r['query']:r for r in records_a}; b={r['query']:r for r in records_b}
 keys=sorted(set(a)&set(b)); formulas={}
 for k in keys: formulas.setdefault(a[k]['formula'],[]).append(float(a[k]['top1'])-float(b[k]['top1']))
 groups=list(formulas.values()); rng=np.random.default_rng(seed); draws=[]
 for _ in range(n):
  ix=rng.integers(0,len(groups),len(groups)); draws.append(np.concatenate([np.asarray(groups[i]) for i in ix]).mean())
 values=np.concatenate([np.asarray(v) for v in groups])
 return {'mean':float(values.mean()),'ci_low':float(np.percentile(draws,2.5)),'ci_high':float(np.percentile(draws,97.5))}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--cache',type=Path,default=ROOT/'data/validation/g8r_noise_v3_c2c_p2b_molecule_cache.npz')
 p.add_argument('--output',type=Path,default=ROOT/'data/validation/g8r_noise_v3_c2c_p2b_residual.json')
 p.add_argument('--device',default='cuda');p.add_argument('--folds',type=int,default=5);p.add_argument('--epochs',type=int,default=5)
 p.add_argument('--bootstrap',type=int,default=5000);p.add_argument('--seed',type=int,default=20260825)
 p.add_argument('--learning-rate',type=float,default=3e-3);p.add_argument('--weight-decay',type=float,default=1e-4)
 p.add_argument('--query-batch-size',type=int,default=32);p.add_argument('--temperature',type=float,default=0.1)
 p.add_argument('--allowed-margin-drop',type=float,default=0.003);p.add_argument('--residual-weight',type=float,default=0.02)
 a=p.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 base=ListwiseCache(a.cache); all_names=base.feature_names
 token=[n for n in all_names if n.startswith('token_')]
 context=['dreams_similarity','p2b_gap_to_top','p2b_top_gap','p2b_applied','p2b_vote_support']
 raw=[n for n in all_names if n not in token and n not in context]
 families={'p2b_context':context,'p2b_raw':[ *context,*raw ],'p2b_raw_token':[ *context,*raw,*token ]}
 folds=np.asarray([deterministic_formula_fold(str(v),a.folds) for v in base.query_formula])
 config=Configuration('c2c_fixed',64,0.05,8.0,2.0); device=torch.device(a.device); reports={}; records={}
 for family,names in families.items():
  cache=ListwiseCache(a.cache); cache.features=cache.features[:,[all_names.index(n) for n in names]]; cache.feature_names=names
  fold_records=[]
  for held in range(a.folds):
   train=np.flatnonzero(folds!=held); test=np.flatnonzero(folds==held); mean,scale=fit_standardizer(cache,train)
   model=train_fixed_epochs(cache,train,mean,scale,config,a.seed+100*held,a.epochs,a,device)
   result=evaluate(cache,model,mean,scale,test,device);fold_records.append(result['records'])
   print(f'[C2-C {family} fold={held}] dR1={result["delta_recall1"]:+.4f} near={result["delta_near_recall1"]:+.4f} C/I={result["corrected"]}/{result["introduced"]}',flush=True)
  pooled=combine_oof(fold_records);pooled['formula_cluster_ci']=formula_cluster_ci(pooled['records'],a.bootstrap,a.seed)
  records[family]=pooled['records'];reports[family]={k:v for k,v in pooled.items() if k!='records'}
 pair_token=paired_ci(records['p2b_raw_token'],records['p2b_raw'],a.bootstrap,a.seed+1)
 gates={'full_positive_vs_p2b':reports['p2b_raw_token']['delta_recall1']>0,
        'full_near_nonnegative':reports['p2b_raw_token']['delta_near_recall1']>=0,
        'full_corrected_ge_introduced':reports['p2b_raw_token']['corrected']>=reports['p2b_raw_token']['introduced'],
        'token_beats_raw_formula_ci_positive':pair_token['ci_low']>0}
 out={'status':'noise_v3_c2c_p2b_residual_complete','protocol':'fixed-recipe formula-OOF; P2b baseline',
      'configuration':asdict(config),'families':reports,'token_vs_raw_paired_formula_ci':pair_token,'gates':gates,
      'pass_to_robust_training':bool(all(gates.values())),'claim_limit':'training-graph OOF screen; no sealed P3'}
 a.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__': main()
