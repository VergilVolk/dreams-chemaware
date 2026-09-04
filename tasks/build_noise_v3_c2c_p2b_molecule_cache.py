"""Build a molecule-level candidate cache anchored to frozen P2b.

Each candidate molecule is represented by one internally consistent spectrum
pair: the pair that wins under the actual frozen P2b decision for that query.
This avoids combining unrelated best spectra across features.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, tempfile
from pathlib import Path
import numpy as np
from build_g8r_real_error_atlas import Cache
from g8r_p2_rank_fusion_core import fuse_one_query, fusion_configuration_from_mapping, normalize_pair_features

ROOT=Path(__file__).resolve().parent.parent

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(8<<20): h.update(b)
 return h.hexdigest()

def mm(x):
 x=np.asarray(x,float); lo,hi=float(x.min()),float(x.max())
 return (x-lo)/(hi-lo) if hi-lo>1e-12 else np.zeros_like(x)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--graph',type=Path,default=ROOT/'data/validation/g8r_noise_v3_c2b_token_pair_cache.npz')
 p.add_argument('--artifact',type=Path,default=ROOT/'data/validation/g8r_p2b_rank_fusion.json')
 p.add_argument('--output',type=Path,default=ROOT/'data/validation/g8r_noise_v3_c2c_p2b_molecule_cache.npz')
 a=p.parse_args()
 for path in (a.graph,a.artifact):
  if not path.is_file(): raise FileNotFoundError(path)
 if a.output.exists() or a.output.with_suffix('.json').exists(): raise FileExistsError(a.output)
 g=Cache(a.graph); art=json.loads(a.artifact.read_text())
 cfg=fusion_configuration_from_mapping(art['configuration'])
 selected=list(map(str,art['selected_features']))
 expected=['dreams_similarity','sqrt_cosine','entropy_similarity','neutral_loss_sqrt_cosine']
 if selected!=expected: raise RuntimeError(f'unexpected P2b schema: {selected}')
 idx=[g.feature_names.index(n) for n in selected]; pair=g.features[:,idx]
 norm=normalize_pair_features(pair,g.molecule_ptr[g.query_ptr],cfg.normalization)
 token_names=[n for n in g.feature_names if n.startswith('token_')]
 raw_names=[n for n in g.feature_names if not n.startswith('token_') and n!='dreams_similarity']
 carry_names=[*raw_names,*token_names]
 carry_idx=[g.feature_names.index(n) for n in carry_names]
 rows=[]; applied=[]
 for q in range(g.n_queries):
  ml,mr=map(int,g.query_ptr[q:q+2]); pl,pr=int(g.molecule_ptr[ml]),int(g.molecule_ptr[mr]); ptr=g.molecule_ptr[ml:mr+1]-pl
  molecule_scores,use,support=fuse_one_query(norm[pl:pr],pair[pl:pr,0],ptr,np.asarray(cfg.weights),(1,2,3),cfg.min_support,cfg.min_advantage)
  baseline=mm(molecule_scores); top=np.sort(baseline); top_gap=float(top[-1]-top[-2])
  for local,(left,right) in enumerate(zip(ptr[:-1],ptr[1:])):
   local_pair_score=(norm[pl+left:pl+right]@np.asarray(cfg.weights)) if use else pair[pl+left:pl+right,0]
   winner=pl+left+int(np.argmax(local_pair_score))
   rows.append([baseline[local],baseline[local]-float(baseline.max()),top_gap,float(use),float(support),*g.features[winner,carry_idx]])
  applied.append(use)
  if (q+1)%2000==0 or q+1==g.n_queries: print(f'[C2-C cache] {q+1:,}/{g.n_queries:,}',flush=True)
 # The shared listwise trainer requires the first baseline-score field to use
 # this historical name.  Its value here is frozen P2b, not raw DreaMS.
 names=['dreams_similarity','p2b_gap_to_top','p2b_top_gap','p2b_applied','p2b_vote_support',*carry_names]
 features=np.asarray(rows,np.float32)
 payload={
  'features':features,'feature_names':np.asarray(names,object),'query_ptr':g.query_ptr,
  'molecule_ptr':np.arange(len(features)+1,dtype=np.int64),'molecule_label':g.molecule_label,
  'query_formula':g.query_formula,'query_has_near':g.query_has_near,'query_ik14':g.query_ik14,
 }
 staging=Path(tempfile.mkdtemp(prefix='c2c_cache_',dir=a.output.parent))
 try:
  tmp=staging/a.output.name; np.savez_compressed(tmp,**payload); tmp.replace(a.output)
 finally: shutil.rmtree(staging,ignore_errors=True)
 report={'status':'noise_v3_c2c_p2b_molecule_cache_complete','queries':g.n_queries,'molecules':len(features),
         'features':names,'raw_features':raw_names,'token_features':token_names,'p2b_application_rate':float(np.mean(applied)),
         'baseline_field_semantics':'dreams_similarity stores the frozen P2b molecule score in this derived cache',
         'protocol':'one P2b-winning spectrum pair per candidate molecule','graph_sha256':sha(a.graph),
         'artifact_sha256':sha(a.artifact),'cache_sha256':sha(a.output)}
 a.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
