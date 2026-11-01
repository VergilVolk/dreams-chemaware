#!/usr/bin/env python3
"""Evaluate fixed transfer-shortcut repairs on consumed GNPS development."""
from __future__ import annotations

import json, os, shutil, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from evaluate_full23_multireference_on_gnps import BATCH_QUERIES, arguments, _load_torch
from full23_multireference_readout_core import TOPK_ARM
from full23_tensor_set_readout_core import model_from_payload
from train_enveda_unified_candidate_integrator import PANELS, _comparison, _joint_query_balanced, _metric_summary
from train_full23_multireference_readout import _load_banks, _make_representation_panels, _pack
from unified_additive_release import CHECKPOINT_SHA256, sha256_file, validate_checkpoint

ARMS = ("no_reference_count", "no_count_query_centered")

@torch.inference_mode()
def predict(model, panels, banks, device):
    output={n:np.full(len(p.base.labels),np.nan,np.float32) for n,p in panels.items()}
    model.eval()
    for panel_name in PANELS:
        panel=panels[panel_name]; queries=np.arange(len(panel.base.query_ptr)-1)
        for start in range(0,len(queries),BATCH_QUERIES):
            chosen=[(panel_name,int(q)) for q in queries[start:start+BATCH_QUERIES]]
            anchor,_,query,refs,mask,ptr,_=_pack(panels,chosen,TOPK_ARM)
            value=model(torch.as_tensor(anchor,device=device),
                        torch.as_tensor(query,device=device),torch.as_tensor(refs,device=device),
                        torch.as_tensor(mask,device=device),banks,query_ptr=ptr).float().cpu().numpy()
            cursor=0
            for _,q in chosen:
                left,right=map(int,panel.base.query_ptr[q:q+2]); width=right-left
                output[panel_name][left:right]=value[cursor:cursor+width]; cursor+=width
    if any(not np.all(np.isfinite(v)) for v in output.values()):
        raise RuntimeError("prediction coverage incomplete")
    return output

def main():
    args=arguments()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("GPU required")
    full23=validate_checkpoint(args.full23_checkpoint)
    if sha256_file(args.full23_checkpoint)!=CHECKPOINT_SHA256: raise RuntimeError("Full23 drifted")
    model_dir=args.readout_checkpoint
    payloads={name:_load_torch(model_dir/f"{name}.pt") for name in ARMS}
    manifest=pd.read_csv(args.benchmark/"manifest.csv.gz",usecols=["row"])
    cpu_banks=_load_banks(args,len(manifest)); panels=_make_representation_panels(args,full23,cpu_banks)
    dtype=torch.float16 if args.device.startswith("cuda") else torch.float32
    banks={n:torch.as_tensor(v,dtype=dtype,device=args.device) for n,(_,v) in cpu_banks.items()}
    predicted={name:predict(model_from_payload(payloads[name],args.device),panels,banks,args.device) for name in ARMS}
    scores={};metrics={};comparisons={}
    for pi,panel_name in enumerate(PANELS):
        panel=panels[panel_name].base
        scores[panel_name]={"full23_v1":panels[panel_name].anchor,
                            **{name:predicted[name][panel_name] for name in ARMS}}
        metrics[panel_name]={n:_metric_summary(v,panel)[0] for n,v in scores[panel_name].items()}
        comparisons[panel_name]={name:_comparison(predicted[name][panel_name],panels[panel_name].anchor,
                                                   panel,5000,20261101+10*pi+i)
                                 for i,name in enumerate(ARMS)}
    joint={}
    for method in ("full23_v1",)+ARMS:
        joint[method],_=_joint_query_balanced({n:p.base for n,p in panels.items()},scores,method)
    eligible=[]
    for arm in ARMS:
        if all(comparisons[p][arm]["recall1_formula_cluster_ci"]["ci_low"]>0
               and comparisons[p][arm]["corrected"]>comparisons[p][arm]["introduced"]
               and metrics[p][arm]["near"]["recall@1"]>=metrics[p]["full23_v1"]["near"]["recall@1"]
               for p in PANELS): eligible.append(arm)
    selected=max(eligible,key=lambda n:joint[n]["top1_utility"]) if eligible else "full23_v1"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=f".{args.output.name}.",dir=args.output.parent))
    try:
        for panel_name in PANELS:
            panel=panels[panel_name].base
            np.savez_compressed(staging/f"candidate_predictions_{panel_name}.npz",
                                query_ptr=panel.query_ptr,labels=panel.labels,**scores[panel_name])
        report={"status":"FULL23_TRANSFER_SAFE_GNPS_DEVELOPMENT_EVALUATION_COMPLETE",
                "dataset":{"name":"GNPS Gold/Silver","role":"consumed development",
                           "queries":{n:len(p.base.query_ptr)-1 for n,p in panels.items()}},
                "selected":selected,"metrics":metrics,"comparisons":comparisons,
                "query_balanced_joint":joint,
                "readout_checkpoint_sha256":{n:sha256_file(model_dir/f"{n}.pt") for n in ARMS},
                "claim_boundary":"Consumed development only; not external evidence."}
        (staging/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        os.replace(staging,args.output)
    except Exception: shutil.rmtree(staging,ignore_errors=True);raise
    print(json.dumps({"status":report["status"],"selected":selected,"output":str(args.output)}),flush=True)

if __name__=="__main__": main()
