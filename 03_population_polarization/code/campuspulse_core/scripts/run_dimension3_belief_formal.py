from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from comparison_harness.adapters import ArgumentCandidate
from comparison_harness.contracts import AgentSeed
from comparison_harness.dimension3_belief_network import BeliefNetworkPopulation
from comparison_harness.real_provider import OpenAIProviderTransport, RealModelProvider

def canon(v:Any)->str: return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def jl(p:Path): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def writejl(p:Path,rows): p.write_text("".join(canon(x)+"\n" for x in rows),encoding="utf-8",newline="\n")

def main()->int:
 p=argparse.ArgumentParser(); p.add_argument("--initializations",type=Path,required=True); p.add_argument("--event-arguments",type=Path,required=True); p.add_argument("--edges",type=Path,required=True); p.add_argument("--provider-boundary",type=Path,required=True); p.add_argument("--auxiliary-boundary",type=Path,required=True); p.add_argument("--credentials",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--seed",type=int,default=20260815); a=p.parse_args()
 final_names=("belief_population_states.jsonl","network_trace.jsonl","run_report.json")
 if any((a.output_root/name).exists() for name in final_names):
  raise FileExistsError("formal Belief Engine output already exists")
 a.output_root.mkdir(parents=True,exist_ok=True)
 init={r["agent_id"]:r for r in jl(a.initializations) if r["method"]=="belief_engine"}; events={r["agent_id"]:r for r in jl(a.event_arguments)}; edges=jl(a.edges)
 if len(init)!=len(events)!=1000 or len(edges)!=8000: raise ValueError("formal grid mismatch")
 cfg=json.loads(a.provider_boundary.read_text(encoding="utf-8")); transport=OpenAIProviderTransport(api_key_env=cfg["api_key_environment_variable"],timeout_seconds=float(cfg["request_constraints"]["timeout_seconds"]),credentials_path=a.credentials)
 env=dict(os.environ); env["AAMAS_REAL_MODEL_ENABLED"]="YES_I_ACCEPT_API_COST"; env["AAMAS_FULL_EXPERIMENT_ENABLED"]="YES_RUN_FROZEN_FULL_EXPERIMENT"; env[cfg["api_key_environment_variable"]]="CONFIGURED_VIA_CREDENTIALS_JSON"
 provider=RealModelProvider(config_path=a.provider_boundary,auxiliary_boundary_path=a.auxiliary_boundary,storage_root=a.output_root/"provider_state",transport=transport,enable_real=True,budget_profile="full",environment=env)
 # The same frozen argument is delivered repeatedly over 72,000 network edges.
 # Embeddings are a deterministic function of the exact text and frozen model,
 # so memoizing them is semantically identical to repeated provider-cache reads
 # while avoiding roughly 146,000 redundant verified file reads.
 embedding_memo={}
 def embed(text):
  if text not in embedding_memo: embedding_memo[text]=provider.embed(text).vector
  return embedding_memo[text]
 def noextract(text,role): raise RuntimeError("formal D3 uses frozen structured arguments")
 seeds={}; seedargs={}; groups={}
 for aid,r in init.items():
  n=r["native_initialization"]; groups[aid]=r["interest_group"]; seeds[aid]=AgentSeed(aid,None,groups[aid],n["role_background"],None)
  shared=canon({"role_background":n["role_background"],"stake_position":n["stake_position"]})
  seedargs[aid]=tuple(ArgumentCandidate(f"shared-initial-{x['polarity']}:{shared}",int(x["polarity"]),float(x["strength"]),"seed") for x in n["structured_argument_library"])
 pop=BeliefNetworkPopulation(seeds=seeds,seed_arguments=seedargs,directed_edges=((x["source_agent_id"],x["target_agent_id"]) for x in edges),extractor=noextract,embedder=embed,evidence_uptake=.2,prior_anchoring=.4,similarity_threshold=.60,arguments_per_edge=2)
 states=pop.population_state(0,groups,a.seed); trace=[]
 for tick in range(1,10):
  if tick==3:
   mapped={aid:tuple(ArgumentCandidate(x["claim"],int(x["polarity"]),float(x["strength"]),"opponent") for x in events[aid]["arguments"]) for aid in sorted(events)}
   pop.inject_event_arguments(tick=3,arguments_by_agent=mapped,event_id="housing-520-320-applications-536-v1")
  deliveries=pop.step(tick); trace.append({"tick":tick,"delivery_count":len(deliveries),"argument_count":sum(len(x.arguments) for x in deliveries)})
  if tick in {3,9}: states.extend(pop.population_state(tick,groups,a.seed))
 if len(states)!=3000 or len(pop.delivery_log)!=72000: raise ValueError("formal output incomplete")
 sp=a.output_root/"belief_population_states.jsonl"; tp=a.output_root/"network_trace.jsonl"; writejl(sp,states); writejl(tp,trace)
 report={"schema_version":"dimension3-belief-formal-run-v1.0.0","status":"complete","scientific_result":True,"seed":a.seed,"state_rows":len(states),"network_deliveries":len(pop.delivery_log),"unique_embedding_texts":len(embedding_memo),"logical_calls":provider.logical_calls,"transport_attempts":provider.transport_attempts,"embedding_inputs":provider.embedding_inputs,"input_characters":provider.input_chars,"output_sha256":{"states":hashlib.sha256(sp.read_bytes()).hexdigest(),"trace":hashlib.sha256(tp.read_bytes()).hexdigest()}}
 (a.output_root/"run_report.json").write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8",newline="\n"); print(canon(report)); return 0
if __name__=="__main__": raise SystemExit(main())
