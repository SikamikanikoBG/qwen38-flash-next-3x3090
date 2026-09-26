#!/usr/bin/env python3
"""Single-stream speed bench for llama-server. Uses the server's own timings.
usage: bench.py <tag> [--port 18100] [--depths 0,4000,16000,32000] [--gen 256] [--reps 2]"""
import json,sys,time,urllib.request,random,argparse
ap=argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--port",default="18100")
ap.add_argument("--depths",default="0,4000,16000,32000"); ap.add_argument("--gen",type=int,default=256)
ap.add_argument("--reps",type=int,default=2); ap.add_argument("--out",default="results/speed.jsonl")
a=ap.parse_args(); URL=f"http://127.0.0.1:{a.port}"
WORDS=("system memory cache kernel thread latency bandwidth expert router token layer tensor gradient "
       "compiler socket channel scheduler buffer pipeline network storage vector matrix").split()
def filler(ntok,seed):
    r=random.Random(seed); return " ".join(r.choice(WORDS) for _ in range(int(ntok*0.95)))
TASK="Write a detailed technical explanation of how a CPU cache hierarchy works, covering L1, L2, L3, coherence protocols and prefetching."
def post(p,d):
    return json.load(urllib.request.urlopen(urllib.request.Request(URL+p,data=json.dumps(d).encode(),headers={"Content-Type":"application/json"}),timeout=3600))
out=open(a.out,"a")
for depth in [int(x) for x in a.depths.split(",")]:
    for rep in range(a.reps):
        ctx=(filler(depth,rep*7919+depth)+"\n\n") if depth else ""
        prompt=f"<|im_start|>user\n{ctx}{TASK}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        t0=time.time()
        r=post("/completion",{"prompt":prompt,"n_predict":a.gen,"temperature":0,"cache_prompt":False,"ignore_eos":True})
        t=r["timings"]; row=dict(tag=a.tag,depth=depth,rep=rep,prompt_n=t["prompt_n"],prefill_tps=round(t["prompt_per_second"],1),
            ttft_s=round(t["prompt_ms"]/1000,3),decode_tps=round(t["predicted_per_second"],2),gen_n=t["predicted_n"],
            draft_n=t.get("draft_n"),draft_acc=t.get("draft_n_accepted"),wall_s=round(time.time()-t0,2))
        print(json.dumps(row),flush=True); out.write(json.dumps(row)+"\n"); out.flush()
