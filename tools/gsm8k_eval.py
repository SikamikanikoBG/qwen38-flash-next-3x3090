#!/usr/bin/env python3
"""GSM8K exact-match against an OpenAI-compatible server. Greedy, thinking off.
usage: gsm8k_eval.py <tag> [--port 18100] [--n 200] [--conc 1]"""
import json, re, sys, time, argparse, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--port", default="18100")
ap.add_argument("--n", type=int, default=200); ap.add_argument("--conc", type=int, default=1)
ap.add_argument("--data", default="data/gsm8k_test.jsonl"); ap.add_argument("--out", default="results/quality.jsonl")
a = ap.parse_args()
rows = [(d["question"], d["answer"]) for d in map(json.loads, open(a.data))][:a.n]
def one(row):
    q, ans = row
    gold = ans.split("####")[-1].strip().replace(",", "")
    body = {"messages": [{"role": "user", "content": q + "\n\nSolve step by step, then give the final answer as \"Final answer: <number>\"."}],
            "max_tokens": 768, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    txt = json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]["message"]["content"] or ""
    m = re.search(r"Final answer:\s*\**\s*\$?(-?[\d,]*\.?\d+)", txt)
    pred = m.group(1).replace(",", "") if m else (re.findall(r"-?\d[\d,]*\.?\d*", txt) or [None])[-1]
    try: return abs(float(str(pred).replace(",", "")) - float(gold)) < 1e-6
    except Exception: return False
t0 = time.time()
with ThreadPoolExecutor(a.conc) as ex: res = list(ex.map(one, rows))
acc = sum(res) / len(res)
row = {"tag": a.tag, "bench": "gsm8k", "n": len(res), "acc": round(acc, 4), "secs": round(time.time() - t0)}
print(json.dumps(row)); open(a.out, "a").write(json.dumps(row) + "\n")
