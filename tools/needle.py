#!/usr/bin/env python3
"""Needle-in-a-haystack: hide a code in N tokens of wikitext at a given depth, ask for it back.
usage: needle.py <approx_tokens> <depth 0..1> [--port 18100]"""
import json, sys, time, urllib.request, random
n_tok, depth = int(sys.argv[1]), float(sys.argv[2]); port = sys.argv[4] if len(sys.argv) > 4 else "18100"
text = open("data/wiki.test.raw", encoding="utf-8").read()
hay = (text * (1 + int(n_tok * 4.2 // len(text))))[: int(n_tok * 4.0)]
rng = random.Random(n_tok); code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(10))
cut = int(len(hay) * depth); cut = hay.rfind("\n", 0, cut) + 1
hay = hay[:cut] + f"\nIMPORTANT: the vault access code is {code}. Remember it.\n" + hay[cut:]
body = {"messages": [{"role": "user", "content": hay + "\n\nWhat is the vault access code mentioned in the text above? Reply with the code only."}],
        "max_tokens": 20, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
t0 = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=7200))
ans = r["choices"][0]["message"]["content"].strip(); t = r.get("timings", {})
row = {"test": "needle", "prompt_tokens": r["usage"]["prompt_tokens"], "depth": depth, "expected": code, "answer": ans,
       "correct": code in ans, "ttft_s": round(t.get("prompt_ms", 0) / 1000, 1), "prefill_tps": round(t.get("prompt_per_second", 0), 1),
       "wall_s": round(time.time() - t0, 1)}
print(json.dumps(row)); open("results/longctx.jsonl", "a").write(json.dumps(row) + "\n")
