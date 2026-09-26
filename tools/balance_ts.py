#!/usr/bin/env python3
"""Pick a -ts (layers per GPU) that balances GPU bytes for a hot/cold split.
usage: balance_ts.py <model first shard> <split.txt> <n_gpus> [extra_gib_gpu0]"""
import sys, glob, re, os
sys.path.insert(0, os.environ.get("GGUF_PY", "/lab/llama.cpp/gguf-py"))
from gguf import GGUFReader
model, split, ng = sys.argv[1], sys.argv[2], int(sys.argv[3])
extra0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
shards = sorted(glob.glob(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", model)))
exp_b, dense_b, nexp = {}, {}, {}
for f in shards:
    for t in GGUFReader(f).tensors:
        m = re.match(r"blk\.(\d+)\.(.*)", t.name)
        if not m: continue
        l = int(m.group(1))
        if "_exps" in t.name:
            exp_b[l] = exp_b.get(l, 0) + int(t.n_bytes) // int(t.shape[-1]); nexp[l] = int(t.shape[-1])
        else:
            dense_b[l] = dense_b.get(l, 0) + int(t.n_bytes)
hot = {}
for line in open(split):
    if line.startswith("#") or not line.strip(): continue
    p = line.split(); hot[int(p[0])] = int(p[1])
layers = sorted(l for l in exp_b if l in dense_b and l < 48)
gb = [ (dense_b[l] + exp_b[l] * hot.get(l, nexp[l])) / 2**30 for l in layers]
tot = sum(gb) + extra0
target = tot / ng
cuts, acc, cur = [], extra0, 0
for i, g in enumerate(gb):
    if acc + g/2 > target * (len(cuts) + 1) and len(cuts) < ng - 1:
        cuts.append(i)
    acc += g
bounds = [0] + cuts + [len(layers)]
counts = [bounds[i+1] - bounds[i] for i in range(ng)]
per = [sum(gb[bounds[i]:bounds[i+1]]) for i in range(ng)]
print("ts=" + ",".join(map(str, counts)), "GiB per GPU (layers only):", [round(x, 2) for x in per], "total", round(sum(gb), 2))
