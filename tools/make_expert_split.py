#!/usr/bin/env python3
"""Build a LLAMA_EXPERT_SPLIT file: which experts of each MoE layer stay on the GPUs (hot) and which
go to CPU RAM (cold), from routing statistics in an imatrix GGUF.

Experts are ranked globally by (routed-token share) and taken hottest-first until the GPU budget for
routed experts is spent, so strongly skewed layers keep fewer experts hot and flat layers keep more.

usage: make_expert_split.py <imatrix.gguf> <model.gguf (first shard)> <hot_budget_GiB> <out.txt>
"""
import sys, glob, re, os
import numpy as np
sys.path.insert(0, os.environ.get("GGUF_PY", "/lab/llama.cpp/gguf-py"))
from gguf import GGUFReader

imat, model, budget_gib, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]

counts = {}
for t in GGUFReader(imat).tensors:
    m = re.match(r"blk\.(\d+)\.ffn_gate_exps\.weight\.counts$", t.name)
    if m:
        counts[int(m.group(1))] = np.asarray(t.data, dtype=np.float64).reshape(-1)

# bytes per expert per layer (gate+up+down), from the model shards
shards = sorted(glob.glob(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", model)))
exp_bytes = {}
n_layer = None
for f in shards:
    r = GGUFReader(f)
    for fld in r.fields.values():
        if fld.name.endswith(".block_count"):
            n_layer = int(fld.parts[fld.data[0]][0])
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", t.name)
        if m:
            il = int(m.group(1))
            exp_bytes[il] = exp_bytes.get(il, 0) + int(t.n_bytes) // int(t.shape[-1])

layers = sorted(l for l in exp_bytes if l in counts and (n_layer is None or l < n_layer))
cand = []  # (share per byte, layer, expert)
for l in layers:
    c = counts[l]; share = c / c.sum()
    for e in range(len(c)):
        cand.append((share[e] / exp_bytes[l], l, e))
cand.sort(reverse=True)

budget = budget_gib * 2**30
hot = {l: [] for l in layers}
used = 0
for _, l, e in cand:
    if used + exp_bytes[l] > budget:
        continue
    hot[l].append(e); used += exp_bytes[l]

cold_share = []
with open(out, "w") as f:
    f.write(f"# hot/cold expert split: budget {budget_gib} GiB for hot routed experts, from {os.path.basename(imat)}\n")
    for l in layers:
        c = counts[l]
        hs = sorted(hot[l], key=lambda e: -c[e])
        cs = sorted(set(range(len(c))) - set(hs), key=lambda e: -c[e])
        if not cs:  # fully hot: no split needed for this layer
            continue
        if not hs:
            hs, cs = cs[:1], cs[1:]
        f.write(f"{l} {len(hs)} " + " ".join(map(str, hs + cs)) + "\n")
        cold_share.append(c[cs].sum() / c.sum())
tot = sum(exp_bytes[l] * len(counts[l]) for l in layers)
print(f"layers={len(layers)} hot={used/2**30:.1f} GiB of {tot/2**30:.1f} GiB ({used/tot*100:.1f}% of expert bytes), "
      f"cold experts serve {np.mean(cold_share)*100:.1f}% of routed tokens (mean over split layers; whole-layer offload of the same bytes: {(1-used/tot)*100:.1f}%)")
print("n_hot per layer:", [len(hot[l]) for l in layers])
