#!/usr/bin/env python3
"""Frequency-tiered expert quantization for qwen4exp (Qwen3.8-Flash-Next) GGUFs.

Every MoE layer's routed experts are ranked by how often the router picks them (imatrix counts) and
split into up to 3 tiers. Hot experts get more bits, rarely used ones fewer, under a total byte budget,
so the whole model can stay in VRAM while most routed tokens still see high-precision weights.

Output GGUF (read by the patched llama.cpp, src/models/qwen4exp.cpp):
  - blk.L.ffn_gate_inp.weight        router rows permuted: slot s = expert perm[s]
  - blk.L.ffn_{gate,up,down}_exps.weight      tier 0 (slots [0, n0))
  - blk.L.ffn_{gate,up,down}_exps.weight.t1   tier 1 (slots [n0, n0+n1)), .t2 likewise
  - qwen4exp.expert_n_tiers (u32), qwen4exp.expert_tiers (i32[n_layer * n_tiers]) tier sizes
Everything else is copied byte-for-byte from the source.

The source must hold the experts at >= 8 bits (Q8_0/BF16/F16): they are dequantized and re-quantized
with ggml's own quantizers (libggml-base via ctypes), using the per-expert imatrix.

usage:
  expert_tiers.py plan  --src Q8_0-00001-of-N.gguf --imatrix imatrix.gguf --budget-gib 52 [--gu Q6_K,IQ4_XS,IQ2_S --dn Q8_0,IQ4_NL,MXFP4]
  expert_tiers.py write --src ... --imatrix ... --budget-gib 52 ... --out model-tiered.gguf [--threads 56]
"""
import argparse, ctypes, glob, heapq, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

sys.path.insert(0, os.environ.get("GGUF_PY", "/lab/llama.cpp/gguf-py"))
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as QT, GGUFValueType as VT

LIBGGML = os.environ.get("LIBGGML", "/lab/llama.cpp/build/bin/libggml-base.so")

# rough relative weight-MSE per quant type (each bit ~4x); only used to rank tier allocations
BPW = {"Q8_0": 8.5, "Q6_K": 6.5625, "Q5_1": 6.0, "Q5_K": 5.5, "Q5_0": 5.5, "Q4_K": 4.5, "IQ4_NL": 4.5,
       "Q4_0": 4.5, "IQ4_XS": 4.25, "MXFP4": 4.25, "IQ3_S": 3.4375, "IQ3_XXS": 3.0625, "IQ2_M": 2.7,
       "IQ2_S": 2.5625, "IQ2_XS": 2.3125, "IQ2_XXS": 2.0625}
ERR = {t: 4.0 ** (-(b - 6.5625)) for t, b in BPW.items()}
ERR["IQ4_XS"] *= 0.8; ERR["IQ4_NL"] *= 0.8; ERR["MXFP4"] *= 1.6   # imatrix-aware non-linear grids vs plain


def load_counts_imatrix(path):
    r = GGUFReader(path)
    counts, sum2 = {}, {}
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.(ffn_(?:gate|up|down)_exps)\.weight\.(in_sum2|counts)$", t.name)
        if not m:
            continue
        il, kind, what = int(m.group(1)), m.group(2), m.group(3)
        a = np.asarray(t.data, dtype=np.float32)
        if what == "counts":
            counts[(il, kind)] = a.reshape(-1).astype(np.float64)
        else:
            sum2[(il, kind)] = a.reshape(-1, int(t.shape[0]))  # [n_expert, n_per_row]
    return counts, sum2


def shards_of(src):
    return sorted(glob.glob(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", src))) or [src]


def plan(args, readers):
    gu, dn = args.gu.split(","), args.dn.split(",")
    assert len(gu) == len(dn) and 1 <= len(gu) <= 4
    counts, _ = load_counts_imatrix(args.imatrix)
    # per layer: params of gate+up and of down per expert
    shapes = {}
    for r in readers:
        for t in r.tensors:
            m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", t.name)
            if m:
                il = int(m.group(1)); k = "down" if m.group(2) == "down" else "gu"
                n_exp = int(t.shape[-1]); per = int(np.prod(t.shape[:-1]))
                shapes.setdefault(il, {"n": n_exp, "gu": 0, "down": 0})[k] += per
    n_layer = int(next(f for f in readers[0].fields.values() if f.name.endswith(".block_count")).parts[-1][0])
    layers = [l for l in sorted(shapes) if l < n_layer]
    K = len(gu)
    tier_bytes = lambda l, k: (shapes[l]["gu"] * BPW[gu[k]] + shapes[l]["down"] * BPW[dn[k]]) / 8
    tier_err   = lambda l, k: (shapes[l]["gu"] * ERR[gu[k]] + shapes[l]["down"] * ERR[dn[k]])
    order, share = {}, {}
    for l in layers:
        c = counts.get((l, "ffn_gate_exps"))
        if c is None:
            c = np.ones(shapes[l]["n"])
        s = (c + 1e-3) / (c + 1e-3).sum()
        order[l] = np.argsort(-s, kind="stable"); share[l] = s[order[l]]
    # start: everything in the lowest tier; upgrades move the next-hottest expert of a layer up one tier
    n = {l: [0] * K for l in layers}
    for l in layers:
        n[l][K - 1] = shapes[l]["n"]
    used = sum(tier_bytes(l, K - 1) * shapes[l]["n"] for l in layers)
    budget = args.budget_gib * 2**30
    heap = []
    def push(l, k):  # candidate: promote the hottest expert of tier k (k>0) of layer l into tier k-1
        if k <= 0 or n[l][k] == 0:
            return
        pos = sum(n[l][:k])            # sorted position of the hottest expert in tier k
        db = tier_bytes(l, k - 1) - tier_bytes(l, k)
        de = share[l][pos] * (tier_err(l, k) - tier_err(l, k - 1))
        heapq.heappush(heap, (-de / db, l, k, pos))
    for l in layers:
        push(l, K - 1)
    while heap:
        _, l, k, pos = heapq.heappop(heap)
        if pos != sum(n[l][:k]) or n[l][k] == 0:
            continue
        db = tier_bytes(l, k - 1) - tier_bytes(l, k)
        if used + db > budget:
            continue
        n[l][k] -= 1; n[l][k - 1] += 1; used += db
        push(l, k); push(l, k - 1)
    exp_err = sum(sum(share[l][sum(n[l][:k]):sum(n[l][:k + 1])].sum() * tier_err(l, k) / (shapes[l]["gu"] + shapes[l]["down"])
                      for k in range(K)) for l in layers) / len(layers)
    tok = [sum(share[l][sum(n[l][:k]):sum(n[l][:k + 1])].sum() for l in layers) / len(layers) for k in range(K)]
    print(f"tiers gate/up={gu} down={dn}: experts {used/2**30:.2f} GiB (budget {args.budget_gib}), "
          f"routed-token share per tier {[round(x*100,1) for x in tok]}%, rel. err {exp_err:.3f}")
    # drop empty tiers globally is not possible (tier count is per model); empty tiers per layer are allowed
    return layers, order, n, gu, dn


def load_lib():
    lib = ctypes.CDLL(LIBGGML)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    lib.ggml_row_size.restype = ctypes.c_size_t
    lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    return lib


def dequant(raw, qtype, n_per_row):
    """raw uint8 rows -> float32 [rows, n_per_row] (Q8_0 / F16 / BF16 / F32 sources)."""
    if qtype == QT.Q8_0:
        b = raw.reshape(-1, 34)
        d = b[:, :2].copy().view(np.float16).astype(np.float32)
        q = b[:, 2:].view(np.int8).astype(np.float32)
        return (q * d).reshape(-1, n_per_row)
    if qtype == QT.F16:
        return raw.view(np.float16).astype(np.float32).reshape(-1, n_per_row)
    if qtype == QT.BF16:
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(-1, n_per_row)
    if qtype == QT.F32:
        return raw.view(np.float32).reshape(-1, n_per_row)
    raise SystemExit(f"source experts must be Q8_0/F16/BF16/F32, got {qtype.name}")


def write(args, readers):
    layers, order, n, gu, dn = plan(args, readers)
    K = len(gu)
    lib = load_lib()
    for t in set(gu + dn):
        lib.ggml_quantize_init(int(QT[t]))
    _, sum2 = load_counts_imatrix(args.imatrix)
    counts, _ = load_counts_imatrix(args.imatrix)
    arch = next(f for f in readers[0].fields.values() if f.name == "general.architecture")
    arch = bytes(arch.parts[arch.data[0]]).decode()
    # two shards: <out>-00001-of-00002.gguf (everything we rewrite) and -00002 (big shared tensors, e.g. n-gram table)
    base = re.sub(r"\.gguf$", "", args.out)
    out1, out2 = f"{base}-00001-of-00002.gguf", f"{base}-00002-of-00002.gguf"
    w = GGUFWriter(out1, arch, use_temp_file=False)
    # copy metadata (skip split keys and the arch key the writer adds itself)
    for f in readers[0].fields.values():
        if f.name.startswith("GGUF.") or f.name.startswith("split.") or f.name == "general.architecture":
            continue
        vt = f.types[0]
        if vt == VT.ARRAY:
            st = f.types[1]
            if st == VT.STRING:
                val = [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]
            else:
                val = [f.parts[i].tolist()[0] for i in f.data]
            w.add_key_value(f.name, val, vt, sub_type=st)
        elif vt == VT.STRING:
            w.add_key_value(f.name, bytes(f.parts[f.data[0]]).decode("utf-8", "replace"), vt)
        else:
            w.add_key_value(f.name, f.parts[f.data[0]].tolist()[0], vt)
    n_layer = max(layers) + 1
    w.add_key_value("qwen4exp.expert_n_tiers", K, VT.UINT32)
    w.add_key_value("qwen4exp.expert_tiers", [int(n[l][k]) if l in n else 0 for l in range(n_layer) for k in range(K)],
                    VT.ARRAY, sub_type=VT.INT32)
    w.add_key_value("general.quantization_note",
                    f"frequency-tiered experts: gate/up {gu}, down {dn}, budget {args.budget_gib} GiB", VT.STRING)

    # plan the output tensor list (name, ggml shape, qtype, nbytes, producer)
    jobs = []
    for r in readers:
        for t in r.tensors:
            m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", t.name)
            if m and int(m.group(1)) in n:
                il = int(m.group(1)); kind = "ffn_" + m.group(2) + "_exps"
                ne = [int(x) for x in t.shape]      # ggml order: [n_per_row, rows_per_expert, n_expert]
                off = 0
                for k in range(K):
                    cnt = n[il][k]
                    qt = QT[(dn if kind == "ffn_down_exps" else gu)[k]]
                    name = t.name if k == 0 else f"{t.name}.t{k}"
                    nbytes = lib.ggml_row_size(int(qt), ne[0]) * ne[1] * cnt
                    jobs.append((name, [ne[0], ne[1], cnt], qt, nbytes, ("exps", t, il, kind, order[il][off:off + cnt])))
                    off += cnt
                continue
            m = re.match(r"blk\.(\d+)\.ffn_gate_inp\.weight$", t.name)
            if m and int(m.group(1)) in n:
                jobs.append((t.name, [int(x) for x in t.shape], t.tensor_type, int(t.n_bytes), ("perm", t, int(m.group(1)))))
                continue
            jobs.append((t.name, [int(x) for x in t.shape], t.tensor_type, int(t.n_bytes), ("copy", t)))
    s2re = re.compile(args.shard2_re)
    jobs2 = [j for j in jobs if s2re.search(j[0])]
    jobs = [j for j in jobs if not s2re.search(j[0])]
    w.add_key_value("split.no", 0, VT.UINT16)
    w.add_key_value("split.count", 2, VT.UINT16)
    w.add_key_value("split.tensors.count", len(jobs) + len(jobs2), VT.INT32)
    def add_info(w, name, ne, qt, nbytes):
        if qt in (QT.F32, QT.F16):
            w.add_tensor_info(name, list(reversed(ne)), np.dtype(np.float32 if qt == QT.F32 else np.float16), nbytes, raw_dtype=qt)
        else:
            byte_shape = list(reversed(ne)); byte_shape[-1] = lib.ggml_row_size(int(qt), ne[0])
            w.add_tensor_info(name, byte_shape, np.dtype(np.uint8), nbytes, raw_dtype=qt)
    if args.shard2_from:
        if os.path.exists(out2): os.remove(out2)
        os.link(args.shard2_from, out2)
        print(f"shard 2 hard-linked from {args.shard2_from}")
    else:
        w2 = GGUFWriter(out2, arch, use_temp_file=False)
        w2.add_key_value("split.no", 1, VT.UINT16); w2.add_key_value("split.count", 2, VT.UINT16)
        w2.add_key_value("split.tensors.count", len(jobs) + len(jobs2), VT.INT32)
        for name, ne, qt, nbytes, _ in jobs2:
            add_info(w2, name, ne, qt, nbytes)
        w2.write_header_to_file(); w2.write_kv_data_to_file(); w2.write_ti_data_to_file()
        for name, ne, qt, nbytes, (how, t) in jobs2:
            w2.write_tensor_data(np.asarray(t.data).view(np.uint8).reshape(-1))
        w2.close()
        print(f"shard 2 written: {out2}")
    for name, ne, qt, nbytes, _ in jobs:
        if qt in (QT.F32, QT.F16):
            w.add_tensor_info(name, list(reversed(ne)), np.dtype(np.float32 if qt == QT.F32 else np.float16), nbytes, raw_dtype=qt)
        else:  # quantized: gguf-py takes the byte shape (last dim = bytes per row)
            byte_shape = list(reversed(ne))
            byte_shape[-1] = lib.ggml_row_size(int(qt), ne[0])
            w.add_tensor_info(name, byte_shape, np.dtype(np.uint8), nbytes, raw_dtype=qt)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    total = sum(j[3] for j in jobs)
    print(f"writing {len(jobs)} tensors, {total/2**30:.2f} GiB -> {args.out}", flush=True)

    pool = ThreadPoolExecutor(args.threads)
    t0 = time.time(); done = 0
    for name, ne, qt, nbytes, (how, *rest) in jobs:
        if how == "copy":
            data = np.asarray(rest[0].data).view(np.uint8).reshape(-1)
        elif how == "perm":
            t, il = rest
            raw = np.asarray(t.data).view(np.uint8).reshape(int(t.shape[-1]), -1)
            data = raw[order[il]].reshape(-1)
        else:
            t, il, kind, experts = rest
            n_per_row, rows = int(t.shape[0]), int(t.shape[1])
            raw = np.asarray(t.data).view(np.uint8).reshape(int(t.shape[-1]), -1)   # [n_expert, bytes]
            imat = sum2.get((il, kind)); cnt = counts.get((il, kind))
            if imat is not None:
                imat = imat / np.maximum(cnt, 1.0)[:, None]
                mean = imat[cnt > 0].mean(0) if (cnt > 0).any() else np.ones(n_per_row, np.float32)
                imat = np.where((cnt > 0)[:, None], imat, mean[None, :]).astype(np.float32)
            rsz = lib.ggml_row_size(int(qt), n_per_row)
            out = np.empty(len(experts) * rows * rsz, np.uint8)
            def one(j):
                e = int(experts[j])
                src = np.ascontiguousarray(dequant(raw[e], t.tensor_type, n_per_row))
                im = np.ascontiguousarray(imat[e]) if imat is not None else None
                dst = out[j * rows * rsz:(j + 1) * rows * rsz]
                got = lib.ggml_quantize_chunk(int(qt), src.ctypes.data, dst.ctypes.data, 0, rows, n_per_row,
                                              im.ctypes.data if im is not None else None)
                assert got == rows * rsz, (name, e, got, rows * rsz)
            list(pool.map(one, range(len(experts))))
            data = out
        assert data.nbytes == nbytes, (name, data.nbytes, nbytes)
        w.write_tensor_data(data)
        done += nbytes
        if how == "exps":
            el = time.time() - t0
            print(f"  {name} {qt.name} x{ne[2]}  {done/2**30:.1f}/{total/2**30:.1f} GiB  {el:.0f}s", flush=True)
    w.close()
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "write"])
    ap.add_argument("--src", required=True); ap.add_argument("--imatrix", required=True)
    ap.add_argument("--budget-gib", type=float, required=True)
    ap.add_argument("--gu", default="Q6_K,IQ4_XS,IQ2_S"); ap.add_argument("--dn", default="Q8_0,IQ4_NL,MXFP4")
    ap.add_argument("--out"); ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--shard2-re", default=r"^per_layer_token_embd\.", help="tensors placed in shard 2 (shared between variants)")
    ap.add_argument("--shard2-from", help="existing shard-2 file to hard-link instead of writing it again")
    a = ap.parse_args()
    rs = [GGUFReader(p) for p in shards_of(a.src)]
    plan(a, rs) if a.cmd == "plan" else write(a, rs)
