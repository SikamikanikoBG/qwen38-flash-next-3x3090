#!/usr/bin/env python3
"""Render the article/README charts from results/*.jsonl and results/expert_counts.npy.
usage: python tools/make_charts.py   (writes charts/*.png)"""
import json, os, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "charts")
os.makedirs(OUT, exist_ok=True)

# validated categorical slots (dataviz reference palette, light mode) + neutral inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.titleweight": "bold",
    "axes.titlesize": 15, "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 14,
})

def rows(name):
    p = os.path.join(RES, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

speed = rows("speed.jsonl")
def med(tag, depth, key="decode_tps", warm_only=True):
    v = [r[key] for r in speed if r["tag"] == tag and r["depth"] == depth]
    if warm_only and len(v) > 1:
        v = v[1:]  # first request of a depth pays one-time warm-up (page faults, CUDA graph capture)
    return st.median(v) if v else None

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", name)

# ---------------------------------------------------------------- 1. the decode journey
steps = [
    ("stock llama.cpp\nQ4_K_XL, experts spill to CPU", med("base-fit-q4", 0), MUTED),
    ("+ weights pinned in RAM\n(load-mode none)", med("nommap-q4-quiet", 0), MUTED),
    ("+ hot/cold expert split\n(cold experts on CPU)", med("split58-skip-q4", 0), BLUE),
    ("tiered experts, 100% in VRAM\n+ MTP speculative decoding", med("tier53-range2", 0), BLUE),
]
fig, ax = plt.subplots(figsize=(10, 4.6))
y = np.arange(len(steps))[::-1]
for yi, (lab, v, c) in zip(y, steps):
    ax.barh(yi, v, height=0.56, color=c, zorder=3)
    ax.text(v + 1.5, yi, f"{v:.0f} tok/s", va="center", color=INK, fontsize=12, fontweight="bold")
ax.set_yticks(y, [s[0] for s in steps], fontsize=10.5)
ax.axvline(135, color=ORANGE, lw=2, ls=(0, (4, 3)), zorder=2)
ax.text(133, len(steps) - 0.55, "Qwen3.8-27B (dense) in\nproduction: ~135 tok/s", ha="right", va="center", color=INK2, fontsize=10)
ax.set_ylim(-0.5, len(steps) - 0.2)
ax.set_xlim(0, 150); ax.set_xlabel("single-user decode speed, tokens/second (higher is better)")
ax.grid(axis="x", color=GRID, zorder=0); ax.tick_params(axis="y", length=0)
ax.set_title(f"Qwen3.8-Flash-Next (125B MoE) on 3x RTX 3090: {steps[0][1]:.0f} \u2192 {steps[-1][1]:.0f} tokens/s")
save(fig, "decode_journey.png")

# ---------------------------------------------------------------- 2. routing skew
cnt = np.load(os.path.join(RES, "expert_counts.npy"))[:48]
share = np.sort(cnt, axis=1)[:, ::-1]
share = np.cumsum(share, axis=1) / share.sum(axis=1, keepdims=True)
x = np.arange(1, share.shape[1] + 1) / share.shape[1] * 100
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.fill_between(x, share.min(0) * 100, share.max(0) * 100, color=BLUE, alpha=0.15, lw=0, label="range over the 48 layers")
ax.plot(x, share.mean(0) * 100, color=BLUE, lw=2.5, label="average layer")
ax.plot([0, 100], [0, 100], color=MUTED, lw=1.5, ls=(0, (4, 3)), label="if every expert were used equally")
for p in (25, 50, 80):
    v = share.mean(0)[int(p / 100 * 512) - 1] * 100
    ax.plot([p], [v], "o", color=BLUE, ms=8, mec=SURF, mew=2, zorder=5)
    ax.annotate(f"top {p}% of experts\nserve {v:.0f}% of tokens", (p, v), xytext=(p + 3, v - 16), fontsize=10.5, color=INK)
ax.set_xlim(0, 100); ax.set_ylim(0, 102)
ax.set_xlabel("experts in a layer, most-used first (%)"); ax.set_ylabel("share of routed tokens served (%)")
ax.grid(color=GRID); ax.legend(frameon=False, loc="lower right", fontsize=10.5)
ax.set_title("A few experts do most of the work: routing is heavily skewed")
save(fig, "routing_skew.png")

# ---------------------------------------------------------------- 3. quality vs size
q = [  # (label, expert GiB, mean KLD, fits in VRAM)
    ("UD-Q4_K_XL (stock)\n25% of experts in CPU RAM", 71.7, 0.0448, False),
    ("tiered-53 (128k profile)", 53.0, 0.0906, True),
    ("uniform IQ3_S, same size", 52.15, 0.0947, True),
    ("tiered-50 (256k profile)", 49.5, 0.1130, True),
]
fig, ax = plt.subplots(figsize=(10, 5))
for lab, gib, kld, gpu in q:
    ax.plot(gib, kld, "o", ms=11, color=(ORANGE if "uniform" in lab else BLUE) if gpu else MUTED, mec=SURF, mew=2, zorder=4)
    dx, ha = (1.0, "left") if gib < 70 else (-1.0, "right")
    dy = 0.004 if "uniform" in lab else (-0.006 if "tiered-53" in lab else 0)
    ax.annotate(f"{lab}\nKLD {kld:.3f}", (gib, kld), xytext=(gib + dx, kld + dy), ha=ha, va="center", fontsize=10, color=INK)
ax.axvspan(40, 54.5, color=BLUE, alpha=0.06, zorder=0)
ax.text(40.6, 0.123, "fits in 3x24 GB VRAM\nwith the rest of the model", color=INK2, fontsize=10, va="top")
ax.set_xlim(40, 76); ax.set_ylim(0.03, 0.125)
ax.set_xlabel("size of the expert weights (GiB)")
ax.set_ylabel("KL divergence vs 8-bit reference\n(lower = closer to the original)")
ax.grid(color=GRID)
ax.set_title("Quality vs size: what 20 GB less expert weight costs")
save(fig, "quality_vs_size.png")

# ---------------------------------------------------------------- 4. speed vs context depth
prof = [("128k profile (tiered-53)", "sweep-t53", BLUE), ("256k profile (tiered-50)", "sweep-t50", ORANGE)]
depths = sorted({r["depth"] for r in speed if r["tag"].startswith("sweep-")})
if depths:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for lab, tag, c in prof:
        ds = [d for d in depths if med(tag, d) is not None]
        pt = [st.median([r["prompt_n"] for r in speed if r["tag"] == tag and r["depth"] == d]) / 1000 for d in ds]
        a1.plot(pt, [med(tag, d) for d in ds], "-o", color=c, lw=2, ms=8, mec=SURF, mew=2, label=lab)
        a2.plot(pt, [med(tag, d, "ttft_s") for d in ds], "-o", color=c, lw=2, ms=8, mec=SURF, mew=2, label=lab)
    a1.set_title("Decode speed vs prompt length"); a1.set_ylabel("tokens/second"); a1.set_ylim(0, None)
    a2.set_title("Time to first token vs prompt length"); a2.set_ylabel("seconds")
    for a in (a1, a2):
        a.set_xlabel("prompt length (thousand tokens)"); a.grid(color=GRID); a.legend(frameon=False, fontsize=10)
    save(fig, "speed_vs_context.png")
