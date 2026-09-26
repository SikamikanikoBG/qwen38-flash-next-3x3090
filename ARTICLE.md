---
title: "Qwen3.8-Flash-Next (125B) on Three RTX 3090s at 80 Tokens/s: Teaching llama.cpp Which Experts Matter"
published: false
description: "Qwen3.8-Flash-Next doesn't fit in 72 GB of VRAM. Stock llama.cpp runs it at 23 tok/s. A frequency-tiered expert format and a ~400-line patch get it to 80, entirely on the GPUs."
tags: llm, qwen, homelab, ai
cover_image: https://raw.githubusercontent.com/SikamikanikoBG/qwen38-flash-next-3x3090/main/diagrams/cover-qwen38-flash-next.png
---

**TL;DR:** Qwen3.8-Flash-Next is a 125B mixture-of-experts model: on paper it beats the Qwen3.8-27B I run in production everywhere, by +16.5 points on agentic coding. It does not fit in the 72 GB of VRAM my three RTX 3090s have, and stock llama.cpp, spilling a quarter of the experts into system RAM, runs it at **23 tokens/second**. Three changes get it to **80 tokens/second, entirely in VRAM**:

1. **Sort the experts by how often they're used.** They are wildly unequal: the busiest 25% handle 52% of the work.
2. **Store them at three precisions.** Popular experts keep more bits and rare ones get squeezed harder, so the whole model fits on the GPUs.
3. **Patch llama.cpp** (~400 lines) so one routing decision drives three expert tables at once, then add speculative decoding on top.

Quality stays close to the 8-bit reference: **+1.9% perplexity, 91% same next token, GSM8K 95.5%** (the 27B scores 95–96.5% on the same harness). There's also a 256k-context profile that finds a random code hidden 173,000 tokens deep. The trade-offs are real, and I've listed them. Everything is open: [patches, tools, raw measurements](https://github.com/SikamikanikoBG/qwen38-flash-next-3x3090).

> **Update, 26 Sep 2026, after the first day behind my assistant:** the 80 tok/s holds for short prompts. In real agent use, with ~25k-token prompts, sampling and thinking on, the model decodes at **30–45 tok/s**. My first deployment also had a caching mistake that cost ~45 s before every answer; a server config change fixed it (now **0.7 s**). Details in the "real-world update" section below.

## the use case

My box, **vader**, runs [Qwen3.8-27B](https://dev.to/sikamikanikobg/qwen38-27b-on-2x-rtx-3090-my-first-local-model-i-actually-trust-fpi) around the clock: two RTX 3090s, vLLM, about 135 tokens/second. It's the first local model I actually trust with agent work, and it does most of the work my assistant does.

Then Qwen shipped **Qwen3.8-Flash-Next**, the preview of their Qwen4 architecture. Their own model card puts it ahead of my 27B on almost everything I care about:

| benchmark (Qwen's card) | Flash-Next | 3.8-27B | gap |
|---|---|---|---|
| DeepSWE (agentic coding) | 58.7 | 42.2 | **+16.5** |
| JobBench (professional tasks) | 55.7 | 33.4 | **+22.3** |
| SWE-bench Multilingual | 81.0 | 73.8 | +7.2 |
| Toolathlon (tool use) | 73.5 | 67.1 | +6.4 |
| HLE | 35.9 | 30.8 | +5.1 |
| GPQA Diamond | 91.7 | 89.2 | +2.5 |
| IFBench | 81.3 | 79.5 | +1.8 |

So the goal was simple to state: **the smarter model, on the same three cards, with decode and time-to-first-token that still feel interactive.** No new hardware. I told my AI engineer (Claude Code, more on that at the end) that "no" was not an acceptable answer.

## the box

- **GPUs:** 3x RTX 3090, 24 GB each: 72 GB of VRAM. No NVLink, and no PCIe peer-to-peer (the board has no Resizable BAR), so the cards can't talk to each other directly.
- **CPU:** dual Xeon E5-2660 v4, 56 threads.
- **RAM:** 60 GB. Only **one** 32 GB stick per CPU socket, so it's single-channel, and that matters a lot below.
- **Disk:** one 1 TB NVMe.

## a small dictionary before we start

If you know what a KV cache is, skip this. If you don't, it's all you need for the rest of the article.

- **Parameters / weights.** The numbers a model learned. "125B" means 125 billion of them. Each one normally takes 2 bytes, so 125B parameters is ~250 GB, far more than 72 GB of VRAM.
- **VRAM.** The GPU's own memory. It's about 20–50x faster than system RAM *on this box*. A model runs fast only if what it needs for each word lives in VRAM.
- **Quantization.** Storing each weight with fewer bits (8, 4, 3…) instead of 16. Like saving a photo as a smaller JPEG: smaller file, a little less detail. "Q8_0", "Q4_K", "IQ3_S", "MXFP4" are llama.cpp's names for different bit-budgets and recipes.
- **Mixture of experts (MoE).** Instead of one giant network, each layer has many small "expert" networks, 512 of them here, and a **router** picks the best 10 for each word. So the model *knows* 125B parameters' worth of things but only *computes* with about 6B per word. That's why it can be fast, and also why it's huge.
- **Token.** A word or word-piece. **Decode speed** (tokens/second) is how fast the answer streams out.
- **Prefill / time to first token (TTFT).** Before answering, the model reads your whole prompt. That's prefill, and TTFT is how long you wait before the first word appears.
- **Context / KV cache.** How much text the model can hold in mind at once ("256k context" is about 500 pages), and the memory it uses to remember it.
- **n-gram embedding.** New in this model: a 51 GB lookup table indexed by 2–3 word phrases. It's only ever *looked up*, a few rows per word, so it can live on the NVMe drive.
- **Speculative decoding / MTP.** A small built-in "draft head" guesses the next few words, and the big model checks all the guesses in one pass. Correct guesses are free words.
- **KL divergence (KLD).** How far the compressed model's word probabilities drift from the original's. 0 means identical; lower is better. It's the most sensitive quality meter there is.
- **llama.cpp / GGUF.** The inference engine I patched, and its model-file format.

## the problem, in one picture

![Where Qwen3.8-Flash-Next (125B) lives: before and after](https://raw.githubusercontent.com/SikamikanikoBG/qwen38-flash-next-3x3090/main/diagrams/before_after_qwen38-flash-next.png)

In 16-bit the model is 360 GB. Even the popular 4-bit build ([unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)) is 111 GB: 72 GB of experts, 27 GB of n-gram table (which can stay on disk) and a few GB of everything else. The experts alone don't fit in 72 GB of VRAM once the rest of the model and the context need room too. So llama.cpp does the sensible thing: it puts the overflow experts in system RAM and computes them on the CPU.

On this machine that means **23 tokens/second**. With one memory stick per socket, the CPU reads those experts about ten times slower than a GPU would. And for every word, the work bounces between the GPUs and the CPU dozens of times, so each GPU waits.

## step 1: the obvious setup, and the first trap

The first run with stock settings also showed prefill all over the place (47 to 306 tokens/second on the same 4k prompt) and decode sagging to 10 tok/s. Thread states told the story: the main thread sat in `D` (waiting on disk). llama.cpp memory-maps the model file, loading the GPU part streams 70+ GB through the page cache, and the CPU-side experts kept getting evicted and re-read from NVMe.

`--load-mode none` loads the CPU-side weights into RAM and keeps them there. Also, **don't download a 188 GB file while you benchmark**: my download filled the page cache and pushed the server into swap. With both fixed, prefill doubled (**658–792 tok/s**) and decode steadied at **23–26 tok/s**.

Speculative decoding (MTP) barely helped: 24–29 tok/s. The reason matters for everything that follows. A draft of 5 words means the big model checks 6 words at once. Each word picks 10 experts, so one check can touch up to 60 *different* experts per layer instead of 10, and the slow ones in CPU RAM get hit harder. **As long as experts live in CPU RAM, nothing else will make this fast.**

## step 2: the experts are not equal

Here's the observation the whole project rests on. llama.cpp's calibration data (the *importance matrix* unsloth publishes with their quants) records how often each of the 24,576 experts was picked on real text:

![Routing skew: a few experts do most of the work](https://raw.githubusercontent.com/SikamikanikoBG/qwen38-flash-next-3x3090/main/charts/routing_skew.png)

In the average layer, **the busiest 25% of experts handle 52% of the tokens, and the busiest 80% handle 95%.** Stock llama.cpp can only place *whole layers* of experts: a layer's 512 experts are one tensor, all on the GPU or all on the CPU. It offloads a quarter of the experts, including popular ones, so a quarter of the expert work lands on the slow path.

## step 3: split every layer into hot and cold

The first patch splits every layer's expert tensor in two at load time:

- **hot experts** go to the GPUs;
- **cold experts** go to CPU RAM;
- the **router** is permuted so its choices land in the right half.

With the same 19% of expert bytes on the CPU, the cold experts now serve only **4.5% of routed tokens** instead of ~19%, about 4x less slow-path traffic.

Getting it right took four bugs, each instructive enough to list in its own section below. The result is **30–32 tok/s** with correct answers. That's +30%, but it's not the leap it should be, so I profiled it:

| where one decoded word spends its time | |
|---|---|
| kernel launches | ~4,300 |
| dense weights (attention, DeltaNet, shared expert), 8-bit | ~4.9 ms |
| routed experts | ~2.5 ms |
| ~3,500 tiny element-wise kernels | ~6–7 ms |
| GPU↔CPU synchronisations | ~390 |

The CPU was barely reading any experts anymore; it was the *round trips* that cost. Every layer handed work to the CPU and waited for the answer. Then a cheap experiment: drop the cold experts entirely (quality garbage, speed only) and see how fast it goes fully on GPU: **52 tok/s**, and **86–88 tok/s** with speculative decoding. That was the target. The question became: **how do you fit every expert in VRAM without wrecking quality?**

## step 4: three shelves, all on the GPU

![How one word gets written](https://raw.githubusercontent.com/SikamikanikoBG/qwen38-flash-next-3x3090/main/diagrams/one_word.png)

The idea: if some experts do most of the work, **give them more bits and squeeze the rarely-used ones harder**. Everything still fits in VRAM, and most tokens still see a well-preserved expert. Think of a library that keeps its bestsellers in hardcover and prints the rarely-borrowed titles as compact paperbacks, so the whole collection fits in the building.

![What I built](https://raw.githubusercontent.com/SikamikanikoBG/qwen38-flash-next-3x3090/main/diagrams/pipeline.png)

Two pieces make it work.

**An offline re-packer** ([`tools/expert_tiers.py`](https://github.com/SikamikanikoBG/qwen38-flash-next-3x3090/blob/main/tools/expert_tiers.py)). It starts from the 8-bit model (188 GB) and uses the importance matrix for both *how often* each expert is used and *which inputs matter* inside it. A greedy planner fills a VRAM budget: each byte goes to the expert where it removes the most expected error. Every expert is then re-quantized with llama.cpp's own quantizers, and the file is written with the experts reordered hottest-first as three tensors per layer. One wrinkle: the down-projection matrices have rows 640 wide, which the fancy 2–3-bit formats can't handle (they need multiples of 256). So they get their own ladder: IQ4_NL / MXFP4 instead of IQ4_XS / IQ3.

**A patched llama.cpp** ([`patches/`](https://github.com/SikamikanikoBG/qwen38-flash-next-3x3090/tree/main/patches)). `mul_mat_id`, the operation that runs "each word through its chosen experts", learns to take an **id range**. Each shelf's matrix multiply sees the router's full choice list, computes only the ids that fall on its shelf, and writes zeros for the rest. The three results simply add up. That meant touching the CPU kernels, three CUDA paths (decode, prefill, and the fused gate+up+activation kernel) and the loader. On top sits Qwen's multi-token-prediction draft head, from a [not-yet-merged llama.cpp PR](https://github.com/ggml-org/llama.cpp/pull/28243), re-quantized to 4 bits so it fits too.

Result, in one chart:

![Decode journey](https://raw.githubusercontent.com/SikamikanikoBG/qwen38-flash-next-3x3090/main/charts/decode_journey.png)

## the four bugs, briefly

For people who'll try this, and because two of them are delightful.

1. **Duplicate experts crash the CUDA MoE kernel.** My first version pointed "not on this shelf" at expert 0. A word with two cold experts then had expert 0 twice, and CUDA's expert-grouping helper assumes each expert appears at most once per word. It silently dropped a slot, and a later kernel read an unwritten index. Fixed by teaching the kernels a real "skip".
2. **A flag stored where a precision setting lives.** I marked the skip-capable nodes in `op_params[0]`, which llama.cpp already uses for the accumulator precision. The flag was quietly overwritten. It moved to slot 7, with a magic value.
3. **The unsigned ternary.** `ids ? ids[i] : blockIdx.x`, where `blockIdx.x` is unsigned. C++ promotes the whole expression to unsigned, so my `-1` ("skip") became 4,294,967,295 and the kernel read 4 billion rows past the buffer. `compute-sanitizer` found it in one run.
4. **Fused kernels hide the tag.** For decode, CUDA fuses gate, up and activation into one kernel whose "destination" is the activation node, not the matrix multiply that carries my id-range tag. The first id-range build printed `//////////`. The kernel now looks up the tag on its source node.

## how much quality did it cost?

The honest meter is KL divergence against the 8-bit model over 24,576 tokens of Wikipedia text:

![Quality vs size](https://raw.githubusercontent.com/SikamikanikoBG/qwen38-flash-next-3x3090/main/charts/quality_vs_size.png)

| model | expert size | fits in VRAM | perplexity vs 8-bit | KLD | same top token |
|---|---|---|---|---|---|
| unsloth UD-Q4_K_XL (stock) | 71.7 GiB | no, 25% in RAM | +0.9% | 0.045 | 93.7% |
| **tiered, 128k profile** | 53.0 GiB | **yes** | **+1.9%** | **0.091** | **91.1%** |
| uniform IQ3_S / MXFP4, same size | 52.2 GiB | yes | +3.0% | 0.095 | 91.0% |
| tiered, 256k profile | 49.5 GiB | yes | +3.2% | 0.113 | 90.1% |

What I take from it:

- **Fitting in VRAM costs quality, and there's no way around that.** 20 GB less expert weight roughly doubles the KL divergence of the stock 4-bit build. That's physics, not a bug.
- **At equal size, tiering beats uniform**: perplexity drift 1.9% vs 3.0%. But the gain is smaller than my planner predicted: its error model is crude, and at this size the total byte count dominates. There's more to win with a better tier recipe.
- **On a task, it holds up.** GSM8K (200 grade-school maths questions, greedy, no thinking) scores **95.5%**. The 27B's published number on the same harness is 95.0–96.5%. GSM8K is saturated, so it mainly proves nothing broke. The model card's gains on agentic work are where the upgrade should show.

## speed vs context, and the 256k profile

![Speed vs context](https://raw.githubusercontent.com/SikamikanikoBG/qwen38-flash-next-3x3090/main/charts/speed_vs_context.png)

Both profiles start at **~80 tokens/second** on a short prompt and slow down as the prompt grows. The per-word attention and indexer work grows with context, and the draft head's guesses get a little worse too. Typical numbers, averaging two runs:

| prompt | 128k profile: decode / time to first token | 256k profile: decode / time to first token |
|---|---|---|
| 4k tokens | 82 tok/s / 7–9 s | 67 tok/s / 7–8 s |
| 15k tokens | 66 tok/s / 18–22 s | 58 tok/s / 19–22 s |
| 30k tokens | 58 tok/s / 35 s | 54 tok/s / 38 s |
| 61k tokens | 53 tok/s / 76–79 s | 51 tok/s / 82 s |
| 95k / 122k tokens | 39 tok/s / 140 s | 35 tok/s / 190 s |

Prefill runs at **540–860 tokens/second**. That's the weak spot next to the 27B's ~1,300, and it's where I'd look next: it's also the part that makes a 120k-token prompt a three-minute wait.

The 256k profile needed two more compromises. The KV cache drops to 8-bit, and experts shrink to 49.5 GiB, because the sparse-attention indexer's scratch memory grows with context and pushed a card out of memory at ~200k on the first try. After rebalancing layers across the cards, it read **173,692 tokens of Wikipedia with a random 10-character code hidden at 45% depth, and returned the code: exactly right (`6BC5XYE8FS`). A second run with the code at 90% depth of 112,711 tokens was also exact.** It took 6.1 minutes (472 tokens/second) to read that much. Long-context prefill on this architecture is still the slow part.

## what it costs to run

Measured at the cards (`nvidia-smi`, 5 Hz, three 1,024-token generations per profile): the three 3090s draw **~540 W together while decoding** and ~124 W at rest with the model loaded. That's **7.5–8.4 joules per token**, 2.1–2.3 kWh per million tokens. On my tariff (0.30 BGN/kWh day, 0.18 night) **a million generated tokens cost 0.63–0.70 BGN in daytime, 0.38–0.42 BGN at night**, about $0.35–0.40. [The 27B costs 0.21–0.34 BGN](https://dev.to/sikamikanikobg/qwen38-27b-on-one-rtx-3090-vs-two-20-decode-14-cold-prefill-and-3x-on-cached-prompts-55kc) for the same million, so the bigger brain costs roughly twice as much per word, all three cards included. It's still coffee money.

## real-world update: day one behind my assistant

Benchmarks are one request with a short prompt and greedy decoding. My assistant is none of those things, so after a day of it running Jarvis, here is what the server logs say about real requests:

| | a real assistant turn | the benchmark |
|---|---|---|
| prompt | ~25,000 tokens (system prompt, tools, memory) | 41 tokens |
| sampling | temperature 0.7, thinking on | greedy |
| tokens per verify step (draft acceptance) | 2.6 (39%) | 3.4 (62%) |
| decode | **41–45 tok/s** (30–45 is what it feels like) | 80 tok/s |

**Decode: 30–45 tok/s, not 80.** Two things compound. The prompt is deep, and every decoded word pays attention over it; the context chart above already shows ~50 tok/s at 60k. And the draft head guesses sampled text, tool-call JSON and reasoning worse than greedy prose, so speculative decoding saves less. That's the honest number for agent work on this box. For comparison, the 27B benchmarks at ~135 tok/s on this box, so the upgrade is intelligence, not speed.

**Time to first token: the mistake was mine.** My assistant runs several *roles*: chat, planner, classifier, triage, background jobs. Each has its own ~25k-token prompt prefix. I deployed the server with one slot (`-np 1`), so every time the role changed, the cached prefix was thrown away and **25,000 tokens were re-read from scratch: ~45 seconds before every answer**. The fix is four slots sharing one KV pool, so every role keeps its prefix warm:

| returning to a 14k-token prompt after another role ran | before (`-np 1`) | after (`-np 4 -kvu`) |
|---|---|---|
| time to first token | 20–45 s | **0.6–0.7 s** |

It isn't free: four slots need a little more memory, so the shared pool dropped from 128k to 96k tokens (128k with four slots ran out of VRAM). A single request can still use all 96k.

**Schedule the background jobs apart.** A model at 30–45 tok/s with 25k-token prompts spends minutes per agent task. My assistant had 13 scheduled jobs between 06:30 and 09:00 on Mondays, four of them at the same minute. I've spread them 30 minutes apart (05:00 to 11:30) so they stop queueing behind each other. I'll report after a week whether that's enough.

**Images and video work too,** once the vision projector is loaded (`--mmproj`, on the GPU with the most free memory: 4 s per photo, against 61 s on the CPU) and the container has `ffmpeg` for video (a 5-second clip in 6.4 s). The repo has the exact launcher.

## what I'd not claim

- **It's single-user.** Every speed number here is one request at a time, and real agent turns decode at 30–45 tok/s, not 80. The 27B on vLLM batches many users far better.
- **The 27B is still faster.** About 135 vs 80 tok/s decode, and roughly 1,300 vs 540–860 tok/s prefill. The 125B is smarter, not quicker.
- **It uses all three cards.** You can't also keep the 27B running.
- **The quality numbers are wikitext + GSM8K.** They say the compression is gentle. They don't prove the agentic gains survive intact; that needs agent benchmarks I haven't run yet.
- **The tier planner is a heuristic.** The measurements say it helps; a calibrated one would help more.

## reproduce it

Everything is in **[SikamikanikoBG/qwen38-flash-next-3x3090](https://github.com/SikamikanikoBG/qwen38-flash-next-3x3090)**: both llama.cpp patches, the re-packer, the benchmark scripts, and every raw number behind the charts. The short version:

```bash
# llama.cpp at 81bc6b8 + MTP PR #28243 + the tiering patch
git apply patches/0001-qwen4exp-mtp-pr28243.patch patches/0002-tiered-experts-hot-cold-split.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j

# re-pack the 8-bit model into three precision shelves (53 GiB of experts)
python tools/expert_tiers.py write --src Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
  --imatrix imatrix_unsloth.gguf --budget-gib 53 \
  --gu Q6_K,IQ4_XS,IQ3_XXS --dn Q8_0,IQ4_NL,MXFP4 --out fn-tier53.gguf

# serve: all layers on GPU, 4 slots sharing a 96k KV pool, vision, MTP drafting 4 tokens
llama-server -m fn-tier53-00001-of-00002.gguf -ngl 99 -ts 16,16,16 -c 98304 -np 4 -kvu \
  -b 1024 -ub 256 -fa on --jinja \
  --mmproj mmproj-F16.gguf -mmdev CUDA1 --image-max-tokens 1024 \
  -md mtp-shared-iq4.gguf --spec-type draft-mtp --spec-draft-n-max 4
```

## credits

- **Qwen**, for the model and an architecture that rewards this kind of work.
- **unsloth**, for the GGUFs, the MTP draft files and the importance matrix.
- **llama.cpp and ggml**, plus the authors of the [Qwen4Exp MTP PR](https://github.com/ggml-org/llama.cpp/pull/28243) and the [sparse-attention decay issue](https://github.com/ggml-org/llama.cpp/issues/28734).
- **[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)**, whose rigor on the 27B set the bar for measuring this.

*This project, from the patches and the tools to the measurements and this write-up, was done with **Claude Code** acting as my AI engineer, working on my hardware under my direction. The numbers are from vader, the raw data is in the repo, and the mistakes are both of ours.*
