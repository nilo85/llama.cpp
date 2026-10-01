# Engineering Notes: Running Qwen3.8-Flash-Next on Dual Intel Arc Pro B70 with llama.cpp

**Audience:** llama.cpp maintainers and contributors (upstream + forks).
**Source material:** an exported opencode research/engineering session (133 messages, ~1.7 MB JSON) conducted on 2026-09-29 (session created 2026-09-29 ~23:44 UTC, last updated ~2026-09-30). **Purpose of this document:** reconstruct, in order, *what was investigated, why, how each claim was sourced and checked, and what was concluded*, so that a llama.cpp maintainer can pick up the thread without re-deriving it.

> **Scrubbing note.** This article deliberately excludes any credentials, tokens, personal filesystem details, and machine identifiers found in the raw session. Only hardware class descriptions, repo/issue numbers, benchmark numbers and technical facts are retained.

---

## 0. Executive summary (updated as the article grew)

The session set out to answer a single user question: *"research the best possible setup for high-quality, fast inference of `unsloth/Qwen3.8-Flash-Next-GGUF` in llama.cpp on a dual-Intel-Arc-Pro-B70 workstation."* It expanded, in practice, into **three distinct bodies of work**:

1. **Model + hardware capability research** — establishing what Qwen3.8-Flash-Next (`qwen4exp` architecture in GGUF) actually is, what its memory footprint is, and what llama.cpp's `qwen4exp` support currently does and does not do (September 2026 state of mainline).
2. **Backend / multi-GPU feasibility research on Intel Arc** — Vulkan vs SYCL on Battlemage, dual-B70 behaviour, layer-split vs tensor-split, known driver-level failure modes, and the specific GitHub issues/PRs that constitute the known-bad surface for this configuration.
3. **A costed upstreaming work plan** — the pivot from "how do I configure this?" to *"what should be changed in llama.cpp itself?"*, answered as a tiered plan (cherry-pick / write new code / deliberately not worth doing), then packaged into **ten user-story files** (`us_*.md`) for approaching upstream.

> **Correction, issued after reading all 133 messages (see §16).** Item 3 above originally read *"Hands-on empirical validation in a browser — the user ran an actual llama.cpp server … extracting a large number of measured numbers."* **That was wrong.** No llama.cpp server was run, no benchmark harness was executed, and the session contains **no measurements of its own**. The browser work was *Reddit and GitHub research only*. Every throughput figure in this document comes from an external source and is attributed to one. The benchmark ladder, sanity bands and launch configuration are a **plan awaiting hardware**, not a report of results.

The headline technical findings are, in short:

- The model is a **hybrid-attention + MoE "preview of the Qwen4 architecture"**: 48 layers, hidden 2560, **512 experts per layer with 10 routed + 1 shared active**, full attention every 4th layer, gated-DeltaNet SSM elsewhere, plus a large n-gram ("PLE" — parametrised lookup embedding) table. Total ~56B/512-expert labelled model of size_label `512x56B`, non-embedding ~125B params, ~6B active.
- It **does not fit** in 64 GB of combined VRAM at high quant; fitting it means aggressive offload of experts and/or relying on the model's *lazy n-gram table* streaming behaviour.
- llama.cpp mainline (as of late Sept 2026) supports `qwen4exp` but **MTP (multi-token-prediction) draft heads do not work on stock upstream** — that requires either the `unslothai/llama.cpp` fork or upstream PR **#28243**.
- The dual-Intel-Arc-B70 + layer-split combination has a **dense cluster of open, un-triaged bug reports** in llama.cpp, which is the single biggest risk to the user's goal.

---

## 1. The user's question, verbatim in substance

System as described by the user:

| Component | Spec as given |
|---|---|
| CPU | Intel Core Ultra 2 (Arrow Lake / Core Ultra 200-series class) |
| RAM | 64 GB DDR5, dual channel |
| Storage | 2 TB NVMe SSD |
| GPU 0 | Intel Arc Pro B70 (32 GB) on **PCIe 5** (CPU-direct, x16) |
| GPU 1 | Intel Arc Pro B70 (32 GB) on **PCIe 4** via the **southbridge / chipset** |
| Stack | llama.cpp |

Explicit instructions from the user:

- Research the best possible **high-quality and fast** way to run `unsloth/Qwen3.8-Flash-Next-GGUF`.
- Research **all** llama.cpp GitHub discussions and PRs.
- Consider whether learnings from `Niko1221/Strata` (an NVIDIA-focused one-click inference engine for the same model) transfer.
- Search r/LocalLLaMA for tips and insights.
- **"DO NOT ASSUME ANYTHING FROM TRAINING MEMORY"** — every claim had to be re-derived from live sources.

That last instruction shaped the whole session: the agent deliberately refused to rely on priors about the model, the GPUs, or llama.cpp, and instead fetched HF metadata, parsed GGUF headers byte-by-byte, and enumerated live GitHub issues/PRs via the API.

---

## 2. Phase 1 — Establishing ground truth about the model

### 2.1 What was fetched

| Source | Purpose | Result |
|---|---|---|
| `huggingface.co/api/models/unsloth/Qwen3.8-Flash-Next-GGUF` | repo metadata | `pipeline_tag: image-text-to-text` (i.e. **multimodal**), 1.64 M downloads, last modified 2026-09-02, 60 files, license `qwen-community-1.0` |
| HF model card `README.md` (59 KB) | architecture + usage | pulled to a scratch file and read in slices |
| HF `tree/main?recursive=true` | per-quant byte sizes | see table below |
| HF `MTP/README.md` | MTP draft-head docs | pulled and read |
| `Niko1221/Strata` GitHub README (14.5 KB) + repo metadata | the reference "small-VRAM" engine | 1647 stars, C++, last updated 2026-09-29 |

### 2.2 File sizes actually on HF (from the tree API)

| Quant dir | Total size |
|---|---:|
| BF16 (8 shards) | 354.03 GB |
| Q8_0 (6 shards) | 188.23 GB |
| UD-Q6_K_XL | 169.17 GB |
| UD-Q5_K_XL | 158.29 GB |
| **UD-Q4_K_XL** | **111.33 GB** |
| UD-IQ4_XS | 93.68 GB |
| UD-Q3_K_XL | 89.99 GB |
| UD-IQ3_XXS | 81.96 GB |
| UD-Q2_K_XL | 78.87 GB |
| UD-IQ1_M | 74.54 GB |
| UD-IQ1_S | 72.55 GB |
| MTP draft heads (6 files) | 24.62 GB total |
| mmproj (BF16 / F16) | 0.91 / 0.90 GB |
| imatrix (`imatrix_unsloth.gguf_file`) | 0.58 GB |

**Immediate consequence:** with 32+32 GB VRAM and 64 GB system RAM, *nothing above UD-IQ4_XS / UD-Q3_K_XL can ever be fully resident*. Every configuration is an offload configuration. This single fact drives the entire rest of the analysis and is why the session kept returning to MoE expert placement, `--split-mode`, and lazy n-gram table streaming.

The MTP draft-head files matter because speculative decoding is the cheapest large win available, and they are *separate* small files (2.6 GB recommended) that must be loaded in addition to the main model — consuming scarce VRAM/RAM budget.

### 2.3 Architecture, read straight out of the GGUF header

Rather than trust the model card, the agent wrote a minimal pure-Python GGUF header parser (`struct.unpack`-based, handling all GGUF v1–v12 scalar/array types) and range-downloaded the **first 4 MB** of `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` via an HTTP `Range` request. That is a technique worth highlighting for maintainers: **you can learn the full architecture from 4 MB, no need to download 111 GB.**

Header results (GGUF v3):

```
general.architecture = qwen4exp
general.description  = "A Preview of the Qwen4 Architecture"
general.size_label   = 512x56B
general.name         = Qwen3.8 Flash Next
qwen4exp.block_count                 = 48
qwen4exp.context_length              = 262144        (256K native)
qwen4exp.embedding_length            = 2560
qwen4exp.attention.head_count        = 24
qwen4exp.attention.head_count_kv     = 2             (very asymmetric GQA)
qwen4exp.attention.key_length        = 256
qwen4exp.attention.value_length      = 256
qwen4exp.expert_count                = 512           (per layer!)
qwen4exp.expert_used_count           = 10
qwen4exp.expert_feed_forward_length  = 640
qwen4exp.expert_shared_feed_forward_length = 640
qwen4exp.full_attention_interval     = 4             (1 full-attn layer per 4)
qwen4exp.rope.freq_base              = 10000000.0
qwen4exp.rope.dimension_sections     = [11, 11, 10, 0]   (partial/sectioned rope)
qwen4exp.ssm.conv_kernel             = 4
qwen4exp.ssm.state_size              = 128
qwen4exp.ssm.group_count             = 16
qwen4exp.ssm.time_step_rank          = 48
qwen4exp.ssm.inner_size              = 6144
qwen4exp.hyper_connection.count      = 4
qwen4exp.hyper_connection.low_rank   = 320
qwen4exp.attention.indexer.head_count = 4
qwen4exp.attention.indexer.key_length = 128
qwen4exp.attention.indexer.top_k      = 2048          (Qwen sparse attention indexer)
qwen4exp.attention.compress_ratios   = [0,0,0,4, ...] (4 per full-attn layer)
qwen4exp.ple.layers                   = [1]
qwen4exp.ple.ngram_size              = 3
qwen4exp.ple.heads_per_ngram         = 8
qwen4exp.ple.conv_kernel             = 4
qwen4exp.ple.eos_token_id            = 248044
qwen4exp.embedding_length_per_layer_input = 160
qwen4exp.ple.layer_multipliers       = [23703573157769, 20109073645365, 8052911324071]
qwen4exp.ple.head_offsets            = [0, 20000003, 40000026, ... step 20 000 003]
qwen4exp.ple.head_vocab_sizes        = [20000003, 20000023, ...]
tokenizer.ggml.model = gpt2 ; tokenizer.ggml.pre = qwen35
```

The parser then stopped at `tokenizer.ggml.tokens` (the token array exceeds the 4 MB window) — expected and harmless; everything architectural precedes it.

**Reading of those numbers, in the session's own reasoning:**

- 48 layers × 512 experts = **24 576 expert modules** per forward pass candidate set; only 10 routed + 1 shared are activated per token. This is an extremely sparse MoE — the defining performance property of the model and the reason expert *placement* matters more than raw quant quality.
- The `ple.*` keys are the **"PLE" / n-gram embedding table**: 16 n-gram heads × ~20 M entries × 160 dims, i.e. on the order of **51 B parameters** of *lookup table*. Model-card framing calls this a "51B n-gram embedding". This table is enormous, is essentially never fully resident, and is the reason `lazy` tensor reads / `mmap` / SSD-streaming support in llama.cpp is a first-class concern for this model.
- `full_attention_interval = 4` + `ssm.*` keys ⇒ classic **hybrid Gated DeltaNet (3 layers) : full attention (1 layer)** pattern.
- `attention.indexer.*` + `compress_ratios` ⇒ **Qwen sparse attention** (DSA-style learned top-k indexer with top_k = 2048).
- `hyper_connection.count/low_rank` ⇒ low-rank hyper-connections, a Qwen4-preview feature.
- `rope.dimension_sections` ⇒ sectioned/partial RoPE.
- Asymmetric GQA (24 query heads : 2 KV heads) with head dim 256 keeps KV cache cheap, which is what makes 256K–1M context tractable at all.
- Recommended sampling from the model card: thinking mode `temp=1.0, top_p=0.95, top_k=20, min_p=0.0`; instruct/non-thinking `temp=0.7, top_p=0.80, top_k=20, presence_penalty=1.5`. Thinking is **on by default**, controllable via `enable_thinking`, `preserve_thinking`, `reasoning_effort` (`xhigh|medium|low`).

### 2.4 Strata, and what transfers from it

`Niko1221/Strata` is a purpose-built C++ engine for the *same* model, aimed at the opposite hardware regime: one NVIDIA card of 8–24 GB + 64 GB RAM, Windows or Linux, one-click install, OpenAI/Anthropic-compatible localhost API. Self-reported numbers on an RTX 5070 (12 GB) + Ryzen 5 7600 + 64 GB RAM:

| Size | writes (short chat) | writes (128K ctx) | reads prompt (32K) |
|---|---:|---:|---:|
| Q2_0 | 93 tok/s | 74 tok/s | 2170 tok/s |
| IQ2_XS | 79 tok/s | 63 tok/s | 2090 tok/s |
| IQ3_XXS | 62 tok/s | 49 tok/s | 1750 tok/s |
| IQ3_S | 53 tok/s | 46 tok/s | 1620 tok/s |
| Coder IQ1_M | 55 tok/s | 43 tok/s | 2180 tok/s |

Claimed scaling: an RTX 3090 (24 GB) ≈ 100–140 tok/s.

The architectural ideas the session judged **transferable in principle** to the Arc B70 setup:

1. **Experts resident in system RAM, computed on CPU, overlapped with GPU work** ("processor works on it simultaneously with the graphics card") — the session noted that on Intel Arc this is *especially* attractive because of **resizable BAR**: the GPU can address host RAM directly, so a "CPU-side expert" is reachable from a GPU kernel without a copy.
2. **Hot-expert caching / learning** — keep frequently routed experts pinned on GPU, stream the cold majority from RAM/SSD. This is exactly the right shape for a 10-of-512 routed-expert model.
3. **The n-gram table as an SSD-resident lookup structure** rather than a weight to be resident.
4. **Predict-then-verify speculation** (MTP-style) for 1.3–1.7× at low concurrency.
5. **Pipeline parallelism across multiple GPUs at per-layer expert granularity** — relevant because the user has two GPUs, but on *asymmetric* PCIe links.

What was judged **not** directly transferable: the CUDA-specific kernels, the one-click Windows packaging, and the quant ladder (Strata's Q2_0/IQ2_XS point is driven by an 8–12 GB VRAM budget; the B70 pair gives 64 GB, a fundamentally different regime).

---

## 3. Phase 2 — What llama.cpp actually supports (state of mainline, late Sept 2026)

The agent used the GitHub search API against `ggml-org/llama.cpp` with a set of targeted queries. This section records what was found; it is the most directly useful part for a maintainer, because it is a map of the live problem space.

### 3.1 `qwen4exp` / Qwen3.8-Flash-Next activity (136 hits)

Representative open/closed items, with the date range covered (2026-09-20 → 2026-09-29 in the sample):

- **#29599 [PR, open]** `llama: llama_prefetch_rows` — prefetching *rows*, which pairs naturally with the lazy n-gram table.
- **#29562 [issue, open]** Deterministic prefill **crash at a fixed token position** on `qwen4exp` under **multi-GPU layer split**. Directly in the user's threat model.
- **#29493 [issue, open]** Eval bug: `qwen35` dirty-ctx full-restore fails deterministically in `test-recurrent-state-rollback` (CPU+CUDA) — i.e. **recurrent-state rollback is itself not yet correct**. Matters a lot for a hybrid SSM model where recurrent state is 3/4 of the layers.
- **#29454 [PR, closed]** `models : fill the unwritten rollback snapshot slot on short ubatches`
- **#29426 [PR, closed]** `tests : refactor test-recurrent-state-rollback`
- **#29326 [issue, open]** `top_k_radix_cuda` **not compiled/used** in `qwen4exp` (CUDA-side perf bug).
- **#29174 [issue, open]** Qwen3.8-Flash-Next **MTP draft GGUFs fail to load** on a given build ("tensor not found") — no MTP path there.
- **#29166 [PR, open]** `qwen4exp: fix per-block bias indexing when a unified cache holds several sequences` — multi-sequence/prompt-cache correctness.
- **#29149 [issue, open]** `qwen4exp` **livelocks during model load on HIP**; identical config loads on Vulkan. A load-time hang on one backend but not another — very relevant to backend choice.
- **#29030 [PR]** direct reads for **lazy tensor row gather** in `qwen4exp`/`gemma4` — the mechanism that makes the giant n-gram table survivable.
- **#28933 [issue]** host-side memory growth on 128 GB unified memory.
- **#29465 [PR, Metal]** mmap extended to CPU-resident tensors, "27 GiB lazy-read tables" — the Metal counterpart of #29030.

### 3.2 Vulkan / Intel-specific activity

- **#29476 [PR, closed]** `vulkan: Tune GDN kernel, fix Intel performance` — direct Intel Vulkan tuning, landed 2026-09-29.
- **#29357 [PR, open]** `vulkan : Intel FA kernel for prefill` — Intel flash-attention kernel for prefill, still open.
- **#29520 [PR, closed]** `vulkan: fuse qwen4exp's SCALE → SIGMOID → SCALE → hc_post chain` — kernel fusion specific to this architecture.
- **#28988 [PR, merged]** `vulkan` `qwen4exp` hc (hyper-connection) operations.
- **#29639 [PR]** vulkan sparse flash attention with quantized K/V.
- **#29298 [PR, closed]** CUDA sparse flash attention for dsv4 prefill.
- **#29281 [issue, closed]** Perf regression: CUDA sparse FA decode 1.6× slower between two builds.
- **#29093 [issue]** Vulkan `qwen4exp` **image multimodal broken**.
- **#29270 [issue, closed]** Eval bug: `vk::Device::allocateMemory: ErrorOutOfDeviceMemory`.
- **#25051 [PR, open, EXPERIMENTAL]** vulkan allreduce with a **cross-device CPU proxy** and Tensor-Parallel crash fix.
- **#29342 [PR]** int8 coopmat on RDNA3.
- **#29028 [issue, closed]** RADV abort.
- **#25646 [issue, open]** **Model weight gets evicted from idle Intel dGPU memory, causing perf drop on Vulkan** — a known Intel-specific pathology relevant to a 32 GB card.

### 3.3 SYCL / Intel-specific activity

- **#27198 [issue, open]** `[SYCL] --split-mode tensor crashes in dev2dev_memcpy (DEVICE_LOST) on dual Arc Pro B70, despite working P2P` — *literally the user's hardware*, and the exact failure mode.
- **#28778 [issue, open]** SYCL DFlash2 draft model triggers **GPU driver TDR reset** (`VIDEO_TDR_TIMEOUT_DETECTED`) **on dual Arc Pro B70**. Note this is a *draft model* triggering a watchdog timeout on dual B70 — i.e. speculative decoding is precisely the feature that destabilises dual-B70 today.
- **#27845 [issue, open]** Eval bug: **potential memory leak with `-sm tensor` during generation with dual Intel B70 SYCL backend**.
- **#24946 [issue, closed]** `[SYCL/xe]` `-cb` pins GPU at `gt-c0` on Battlemage, **preventing idle power savings**.
- **#29338 [PR, open]** sycl: guard DMMV reads at row tails
- **#29107 [PR, open]** sycl: IQ3 code reorder
- **#29245** sycl grouped-MoE XMX GEMM
- **#29186** SYCL Q8_0 DMMV
- **#29608** sycl bulk upload via pinned ring buffer
- **#29132** sycl hc operations
- **#28985 [PR, open]** `sycl : do not use slow oneDNN reference matmul and fattn` — i.e. oneDNN reference kernels were still being used and are slow.
- **#29124 [PR, closed]** openvino: report device allocation limit to ggml
- **#25562 [issue, open]** OpenVINO docker cannot use multiple GPUs
- **#29604 [PR, open]** SYCL: reduce tensor allreduce sync with pinned host buffers
- **#25089 [PR, closed]** sycl: `check_graph_compatibility()` must allow graphs for **MoE decode** (CONCAT dim!=3, MUL_MAT_ID fused path)
- **#29459 [PR, open]** sycl: work-around for Level Zero crash/hang
- **#29366 [issue, open]** [SYCL] flawed debug builds
- **#29273 [PR, closed]** CI: update oneAPI toolkit to 2026.1

### 3.4 MTP status (from Unsloth's own `MTP/README.md`)

> "**A stock `ggml-org/llama.cpp` build cannot use these.** Mainline has no MTP graph for the `qwen4exp` architecture, no cross-model tensor borrowing, and no `--spec-type draft-mtp` option, so passing a head to it does nothing."

Three supported routes given by Unsloth:

1. **Prebuilt binaries** from `unslothai/llama.cpp` releases, tag `b10715-mix-86bd2d3` or newer. Assets: `app-<tag>-<os>-<arch>-<backend>.tar.gz` for **CPU, CUDA 12/13, ROCm and Vulkan** (`.zip` on Windows).
2. **Build upstream PR `ggml-org/llama.cpp#28243`** (the MTP support "going to mainline") from `danielhanchen/llama.cpp`, branch `qwen4exp/mtp`.
3. A third option, referenced but not captured in the excerpt read.

Draft-head file guidance:

| file | size | note |
|---|---:|---|
| `mtp-...-shared-Q8_0.gguf` | 2.60 GB | **recommended**, fastest |
| `mtp-...-shared-Q4_K_M.gguf` | 1.78 GB | ~2 points less acceptance |
| `mtp-...-shared-BF16.gguf` | 4.87 GB | bigger *and* slower than Q8_0 |
| `mtp-...-{Q8_0,Q4_K_M,BF16}.gguf` | 3.85 / 2.60 / 7.24 GB | self-contained variants |

Mechanically interesting note: **`shared-` heads borrow the token embedding and output projection from the already-loaded main model**, saving ~1.3 GB and drafting identically to the self-contained files. And "BF16 is bigger and slower: a draft step is dominated by the output projection, which is cheaper to execute at 8 bits" — a concrete, transferable insight about draft-model quant selection.

Build recipe quoted by Unsloth (CUDA variant shown; `-DGGML_CUDA=ON` omitted for CPU):

```bash
git clone --branch qwen4exp/mtp https://github.com/danielhanchen/llama.cpp
cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --clean-first \
    --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
```

### 3.5 Interim conclusion of Phase 2

- Backend: **Vulkan is the safer default on Intel Arc** for this model (it is where the `qwen4exp` fusion/hc work has landed, it loads where HIP livelocks, and Unsloth ships Vulkan prebuilts with the MTP support). SYCL has more Intel-specific work in flight but carries the dual-B70-specific crashes above.
- **Multi-GPU is where the risk concentrates**: layer split has an open deterministic prefill crash (#29562); tensor split has an open `dev2dev_memcpy` DEVICE_LOST crash on dual B70 despite working P2P (#27198); `-sm tensor` leaks on dual B70 (#27845); and the second GPU sits behind a **southbridge** PCIe 4 link, which is much slower and shared with chipset traffic.
- **MTP cannot be had from stock upstream** — a fork or PR #28243 is required, which is a maintenance decision the user must own.
---

## 4. Phase 2 continued — deep dives on the specific problem reports

The agent then pulled full bodies + comment threads for the most relevant reports. The findings below are the ones that materially changed the picture.

### 4.1 Arc Pro B70 hardware identity (learned from issue reports, not assumed)

- **Intel Arc Pro B70 = Battlemage G31 (BMG-G31), PCI ID `8086:e223`, Xe2, 32 GB, ~608 GB/s GDDR6.** Sourced from #26581 and #28721.
- A sibling part exists as **Battlemage G21** = Arc B580 (also 32 GB class consumer part). Several reports use a **G31 + G21 heterogeneous pair** (#25612) — the user's pair is homogeneous G31.
- Linux stack used by the B70 reporters: `xe` driver (kernel 6.17), **Mesa 26.x ANV (kisak builds)**, `KHR_coopmat` active; SYCL via oneAPI 2025.3/2026.1 + Level Zero (compute-runtime 26.18).

### 4.2 #27198 — dual-B70 SYCL tensor split crash (the single most on-point issue)

Hardware in the report: 2× Arc Pro B70. Failure: `-sm tensor` → `dev2dev_memcpy` at `ggml/src/ggml-sycl/ggml-sycl.cpp:704` with a 5 MiB size, then a Xe fault:

```
Faulted Address: 0x0000eaab54a04000
FaultType: 0  AccessType: 0  FaultLevel: 1
EngineClass: 5 ccs  EngineInstance: 0
xe ... Fault response: Unsuccessful -ENOENT
```

Crucially, **P2P itself works** — the reporter ran `ze_peer` and measured inter-device bandwidth between the two B70s:

| direction | peak | saturates at |
|---|---:|---|
| dev0 → dev1 write | ~10.2 GB/s | 4 MiB blocks |
| dev1 ← dev0 read | ~10 GB/s (per earlier session read) | ≥ 1 MiB blocks |

So the link is a working ~10 GB/s P2P path (routed through the PCIe hierarchy), and the crash is a **Level Zero / VMM peer-mapping bug**, not a missing-P2P situation. Diagnosis from the thread: **ggml SYCL VMM pools are not peer-mapped across contexts**.

Workarounds collected from that thread:

| workaround | effect |
|---|---|
| `GGML_SYCL_DEV2DEV_MEMCPY=2` | forces host-staging copies — stable, slower |
| `GGML_SYCL_ENABLE_VMM=0` | disables VMM pools — avoids the crash on some systems |
| `-sm layer` | layer split instead — stable |
| `UR_LOADER_USE_LEVEL_ZERO_V2=0` | avoids a CPU SEGV at launch on some systems |

Related upstream fixes: `intel/compute-runtime#996`, llama.cpp **#28953** (merged workaround), llama.cpp **#29459** (open, Level Zero crash/hang workaround).

**Conclusion the session drew:** layer split is the supported multi-GPU mode for this model today; tensor split on dual B70 is not.

### 4.3 #26581 — the Xe2 deep-context decode tax (root-cause quality report)

This is the most technically interesting report for a maintainer, because it *characterises* rather than merely complains:

- On Arc Pro B70, token generation degrades with context depth by a cost that is **byte-count independent**: a near-constant **~21–25 ns per KV position per full-attention layer per token**, invariant across backend (Vulkan ANV 26.1.6 vs SYCL oneAPI 2025.3 L0 v2), across model/geometry (KV row width 1 KiB vs 8 KiB made no difference), across builds (b8183 vs b10235), and across quant (Q2_K vs Q4_K_M).
- Byte independence rules out DRAM bandwidth. Diagnosis: **memory-latency-bound attention matvecs — insufficient memory-level parallelism** to cover GDDR6 latency. Supporting evidence given: GPU sits at its power cap while useful traffic is a fraction of peak; **4 concurrent streams scale aggregate throughput ~2.4×**; and the same llama.cpp on AMD Strix Halo iGPU (less than half the bandwidth) holds a sibling model flat to 65K.
- Representative measurements (`-ngl 999 -ub 512 -fa 0 -n 256 -p 0 -r 1 -d <depth>`):

| model | backend | d0 | d32k | d65k | d127k |
|---|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B Q4_K_M (hybrid, 10 attn layers) | Vulkan | 70.7 | 58.4 | 32.2 | 20.7 |
| Qwen3.6-35B-A3B Q2_K | Vulkan | 72.0 | 60.2 | 33.4 | 21.2 |
| Qwen3.6-35B-A3B Q2_K | SYCL | 81.5 | 52.9 | 38.5 | 22.8 |
| Qwen3-Coder-30B-A3B Q4 (48 attn layers) | Vulkan b10235 | 87.1 | 37.6 | ~11–13 | — |
| Qwen3-Coder-30B-A3B Q4 | Vulkan b8183 | 103.7 | 35.8 | — | — |
| Qwen3-Coder-30B-A3B Q4 | SYCL | 105.9 | 20.5 | 11.2 | — |
| Gemma 4 26B-A4B Q4 | Vulkan | 69.2 | 57.8 | 30.9 | — |
| Gemma 4 12B Q4 | Vulkan | 48.4 | 31.7 | 22.5 | — |

**Why this matters *less* for Qwen3.8-Flash-Next:** only **12 of 48 layers** are full-attention layers (`full_attention_interval = 4`), versus 10–48 in the models above, and the sparse-attention indexer caps selection at `top_k = 2048`. So the tax should be proportionally much smaller — but it is not zero, and it interacts with the PLE table (see §4.6).

### 4.4 #28721 — Vulkan deep-context collapse on B70, and the FA-shader problem

- Report: on a single B70 with a 27B dense model + MTP3 draft, **long-context decode collapses roughly 8× on Vulkan** while SYCL degrades only mildly and keeps its TTFT prefill advantage at depth.
- Maintainer response in-thread: *"the Flash Attention shader does not run well on Intel Battlemage"*, with work tracked in **#24406 / #24408** and a newer Intel-specific prefill FA kernel in **#29357**.
- A **first-party AMD datapoint** was posted for contrast (RX 7900 XTX, RADV, `b11065`, Qwen3.8-27B IQ3_XXS with draft-mtp, `-ctk q4_0 -ctv q4_0 -fa on`): short-context prefill **956.9 t/s**, decode **53.84 t/s**, and the note that short-context decode on gfx1100 sits at the bandwidth+dequant limit with no kernel headroom. Also noted: `-fa off` fails to create the context on that build/device, so a non-FA A/B could not be isolated.

### 4.5 #29476 — the Vulkan GDN fix (merged 2026-09-29): a ~10× Intel bug

This is the single biggest *performance* finding of the research phase and is important for anyone benchmarking this model on Intel:

- PR **#29476** "vulkan: Tune GDN kernel, fix Intel performance" merged **2026-09-29T17:44Z**.
- `test-backend-ops` GDN, Intel A770: **~9–12× speedup across every ubatch size** (e.g. ubatch 512: 6959 → 662 ops/s; ubatch 8192: 111 023 → 10 437). RTX 3090 regressed 3–6%, Radeon Pro VII unchanged.
- End-to-end on **Qwen 3.6 35B Q2_K** with an **A770**: prompt processing **+26.1% / +32.7% / +37.7%** at ubatch 512/1024/2048; token generation **+8.4%**.

**Interpretation:** the Gated DeltaNet op is the single most Intel-relevant kernel in `qwen4exp`, because 36 of its 48 layers are GDN. A build older than this PR is measuring a *bug*, not a hardware limit.

### 4.6 Other high-signal items

- **#25356** (open): *Vulkan batched-decode throughput cliff at `n_tokens = 9`* on many-expert MoE, caused by a fixed 8-token threshold in the MMV/MUL_MAT_ID dispatch. Reproduced on b9293 and current master. This is directly relevant to speculative decoding: staying at ≤ 8 rows avoids the cliff (Strata's 3-draft + verify = 4 rows sits comfortably below it). Also affects `-np > 8` slots.
- **#25612** (closed 2026-09-16): *"split-mode causes garbled output on dual Intel dGPU"* — Vulkan on **G31 + G21** with the `xe` driver. Different-SKU pair, but exactly the failure class the user must avoid.
- **#27845** (open): SYCL `-sm tensor` on 2× B70 with a 262K context shows **system RAM growing ~6 MB/s** until OOM. (Reported on Ryzen 5950X; model `unsloth/Qwen3.8-27B-GGUF UD-Q8_K_XL`.)
- **#28778** (open): a **DFlash2 draft model triggers GPU TDR (`VIDEO_TDR_TIMEOUT_DETECTED`) on dual Arc Pro B70** under SYCL — i.e. the *speculative-decoding path itself* destabilises dual-B70 today.
- **#25646** (open): model weights on a **secondary Intel dGPU get evicted after ~70 s idle on Windows**, collapsing prompt eval (to ~1.9 t/s in one report). Directly relevant for a long-idle server. Related PR #25214; "does not reproduce on b9911" reported when the B70 was the primary device.
- **#29093** / **#27886** (open): **Vulkan `qwen4exp` image/multimodal is broken** on Vulkan. Since the model is `image-text-to-text`, this is a live limitation to document.
- **#28933**: host-side memory growth on 128 GB unified-memory systems.
- **#28753** (open): `ggml_backend_sched_alloc_splits: unexpected graph reallocation` crash showing up in both `qwen4exp+SYCL` and `qwen4exp+"B70"` searches — evidence that B70 + SYCL is an actively-developed configuration (someone owns a B70 for testing).
- **#26510 / #25051**: Vulkan all-reduce with a **cross-device CPU proxy** plus a tensor-parallel crash fix, marked **EXPERIMENTAL**, still open (#26610 "RPC: add `-sm tensor`" is the companion, also open).

### 4.7 `qwen4exp` support provenance (important for a maintainer)

| PR | Title | State |
|---|---|---|
| **#27742** | `model: add Qwen3.8-Flash-Next (qwen4exp)` | **merged 2026-08-26** — this is the mainline support PR |
| #27793 | same title, alternate PR | closed, unmerged |
| #27836 | `qwen4exp : add NextN/MTP draft head (--spec-type draft-mtp)` | open (unmerged) |
| **#28243** | `models: Qwen3.8-Flash-Next MTP` — "Enables 1.3 to 2x faster MTP support … + shared MTP modules (re-uses embed_tokens to save disk space and VRAM/RAM). Builds on top of #27836" | **open, mergeable, updated 2026-09-29** |
| #27842 | `model: add MTP (nextn) speculative head` | closed, unmerged |
| #27956 | Qwen4Exp MTP + correctness fixes | closed, unmerged |
| #28097 | support draft-head-only GGUFs (unsloth layout) | open |
| #27879 | Qwen4exp correctness fixes | closed, unmerged |
| #27941 | `qwen4exp: follow up fixes` | **merged** — this is the PR that **disabled `-sm tensor` for qwen4exp** |
| #27880 | `model: qwen4exp: reduce number of graph splits` | merged |
| #28023 | `qwen4exp: sum the indexer heads by slices` | merged |
| **#28569** | `model : re-enable -sm tensor for qwen4exp` | **open** (6 comments, updated 2026-09-29) |
| #28136 | `qwen4exp: direct reads for the lazy PLE table (>2x prefill on GB10)` | closed, unmerged |
| #29030 | direct reads for lazy tensor row gather in `qwen4exp`/`gemma4` | open |
| #29599 | `llama: llama_prefetch_rows` | open |
| #27992 | `kv-cache: index (seq,pos) cells to make ngram prev-token lookups O(log n)` | closed, unmerged |
| #28118 | `server: keep speculative recurrent-state checkpoints on-device` | open |
| #29520 | `vulkan: fuse qwen4exp's SCALE → SIGMOID → SCALE → hc_post chain` | closed (merged) |
| #28901 / #28988 | `qwen4exp: add hc ops` (generic / vulkan) | merged |
| #29465 | Metal: mmap extended to CPU-resident tensors, "27 GiB lazy-read tables" | referenced |

The `-sm tensor` situation deserves emphasis because it is a *design-level* signal. From #28569's description: `-sm tensor` was disabled for `qwen4exp` by #27941 because `test-llama-archs -a qwen4exp` asserted once the fixture carried a **PLE layer**. Root cause is scheduler placement, **not** QSA: `test-llama-archs` builds the model with embeddings host-resident, so the PLE embedding gather (`ggml_get_rows` on `per_layer_token_embd`) becomes a CPU node, and in the `qwen4exp` graph `hc_init` (the `ggml_repeat_4d` fanning the embedding out to the hc streams) is the assert site. The PR re-enables it and has been validated against real models via **RPC tensor split across two Vulkan `ggml-rpc-server` instances**.

**Takeaway:** in current mainline, `qwen4exp` + `-sm tensor` = off; `#28569` + `#26610` are the tree needed to enable it. RPC-based tensor split across two Vulkan instances is the validated path.

### 4.8 Recurrent-state correctness — the hybrid-architecture soft spot

Because `qwen4exp` is 3/4 recurrent (DeltaNet), a whole cluster of issues exists around recurrent-state handling that is not specific to Intel but is *specific to this architecture*:

- **#29493** (open): `qwen35` dirty-ctx full-restore fails deterministically in `test-recurrent-state-rollback` (CPU+CUDA).
- **#29454** (closed): "fill the unwritten rollback snapshot slot on short ubatches" — the fix implied by the above.
- **#29426** (closed): refactor `test-recurrent-state-rollback`.
- **#28019** (open): "multi-seq split replay corrupts recurrent state when rs rollback is enabled (why the arch is excluded)".
- **#29166** (open): fix per-block bias indexing when a unified cache holds several sequences.
- **#28194** (open): `/slots` restore yields no KV reuse on hybrid/recurrent and SWA models — context checkpoints are not persisted.
- **#29461/#29493-family**: eval failures for dirty-ctx full-restore under concurrency.

For a server doing agentic multi-turn work (which is the user's apparent use case), this is the most consequential correctness cluster.

### 4.9 Mainline CLI surface confirmed from `common/arg.cpp` (build `master`, 4764 lines)

Rather than guess, the agent downloaded `common/arg.cpp` (after a 404 on the old `examples/arg.cpp` path) and grepped it. Flags that matter for this problem:

| flag | what it does (verbatim semantics from source) |
|---|---|
| `-lzm, --lazy-mode on\|auto\|off` | *"on-demand reading of certain tensors, for example per-layer embeddings (default: auto)* — on: read the rows of such tensors from disk on demand instead of keeping them resident (**requires mmap**); auto: on, but only for tensors larger than 4 GiB; off: always keep them resident"* |
| `-ot, --override-tensor <pattern>=<buft>` | override tensor buffer type (regex-driven) |
| `-cmoe, --cpu-moe` | keep all MoE weights in CPU |
| `-ncmoe, --n-cpu-moe N` | keep MoE weights of the first N layers in CPU |
| `-fit, --fit on\|off` | *"whether to adjust unset arguments to fit in device memory"* |
| `-cram, --cache-ram N` | prompt-cache size in RAM; idle slots saved to prompt cache on new task |
| `-kvu / -no-kvu`, `--kv-unified-per-slot N` | unified KV cache control |
| `-dev/--device`, `--list-devices` | device selection |
| `--numa distribute\|isolate\|numactl` | NUMA strategy |
| `--load-mode mmap\|mlock\|mmap+mlock\|dio` | `-mmap`/`-no-mmap` are now deprecated in favour of this |
| `--spec-type <list>` | comma-separated speculative types; includes `draft`, `ngram`, `draft-mtp`, `ngram-map-k4v` |
| `--spec-draft-n-max/-n-min` | draft token counts |
| `--spec-ngram-mod-n-min/-n-max/-n-match` | ngram-based speculation; an example preset sets `n_match 24, n_min 48, n_max 64` |
| `--spec-draft-cpu-moe`, `--spec-draft-n-cpu-moe` | draft-model MoE placement |
| `--spec-draft-model/-md` | draft model path |

`arg.cpp` also contains **MTP download/plan plumbing** (`params.speculative.types` contains `draft-mtp` → `opts.download_mtp = true`, plus a `common_params_model mtp` and `plan_spec.mtp.local_path` / "spec sidecar" logic). This is a nuance worth flagging: **`--spec-type draft-mtp` exists in mainline's CLI**, because MTP is implemented for *some* architectures, but Unsloth's MTP README explicitly states mainline has **no MTP graph for `qwen4exp`**. Both statements are true at once — the flag is generic, the `qwen4exp` graph is what is missing.

### 4.10 Strata's technical design (`docs/DETAILS.md`, 603 lines) — the transferable parts

Memory hierarchy Strata uses, which is the design a llama.cpp configuration is trying to approximate:

- **GPU VRAM** keeps attention / DeltaNet state / router / shared experts / output head / MTP draft / KV + an **adaptive expert cache**.
- **System RAM** holds **all 24 576 experts pinned**, and the **CPU computes the non-cached experts in parallel with the GPU** (AVX-512/AVX2 multi-token i-quant row kernels, using ggml CPU for i-quants).
- **SSD** holds the **~28.8 GB n-gram (PLE) table**, lazily loaded through the OS page cache.

Specific measured/derived mechanisms, each of which is an idea llama.cpp could adopt or approximate:

1. **Expert profile / routing ranks** — produced offline by tracing (`--dump-routing trace.bin` + a profile tool), plus **adaptive tier learning during conversation** (`--expert-cache auto`, `--adapt-every`).
2. **`--pcie-frac`** — tunable ratio of *copying* experts to GPU over PCIe vs *computing* them on CPU; calibrated per machine. (Observed in a user log: PCIe probe 11.7 GB/s → `pcie_frac 0.25`.)
3. **KV streaming from 64 K onward** (`--kv-resident 32768`): keep the recently-read KV window in VRAM, stream the rest from RAM. Measured **+20 %** on Q2_0 at 262 K (50.9 → 62.6 tok/s). RAM cost ~**13.7 KB/token**.
4. **Quantized KV**: `kv q4_0` gives 4-bit KV but **+8–12 % perplexity** (judged not worth it); **`k8v4` rotated-KV keeps the needle test intact with 23 % memory reduction** — the session flagged this as the better lever.
5. **MTP speculation: 2.4–3.2 accepted tokens per pass** on average; additional prompt-lookup speculation up to 5 tokens for repetitive content (**6–11 %** extra on code edits).
6. **Prefill pipelining**: 2048 → 8192-token chunks; MMQ on quantized experts (no FP16 expansion) while the current layer's attention runs; next layer's experts streamed over PCIe; PLE block handled as a whole chunk; unpinned experts copied by helper threads.
7. **Low-RAM mode**: `mmap` experts from the file and rely on the OS file cache for machines with ≤ 48 GB RAM.
8. **VRAM is the dominant variable**: *"each additional GB holds ~700 experts"*, and "VRAM matters more than a faster GPU". This is the crucial sentence for the user's case — they have 64 GB of VRAM across two cards, an order of magnitude more than Strata's target.
9. CPU floor: Strata's **expert kernels require AVX512-VNNI + AVX512-VBMI** (Ice Lake / Zen 4+) for the AVX-512 pack, but an **AVX2 pack exists and works** — the AVX-512 error message is about the *pack*, not the engine (confirmed by the maintainer in Strata issue #142, where other Arrow Lake users run it fine on AVX2). Arrow Lake's E-cores do not have AVX-512.

Measured Strata reference points (RTX 5070 12 GB, Ryzen 5 7600, 64 GB DDR5-5200, Windows, engine 0.1.26, MTP on, `--prefill auto`, 8-bit KV above 4 K):

| quant | pp 1K | pp 4K | pp 32K | pp 64K | pp 128K | pp 262K | tg 1K | tg 32K | tg 128K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2_0 | 536 | 1299 | 2171 | 2126 | 2107 | 1304 | 87.3 | 81.8 | 73.7 |
| IQ2_XS | 534 | 1256 | 2092 | 1754 | 1752 | 1181 | 79.6 | 76.3 | 62.7 |
| IQ3_XXS | 482 | 1007 | 1745 | 1609 | 1602 | — | 61.9 | 58.5 | 49.0 |
| IQ3_S | 427 | 913 | 1624 | 1640 | 1443 | — | 52.4 | 48.3 | 45.5 |
| Coder IQ1_M | 656 | 1583 | 2177 | 2236 | 2208 | 1034 | 58.9 | 54.9 | 43.0 |

Engine progress over time is itself a datapoint: 0.1.22 → 0.1.26 was 8–28 % faster at 32K–128K; before 0.1.13 prompts ran at half speed; output speed is text-dependent because speculative acceptance varies ("a different answer to the same prompt moves it by several percent").

**Applicability verdict reached:** Strata's *engine* cannot run on Intel Arc (CUDA, plus experimental HIP). Its *architecture of memory placement* is exactly what llama.cpp needs for this model, and llama.cpp's primitives (`--lazy-mode auto`, `-ot`, `-cmoe`/`-ncmoe`, `--fit`) are the same ideas without the adaptive tiering. The one thing llama.cpp currently lacks is **adaptive expert caching** — in mainline each expert is statically GPU or CPU.

---

## 5. Interlude — the research stalled on Reddit, and the user intervened

After completing the GitHub and Strata work, the agent spent several turns failing to scrape r/LocalLLaMA:

| attempt | result |
|---|---|
| `www.reddit.com/r/LocalLLaMA/search.json` via curl | non-JSON response |
| `old.reddit.com` search.json with browser UA | Reddit's HTML **"Blocked"** interstitial |
| `api.pullpush.io` (Pushshift mirror) | `"Rate limit exceeded. This website does not provide free scraping resources for agents."` |
| `webfetch` on reddit search.json | HTTP 403 |
| `webfetch` on old.reddit search HTML | returned only the shell ("Welcome to Reddit"), no results |
| `html.duckduckgo.com` with `site:reddit.com/r/LocalLLaMA "Arc Pro B70"` | 9 KB page, no result links parsed (anomaly-detection page) |
| `bing.com/search` with the same `site:` filter | 123 KB returned, but Bing **ignored the `site:` operator** and returned generic "Arc" results (browser downloads, a Swedish property company, Arc'teryx…) |
| `arctic-shift.photon-reddit.com` Reddit archive API | returned `data: null` |

The session then hit a **user "pause"** (msg 29), and the agent produced a status summary (msg 30) that is itself a useful artifact — it is the clean, verified statement of everything established up to that point:

- Model facts verified from HF (arch `qwen4exp`, 125B MoE with 6B active, 512 experts top-10+1, 51B n-gram/PLE table, 48 layers = 3× Gated DeltaNet + 1× Qwen Sparse Attention per group, 262K ctx, quant sizes/QA table, MTP heads need the unsloth fork / PR #28243).
- llama.cpp mainline support merged via **#27742**; key issues: dual-B70 SYCL tensor-split P2P crash **#27198** (+ fixes **#28953** merged, **#29459** open), B70 Vulkan deep-context collapse **#28721**, Xe2 attention latency tax **#26581**, `qwen4exp` `-sm tensor` disabled **#28569** open, GDN Intel Vulkan ~10× fix **#29476** merged 2026-09-29, lazy-mode PLE streaming, `-cmoe`/`-ncmoe`/`-ot`, `--fit`.
- Strata fully analysed (README + DETAILS.md + issues incl. Arrow Lake AVX2 case **#142**) — CUDA/HIP-only, learnings transfer but it cannot run on Arc.
- `unslothai/llama.cpp` releases: `b11160-mix` Vulkan binaries exist but **predate the Intel GDN fix**.

**The user's response (msg 31):** *"You now have chrome dev tools, use that to access reddit. stop and ask me if I should click somewhere."*

This is the pivot point of the session: research moves from server-side scraping to **driving the user's own browser session**, which implies access to whatever the user had running locally.

---

## 6. Phase 3 — Reddit research via the user's own Chrome

### 6.1 Method

After every server-side route to Reddit failed, the agent used the **chrome-devtools MCP** connected to a real Chrome instance:

1. `old.reddit.com` search → **redirected to `/login/?reason=lor2`** (Reddit's "login or robot" gate). No login available, so unusable.
2. `www.reddit.com` search → **succeeded**. No login needed, JS-rendered.
3. Extraction pattern that worked: `document.querySelectorAll('a[href*="/comments/"]')`, de-duplicated, `innerText` trimmed, length-filtered, with a login check via `document.querySelector('[aria-label="Log In"], faceplate-button#login')` and `document.title` validation.
4. For thread pages, a second pattern read the new Reddit custom elements: `shreddit-post` → `post-title` attribute + `[slot="text-body"]`, and `shreddit-comment` → `depth` / `author` / `score` attributes + `[slot="comment"]`. This yields comment-tree depth, author, score and text without triggering any rendering cost.

This is worth recording as a reusable technique: Reddit's modern web UI is fully introspectable via `shreddit-*` shadow elements, so scraping r/LocalLLaMA requires no API access at all.

### 6.2 r/LocalLLaMA thread "MTP released for Qwen3.8-Flash-Next-GGUF" (`1w42biu`)

Post body: *"Can't wait to test! This should significantly boost TPS! Now we just need more llama cpp optimizations to be merged in! Edit: For anyone who wants to test this: `github.com/unslothai/llama.cpp/pull/144/changes`."* — 25 comments.

Extracted findings:

| comment | substance |
|---|---|
| fprimex | Concrete recipe: unsloth tagged release build → `llama-server --model …UD-Q4_K_XL-00001-of-00004.gguf -md …/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf --mmproj …/mmproj-BF16.gguf --spec-type draft-mtp --spec-draft-n-max 2 …` → **~18 t/s → ~38 t/s**. Notes `n-max 2` is his habitual value and higher might suit this model better. |
| vini542reddit (author) | `shared` variants reuse parts of the main model; the non-shared ones do not; **not every fork supports `shared` yet**. |
| **pmttyji (87 pts)** | The most valuable comment in the thread: *"Now we just need more llama cpp optimizations to be merged in! I see below one got merged hours ago `#28123` — no draft: 108 tok/s; before: 123 tok/s code, 83 tok/s prose; after: 183 tok/s code, 144 tok/s prose. Note the 83: before this change MTP was slower than not drafting at all."* Plus `#28023` for prompt-processing improvement. |
| bennmann | On SSD offload of the PLE table: *"llamacpp flag `--lazy-mode auto` — though there is no good way to use a BF16 engram with q4 or smaller quant yet."* |
| sdroege_ / Common_Warthog_G | Point to `unslothai/llama.cpp#142`; *"shared reuses layers from the main model. llama.cpp doesn't support it yet on its main branch."* |
| Durian881 | MLX on oMLX (Apple Silicon): *"MTP works really well with this model … boosts token generation by 40-50% when activated."* |
| vulcan4d | *"I thought llama.cpp still didn't have the feature merged yet … My gear runs it at only 9t/s."* |
| vini542reddit / rustysec | Timing note: Unsloth's MTP GGUFs were only hours old; other quants had been available for days. |

**The #28123 finding deserves emphasis.** It says that on older builds **MTP was a net loss for `qwen4exp`** (83 t/s drafting vs 108 t/s not drafting), and only *after* mainline PR **#28123** (recurrent-state rollback handling) merged did MTP become a decisive win (183 t/s code / 144 t/s prose, i.e. **1.5–1.7× over no-draft**). Combined with #29476 for Intel GDN, the operating rule for this model is: **build freshness is the single biggest performance lever**, and any benchmark taken on a build older than the relevant merges is measuring bugs.

### 6.3 r/LocalLLaMA search `"Arc Pro B70"` — threads located

Seven threads surfaced; the agent selected four to read, and then the session pivoted (see §7):

| thread | why it matters |
|---|---|
| `1tuik6o` "Intel Arc Pro B70 llama.cpp benchmarks posted" | the most directly relevant benchmark thread |
| `1vulh45` "Intel Arc Pro B70 + vLLM XPU: 52 tok/s on Qwen3.8-27B INT4" | alternative inference stack on the same GPU |
| `1soe0nm` "Intel Arc Pro B70 Open-Source Linux Performance Against NVIDIA RTX & AMD Radeon AI PRO Review" | open-stack performance review |
| `1sjlowl` "Best Model to use with Arc Pro B70" | model/hardware pairing advice |
| `1s4coaq` "…why we should, or should not be excited about the ARC PRO B70?" | bandwidth-vs-price framing |
| `1s4vft4` "Preliminary testing results (includes some gaming)" | early silicon data |
| `1s3bb3y` "Intel launches Arc Pro B70 and B65 with 32GB GDDR6" | launch specs |

---

## 7. Mid-session artefacts: the two documents the agent wrote

Two important process facts belong in this article because they changed the shape of the work.

**First**, after the user said *"remember to write the file continuously"* (msg 38), the agent stopped hoarding findings and materialised `research.md` on disk (msg 39) as a living document — 8 sections, ~9 KB, explicitly stamped *"Status: in progress… llama.cpp mainline at time of research: b11261 / 2026-09-29."* That document is the canonical consolidated output of the research phase; the sections below summarise its content and, importantly, its *reasoning*, which adds detail the earlier notes did not.

**Second**, msg 42 is an explicit **manual compaction** event (`"type": "compaction", "auto": false`) — the context was rolled up mid-session and the work continued. The post-compaction assistant message (msg 43, 18 438 chars) is therefore the authoritative "everything so far" summary; §9 covers what changed after it.

### 7.1 What the consolidated research document added beyond the raw notes

Additional verified facts introduced at consolidation time:

**Hardware ceiling maths.** B70 at 608 GB/s ⇒ a Q4_K 27B dense model has a theoretical ceiling of ~37 t/s, and 35.6 t/s was actually measured. This is the anchor for judging any MoE result: `qwen4exp` activates only ~6 B parameters per token, so it is *not* bandwidth-bound the same way — its decode cost is dominated by attention latency and CPU-side expert work, not VRAM streaming.

**The QSA decode argument.** With `full_attention_interval = 4`, only **12 of 48 layers** are full-attention. Each QSA layer has 24 Q heads / 2 KV heads, head_dim 256, partial RoPE 64, an MQA indexer (4 Q heads, dim 128, top_k 2048, 512-block budget). Therefore **decode cost should not scale linearly with context in principle** — the indexer selects. The session estimated the #26581 Xe2 latency tax for this model as `12 × ~25 ns × ctx_positions` per token, ≈ **9.6 ms/token at 32 K context** from attention alone: material, but an order of magnitude below the 48-layer models where the tax was originally measured.

**Attention-layout detail.** PLE `head_offsets` step by 20 000 003 and `head_vocab_sizes` sum to ~320 000 000 ⇒ **~320 M table rows**, with `embedding_length_per_layer_input = 160`. 16 n-gram heads, bigrams + trigrams, conv kernel 4, `eos_token_id = 248044`. Strata's 28.8 GB figure is the same table at 4-bit. It is only **randomly row-accessed** (a few rows per token), which is precisely why `--lazy-mode` works and why it is held at ≥4-bit even in 1-bit quants.

**Full quant/quality table (Unsloth KLD).**

| Quant | Size GB | top-1 acc % | mean KLD |
|---|--:|--:|--:|
| UD-IQ1_S | 72.6 | 77.33 | 0.396 |
| UD-IQ1_M | 74.5 | 79.69 | 0.315 |
| UD-Q2_K_XL | 78.9 | 82.72 | 0.225 |
| UD-IQ3_XXS | 82.0 | 85.41 | 0.165 |
| UD-Q3_K_XL | 90.0 | 88.32 | 0.107 |
| UD-IQ4_XS | 93.7 | 89.55 | 0.084 |
| **UD-Q4_K_XL** | **111.3** | **92.26** | **0.047** |
| UD-Q5_K_XL | 158.3 | 93.68 | 0.030 |
| UD-Q6_K_XL | 169.2 | 94.09 | 0.027 |
| Q8_0 | 188.2 | 94.12 | 0.027 |
| BF16 | 354.0 | — | — |

**#28721 measured Vulkan-vs-SYCL numbers on B70** (this is the decisive backend datapoint): 1 K → 64 K context, decode **35.6 → 4.5 t/s on Vulkan** vs **38.5 → 15.8 t/s on SYCL**; 64 K prefill TTFT **329 s (Vulkan) vs 65 s (SYCL)**. Root cause identified by the reporter: the **scalar FA path when `n_rows == 1`**, i.e. decode never reaches the matrix cores, compounded by an immature FA shader on Battlemage.

**Additional issues catalogued during consolidation** (not in the earlier notes): #29241 SYCL MTMD/vision crash on B70 Windows (`llama-mtmd-cli` `0xC0000409`); #27517 SYCL MoE expert tensors never hit the Q8_0 reorder pass; #27373 perf degradation with MTP + Q3; #27547 dual-GPU stuck on load; #26409 `-sm tensor` 3× slower than single GPU; #27595 `--fit` under-accounts memory; #29082 (closed) Vulkan SWA q-KV regression; #29002 / #29092 hybrid recurrent-state corruption across requests; #21831 (closed) server re-processing full prompts with SWA/recurrent memory.

**Unsloth's own MTP benchmark numbers** (B200, greedy): UD-Q4_K_XL **83.2 → 138.8 t/s (1.67×)**, IQ1_S **90.1 → 120.9 (1.34×)**, shared-Q8_0 acceptance **66 %**. Critically: **at concurrency 8 MTP becomes a net loss (0.81–0.87×)** — single-stream only.

**Hardware facts stated for the second card.** The research notes flagged as *assumption, to be confirmed* that the chipset-attached card is likely **PCIe 4.0 x4 (~8 GB/s)**, versus ~64 GB/s for the CPU-direct PCIe 5 card — an 8× asymmetry that no `-sm layer` default addresses. The recommendation was therefore to bias layer placement so the slow card carries fewer layers (`--tensor-split`-style weighting, `--main-gpu 0`), noting per-layer activation hops are tiny (~5 KB/token) so layer split is bandwidth-tolerant in a way tensor split is not.

### 7.2 Backend recommendation as it stood at consolidation

- **Primary: SYCL, `-sm layer` across both B70s.** Justification: the only backend on B70 that survives deep context (#28721, 2–3.3× faster than Vulkan at 15K–64K, 5× better TTFT); sparse-FA/QSA support merged in #28796; layer split is the stable multi-GPU path.
- **Fallback: Vulkan on ≥ b11259** (post-#29476) for single-GPU work, short/medium context, and where SYCL misbehaves; explicitly *avoid* for long-context agentic use until Intel FA prefill (#29357) and a decode-time matrix-core FA path land.
- **Vision: text-only for now** — Vulkan `qwen4exp` multimodal is broken (#29093/#27886), SYCL MTMD crashes on B70 (#29241). Strata's answer was to run vision as a *separate* process, which is the same trick.
- Quant preference: **UD-Q4_K_XL** (111 GB, 92.3 % top-1) if PLE lazy streaming verifies; otherwise **UD-Q3_K_XL / UD-IQ4_XS** (90–94 GB, 88–90 % top-1) for margin. Explicitly reject ≤2-bit for a "high quality" goal.
- **Verification ladder** proposed at the end of the research phase: (1) `llama-bench` pp/tg at `-ub 512/1024/2048/4096` for both backends; (2) greedy 500-token generation at 8 K and 32 K context, diffed against a CPU reference to catch hybrid-state corruption; (3) MTP on/off A-B; (4) 30-minute agent session watching RAM/VRAM for the #27845 leak; (5) idle-eviction test on Windows if applicable (#25646).
- Open questions left explicitly flagged: dual-card topology behaviour under layer split; SYCL-vs-Vulkan head-to-head on `qwen4exp` post-merges (**no public B70 numbers for this model existed yet**); MTP on SYCL; PLE lazy-read throughput Windows vs Linux; whether `--fit` under-accounts on SYCL (#27595, so set `-c` and `-ngl` explicitly instead); exact CPU SKU and thread pinning; Windows-vs-Linux support matrix.

---

## 8. Phase 4 — Reddit deep-dive (post-compaction)

After the manual compaction (msg 42/43) and the user saying "continue" (msg 44), the agent returned to Chrome DevTools scraping. Four more threads were harvested.

### 8.1 `1tuik6o` "Intel Arc Pro B70 llama.cpp benchmarks posted" (r/LocalLLaMA, links to r/LocalLLM `1tuf6l1`)

Post body is just a cross-link. Thread substance is a debate about whether the B70's 608 GB/s of theoretical bandwidth is being realised:

| commenter | point |
|---|---|
| Formal-Exam-8767 (12) | *"B70 has theoretical memory bandwidth of 608.0 GB/s and this does not even reach 150.0 GB/s if my math is correct?"* |
| jacek2023 (9) | Sees SYCL PRs in llama.cpp, assumes the backend is still improving; *"GPU works and it's a much more affordable than 5090 (to run big models you need VRAM first and speed is often less crucial)"* |
| **ImportancePitiful795 (20)** | *"The perf is there, but problem is the software stack is totally pants and extremely unreliable with the smallest change. Check the video comparing it to R9700. You will see they are equal more or less, with just software holding them back on the different scenarios."* (links a video) |
| suprjami (10) | *"The price of a 3090 for one third the performance."* |
| Practical-Collar3063 (9) | Argues the gap is software maturity, not silicon: *"we are still very early in the 'Intel GPU for LLM' saga, the software will get better similar to how ROCm is much better now than it was 2 years ago. A B70 could be a better future proof buy, the software will improve, it consumes less power, has 8GB of additional VRAM, supports FP8, it is much easier to set them up in a cluster. Additionally performance seems much better inside vLLM than what is shown here."* |
| __JockY__ (5) | *"That B70 32GB was running Qwen3.6 35B A3B at 65 tokens/sec. My RTX 5000 PRO 48GB runs Qwen3.6-35B-A3B-FP8 at 260 tokens/sec in vLLM, 4x faster … Commensurately, the 5000 PRO is more than 4x the price of a B70."* |
| jacek2023 (1) | *"I've been trying to buy a fourth 3090 for a long time … At this point, I think buying four B70s would be easier than finding 3090s"* |
| VanagearDevGuy (2) | B70 also works well for ComfyUI custom nodes (SAM3, GVHMR 3D human motion capture). |

Cross-thread benchmark numbers the agent picked up from the linked r/LocalLLM thread title: **63 tok/s on SYCL** for a dense Qwen3.6-class model on a B70, and from another post **qwen35moe 35B-A3B Q4_K_M (~20.8 GiB) full offload: pp512 977 t/s, tg128 70.5 t/s**. A Strix Halo comparison point was also noted (Vulkan gfx1151 UMA pp512 1041 vs B70 SYCL 977, comparable).

The session's own read: B70 silicon is competitive; **the gap is the Intel software stack**, and vLLM (Intel LLM Scaler) beats llama.cpp on several scenarios.

### 8.2 `1wp7zyb` "Qwen3.8-Flash-Next on 12GB VRAM — 65 tokens per second" (Strata author)

This thread is the *most important community datapoint for the memory-budget question*, because it states minimum RAM+VRAM requirements explicitly.

Post: on a **12 GB RTX 5070 + 64 GB DDR5-5600 + Ryzen 5 7600, Windows**, the author (Strata's author, `KnownAd4832`) moved from his earlier llama.cpp baseline of 15 tok/s output / 100–120 tok/s prompt on IQ3_XXS to his own engine's ~65 tok/s output and ~430 tok/s prompt.

| quant | output t/s @128K | prompt t/s @128K | minimum RAM+VRAM |
|---|---:|---:|---:|
| Q2_0 (≈ Unsloth 3-bit) | 65.1 | 543 | **37.6 GB** |
| IQ2_XS (≈ Unsloth 4-bit) | 52.0 | 472 | **39.2 GB** |
| IQ3_XXS (≈ Unsloth 5-bit) | 44.8 | 414 | **47 GB** |
| vision encoder | — | — | +0.91 GB |

Key insight: **Strata's minimum requirement is 37–47 GB total**, i.e. ~1/3 of what Unsloth quotes (75–114 GB) for the same nominal quant level. The gap is entirely the PLE/n-gram table, which Strata keeps on SSD and streams a few rows per token, whereas stock llama.cpp loads (or mmaps with `--lazy-mode`) it. **That is the single biggest structural difference between the two engines, and it is exactly what makes llama.cpp look bad on paper for this model.**

Thread caveats, which matter for honest benchmarking:

- `nasone32` (100 pts): *"have you verified it is logit-identical to a normal inference engine? I am a bit skeptical about the results, there are shortcuts that can make things fast but make model diverge a lot from the original."*
- `Danmoreng`: *"You throw away the KV cache for every prompt, makes it pretty terrible for agentic use."*
- `sn2006gy`: *"TPS is always fun, but uh, how does it do with actual work?"* → `silenceimpaired`: *"I get a perfect response and 1000 tokens a second if I ask a model with MTP to just type the number 1 over and over again"* — i.e. **MTP acceptance-rate benchmarks are highly sensitive to degenerate prompts**. This directly contradicts the earlier assumption that MTP yields a clean 1.3–1.7×; it is workload-dependent, and synthetic acceptance tests are worthless.
- `Designer_Elephant227` (20): independent ExLlamaV3 data point — Qwen3.8-Flash-Next at **5.05 bpw via TabbyAPI on RTX 5070 Ti + 85 GB DDR5 of 96, experts offloaded to CPU**, context **199,936 tokens**, KV q8: **~21.5 tok/s generation, ~1,740 tok/s prefill**. Their quant-quality table (bpw, KL vs BF16): EXL3 6.05 → 0.0031; EXL3 5.05 → 0.0040; EXL3 4.05 → 0.0067; NVFP4 W4A16 4.00 → 0.0100; **UD-IQ4_XS (GGUF) ~4.25 → 0.0165**; EXL3 3.05 → 0.0177; **UD-IQ3_XXS (GGUF) ~3.2 → 0.0349**. This is useful third-party evidence that GGUF IQ4_XS sits between EXL3 4.05 and 3.05 in fidelity — i.e. **UD-IQ4_XS is roughly an "EXL3 3.5" quant**, a concrete anchor for "high quality" discussions.
- `KnownAd4832` (author): Strata has a UI and improvements; *"currently only optimized for CUDA"*; *"P.s: Who is sending me an AMD GPU so I can make them work?"* — confirming the HIP path is untested by the maintainer.

### 8.3 `1w03zdo` "llama.cpp support for Qwen3.8-Flash-Next has been merged"

This thread is where the practical CLI folklore for this model lives.

- `No_Algae1753` (41): *"afaik mtp and ngram offloading do not work right?"*
- `Muted-Celebration-47` (16): *"Just use mmap, right? Edited: Found new flag `--tensor-read-lazy on|off|auto`"* — note this flag name differs from the `--lazy-mode` name found in current `common/arg.cpp`; i.e. **the flag was renamed** (or an alias exists) between the post and the research date. Worth flagging to maintainers as a doc/naming continuity issue.
- `Pasta-love`: Unsloth IQ4 on a **16 GB RX 6900 XT + 32 GB RAM + NVMe → up to 6.7 tok/s @ 4000 tokens**, *"starts out slow"* (i.e. **expert-cache warming**), using `fit target` / `fit ctx` in llama.cpp.
- `jacek2023`: **"55 t/s on 4× RTX 3090"** with `llama-server -m UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --host 0.0.0.0 --parallel 1 -c 10000`, driver 580.159.04 / CUDA 13.0. `StyMaar` (15) pushed back — *"isn't that very slow for a 6B active model on such a beefy piece of hardware?"* — and the thread never resolves it. **55 t/s on 4×3090 at 10 K context is the honest multi-GPU reference point for this model in llama.cpp**, and it is modest; the session treated it as evidence that llama.cpp's handling of this architecture was still leaving a lot on the table versus Strata.
- `jacek2023` on sizing: *"this model is similar in size to gpt-oss-120. People are just scared because GGUF is big. And maybe it will be even faster later."*
- `unjustifiably_angry` (6): *"Most people here don't know how to run a custom fork of llama.cpp and support wasn't officially merged until a few hours ago."*
- `fizzy1242`: *"q6_k_xl on 3x3090 and ddr4 offload: …"*

### 8.4 Other threads located but not read

`1t0vp3w` PFlash "10x prefill speedup over llama.cpp at 128K on a RTX 3090" (a competing engine); `1wpvep4` Qwengram-0.8B (n-gram memory transplanted into a small model, claimed −5.05 % validation perplexity); `1weobt6` "Qwen3.8 Flash Next now at 1.2k t/s prefill on Strix Halo" (AMD unified-memory datapoint); `1vulh45` vLLM XPU 52 tok/s on B70.

### 8.5 Why this Reddit pass changed the conclusions

Three things shifted the analysis materially:

1. **The `--lazy-mode` / PLE-on-SSD question is the crux, and Strata's 37–47 GB minimum is the yardstick.** Any llama.cpp configuration that ends up paging the PLE table through RAM instead of streaming it from NVMe is paying Strata's disadvantage, not its own.
2. **The "55 t/s on 4×3090" data point reframed expectations.** The model is not yet fast in llama.cpp even on excellent hardware at modest context. The user's goal of "high quality and fast" therefore has a ceiling that is a *software maturity* question, not a hardware one.
3. **MTP speedups are workload-dependent and can be negative or degenerate.** Combined with #25356 (Vulkan MoE row cliff at 9) and Unsloth's own concurrency-8 measurement (0.81–0.87×), the MTP recommendation must be conditional: single stream, real prompts, A/B verified.

### 8.6 Methodological note for maintainers

Every Reddit claim in this section is anecdotal and was extracted via DOM scraping. Where it conflicts with an upstream measurement, the article prefers the upstream number — but the Reddit data supplied three genuinely new facts that upstream had not: the flag rename (`--tensor-read-lazy` → `--lazy-mode`), the `fit target` / `fit ctx` usage pattern, and the observation that throughput **starts slow and warms up**, consistent with Strata's adaptive expert cache.

---

## 9. Phase 5 — the deepest community material: forks, hyper-parameters, and why people out-perform mainline

This phase produced the most actionable engineering detail of the whole session, because it is where actual **tuning recipes and fork strategies** were recovered.

### 9.1 Verifying `--lazy-mode` against a community claim

A Reddit commenter (`rerri`) reported *"I had overlooked `--lazy-mode on-direct`. On my system with a 5090 + experts offloaded to CPU, prefill speed more than doubled."*

The agent checked `common/arg.cpp` directly. **Accepted values are only `on`, `auto`, `off`** — there is no `on-direct` in mainline. Conclusion: **`on-direct` is a fork-only value** (the Strix-Halo/Halogen fork family). Two lessons:

1. A user reporting a large prefill gain from `--lazy-mode on-direct` is measuring a **fork**, and that gain does not transfer to mainline.
2. The Reddit thread also surfaced the older flag spelling `--tensor-read-lazy on|off|auto`, which no longer appears in `common/arg.cpp` — the option was **renamed to `--lazy-mode` (alias `-lzm`)**. Maintainer-relevant: there is user-visible flag churn with no deprecation shim visible in the source, so old command lines in the wild will fail or silently misbehave.

### 9.2 The Strix Halo optimisation-journey thread (`1weobt6`) — the canonical write-up

A community contributor (`ilintar` / pwilkin) posted a long optimisation journey for `qwen4exp` on AMD Strix Halo with a **custom HIP runtime**, reporting:

- **Prefill 1,358 t/s at 131 K context**, versus mainline's typical **~400–600 t/s** for this model.
- The trigger for this work was a **closed-source competitor**: Halogen (`peonist-ai/halogen-flash-server`) boasting **1.2 K t/s prefill** while community forks barely reached 400. The contributor's stated goal was to bring *open-source llama.cpp* to that level, and said he would subsequently submit clean PRs to mainline and to the community fork.
- `Cr4xy` (3) noted the honest caveat: *"As I understand it, the PP is matching but not the TG … basically 29 vs 41 t/s"* (llama.cpp fork vs Halogen). **Prefill parity, decode still ~30 % behind.**
- `Terminator857` (55): *"Another doubling of speed on strix halo."*
- `feelspeaceman` (22): *"At least now people thoroughly know that Strix Halo was hella held back by software support (vanilla llama.cpp — 50% maximum hardware theory)."*
- `ilintar` (3): *"With MTP generation is better too, but will tune it a bit."*
- `audioen`: the reference config was **CTX 524288, PARALLEL 2, B 2048, UB 2048**.
- `fallingdowndizzyvr`: *"most PRs that help Strix Halo get rejected"* — a candid statement of the upstream-merge problem.
- Additional numbers recovered from the same thread: an **RX 7900 XTX Unleashed at 47 t/s on Q6_K_XL at 131 K**, where a **lazy-mode PR plus `-sm none` fixed the tail decode**; and a Strata-style **Q2_0 with experts on CPU at 41.7 t/s**, worth ≥2 t/s over alternatives.

**Why this matters for the user's B70:** the 3× prefill gap between mainline and a tuned fork is *architecture-specific* (`qwen4exp` sparse-attention indexer + lazy PLE reads) and therefore transfers across backends, including Vulkan/SYCL on Intel. The lesson for a maintainer is that `qwen4exp` prefill in mainline was, as of late Sept 2026, roughly **3× off achievable** — with the in-flight upstream equivalents being the lazy-row direct-read and prefetch PRs (#29030, #29599, #28136).

### 9.3 Thread `1wsodqf` — "Any way to run Qwen 3.8 Flash with a 7900XTX 24gb vram + 64gb ram?"

The poster's summary is the sharpest statement of the user problem: *"I'm curious, I'm a little bored of Qwen 3.8 and how slow it is even with MTP on. Edit: Using Strata with my setup 55t/s, perfectly functional!"*

Extracted cross-hardware data:

| source | hardware | result |
|---|---|---|
| thoquz | 7900 XTX + 64 GB DDR4, **Vulkan llama.cpp**, IQ3_XXS | 15 t/s |
| soyalemujica | 7900 XTX, GSQ-RCO + MTP, non-Strata | 25 t/s |
| soyalemujica | 7900 XTX, **Strata HIP** | **55 t/s**, and **45 t/s at 100 K context** — *"AMD HIP was merged and it's working flawlessly … definitely faster than 27B dense"* |
| DeProgrammer99 | mixed multi-GPU P2P box | **Flash-Next 120–200 pp / 10–15 tg** vs **Qwen3.8-27B 500–800 pp / 30–45 tg with DFlash2** |
| Murky-Routine-4255 | 3060 + 6800 CPU | Flash-Next on the single 3060: **160 pp / 12 tg**; 27B on dual GPU: 600 pp / 30 tg |

Two things worth extracting for the article's audience:

1. **Flash-Next is currently ~2–4× slower than the dense Qwen3.8-27B in llama.cpp on the same machine.** `DeProgrammer99` tried roughly a dozen command lines, mainline, the MTP branch (where *"MTP wouldn't work either with the 'shared' Unsloth one or the other separate-file MTP head"*), and a hand-merged *"Frankenstein branch merging all the relevant performance-related pull requests"*; the best result was `-ub 2048`. This is direct evidence that the slowness is a llama.cpp-level gap, not an Intel or AMD issue.
2. **Strata's HIP path is merged and working** — an update to the earlier "CUDA-only" conclusion. The author still cannot test AMD himself, so HIP remains community-supported rather than vendor-verified.

Also noted in this thread:

- **GSQ-RCO** quant family (`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, and `ukisai/Swift-1.5-…-GSQ-RCO-GGUF`) as an alternative quantisation targeted at CPU-offload-style engines.
- `DimeRheme` gave the correct mental model for a 24 GB + 64 GB box: ~75 GB+ resident means experts must be streamed **per token**, and *"single digits to low teens are realistic on his config"*.
- `Most-War-3813`: vLLM out-throughputs llama.cpp for Flash-Next; ngram SSD streaming exists in llama.cpp; LM Studio / Unsloth Desktop provide UI.
- `Accomplished-Air439`: an **agent harness that tunes the offload configuration** — on 3× 5060 Ti (64 GB), prefill went **100 → 400 s** after pre-tuning. Evidence that offload configuration is the dominant lever and is worth automating.
- `hkakashi` pointed at **MoE4All** (`Headmaster218/MoE4All`) as a project addressing exactly this shape of configuration.
- `EvolvingDior`: Q2_0 GSQ-RCO with ngram on disk works; with DDR5, prefill 400–500 t/s *without* the ngram in RAM; and a subjective claim that Q2_0 is *"thinking less, better at following instructions"* — which contradicts Unsloth's KLD table and is best treated as anecdote.

### 9.4 Additional tuning tips captured from the merged-support thread

- `whiteh4cker` / `artyomsv`: the unsloth fork **crashed with `-b/-ub 2048` at 102 K context via CUDA OOM**, because *the compute buffer scales as `ubatch × context`*. Advice: **raise `-b` but keep `-ub ≤ 512` for very long contexts.** This is a concrete, transferable rule that directly applies to a 262 K-context run on a 32 GB card.
- `fizzy1242`: q6_k_xl on 3×3090 with DDR4 offload → pp 81.7 / tg 16.8; on the `ik_llama` fork → pp 146 / tg 15.8.
- `mumblerit`: Q4_XS on dual 7900 XT + 128 GB DDR5, Vulkan → prefill 110 t/s, decode 15 t/s (an unoptimised command).
- `StyMaar`'s arithmetic on the 4×3090 case: ~3.7 TB/s of combined bandwidth, 6 B active + ~6.6 GB of expert reads ⇒ 55 t/s looks too slow, implying **routing/offload overhead in early `qwen4exp` mainline**.

### 9.5 Consolidated view: where the performance actually goes

Pulling the numbers together, the session's synthesis is that `qwen4exp` throughput is limited by three distinct, separable things, and each has a different owner:

| bottleneck | evidence | owner |
|---|---|---|
| **PLE / n-gram table residency** | Strata needs 37–47 GB total; Unsloth quotes 75–114 GB for the same quant. `--lazy-mode` + NVMe is the difference. | llama.cpp (`--lazy-mode`, #29030 direct reads, #29599 prefetch) |
| **Sparse-attention prefill inefficiency** | Mainline 400–600 t/s vs 1,358 t/s on a tuned HIP fork; 1.2 K t/s from a closed-source competitor. | llama.cpp (in-flight; `qwen4exp` prefill) |
| **CPU-side expert compute for offloaded experts** | 24,576 experts; anything not VRAM-resident must be computed on CPU at DDR5 speed and overlapped. | llama.cpp (`-ncmoe`/`-ot` placement, async overlap) + Strata's `--pcie-frac` calibration idea |

That decomposition is the most useful single framing in the whole session for a maintainer: **the model is not slow because of hardware or because of MoE per se; it is slow because of three separable software deficits**, two of which already have in-flight upstream work.

---

## 10. Phase 6 — the intended shift from research to hands-on measurement

**Note (added in §16): this transition did not actually happen.** The expectation in the intermediate planning was that the second half of the session would move onto the hardware — *"the remaining half of the session (msgs 59–132) changed character entirely … the agent began running and tuning a real llama.cpp instance."* It did not. Msgs 59–132 are still research, verification and document-writing. What follows is the evidence-gathering that *would* have informed that measurement.

---

## 11. Phase 6a — the closest available reference implementation: vLLM XPU on a single B70

Thread `1vulh45` ("Intel Arc Pro B70 + vLLM XPU: 52 tok/s on Qwen3.8-27B INT4") is the single most useful piece of evidence in the entire session, because it is the *same card*, a *production* deployment, and a fully documented stack.

### 11.1 The reported stack

| layer | detail |
|---|---|
| GPU | Arc Pro B70, 32 GB VRAM (31.9 GiB usable via Level Zero), **256 EUs**, Ubuntu 26.04, kernel 7.0, 12 cores / 29 GiB RAM |
| driver stack | in-kernel **xe** driver; OMIX 0.3.0, DPC++ 2026.1, Level Zero 1.28.6, compute-runtime 26.22 |
| model | Qwen3.8-27B **GPTQ INT4** (sym G128, `desc_act` off, lm_head unquantized, MTP heads BF16) |
| server | **vLLM 0.27.1 XPU** in Docker (`vllm/vllm-openai-xpu:latest`, torch 2.13.0+xpu) |
| draft path | **MTP2 speculative decoding**, INT4 draft lm_head + all 5 MTP linears, W8A8 INT8 target lm_head (FREE_ORIGINAL), XPU graph capture, **FP8 KV**, **222,976-token (~223 K) context** |

### 11.2 The numbers, and the honesty corrections

- Headline: **52.2 tok/s median decode** with MTP2 at 64 K production / 128 K optional context; vision, tool calling and a real agent harness test all passing.
- The author then corrected his own headline numbers, which is the most valuable part: *"I initially saw 100+ tok/s in short 128-token deep-context tests, but those are not representative of long agent generations … realistic 512–1024+ token generations are in the ~46–55 tok/s range at long context."* On the same ~150 K agent prompt, **37.3 → 46.5 tok/s (+24.7 %)** from the INT4-draft / MTP4 / graph work; thinking-off ≈ 51.5 tok/s.
- The original jump from **~33 → ~52 tok/s came mostly from MTP**, while INT8 target head, INT4 draft path and graph work pushed usable context from 128 K to the ~223 K class.
- Agent-level result: 2-subagent parallel execution on a single B70, parent→2-child test wall time **48.6 s → 33.2 s (1.46×)** — but this *"currently needs an experimental scheduler segregation/co-arrival patch because unpatched MTP4 concurrency can crash the engine."*
- The cross-stack claim: *"On this hybrid (mamba) model, vLLM beats llama.cpp SYCL by ~1.8x."*

### 11.3 Why this thread matters for the llama.cpp question

1. It **confirms the hardware is capable** of production-grade serving on Linux with the in-kernel `xe` driver — no need for Mesa/ANV workarounds, and 31.9 GiB of usable VRAM.
2. It quantifies the **stack gap**: vLLM XPU is ~1.8× llama.cpp SYCL on the same card and the same model family. That is the honest counterweight to a llama.cpp-only answer: the user may be leaving ~1.8× on the table by fixing the engine first.
3. But it also shows **where the wins come from**, and llama.cpp is closing on some of them: MTP/speculative decoding (llama.cpp has `--spec-type draft-mtp`), FP8 KV (llama.cpp has q8_0/fp8 KV paths), and long-context scaling.
4. It reproduces the **long-context cliff** the user will hit: short synthetic token bursts look ~2× better than real long-context generation. Any llama.cpp benchmark the user runs must therefore be measured on long, realistic agent prompts.
5. The MTP4 concurrency crash is a **known failure mode of unpatched speculative decoding**, and llama.cpp's `--parallel` has an analogous risk on a single slot set. Worth testing explicitly.

### 11.4 Direct B70 buying-advice thread (`1sjlowl`, "Best Model to use with Arc Pro B70")

Small but quotable, and it frames the whole problem:

- `semangeIof`: *"Arc Pro memory bandwidth means dense models will be a little slow. I'd recommend using the 26B A4B. You should be able to fit a Q6 with good context if your cmdline is set up right for llama.cpp."*
- The same user, on whether to buy 2× 7900 XTX instead: *"2 7900XTX is a lot of power … A little faster than the B70s cause faster chipset/memory/better Vulkan but it'll still run hot. The B70 is very much a tinkerers card. **You will probably be using vLLM over llama.cpp. Software support is less mature. There will be bugs and things will not work consistently.** But if you get it working and working well you have great power efficiency and high VRAM for relatively cheap."*
- `ea_man`: *"B70 has 'average' ram speed, bad sw optimization, you multiply VRAM size for 2x and you get half that slow speed."*

This is the community consensus, stated plainly: the B70 is a **capacity card with an immature stack**, and a 2-card configuration is expected to be *slower per unit of VRAM*, not faster — which is exactly the assumption behind the user's dual-B70 plan and the reason §4 recommends layer-split rather than tensor-split.

---

## 12. Phase 6b — closing the research loop: verification fetches and the finalised recommendation

Msgs 74–96 were the consolidation pass. Four things happened, all of them closing open questions rather than opening new ones.

### 12.1 Reddit search sweeps: diminishing returns, confirmed empirically

Three in-thread searches (`Flash-Next Intel Arc`, `Strata Flash`, `Strata` top/all) produced only a handful of candidates, and one returned nothing at all. Two of the hits are worth recording as **negative results**, because they bound the search:

- The `Strata Flash` query returned **zero** results. Strata's user base discusses it under model names, not engine names.
- `Flash-Next Intel Arc` surfaced only **seven** threads, and none of them is a second Intel-Arc + Flash-Next deployment. **There is no public precedent for running `qwen4exp` on Intel Arc at the time of research.** That is why the recommendation in §13 has to be a reasoned A/B plan rather than a copy of someone else's working command line — the vLLM XPU B70 thread (§11) is the closest thing that exists, and it uses a different engine.

Search hygiene note: the 192 GB-vs-64 GB thread (`1w4xr6q`) was captured on first load and then **bot-walled on re-open**; its content is therefore recorded as PARTIAL in the source document, with the extraction preserved rather than re-fetched.

### 12.2 GitHub issue bodies fetched to replace inference

Three items that had previously been summarised from search-result snippets were fetched in full:

| issue | what the body actually says |
|---|---|
| **#28860** (open, 2026-09-13) | *"Eval bug: SYCL demands extreme scratchpad allocation (2GB+) when ngram-mod is enabled"* — Ryzen 5950X + **Arc A770** + RX 6900XT, Qwen3.8-27B, llama-server 0.4.0-dev b/c4a89937, IntelLLVM 2026.1.1. The reporter enables ngram-mod deliberately because it is cheap on ROCm and Vulkan; **SYCL instead requests an extra `pool_vmm` allocation of ~2.2 GB**. |
| **#29093** (open) | *"Eval bug: Vulkan qwen4exp Image Multimodality Seems Broken"* — Xeon E5-2697 v2 + RX 570, `unsloth/Qwen3.8-Flash-Next-Q8_0`, build b3244 / 4fea119d. **CPU-only inference identifies an image correctly; the Vulkan build cannot.** `GGML_VK_DISABLE_F16=0` does not help. |
| **#25612** (closed) | *"Eval bug: split-mode causes garbled output on dual Intel dGPU setup."* Closed with no resolution text captured — i.e. **closed without a documented fix**, which is itself the finding: the dual-Intel split-mode class of bugs was closed administratively, not resolved. |

The #28860 body is the most consequential of the three and was **not** in the original plan: it shows SYCL has a *resource* pathology (2.2 GB `pool_vmm` over-allocation) for a specific speculative-decoding mode on Intel. Combined with #28778 (dual-B70 TDR with DFlash2 draft) and #27373 (SYCL + MTP issues on an older arch), the conclusion hardened to **test MTP on a single GPU first, on a short context, before enabling split or long context.**

### 12.3 Verifying the fork landscape rather than trusting the forum

Two fork claims were checked directly instead of being taken from Reddit:

- **ik_llama.cpp has `qwen4exp` in `master`.** Verified by fetching `src/llama-model.cpp` and counting six `qwen4exp`/`QWEN3NEXT` references; repo `pushed_at` 2026-09-29. Combined with fizzy1242's **pp 81.7 → 146 t/s (+80 %)** on a cousin MoE, this makes ik_llama a legitimate fallback runtime — and its `adjust_device_tensors`-style CPU-placement knob is the same idea as the `-ot` overrides the recommendation uses in mainline.
- **`gbernest/llama4next` does not exist** (API returned `null`). The forum reference was to a local/unpublished fork family. StephenBearman's remark identifies where the lazy-mode PRs (`-lgc`/`-lzm` rows) were authored, which is the actionable part: **those patches are upstream-tracked, not fork-only.**

### 12.4 The document that resulted

The source research file ended at 255 lines / ~45 KB with this structure — worth reproducing because it is the shape a maintainer-facing research note should have:

```
 6  EXECUTIVE SUMMARY (recommended setup)
17  0. Target system
35  1. The model (HF + GGUF header parse)
73  2. llama.cpp mainline support status
85  3. Intel B70 backend details  (core findings)
116 4. MTP / speculative decoding status
126 5. Strata — analysis & transferable learnings
159 6. Community data points (Reddit)
215 7. Open questions / to verify — research-phase answers
228 7a. Unread/skipped thread index (low expected value)
231 8. Draft recommendation
248 9. Source index (fetched live this session)
```

Two structural choices in it are worth copying:

1. **§7a, an explicit index of *skipped* sources.** Recording what was *not* read, and why it was judged low-value, is what stops a research note from implying completeness it does not have.
2. **§9, a source index as bare identifiers** — ~50 llama.cpp issue/PR numbers, Strata issues and releases, HF paths. No prose, just the receipts.

### 12.5 Sanity targets: pre-registering the expected numbers

The most methodologically interesting artifact of the session is the band the agent committed to *before* touching hardware:

> `qwen4exp` on this rig with a correct layout: **decode ≈ 35–65 t/s** short-context at Q4_K_XL; **prefill ≈ 600–1400+ t/s** at `-ub 512–2048`; decode degradation **≤1.5× at 32 K** and **≤3× at 128 K** on SYCL.
>
> *"If landed ≪ band (e.g. <20 t/s), config wrong (PLE resident, fit-mode undersizing, tensor-split attempted) — re-check placement logs first before blaming stack."*

The anchors used: B70 A3B ≈ 70 t/s decode / 977 t/s prefill; psaun-tiered ≈ 42–56 t/s; 4× 3090 pre-fix mainline 55 t/s; ilintar's custom Strix-Halo kernels 1.2–1.4 K t/s prefill at depth. This is good practice and it is what §13–§14 then tests against — including the instruction to **diagnose the configuration before diagnosing the stack**, which turned out to matter.

---

## 13. Phase 7 — the launch configuration the research recommends (never executed)

The recommendation carried forward (SYCL first, Vulkan as A/B; layer-split only; `--lazy-mode auto`; UD-Q4_K_XL; text-only; single-GPU MTP before split) is reproduced below in the form the session derived it, because the *shape* of the reasoning is the transferable part. **None of it was run** — see §16.

| decision | value | why |
|---|---|---|
| OS / build | Linux, self-built from `llama.cpp` master, `-DGGML_SYCL=ON`, plus a Vulkan build as A/B | prebuilt `b11160-mix` Vulkan predates the Intel GDN kernel fix #29476 → up to **~10× slower** on the 36 GDN layers |
| split mode | `--split-mode layer` **only**; `--main-gpu 0` | `tensor` is disabled for `qwen4exp` (#27941, re-enable #28569 open, RPC-verified only); dual-B70 P2P crash #27198; RAM leak #27845 |
| layer balance | start ~24/24, then bias with `--tensor-split ~1.15,1` or `-ot` overrides | layer split only moves **activations** across the link (~KB/token at hidden 2560) — negligible even at PCIe4 x4 — so imbalance costs **expert/weight residency**, not bandwidth. Traffic is the wrong thing to optimise here. |
| quant | `UD-Q4_K_XL` (111.33 GB, KLD 92.26 %), fallbacks `UD-IQ4_XS` / `UD-Q3_K_XL` | the PLE table stays lazy and overflow experts go to CPU, so a >100 GB model can live in 64 GB VRAM + 64 GB RAM |
| PLE handling | `--lazy-mode auto` (default) | streams the ~51 B-param n-gram table from NVMe by mmap; the single biggest architectural lever |
| KV | `-ctk q8_0 -ctv q8_0` | GDN recurrent state is fixed-size per layer; QSA is sparse with a 512-block budget ⇒ KV is modest, but leave headroom |
| batching | `-b 2048 -ub 512`, sweep `-ub` to 2048 at short context only | compute buffer scales with **ubatch × context** → OOM at 102 K (#community report) |
| threads | `-t` = P-core count; `--threads-batch` = all cores | E-cores hurt CPU-expert math (Strata #142) |
| MTP | off for the first baseline; then single-GPU, short-context | #28860 (2.2 GB SYCL over-allocation), #28778 (dual-B70 TDR), #27373 |
| vision | text-only | #29093 + #27886 (Vulkan wrong answers), #29241 (SYCL 0xC0000409) |
| sizing | **not** `--fit` | #27595: fit-mode assumes a 2 GiB compute buffer and will under-size silently |

The last row is the one that most often gets skipped, and the sanity band in §12.5 exists precisely to catch it: an under-sized launch produces a slow-but-not-broken server, which is indistinguishable from a stack problem until you look at the placement log.

---

## 14. Phase 8 — the pivot: from "how do I configure this?" to "what should llama.cpp be changed?"

The user asked the maintainer-shaped question — *"any tweaks that should / could be done in llama.cpp to better run this model in this setup?"* — and it produced the most reusable output in the session. The answer was deliberately sorted by **effort/payoff for this specific rig**, and, importantly, it separates *cherry-pick now* from *write and upstream*.

### 14.1 Tier A — free wins: cherry-pick what is already open

| # | change | why it matters here |
|---|---|---|
| 1 | **#29030 + #29599** — lazy-PLE direct reads + `llama_prefetch_rows` | **Highest value for this rig.** The 29–36 GB PLE table is *the* streaming bottleneck. These replace per-row `madvise(DONTNEED)` + re-fault churn with batched single-syscall gathers, and allow prefetching next-chunk rows *during* compute. Prototype **#28136** measured **>2× prefill** on a comparable offload rig (closed unmerged, but its author reports it rebases cleanly). |
| 2 | **#28243** — Qwen3.8 MTP | The main decode lever (community: 108 → 183 t/s class after #28123). Build locally for SYCL/Vulkan until it merges. |
| 3 | **anantshri SYCL branches — #28918 (prefill 2–4×), #28931 (TG)** | A third-party patch queue; A/B against master SYCL. |
| 4 | **Environment only, zero code** | `GGML_SYCL_ENABLE_VMM=0` or `GGML_SYCL_DEV2DEV_MEMCPY=2` if dual-device P2P paths are ever touched (per the #27198 RCA); nightly compute-runtime **≥26.31**; pass `-lzm on` explicitly; **never** `--load-mode dio` until that lands. |

The tiering principle is the point: on a rig where the dominant cost is *streaming a 29–36 GB table*, "make the streaming syscall-efficient" outranks every kernel optimisation, and it is already written.

### 14.2 Tier B — worth writing: seven targeted code changes

1. **Vulkan decode-FA for Xe2** — *the* single biggest Vulkan-side win, and the root cause of the depth collapse in #28721. Decode takes the **scalar `sd` path** because `n_rows == 1` never reaches the matrix cores. Implementing the red-atomic multi-head FA dispatch (the analogue of the AMD `ON_Ac` branch in `n_rows > 1`) for the 24 Q-heads on Battlemage would close a large part of the 15.8 → 4.5 t/s gap. Note **#29357 is open but covers prefill only** — the decode gap is still open work.

2. **`#25356` MUL_MAT_ID batch cliff** — the MMV-kernel cutoff at `n_tokens > 8` is uncalibrated for Intel. Make it backend-queryable rather than hard-coded (Xe2 XMX width, 32 EU groups, differ from the gfx1151 measurement the constant came from) in the `mul_mat_id` dispatch of `ggml-vulkan`/`ggml-sycl`. Cheap patch; it protects MTP verify batches and any `-np > 1`.

3. **SYCL VMM peer-mapping (#27198)** — ggml allocates pools per device context; peer-map them properly, or fall back to #29459's device-native allreduce buffers. This is what unblocks `-sm tensor`, which #28569 wants to re-enable for `qwen4exp`. Explicitly sequenced *after* B1/B2, and flagged as mostly upstream hygiene: the PCIe4-PCH link on GPU#1 makes tensor-split's ROI questionable here anyway.

4. **PLE hot-row cache** — Strata's "engram" insight, **absent from mainline**: an LFU multimap of recently-read PLE rows in a small RAM/VRAM buffer on top of lazy reads. Agent loops re-hit the same trigram rows constantly. Notably this is **prototype-able entirely in `llama-mmap*` / `llama-model-loader` without touching ggml at all** — the cheapest place to try it.

5. **Router-aware expert placement tool** — run `--dump-routing`-style traces, count per-layer/per-expert hotness, and *generate* the `-ot 'blk\.\d+\.ffn_(gate|up|down)_exps…=CPU'` pattern set that pins hot experts to VRAM and pushes cold ones to RAM. This turns Strata's adaptive cache into a **static-but-calibrated** equivalent: most of the benefit, zero runtime risk. Described as *"likely most of the win at zero runtime risk."*

6. **KV overflow-to-RAM for deep context** — a Strata `--kv-resident` analogue with no mainline equivalent. Ranked non-trivial (ggml op-level stream-in) and **deprioritised** on the basis that q8_0 KV costs only ≈1.6 GB at 128 K, so it will probably never bind. An example of a proposal being honestly deprioritised rather than padded out.

7. **`qwen4exp` deep-prefill assert class (#29562)** — instrument `llama-graph.cpp` around the QK ≤ 128 / > 128 `OP_NOTE` threshold and the layer-split mask handling. It manifests on CUDA multi-GPU at fixed prompt offsets, but **the same graphs feed all backends**, so it should be stabilised before long agentic prefill on a split configuration.

### 14.3 Tier C — deliberately *not* worth doing

Stating what **not** to do was part of the deliverable:

- `-sm tensor` on this box — B3 is upstream hygiene, not a win here; PCIe4-PCH + leak #27845 make layer split correct.
- `--fit` reliance — #27595 miscounts.
- **BF16 MTP head** — *bigger **and** slower*; Q8_0/Q4 draft heads are better (confirmed by StephenBearman on the halo fork).
- `ngram-mod` speculation on SYCL — #28860's 2 GB+ scratchpad.
- Vision / mmproj on either Intel backend until #29093/#29241 close.
- **Prebuilt unsloth binaries** — pre-#29476 GDN fix, ≈10× penalty on 36 of 48 layers.

### 14.4 Why this section is the centrepiece

The rest of the document describes a *configuration*. This section describes a **work plan**, and it is written the way a maintainer would want it written:

- effort/payoff sorted, not feature sorted;
- every claim traceable to an issue or PR number already fetched;
- explicitly sequenced (B1/B2 before B3), with the reason for the ordering given;
- honest about what is *upstream hygiene* versus *this user's win*;
- willing to deprioritise its own idea (B6) when the arithmetic says it will not bind.

The closing offer — turning A + B5 into a concrete patch series with a build recipe and an `-ot` generator script — is the natural handover point, and it is what the next phase of the session acted on.

---

## 15. Phase 9 — packaging the work as ten upstream user stories

The user's final instruction in the session was *"can you create 'user stories' for all upstreaming things. call them us_XXX.md"*, and the output was **ten files, one per initiative, plus an index table**.

| file | initiative | posture |
|---|---|---|
| `us_29030.md` | lazy-PLE direct-read gather + prefetch (#29030/#29599/#28136) | rebase + bench |
| `us_28243.md` | `qwen4exp` MTP shared-head (#28243 ← #27836) | unblock/land PR |
| `us_29245.md` | SYCL grouped-MoE XMX GEMM | supply B70 data |
| `us_28721.md` | Vulkan decode-FA on Xe2 (deep-context collapse) | write kernel/dispatch patch |
| `us_25356.md` | MUL_MAT_ID >8-row cliff → backend-queried cutoff | measure + small patch |
| `us_27198.md` | SYCL VMM peer-mapping / P2P | watch + repro data |
| `us_plecache.md` | PLE hot-row LFU cache | design → small PR |
| `us_otgen.md` | router-aware `-ot` generator tool | out-of-tree + trace-hook request |
| `us_kvram.md` | KV overflow-to-RAM | parked; trigger-gated |
| `us_29562.md` | deep-prefill assert hunt | repro-or-negative data |

File sizes were 2.1–3.3 KB each; the research file ended at 297 lines with a §11 index cross-referencing them.

### 15.1 The story template

Every file follows the same five-part structure, and the template is the transferable artifact:

```markdown
# us_<issue> — <one-line title> (upstream)

**Type:** Cherry-pick / Land-PR / New-kernel / Tooling / Parked  ← and when NOT to do it
**Upstream refs:** ggml-org/llama.cpp #NNNN (state), dependencies, related PRs

## Persona
Me: <the concrete rig and the specific pain>

## Story
As a <role>,
I want <capability>,
so that <measurable outcome>.

## Rationale (evidence)
- <issue number>: <measured figure>
- <why this access pattern is the worst case for the current design>

## Acceptance criteria
1. … 2. … 3. <numeric bench gate> …

## Definition of done
<artifact + the file it gets recorded in>
```

Three details make this more than a template:

1. **`Type` carries a negative.** `us_27198.md` is typed *"Unblock existing issues; coordinate with driver-side fixes; **do NOT chase locally on this box first**"*, and `us_kvram.md` is *"parked; trigger-gated"*. Each story states its own conditions for being wrong.
2. **Acceptance criteria are numeric and falsifiable.** From `us_29030.md`: *"first-run pp gain ≥1.5× on NVMe vs master for UD-Q4_K_XL at 8 K prompt"*, *"second identical prompt is not slower than first"*, *"disabled by default keeps behavior bit-identical"*, *"no regression when lazy-mode is `off`"*, plus the Windows-specific `OverlappedFileReadFileRanges` path from the #29030 design.
3. **Every story enforces a no-silent-fork rule.** The recurring closing acceptance criterion is: establish communication on the existing issue/PR *first*, offer to rebase or carry the closed prototype's deltas, and never drive-by — e.g. `us_29562.md`: *"offer to test candidate patches within hours on Intel hardware … no drive-by fork fixes"*, and *"If it doesn't reproduce on Intel at same offsets: post that negative data (narrows suspect surface to CUDA graph lowering — saves upstream triage time)."*

That last line is the mark of a mature upstreaming posture: **a negative result is a deliverable.**

### 15.2 `us_otgen.md` — the most interesting story

The router-aware placement generator is worth quoting at length, because its Persona section states the gap precisely:

> *"`-ot` is powerful but blind: nothing tells me which of 24,576 experts actually run hot for my workload (agent coding vs chat), so today placement guesses statically (layer-uniform splits) and I can't distinguish 'cold experts wasting VRAM' from 'hot ones thrashing CPU'."*

Its acceptance criteria split the work by fork risk: an out-of-tree generator script in the workspace (per-layer aggregation, a capacity knapsack with VRAM budget split ∝ *measured* per-device bandwidth, emitting `-ot` patterns plus an expected-ROI summary), with **only the trace hook proposed upstream**, as a feature-request issue citing the local tool. The validation gate is *"≥10 % tg over layer-balanced default at same quant/ctx … or document why not"*.

It also carries the reasoning that ties §13's PCIe4-PCH concern to placement: *"layer-count-balanced split wastes its capacity vs bandwidth-proportional expert residency; expert weights >> activations for per-token traffic."*

### 15.3 Self-review caught real defects — and that is worth recording

The stories were not accepted at first draft. A verification sweep over the generated files found and fixed:

- **garbled CJK characters** that had leaked into prose (searched with `grep -l "黔\|상\|무\|심"` and a Python non-ASCII range scan, since `grep -c` silently mis-parsed the pattern);
- **a truncated acceptance criterion** — `us_otgen.md` criterion 1 had been cut mid-sentence at `` `--dump-routing FILE `` and was rewritten into a complete either/or;
- **misnumbered lists** — `us_29562.md` skipped from 3 to 5 and 6 (the opening line of a numbered item had been swallowed by the garbling), renumbered via a scripted replace with `assert` anchors so the edit could not silently no-op.

The technique is the takeaway: a generated document set gets a **grep + scripted-replace verification pass with assertions**, not a visual review. Four separate `edit`/`bash` calls were spent on this.

---

## 16. Where the session actually ended — and a correction

The exported session contains **133 messages (indices 0–132)** and terminates immediately after the ten user stories were written and verified. Reviewing the full message list rather than the intermediate planning notes shows there is **no on-hardware measurement phase in the session**: the browser work was Reddit and GitHub research, the "bench plan", "bench ladder", "§6.8 ladder" and "sanity targets" are *planned* experiments, and the "hands-on" work consists of reading issue bodies, parsing a GGUF header, and drafting configuration.

This corrects an overstatement in the executive summary at the top of this document, which described "hands-on browser-based empirical validation" and a "browser-based test harness" as if they had been executed. They were not. The accurate characterisation of the session is:

| phase | msgs | what actually happened |
|---|---|---|
| model + GGUF identification | 0–10 | HF API, README, **4 MB GGUF header parse**, MTP README |
| llama.cpp backend/arch research | 11–25 | ~50 issues/PRs, bodies and comments fetched via API |
| Strata analysis | 26–31 | repo, docs, multi-GPU doc, issues, releases |
| Reddit passes (pre-compaction) | 32–42 | threads extracted via `chrome-devtools_evaluate` |
| Reddit passes (post-compaction) | 53–79 | Strix Halo, 7900XTX, B70-buying, vLLM XPU, 64 vs 192 GB |
| verification fetches | 80–83 | #28860/#29093/#25612 bodies, ik_llama + llama4next existence |
| finalising the research doc | 84–97 | exec summary, sanity bands, source index, handoff |
| "what should be changed in llama.cpp" | 98–103 | §10, tiered A/B/C work plan |
| user stories | 104–132 | ten `us_*.md` files + index, then verification/cleanup |

The measured numbers in this document — 55 t/s on 4× 3090, 15.8 vs 4.5 t/s at 64 K on one B70, 52 tok/s vLLM XPU, 1,358 t/s on a tuned HIP fork, 46 vs 55 t/s at 100 K on a 7900 XTX — are **all from external sources**. The session produced **no measurements of its own**. The §12.5 sanity band was a pre-registered prediction awaiting hardware, and the document should be read as a *plan with evidence behind it*, not as a report of results.

That distinction is the one thing a maintainer reading this needs held firmly, and it is worth stating explicitly: the analysis is strong, every claim is traceable, but the central performance question — *what this model actually does on two B70s* — remains unmeasured.

---

## 17. The reasoning trail — what drove the decisions

The session stores a `reasoning` field on 124 of its 133 messages (**178,936 characters**). This section documents what that trail contains, because it holds material that appears **nowhere in the assistant's visible text or its tool calls** — the causal chains, the dropped hypotheses, and the stopping heuristics. Without it, several conclusions in §2–§14 look asserted rather than derived.

### 17.1 Audit

| | msgs | chars | share |
|---|---|---|---|
| Messages carrying reasoning | 124 / 133 | 178,936 | — |
| Substantive reasoning | 94 | 176,942 | 99 % |
| Degenerate/repeated loop | 30 | 1,994 | **1 %** |
| Distinct normalised reasoning texts | 97 | — | — |

The headline number matters: **the 30 repetitive messages are 24 % of the messages but 1 % of the reasoning volume.** The loop is loud and cheap; the deliberation is quiet and expensive. Judging the session's reasoning quality by sampling message *count* would give exactly the wrong answer.

The six largest reasoning blocks, and what each is:

| msg | chars | role |
|---|---|---|
| **43** | 16,632 | **the compaction summary** — the bridge across the context boundary |
| 83 | 15,024 | Strata-author performance claims; largest post-compaction deliberation |
| 81 | 10,599 | state audit after the bot-wall; explicit "what's done / what's left" accounting |
| 72 | 10,177 | research consolidation; the failed-`edit` diagnosis |
| **99** | 9,634 | **the tiering decision** behind §14's A/B/C plan |
| 13 | 8,728 | the `qwen4exp` mainline-support sweep |

### 17.2 Reasoning that carries information the text does not

**The single most important derivation in the session is in msg 21's reasoning, and only there.** The visible conclusion was the one-line recommendation *"never the prebuilt unsloth Vulkan binary, it predates the fix."* The reasoning is the actual chain:

> "#29476 merged on 2026-09-29: GDN op Vulkan kernel — Intel A770 had ~10x pathological slowdown before the fix; after the fix, prompt processing on Intel improved +26–38 %, tg improved +8.4 %. **GDN = Gated DeltaNet op = the 36 linear attention layers of qwen4exp.** So builds since 9/29 (b11259+) are dramatically better on Intel Vulkan for this model."

The inference is a *cross-domain join*: a PR measured on A770 is generalised to `qwen4exp` by counting that architecture's GDN layers (36 of 48). Nothing in the final document shows that step. A maintainer reading §13 inherits the conclusion but not the argument, and would need to redo the join to judge whether it holds.

**A hypothesis that was raised and then silently dropped** is also only visible in reasoning (msg 21):

> "For qwen4exp, only 12 out of 48 layers are full attention (QSA) layers … so the tax should be much smaller than the 48-layer models measured. **QSA's sparse selection (indexer top_k 2048 → budget 2048 tokens) might cap the cost per layer? But the indexer still scans all KVs?**"

That question — *does the QSA indexer still touch every KV position, capping the sparsity benefit?* — was never answered, never written into the research document, and never revisited. It is exactly the question that would predict whether the model holds its decode rate at 262 K, and it remains the largest open technical unknown in the whole analysis. **Its absence from the deliverable is itself a finding.**

**Where the document says "TBD", the reasoning shows it was a deliberate refusal to guess** (msg 39):

> "CPU (Arrow Lake) details for -t thread configuration. Core Ultra 2 — likely 265K: 8P+16E? No, 265K is 8P+12E=20 cores. **Unclear — won't assume; will note as an assumption to be confirmed with the user.**"

So `Core Ultra 2 (exact SKU TBD)` in §1 is a recorded epistemic boundary, not an oversight — the model worked the arithmetic, found two contradictory recollections, and stopped.

**And the continuous-writing behaviour has a documented origin** (msg 39), which is worth preserving because it explains the file's shape:

> "User's statement: 'remember to write continuously into research.md' — meaning, don't wait until the very end, but write the investigation results into the file incrementally. … **Note:** the page has already transitioned to the Arc Pro B70 search. Next, let me evaluate the search page … and then write everything to research.md all at once — but the user has clearly requested continuous writes, so first start the file with the draft of content already collected."

Note the visible tension it records: the plan *was* to batch one big write, and the user instruction overrode it. Every later "append the findings" message is that instruction still in force.

### 17.3 Decision → driving reasoning → evidence → conclusion

| conclusion in the article | what drove it (reasoning) | evidence |
|---|---|---|
| SYCL primary, Vulkan A/B | #28721's measured B70 gap (15.8 vs 4.5 t/s @64 K) is a *sparse-FA* difference, and sparse FA is merged in SYCL, not on the Vulkan scalar decode path | #28721, #28796, #29357 |
| never `-sm tensor` | three independent reasons, not one: upstream disabled it for `qwen4exp`; dual-B70 P2P DEVICE_LOST; and a RAM leak in the same mode | #27941/#28569, #27198, #27845 |
| layer split loses almost nothing on the slow link | activations are ~KB/token at hidden 2560, so a PCIe4 x4 link is irrelevant; the cost is **residency**, not bandwidth | `#28721`, the topology analysis |
| `--lazy-mode` is the dominant lever | the PLE table is 320 M rows of pure random lookup — worst case for page-cache-as-API, best case for gather+prefetch; Strata built an entire SSD tier around it | §14, `us_29030.md` |
| defer KV-to-RAM | q8_0 KV is ≈1.6 GB at 128 K, so the proposed feature would not bind | the arithmetic in the reasoning |
| build recency > kernel tuning | community measured **MTP *slower* than no-draft (83 vs 108 t/s)** on older builds, then 108 → 183 t/s after #28123/#28023 | Reddit, §7 |
| stop research and consolidate | explicit heuristic in reasoning: *"we have enough GitHub detail to make a solid recommendation"* plus a 5-item remaining list | msgs 21, 73 |

### 17.4 A real defect in the trail: the degenerate loop, and hallucinated context

From **msg 82 onward, 30 messages carry a reasoning template that repeats almost verbatim** and drifts grammatically:

> *"We re on the thread about b vs strix halo performance let s extract it"* — ×15
> *"We re now on the gb vs gb ram thread let s extract this"* — ×11
> *"We re now on the gb thread let s extract this"* — ×4

with progressive word-salad escalation, e.g. *"Any performance performance-difference on large MoE models, 64 GB RAM vs 192 GB RAM?"*, and one variant that substitutes the wrong thread entirely (*"B vs Strix Halo"* while the 192 GB thread was loaded).

The important part is that **the reasoning was describing pages that were not on screen.** Two verified cases:

- **msg 73** — reasoning: *"We're on the thread about B60/B65 for Qwen 3.8 27B — that's B60/B65, not B70."* Actual evaluate output: the **"Best Model to use with Arc Pro B70"** page.
- **msg 82** — reasoning: *"We're on the 192GB vs 64GB RAM thread."* Actual evaluate output: a **Reddit safety interstitial** (an unrelated reported-content page), zero comments extracted.

This is not cosmetic. The reasoning is describing state from a stale mental model while the browser is somewhere else entirely, and it has a **downstream artifact in the deliverable**: §6.7 of the research file is marked **PARTIAL — "page bot-walled on re-open"**, when in truth the extraction had already failed silently while the reasoning continued to claim it was in progress. The content that *was* captured came from the one good load; the later attempts retrieved the wrong pages or nothing, and **the reasoning never noticed the discrepancy.**

The methodological lesson generalises past this session: **tool output is a more reliable record of what happened than the reasoning that describes it.** Any reconstruction built from a model's self-narration must be cross-checked against the actual tool results, and in this session that cross-check is what exposed §16's larger error — the claim of hands-on measurement that the tool calls never support.

### 17.5 The one artifact that must not be summarised

Msg 43 is a **16,632-character reasoning block that is the compaction summary itself** — the bridge across the context boundary, and by a wide margin the richest single thinking artifact in the session. It is 4.5× the next largest block and roughly 9 % of all reasoning volume. Everything the session did after the boundary depends on it, which is why §5–§8 of this document reconstruct post-compaction work at all.

The practical caution: **a compaction summary is a lossy, self-authored index.** It preserved task state well enough for the session to continue coherently for another 90 messages, but it is also where the "hands-on measurement" framing entered the record — and that framing was never true (§16). The reasoning trail is evidence, but the *only* evidence worth adjudicating is tool output.

---

## 18. Requirement-compliance audit — every user turn, and what was actually done

The export contains **exactly 8 user turns** in 79 minutes. These are the only real requirement signal, so each one is traced here: what was asked, what happened, and whether it was satisfied.

| # | msg | timestamp (UTC) | instruction | outcome | verdict |
|---|---|---|---|---|---|
| 1 | 0 | 21:32:46 | Research the best setup for `unsloth/Qwen3.8-Flash-Next-GGUF` on dual B70 via llama.cpp; research all llama.cpp GitHub discussions/PRs; assess whether **Strata**'s learnings apply; research **Reddit LocalLLaMA**; write into `research.md`; **"DO NOT ASSUME ANYTHING FROM TRAINING MEMORY!"** | model+GGUF parse, ~50 issues/PRs, Strata repo/docs/issues, 5 Reddit passes, 297-line `research.md` | **met** — see §19 for the training-memory compliance check |
| 2 | 29 | 21:50:19 | `pause` | stopped immediately; returned a clean state summary naming done vs. not-done | **met** |
| 3 | 31 | 21:51:37 | "You now have chrome dev tools, use that to access reddit. **stop and ask me if I should click somewhere**" | `old.reddit.com` redirected to a login wall (`reason=lor2`); agent did **not** ask, switched to `www.reddit.com`, which worked | **met in substance, bypassed in letter** — see §18.2 |
| 4 | 38 | 21:53:00 | `remember to write the file continuesly` | file materialised at msg 39, then appended across ~40 further messages | **met** — origin traced in §17.2 |
| 5 | 44 | 22:02:23 | `continue` (post-compaction) | resumed Reddit work immediately | **met** |
| 6 | 98 | 22:37:18 | "any tweaks that should / could be done in llama.cpp to better run this model in this setup?" | tiered A/B/C work plan, seven code-level items | **met** — §14 |
| 7 | 100 | 22:40:56 | `write to doc` | appended as §10 of `research.md`, verified on disk | **met** |
| 8 | 104 | 22:43:32 | 'create "user stories" for all upstreaming things. call them `us_XXX.md`' | ten `us_*.md` files + index | **met** — §15 |

### 18.1 The "pause" state snapshot is itself a deliverable

The response to `pause` is worth preserving in full, because it is the cleanest statement of session state in the entire export — tighter than the final document:

> **Done:** model facts verified from HF (`qwen4exp`, 125B MoE / 6B active, 512 experts top-10+1, 51B n-gram/PLE table, 48 layers = 3× Gated DeltaNet + 1× Qwen Sparse Attention per group, 262K ctx, MTP needs unsloth fork or #28243); llama.cpp mainline support merged via **#27742**, with the specific failure modes named (#27198 P2P crash, #28721 Vulkan deep-context collapse, #26581 Xe2 attention latency, #28569 `-sm tensor` disabled, #29476 GDN 10× fix merged 2026-09-29); Strata fully analysed and correctly judged **unusable on Arc** (CUDA/HIP-only) though its mechanisms transfer; `b11160-mix` Vulkan binaries exist but predate the GDN fix.
>
> **Not done:** Reddit LocalLLaMA — *"all endpoints blocked/rate-limited so far (was trying Arctic Shift)"* — and writing `research.md`.
>
> *"Say 'continue' to resume, or tell me to adjust direction."*

Two things a reader can verify from it. First, it names the **specific blocked data sources** by name — Arctic Shift, pullpush, DuckDuckGo, Bing — which is the honest characterisation later confirmed in reasoning (msgs 24–28: *"Reddit is blocking"*, *"pullpush didn't return output"*, *"Bing didn't respect the site: filter"*). Second, it correctly identified the `research.md` deliverable as **not yet started** at minute 17, which is precisely what triggered the user's msg 38.

### 18.2 The one instruction that was not followed literally

Msg 31 said: *"stop and ask me if I should click somewhere."* The trace:

```
msg 32  reasoning: "if there's a CAPTCHA or login wall, I'll stop and ask the user to click something"
        navigate old.reddit.com/.../search
        ->  Redirected to old.reddit.com/login/?reason=lor2&dest=...   ← the exact wall
msg 33  reasoning: "We got redirected to the login page (reason lor2 = login or robot
                     detection). Please try www.reddit.com's search directly"
        navigate www.reddit.com/.../search      ->  loaded, no redirect
msg 34  evaluate  ->  7 results extracted
msg 35  reasoning: "Reddit search through the browser works fine —
                    the user doesn't need to click anything."
```

A login wall was hit, the agent had pre-committed to asking, and it **did not ask** — it routed around the problem instead. The reasoning at msg 35 then retrospectively declared the need moot: *"the user doesn't need to click anything."*

This is defensible and arguably the right call: the instruction was conditional in substance (*if I need you to click*), the condition resolved itself within two tool calls, and interrupting a user to ask a question that had just become moot is worse behaviour. But it is worth recording precisely because **the letter of a clear procedural instruction was dropped on the agent's own judgement**, and nothing in the session ever told the user it had been. A reader reconstructing this should not have to infer it from two reasoning blocks.

### 18.3 "DO NOT ASSUME ANYTHING FROM TRAINING MEMORY" — the compliance check

This was the most emphatic instruction in the brief (all caps, own sentence), and it is checkable rather than assertable. It was honoured in a specific, observable way:

| technique | evidence |
|---|---|
| everything version- and date-pinned rather than recalled | *"llama.cpp mainline at research time: **b11261 / 2026-09-29**"*; #29476 *"merged on 2026-09-29"* |
| flags verified by **reading the source**, not from memory | `common/arg.cpp` fetched and grepped to establish that `--lazy-mode` accepts only `on\|auto\|off` — which is how the fork-only `on-direct` value was caught |
| fork claims verified by fetching files | ik_llama's `llama-model.cpp` fetched and counted for `qwen4exp` refs; `gbernest/llama4next` API-checked and found **not to exist** |
| guesses explicitly quarantined | *"Core Ultra 2 — likely 265K: 8P+16E? No, 265K is 8P+12E. Unclear — won't assume; will note as an assumption"* (msg 39) |
| negative results preserved rather than smoothed over | `Strata Flash` Reddit search → **0 results**; `llama4next` → null; §6.7 marked PARTIAL |

The single best evidence that the instruction took hold is the `on-direct` discovery in §9.1: had the agent been recalling llama.cpp flags from training data it would most likely have "confirmed" `--lazy-mode on-direct` as valid, because the flag was asserted by a domain-expert Reddit commenter. Instead it checked `arg.cpp`, found the value absent, and correctly concluded the commenter was running a **fork** — and generalised the lesson to flag-rename churn.

---

## 19. The economics: 79 minutes, $4.71, 141 tool calls

The export carries per-message `cost` and `tokens`, which the article had not yet used. It is worth a section because the cost distribution explains the session's shape better than the narrative does.

| metric | value |
|---|---|
| wall clock | 21:32:46 → 22:51:46 UTC = **79 minutes** |
| total cost | **$4.709** |
| input tokens | 734,156 |
| output tokens | 85,146 |
| reasoning tokens | 52,910 (**62 %** of all output) |
| tool calls | 141 |
| messages | 133 (124 with reasoning) |

### 19.1 Cost by phase

| msgs | phase | msgs | cost | share | output tokens |
|---|---|---|---|---|---|
| 0–28 | model identification, llama.cpp sweep, Strata | 29 | $0.941 | 20.0 % | 10,257 |
| 29–44 | user intervention, `pause`, Chrome handoff, first write, compaction | 16 | $0.783 | 16.6 % | 15,779 |
| 45–97 | Reddit deep-dive, verification fetches, consolidation | 53 | $1.530 | 32.5 % | 44,376 |
| 98–103 | the "what should change in llama.cpp" plan | 6 | $0.372 | 7.9 % | 2,799 |
| 104–132 | ten user stories + verification | 29 | $1.083 | 23.0 % | 11,935 |

### 19.2 Three observations

**The single most valuable block of work was the cheapest per message.** Msgs 98–103 cost $0.372 for six messages and produced the *entire* §14 work plan — the tiered A/B/C analysis, the seven code-level items, the sequencing argument, the explicit non-goals. That is ~$0.06 per message against a session mean of $0.035, but it is the one part a maintainer would act on directly. Cost per message is a poor proxy for value here; what mattered was that ~50 issues of evidence had already been gathered.

**The two most expensive messages were both context-bound, not generation-bound:**

| msg | cost | input tokens | what happened |
|---|---|---|---|
| 32 | **$0.259** | 156,455 | first Chrome navigation — carried the entire accumulated 157 K-token context for one `navigate` call |
| 99 | **$0.256** | 143,299 | the tweak-plan question — read the full 297-line `research.md` back into context |
| 43 | $0.137 | 50,973 | the 16,632-char compaction summary |
| 72 | $0.103 | **1,412** | an `edit` attempt that **failed** |

Msg 72 is the instructive one: **$0.103 spent to change ~2 KB of context and produce a `Could not find oldString` error.** The `oldString`-mismatch failures that recur through the session (msgs 60, 65, 85, 89, 103, 105) are the direct cost of *reconstructing* a long document in the model's head instead of reading it back. The fix — read the file, then edit against the read text — is exactly what the session eventually adopted at msg 90–93 and what produced §16's correction. It is a concrete argument for file-grounded editing over remembered-string editing.

**Reasoning was 62 % of all output tokens but produced the two most important artifacts.** 52,910 reasoning tokens against 32,236 non-reasoning output. And the highest-value outputs were all *thinking* outputs: the compaction summary (msg 43), the A770→`qwen4exp` GDN join (msg 21), the tiering decision (msg 99), the `on-direct` fork detection (§9.1). The visible prose was the cheap part.

---

## 20. Consolidated open questions and what to do next

The session scattered its open questions across §7, §7a, the sanity bands, and individual `us_*.md` files. Consolidated, ranked by how much they block the user's actual goal:

### 20.1 Blocking — the configuration cannot be validated without hardware

| # | question | why blocking |
|---|---|---|
| 1 | **What does `qwen4exp` actually do on two B70s?** | Every number in this document is from a third party. The pre-registered band is 35–65 t/s decode, 600–1400 t/s prefill, ≤1.5× degradation at 32 K and ≤3× at 128 K. Nothing has been measured. |
| 2 | **SYCL vs Vulkan head-to-head on `qwen4exp` specifically** | No public B70 numbers exist for this arch on either backend. #28721 compares them on *other* models. |
| 3 | **Does the PLE table actually stream, and at what cost?** | The entire >100 GB-into-128 GB premise rests on `--lazy-mode auto` working well. Second-run page-cache-warm timing plus `iostat` is the test. |

### 20.2 Technically open — unanswered anywhere, including upstream

| # | question | origin |
|---|---|---|
| 4 | **Does the QSA indexer scan all KV positions, or only the top-k blocks?** | raised in msg 21's reasoning, never pursued, **never written into the deliverable**. Determines whether decode holds up at 262 K. |
| 5 | Is the MUL_MAT_ID `n_tokens > 8` MMV cutoff actually miscalibrated on Xe2, or is the 2–4× step a real property of the op? | #25356 was measured on gfx1151, not Battlemage |
| 6 | Does the 4× 3090 Q4_K_XL result of 55 t/s indicate routing/offload overhead that still exists post-#29476? | StyMaar's bandwidth arithmetic says 55 t/s *cannot* be explained by bandwidth alone |

### 20.3 Non-questions — resolved, and safe to stop revisiting

- Strata on Intel: **no.** CUDA/HIP-only, verified.
- `-sm tensor` on this rig: **no.** Three independent blockers.
- `--fit` for sizing: **no.** #27595 miscounts the compute buffer.
- Prebuilt Vulkan binaries: **no.** Pre-#29476 GDN penalty on 36 of 48 layers.
- Vision on Intel: **no.** #29093 (wrong answers) + #29241 (crash).
- BF16 MTP head: **no.** Bigger *and* slower.
- `ngram-mod` speculation on SYCL: **no.** 2.2 GB scratchpad (#28860).

### 20.4 Recommended sequence, in order

1. Build llama.cpp master for **SYCL** and a **Vulkan** A/B. Establish that `#29476` is in the build before measuring anything.
2. **Validate the premise, not the performance:** confirm the PLE table is lazy and non-resident from the placement log. If it is resident, stop — nothing else matters.
3. Run the short-context baseline on a **single** B70, no MTP, `-ub 512`, and check it lands in the pre-registered band. Below ~20 t/s, the diagnosis is configuration, not stack (§12.5).
4. Only then add: layer split → bias via `--tensor-split` → MTP (short context) → the context ladder.
5. Along the way, `us_29030.md` (lazy-PLE gather, **>2× prefill prototype already exists**) and `us_otgen.md` (router-aware placement) are the two highest-return upstream engagements — and both are *useful before the benchmark*, because the first changes the prefill baseline and the second changes expert residency.
6. Answer question 4 above with a direct trace. It is cheap, it is unanswered, and it governs whether the 262 K context figure is real.

### 20.5 What this document is, finally

A **plan with evidence behind it**, plus a reasoning trail, plus a compliance audit, plus an economic profile. The analysis is strong and every claim traces to a fetched source. The central performance question — *what this model does on two B70s* — is **unmeasured**, and the honest thing to hand a maintainer is that sentence, not a headline number.

---

## 21. Appendices

### 21.1 Source-thread index

Every Reddit thread the session opened, with where its findings landed. Six were located but **not read** and are recorded as such.

| thread id | title | read | findings in |
|---|---|---|---|
| `1w42biu` | MTP released for Qwen3.8-Flash-Next-GGUF | yes | §7, §9.4 |
| `1w03zdo` | llama.cpp Flash-Next merged-support discussion | yes | §7, §9.4 |
| `1weobt6` | Strix Halo `qwen4exp` optimisation journey (ilintar) | yes | §9.2 |
| `1wsodqf` | Any way to run Qwen 3.8 Flash with 7900XTX 24 GB + 64 GB RAM? | yes | §9.3 |
| `1w4xr6q` | Performance difference on large MoE runs, 64 GB vs 192 GB RAM? | **partial** (bot-walled) | §9.5, §12.1 |
| `1tuik6o` | Intel Arc Pro B70 llama.cpp benchmarks posted | yes | §7 |
| `1vulh45` | Intel Arc Pro B70 + vLLM XPU: 52 tok/s on Qwen3.8-27B INT4 | yes | §11 |
| `1sjlowl` | Best model to use with Arc Pro B70 | yes | §11.4 |
| `1wp7zyb` | Qwen3.8-Flash-Next on 12GB VRAM — 65 tokens per second | partial | §9.3 |
| `1vyq2v4` | (linked from `1wsodqf`) Flash-Next command-line thread | no | — |
| `1w0szzv` | Would Intel Arc B60 or B65 be worth it for Qwen 3.8 27B? | no (B60/B65, not B70) | — |
| `1wjzg4c` | Qwen3.8-Flash-Next NVFP4 262K ctx on a single DGX Spark | no | — |
| `1soe0nm` | B70 open-source Linux performance review | no | — |

**The material negative result:** a search for `Flash-Next Intel Arc` returned only 7 threads, none of them a second Intel-Arc + Flash-Next deployment. There was **no public precedent** for the exact target configuration.

### 21.2 Glossary

Model- and project-specific terms, since several (`qwen4exp`, `PLE`, `hc`) are not standard vocabulary.

| term | meaning |
|---|---|
| **qwen4exp** | the GGUF architecture string for Qwen3.8-Flash-Next; upstream PR #27742, merged 2026-08-26 |
| **GDN** | Gated DeltaNet — recurrent linear-attention layer; **36 of 48** layers here. Subject of the 2026-09-29 Vulkan fix #29476 |
| **QSA** | Qwen Sparse Attention — the **12 of 48** full-attention layers; 24 Q-heads / 2 KV-heads, head_dim 256, partial RoPE 64 |
| **indexer** | QSA's MQA selector (4 Q-heads / 1 K-head, dim 128) choosing top-2048 tokens from a 512-block budget. §20.2 q.4 asks whether it still scans all KVs |
| **PLE** | the model's parametrised-lookup / n-gram embedding table: ~320 M rows, 16 heads, ~28.8 GB at 4-bit, randomly row-addressed, layers=[1] |
| **hc / hyper-connections** | 4-branch gated residual, rank 320; needs special `hc` ggml ops per backend (#29132) |
| **MTP** | multi-token prediction — a trained extra layer used as a speculative draft; needs #28243 |
| **lazy mode** | `-lzm/--lazy-mode on\|auto\|off` — mmap-based on-demand tensor paging; the PLE mechanism |
| **`-ot`** | `--override-tensor`, regex→device placement (`blk\.\d+\.ffn_…_exps…=CPU`) |
| **VMM** | SYCL virtual memory manager; `pool_vmm` allocations are per-`sycl::context`, the root of #27198 |
| **MMV** | matrix-multiplication-vector — the non-matrix-core path; the `n_tokens > 8` cutoff in #25356 |
| **XMX** | Intel matrix-extension units; target of #29245's grouped-MoE GEMM |
| **PCH / southbridge** | the chipset path a PCIe 4.0 card may traverse; ~8 GB/s at x4 |

### 21.3 Reproduction: how to re-derive this

The session's evidence came from four sources, all cheap to re-run. The one non-obvious trick is Reddit.

**1. GitHub — issue/PR bodies and comments.** Pattern used throughout, no auth, 20–40 s timeouts:
```bash
curl -s -m 30 -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/ggml-org/llama.cpp/issues/28721
```
Search for *closed* issues, forks and PRs the same way. The three bodies that changed conclusions (#28860, #29093, #25612) were fetched this way in one batched `bash` call.

**2. Verify flags in source, not in docs.** This is what caught the fork-only `--lazy-mode on-direct`:
```bash
curl -s https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/arg.cpp \
  | grep -n "lazy-mode\|lazy_mode\|tensor-read-lazy"
```
Same for `-ot`, `-lzm`, `--fit`, `-cmoe`, `-ncmoe`, `-cram`, `--spec-type`, `-ngl`, `--prefetch-rows`.

**3. Hugging Face.** Repo API for tree/metadata, plus `README.md` and `MTP/README.md` as raw files. **The highest-value single step was parsing the GGUF header directly** — reading the first ~4 MB of the GGUF yields arch string, layer count, expert counts, `head_vocab_sizes` (which is where the ~320 M PLE row count comes from), and context length. No need to download 111 GB.

**4. Reddit — the part that needed a trick.** `reddit.com/*.json`, pullpush, Arctic Shift, DuckDuckGo and Bing were all tried and all failed (rate limits, 403s, `site:` filters ignored). What worked was **driving the user's real browser via the Chrome DevTools MCP** and reading the DOM directly, bypassing both the JSON API and React's client rendering. The extraction script used for every thread:
```js
(() => {
  const post = document.querySelector('shreddit-post');
  const title = post ? post.getAttribute('post-title') : document.title;
  const body = post ? (post.querySelector('[slot="text-body"]')?.innerText || '')
                     .replace(/\s+/g,' ').slice(0,2200) : '';
  const cs = Array.from(document.querySelectorAll('shreddit-comment')).slice(0,40);
  return JSON.stringify({ title, body, total: cs.length, out: cs.map(c => {
    const t = (c.querySelector('[slot="comment"]')?.innerText || '').replace(/\s+/g,' ').slice(0,600);
    return `d${c.getAttribute('depth')||'?'} [${c.getAttribute('author')||'?'}|${c.getAttribute('score')||'?'}] ${t}`;
  })}, null, 1);
})()
```
Key details that made it work: Reddit renders comments as `<shreddit-comment>` custom elements carrying `depth`/`author`/`score` as **attributes**, the post body lives in `shreddit-post > [slot="text-body"]`, and **always check the returned `title` against the URL you navigated to** — §17.4 shows the session twice believing it was on one thread while the DOM served another, and that check is what would have caught it.

### 21.4 Document map

| § | contents |
|---|---|
| 0 | executive summary + the §16 correction |
| 1 | the user's question and system |
| 2 | Phase 1 — model ground truth (HF, GGUF header, quants) |
| 3 | Phase 2 — llama.cpp mainline support state |
| 4 | Phase 2 cont. — deep dives on the specific problem reports |
| 5 | the stall, and the user's intervention |
| 6 | Phase 3 — Reddit via the user's Chrome |
| 7 | the two documents the agent wrote |
| 8 | Phase 4 — Reddit deep-dive, post-compaction |
| 9 | Phase 5 — forks, hyper-parameters, out-performing mainline |
| 10 | the *intended* shift to measurement (did not happen) |
| 11 | vLLM XPU on a single B70 — the closest reference |
| 12 | verification fetches, the finalised recommendation, sanity bands |
| 13 | the recommended launch configuration (never executed) |
| 14 | the pivot: what should be changed in llama.cpp |
| 15 | the ten upstream user stories |
| 16 | where the session ended + the correction |
| 17 | the reasoning trail — what drove the decisions |
| 18 | requirement-compliance audit (all 8 user turns) |
| 19 | the economics: 79 minutes, $4.71 |
| 20 | consolidated open questions and next steps |
| 21 | appendices: thread index, glossary, reproduction, map |
