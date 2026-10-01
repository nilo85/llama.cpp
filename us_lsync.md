# us_lsync — SYCL layer-mode cross-device handoff latency (local profiling)

**Type:** Local profiling; upstream only if a real serialization penalty is confirmed.
**Priority (2026-10-01):** P2, CLOSED as inherent (not a bug). Re-tested with the correct framing after an earlier misframe.
**Status:** The two-GPU power flip is REAL and confirmed, but it is inherent to layer block-split (hard data dependency: GPU1's block needs GPU0's output each forward pass; llama.cpp batches all sequences per layer, so no cross-sequence overlap). The copy-path env variants do NOT recover meaningful throughput (the first +6.7% prefill was a cold-run outlier; warm repeats show env0~env1). No small fix exists; removing the handoff dead-time would need cross-ubatch pipelining (major change, out of scope). The flip does NOT hurt concurrent throughput: decode is memory-bound, so `-np 4` still gives ~4x aggregate t/s.
**Upstream refs:** related to #27198/#29459 (`dev2dev_memcpy`/VMM), but this story is about layer-mode GPU overlap, not the tensor-split `DEVICE_LOST` crash.

## Persona
Me: running Qwen3.8-Flash-Next on 2x B70 with `--split-mode layer --tensor-split 50,50`. The Q2/Q3 PLE models only fit when split across both GPUs. When I watch `nvtop` during a run, the two GPUs' power/utilization FLIP between each other (one ramps up while the other drops), even when I hammer with concurrent load. That looks like the two GPU blocks are not overlapping - they take turns - which smells like a synchronization problem at the layer-block boundary.

## Story
As a llama.cpp+SYCL user on a dual-Intel box,
I want to know whether the layer-mode cross-device handoff forces the two GPUs to take turns (power flip) instead of overlapping,
so I can decide whether the synchronous `dev2dev_memcpy` + full-queue-drain path is a real serialization penalty worth fixing upstream.

## Rationale (evidence)
- `ggml_backend_sycl_buffer_cpy_tensor` waits on all queues of both devices, then calls `dev2dev_memcpy` (`ggml/src/ggml-sycl/ggml-sycl.cpp:827-848`).
- `dev2dev_memcpy` default SYCL path also `.wait()`s on the copy; the L0 path is immediate but still runs after the queue waits; the host-forward path is two waited copies.
- `--split-mode layer --tensor-split 50,50` is a block split (`src/llama-model.cpp:1586`), so the main hidden state crosses GPUs once per forward pass, but that copy can still serialize the two devices.
- Lazy PLE adds host-to-GPU uploads per layer; that can look like sync pressure but is a different path.

## Acceptance criteria
1. Sample per-GPU power + util at fine resolution (`nvtop -l -d 1`, 0.1s) during a sustained Flash-Next Q2/Q3 layer `50,50` run; capture the time series for both B70s.
2. Classify the pattern:
   - clean flip (GPU0 high while GPU1 low, then vice versa, no gap) = inherent layer-split serialization;
   - flip with dead-time gaps (both GPUs drop low at the handoff) = sync/copy penalty;
   - both GPUs high concurrently = overlap works, no issue.
3. Compare `GGML_SYCL_DEV2DEV_MEMCPY=0/1/2` and see whether any variant reduces the dead-time gap or increases concurrent overlap.
4. Compare layer `50,50` vs tensor `50,50` vs single-GPU under the same load: tensor split keeps both GPUs busy every layer (control for "should both be busy"), single-GPU is the no-split baseline.
5. Confirm a fixable issue only if there is measurable dead-time (both GPUs idle at handoff) that an async/double-buffered copy could remove.
6. If the flip is purely inherent to batched layer-by-layer processing (no dead-time), document that and close.

## Definition of done
Either:
- a confirmed serialization penalty (dead-time at handoff) with numbers plus a proposed fix/issue, or
- a documented finding that the flip is inherent to layer split (no fixable dead-time) on this rig.

## TODO
- [x] Create story
- [x] (first pass) single-stream tg + dev2dev env variants + Qwen3.6 control  -> tested the WRONG thing (speedup, not flip)
- [x] Re-open with correct framing (power flip under load)
- [x] Stop 27B service, verify GPUs free
- [x] Build nvtop power/util time-series logger (0.1s, both B70s)
- [x] Run Flash-Next Q2 layer 50,50 sustained prefill+decode + sample power trace
- [x] Classify flip pattern (clean flip / dead-time gap / concurrent)
- [x] Compare dev2dev env variants (0/1/2) for gap reduction
- [x] Compare np1 vs np4 (concurrency) for GPU overlap
- [x] Confirm prefill env-variant gain is noise via warm repeats (Q2/Q3)
- [x] Decide: fixable serialization penalty vs inherent layer-split behavior
- [x] Restart 27B service, verify health
- [x] Update story/research/host-info (no code change, so no branch push needed)

## Work Log & Resume Context
_State: CLOSED as inherent. The power flip is real and confirmed by fine-grained per-GPU sampling, but it is the expected consequence of layer block-split (data dependency + llama.cpp's per-layer batching), not a fixable sync bug. The copy-path env variant does not recover throughput; concurrency throughput is unaffected._

### 2026-10-01 - story created
Layer mode is not the tensor-split P2P crash path, but it can still call the same SYCL cross-device copy helper. The helper currently does full queue waits plus a synchronous `dev2dev_memcpy`. Need numbers before deciding whether this is worth chasing upstream.

### 2026-10-01 - profiling results (first pass, single-stream tg)
- Qwen3.8-Flash-Next Q2/Q3, combo build, layer `50,50`, warm 512-token `--ignore-eos` runs:
  - Q2: env0 `27.5 t/s`, env1 `28.2 t/s`, env2 `28.2 t/s`
  - Q3: env0 `24.5 t/s`, env1 `24.3 t/s`, env2 `24.2 t/s`
  - `GGML_SYCL_DEV2DEV_MEMCPY=0/1/2` changes tg by only a few percent and not consistently in one direction.
- `GGML_SYCL_DEBUG=1` on Q2 shows `1030` `ggml_backend_sycl_buffer_cpy_tensor` calls over 512 tokens (~2 per token). The decode copies are small (~40 KB each), so total cross-device traffic is not bandwidth-bound.
- Qwen3.6-35B-A3B Q4_K_XL control, same combo build:
  - single-GPU: pp512 `1377.8 t/s`, tg128 `83.07 t/s`
  - both-GPU layer `50,50`: pp512 `1377.5`/`1318.9 t/s`, tg128 `83.24`/`83.14 t/s`
  - adding the second GPU in layer mode does not produce a single-stream tg speedup.
- NOTE: these single-stream tg numbers are still valid but do NOT address the user's hypothesis (power flip under load). They only show layer split doesn't speed up single-stream tg, which is expected for a block split.

### 2026-10-01 - re-framing after user correction
User clarified the actual observation: running a large Flash-Next model, the two GPUs' power usage FLIPS between GPU0 and GPU1 even when hammering concurrently. That is a serialization signature, not a speedup question. The first-pass "abandon" was premature because it measured the wrong thing. Re-open the track.

Tooling: `nvtop` 3.3.2 is installed with Intel support (`-s` snapshot / `-l` loop, JSON output with `power_draw` and `gpu_util` per device). Needs root (or `cap_perfmon`) for accurate readings. `intel_gpu_top` does NOT work for the Arc B70s (they use the `xe` driver, not `i915` PMU) - it only sees the Meteorlake/Arrow Lake iGPU. Device map: `card0`/`renderD129` = B70 #0, `card2`/`renderD130` = B70 #1, `card1`/`renderD128` = iGPU.

### 2026-10-01 - power-flip measurement (correct test)
Logger: `sudo nvtop -l -d 1` (0.1s) piped to a JSON-array parser writing per-GPU `power_draw`/`gpu_util` CSV. Workload: combo build, Qwen3.8-Flash-Next, layer `50,50`, `-c 8192`, big prompt (~4859 tok) prefill + decode. Idle baseline ~50W/43W, util 0.

PREFILL (np1, env0) at full resolution: unmistakable flip. GPU0 runs ~99% util / ~235-247W for ~0.3-0.5s, then GPU1 runs ~99% / ~230-240W, alternating, one GPU idle while the other works. Phase-split stats (first 10s of compute window):
- env0: both-idle 9.0%, gpu0-only 39.3%, gpu1-only 24.7% (flip = 64.0%)
- env1 (L0 async): both-idle 7.5%, flip 65.6%
- env2 (host-forward): both-idle 8.6%, flip 66.7%
So ~64-67% of prefill time exactly ONE GPU computes (inherent block-split serialization), ~8% both-idle (handoff dead-time).

DECODE (np1): per-token flip is faster than the 0.1s window, so both GPUs show partial util simultaneously (g0 ~70%, g1 ~47%, sum ~117% < 200% => no true overlap, just averaged alternation).

CONCURRENCY (np4, short prompt, 512 tok/seq): per-sequence decode ~28 t/s, aggregate ~4x112 t/s vs np1 ~27 t/s. Both GPUs show partial util together (g0 ~70%, g1 ~50%). Throughput scales ~linearly with batch despite the flip, because decode is memory-bound (weights read once per step, reused across the 4 batched tokens).

### 2026-10-01 - env-variant prefill gain is an OUTLIER (warm repeats)
First single run showed env1 prefill 534.4 vs env0 500.9 (+6.7%), but that env0 number was a cold-run outlier. Alternating warm repeats (Q2 and Q3, `-n 16`, pp only):
- Q2 env0 {532.6, 530.1, 530.1} vs env1 {536.6, 537.6, 536.4} -> ~1.1% (marginal, within run-to-run)
- Q3 env0 {464.9, 464.0, 484.2} vs env1 {472.5, 467.8, 472.1} -> ~0% (noise)
So the copy-path mechanism (SYCL sync vs L0 async vs host-forward) does NOT meaningfully change prefill throughput. The ~8% both-idle handoff dead-time is not recoverable via the copy path.

### 2026-10-01 - decision: inherent, not a fixable bug
- The power flip is REAL and confirmed, but it is the expected behavior of layer block-split: GPU1's layer block has a hard data dependency on GPU0's block output within each forward pass, and llama.cpp batches all sequences per layer (one forward pass covers the whole batch), so the two GPU blocks cannot overlap. Concurrency does not change this.
- The only "extra" cost is the ~8% handoff dead-time, which the copy-path env variants do NOT recover (repeats show env0~env1). Eliminating it would require cross-ubatch / cross-sequence pipelining (start GPU0 on the next ubatch while GPU1 finishes the current one) - a large architectural change that conflicts with llama.cpp's simplicity ethos and is out of scope for this track.
- Practical takeaway for the user: the flip is cosmetic, not a throughput killer. Layer split gives model CAPACITY (the PLE Q2/Q3 only fit when split), and concurrent decode still scales throughput ~linearly with batch size. Do NOT expect single-stream latency speedup from the second GPU, but do expect good aggregate throughput under concurrency.
- No code change, so no branch push to nilo85 for this story.
