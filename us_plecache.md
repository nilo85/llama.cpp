# us_plecache — PLE hot-row cache over lazy reads (new contribution, medium)

**Type:** New feature, mainline-compatible plumbing only (mmap/loader path, no ggml changes). Complements us_29030 (do that first; this rides its gather API).
**Priority (2026-09-30):** P3, after `us_29030` is tested/landed.
**Upstream refs:** builds on #29030/#29599 gather+prefetch; inspired by Strata's SSD-streamed engram via OS page cache (docs/DETAILS.md: "only randomly row-accessed, few rows/token"); no existing llama.cpp issue found — would be new (search first: `lazy row cache`, `ple cache`).

## Persona
Me: agent-style Flash-Next usage (Claude-Code-like loops, iterative prompts) where consecutive tokens/prompts repeatedly hit the *same* PLE trigram rows (code identifiers, repeated n-grams, session-vocabulary skew). Today each hit re-reads NVMe (or fights the OS page cache under RAM pressure — my RAM is mostly pledged to CPU-expert overflow anyway).

## Story
As a llama.cpp user with a lazily-streamed PLE table,
I want a small application-level LFU/LRU row cache (configurable MB, RAM- or VRAM-resident by backend) serving hot rows without touching the filesystem,
so agent loops and long sessions stop re-streaming the same few thousand rows.

## Rationale (evidence)
- Strata's measured win partly comes from row-locality OS-cache behavior; the explicitly-engineered equivalent (`--kv-resident`-style pattern) works — the model supports hot-set reuse (PLE heads with fixed vocab offsets ⇒ bounded hot set per domain/session; #29030's gather API gives the exact hook to serve a batch from cache).
- Our rig specifics make this more valuable than average: 64 GB RAM is contended by expert-overflow, so page-cache retention for PLE rows is unreliable precisely when sessions get long; explicit cache survives that pressure and competes better.
- Zero-arch-flag footprint (model-agnostic: any lazy tensor path, qwen4exp PLE and mmproj-style tables included) — matches upstream taste for small, composable loader features.

## Acceptance criteria
1. Design first comment upstream (new issue), referencing #29030 as dependency carrier; keep API surface tiny: `--ple-cache-mb N` (or generic `--lazy-cache-mb`), backend choice auto (RAM first).
2. Prototype on us_29030 branch: cache-hit path returns batched rows from a pinned host buffer; miss path falls through to direct-read gather. Bit-identical outputs (rows are immutable weights).
3. Bench on rig: 2nd-prompt repeat of 4K-token code loop — pp gain ≥25% at `-lzm on Q4_K_XL` over us_29030-alone baseline; establish hit-rate logging (`--stats` field).
4. RAM discipline: default off; bounded (evict by LFU + TTL-free); never deadlocks prefetch threads; no VRAM path v1 (RAM only → keeps it ggml-free).
5. Upstream flow: comment plan in #29030 thread; separate small PR after that merges (or coordinated single PR if author agrees); fallback: documented fork patch + runbook entry if rejected.

## Definition of done
Plan accepted into an upstream issue/PR with at least reviewer engagement, or fork patch proven on rig (criterion 3) and recorded in research.md.

## TODO (branch: us-plecache-hot-rows)
- [ ] Self-contained LFU/LRU row cache over the lazy-read path (loader, no ggml change)
- [ ] `--lazy-cache-mb N` arg; RAM-resident; default off; bounded eviction
- [ ] Build SYCL - verify compiles
- [ ] Test GPU 84:00.0: repeated-prompt pp gain vs no-cache; hit-rate logging
- [ ] Commit frequently; push to nilo85

## Work Log & Resume Context
_State: NOT STARTED. Update after each step (what was done, key decisions, how to verify/resume)._
