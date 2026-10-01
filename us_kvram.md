# us_kvram — KV overflow-to-RAM for deep context (upstream feature, deliberately deprioritized)

**Type:** New ggml-level capability; parked until a measured trigger appears.
**Priority (2026-09-30):** parked; trigger-based re-evaluation only.
**Upstream refs:** none open (confirmed absence in research §5 mapping: `-cram` prompt-cache only, no KV-resident/stream split); Strata analog `--kv-resident 32768` (+13.7 KB/tok stream from RAM; Q2_0 @262K: 50.9→62.6 t/s in their 12 GB-GPU case); adjacent merged machinery worth riding someday: #28953/#29459 buffer-placement plumbing, backend alloc-query hooks discussion in #25356/us_25356.

## Persona
Me at 256K+ context, where QSA KV outgrows leftover VRAM after weights. Not today's persona: at our target ladder (≤128K, q8_0 KV) the math says KV ≈1.6 GB — VRAM is not the constraint, expert residency is.

## Story
As a llama.cpp user pushing 128K–1M ctx on VRAM-limited GPUs,
I want a KV split policy — recent/hot window VRAM-resident, remainder streamed from pinned host RAM per attention op — rather than all-KV-in-VRAM or fail,
so context length decouples from leftover VRAM.

## Rationale (evidence) + parking rationale
- Strata measured concrete wins (cited above) in exactly the "attention-heavy tail" regime; #26581's latency-bound attention (21–25 ns/pos/layer, streams scale ~2.4x) means a streamed-KV decode path can hide latency across streams reasonably well on Xe2 — the physics checks out.
- BUT: (a) our q8_0-KV budget leaves ~10s of GB VRAM headroom at 128K (compute-buffer trap already documented separately, §6.2); (b) ggml change surface is large (op-level in/out placement, backend buffer allocator semantics); (c) maintainer attention is on FA-decode (#28721/us_28721) and grouped-MoE (#29245/us_29245), which dominate our tg error budget first.
- Correct sequencing: prove the need with the §8 ladder's 128K rung before drafting anything.

## Acceptance criteria (trigger-based, re-evaluate when met)
1. Trigger: llama-bench/integrity ladder at 256K target shows OOM or ≥10% tg regression attributable to KV placement (not attention kernel cost of us_28721) while ≥8 GB RAM is uncommitted.
2. If triggered: design doc upstream first (issue, not PR) — sketch Strata-parity semantics (`--kv-resident N`), ggml buffer-allocator interaction, backend requirements (host-visible async copies; Vulkan + SYCL first, CPU fallback exists trivially), bit-exactness expectations (streaming is a placement change, not math change).
3. Prototype gate: single-backend (SYCL first, its deep-ctx lead per #28721 makes the win smaller but the test cleaner on our rig) demonstrating ≥15% tg at 256K vs OOM/fit-mangled baseline, with `-fa on` parity.
4. Upstream flow: coordinate with #29459/#28953 buffer-placement authors; explicitly NOT part of the first local-config runbook (avoid destabilizing a working layer-split setup).

## Definition of done
Either trigger never fires (document the negative result in research.md and close this story), or upstream design thread opened with bench data attached. No fork.

## TODO (branch: us-kvram-overflow)
- [ ] Parked story: design doc (docs/) for KV split policy (recent VRAM-resident, rest RAM-streamed)
- [ ] Minimal API surface: `--kv-ram N` parsed + validated + logged (not yet implemented)
- [ ] Build SYCL - verify compiles
- [ ] Test GPU 84:00.0: confirm flag parses; document that trigger (128K OOM) not yet met
- [ ] Commit frequently; push to nilo85

## Work Log & Resume Context
_State: NOT STARTED. Update after each step (what was done, key decisions, how to verify/resume)._
