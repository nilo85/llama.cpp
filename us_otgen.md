# us_otgen — Router-aware `-ot` pattern generator (offline tool + small core hook)

**Type:** Most is out-of-tree tooling (zero fork risk); small optional core addition to make traces first-class.
**Priority (2026-10-01):** ACTIVE next local item (A stories done/blocked; bench ladder run). Design agreed 2026-10-01; see Work Log.
**Upstream refs:** consumes merged `--override-tensor/-ot` (patterns incl. `blk\.\d+\.ffn_(gate|up|down)_exps\:`, `\.exps\.` variants — arg.cpp master-verified); Strata's `--dump-routing` profile concept (repo/tool-local, their MIT code is referenceable); related open material: ik_llama.cpp `adjust_device_tensors` (fork precedent showing appetite for placement tooling).

## Persona
Me: qwen4exp on mixed-speed GPUs (PCIe5 + PCIe4-PCH B70 pair) + 64 GB RAM. `-ot` is powerful but blind: nothing tells me *which* of 24,576 experts actually run hot for *my* workload (agent coding vs chat), so today placement guesses statically (layer-uniform splits) and I can't distinguish "cold experts wasting VRAM" from "hot ones thrashing CPU".

## Story
As a llama.cpp tuner,
I want to run a short representative-workload pass, capture per-layer expert routing counts, and get an emitted `-ot` file (regex patterns per layer, CPU/GPU0/GPU1 targets) that pins hot experts to VRAM proportionally to each device's bandwidth,
so calibration replaces guesswork and adapts to the workload class the way Strata's adaptive cache does — without any adaptive machinery at runtime.

## Rationale (evidence)
- Strata docs/DETAILS.md: profile rows rank all experts from routing traces, cache-fit maximizes overlap — their headline win source; static-but-calibrated `-ot` captures most of it (hot sets are stable across sessions per their findings; warmup phase noted in issues).
- Accomplished-Air439 (reddit): agent-harness-driven offload search took pp 100→400 t/s on 3×5060Ti — evidence users already do poor-man's versions of this; a first-class tool is evidently wanted.
- Our rig math: GPU#1 may sit at x4 PCIe4 (~8 GB/s) — layer-count-balanced split wastes its capacity vs bandwidth-proportional expert residency; expert weights >> activations for per-token traffic (us_27198 corollary).

## Acceptance criteria
1. Trace source, pick the lowest-risk path: (a) propose an upstream `--dump-routing FILE` trace hook for MoE models, or (b) locally patch our test build with trace logging (respect router-layer graph node reads) — whichever ships first; feeds criterion 2.
2. Generator script (out-of-tree, in this workspace): parses traces, per-layer aggregates, solves simple capacity knapsack (VRAM budget split ∝ measured per-device BW via quick llama-bench), emits `-ot` pattern list + expected-ROI summary.
3. Validate on rig: generated placement ≥10% tg over layer-balanced default at same quant/ctx (llama-bench + real agent prompt mix), or document why not (e.g., placement already saturated — still a data point for research.md).
4. Upstream hygiene: only the trace hook (criterion 1a) goes upstream as feature-request issue with our tool as the citation; everything else stays local unless the community asks for it in-tree.

## Definition of done
Tool exists in this workspace with a `--help` matching the emitted-format spec, run against both backends, results (Δpp/Δtg/VRAM maps) appended to research.md §6.8 + a new §10.5 tooling note; upstream issue filed for trace hook.

## Branch strategy (agreed 2026-10-01)
- **Scoped dev branch:** `us-otgen-router-ot` from **master** (`19e28a277`) — the `--dump-routing` core hook lives here as a small, self-contained, reviewable change (the only in-tree part; the piece that could go upstream).
- **Validation combo:** new 4-way `us-otgen-29030-29245-28243` = merge `us-otgen-router-ot` into the existing 3-way combo (`us-29030-29245-28243`). The hook keys on the tensor *name* `ffn_moe_topk`, which is stable across master and the combo, so the merge is expected to be clean.
- **Model:** Qwen3.8-Flash-Next (cached, qwen4exp, 24,576 experts) — not the Qwen3.6-35B-A3B the original TODO named.
- **Out-of-tree generator** lives in this `b70_opt` workspace (zero fork risk), never in the llama.cpp branch.

## TODO (branch: us-otgen-router-ot)
- [x] Locate MoE router/top-k in graph: `src/llama-graph.cpp:2167` `selected_experts = ggml_argsort_top_k(selection_probs, n_expert_used)` (i32 `[n_expert_used, n_tokens]`), named `ffn_moe_topk` at `:2170`; seen by `graph_get_cb` (`src/llama-context.cpp:2609`).
- [x] Core `--dump-routing` hook implemented, validated, pushed to `nilo85/us-otgen-router-ot`.

## TODO (per-expert placement core, branch: TBD)
- [x] Create/switch llama.cpp branch for per-expert core work (`us-otgen-expert-ot` from `us-otgen-router-ot`).
- [x] M1: env-driven K-way expert split for Qwen3.8-Flash-Next; validate greedy parity + perf on Q2_K_XL. Q3_K_XL K=8 proves capacity; uniform K=3 is not the right capacity path on this rig.
- [x] M2: `-ot` override plumbing for synthetic `.partN` expert tensors. Arbitrary per-layer contiguous ranges are still future work, but generated placement can now use existing `-ot` patterns.
- [x] M3: out-of-tree trace-to-placement generator that emits `-ot` patterns from `--dump-routing` traces.
- [ ] M4: stronger validation + research.md update; push branches. First Q3 generated-placement win is recorded, but needs a better baseline and trace from Q3 itself.
- [ ] M5 (local-only placement): change generator policy so a layer's hot parts go to its owning GPU and everything else goes to CPU (never the remote GPU); A/B vs current remote-spill. This is the in-line next step, not a detour.
- [ ] M6 (DEFERRED, low priority): global expert-index tiering (hottest experts' all-layer weights on GPU1, moderate GPU2, cold CPU). Checked 2026-10-01 against our traces: per-layer top-set overlap is ~chance (1.06-1.19x), so hotness is per-layer-scattered, not index-consistent; the mild global skew (top-10% holds ~21% mass) can't be placed without fighting the layer split. Revisit only if a future model shows index-consistent hotness or we drop the layer split.
- [x] M7 (workload/persona-adaptive traces) VALIDATED 2026-10-01: 3 prompts x 4 personas (coder/reporter/toolcaller/chat), Q3, medium prompts, single-GPU. Per-layer top-set: within-persona Jaccard `0.470` (9.0x chance) vs between-persona `0.152` (2.9x) at top-10%; `0.568` (4.0x) vs `0.302` (2.1x) at top-25%. Personas are a real, robust placement axis (not one-prompt noise). Within-persona union ~1.6x K (stable); cross-persona union ~4.2x K at top-10% (~42% of a layer's experts) and ~2.8x at top-25% (~69%, at/over the Q3 capacity wall). Toolcaller traces thinner (short JSON, ~20-35 steps) -> noisier, but contrast holds.
- [x] M8 (decided 2026-10-01): user's real mix is tool-caller + coder + reasoning (not reporter/chat). 3-persona cross-union = 179 experts (35% of a layer) at top-10%, 321 (63%) at top-25%. 35% is under the 50% that already loaded on Q3 (K=8 hot-50), so a single union-default file fits at a tight hot-fraction; 63% is over the wall. Plan: ship a union-default (3 personas) + per-persona files, switched per session (static-but-calibrated "dynamic"; runtime migration stays out of scope). Caveat: scattered union-hot vs contiguous parts -> higher K for cleaner fit (K-vs-overhead knob).
- [ ] M9 (build): emit the union-default `-ot` from the 3-persona traces using the M5 local-only policy; A/B vs uniform K=8 and vs a single-persona file. Tune K + hot-fraction for the fit/overhead tradeoff.
- [x] Add `--dump-routing FILE`: arg.cpp (next to `-ot` :2752) + common.h `common_params` + common.cpp cparams plumbing + `include/llama.h` (public `const char *`) + `src/llama-cparams.h` (internal `std::string`).
- [x] Core hook: in `graph_get_cb`, capture `ffn_moe_topk` tensors when the flag is set (mutable vector, re-captured on graph rebuild); after `graph_compute` (`:1456`) `routing_trace_flush()` reads back via `ggml_backend_tensor_get`, counts per-layer expert hits, appends one line per MoE layer per step to FILE.
- [x] Build SYCL on `us-otgen-router-ot` - verify compiles + emits a trace.
- [ ] Out-of-tree generator (in b70_opt): parse trace, per-layer expert hotness, capacity knapsack (VRAM split ∝ per-device BW), emit `-ot` patterns + expected-ROI.
- [ ] 4-way combo: merge `us-otgen-router-ot` into `us-29030-29245-28243`, build SYCL.
- [ ] Validate on rig (Qwen3.8-Flash-Next): run trace, generate `-ot`, measure tg vs layer-balanced default (target >=10%, else document why not).
- [ ] Commit frequently; push both branches to nilo85.
- [ ] Upstream: file feature-request issue for the trace hook only; generator stays local.

## Per-expert placement design (full CPU+GPU, vendor-agnostic)

**Decision (2026-10-01):** pursue full CPU+GPU expert placement as a local, vendor-agnostic llama.cpp extension. Goal: a reference implementation that maintainers can take code/inspiration from, even if it is not merged as-is.

**Why not the existing mechanisms:**
- `-ot` selects one buffer type per whole tensor; it cannot address individual experts inside a fused expert tensor.
- SYCL/CUDA split buffers split a tensor across devices of the *same* backend, and the SYCL `mul_mat_id` kernel asserts that split buffers are unsupported.
- `LLAMA_SPLIT_MODE_TENSOR` is disabled for `QWEN4EXP` (`llm_arch_supports_sm_tensor` returns false), and even where enabled it splits hidden/intermediate axes, not the expert axis.

**Chosen approach: static expert partitioning + partial MoE subgraphs**
1. **Loader side:** when an expert placement override is present for a MoE expert tensor (`ffn_down_exps.weight`, `ffn_gate_up_exps.weight`, etc.), split the fused tensor into K contiguous parts along the expert axis (axis 2). Each part is a separate `ggml_tensor` with its own buffer type (CPU, SYCL0, SYCL1, ...). The parts are registered as synthetic weights that are byte-slices of the original GGUF tensor, so `load_all_data()` can mmap/copy each part independently.
2. **Graph side:** `build_moe_ffn()` detects that a layer's expert tensors are partitioned. For each part it builds a complete MoE FFN subgraph (gate/up -> activation -> down) using:
   - the part's expert tensors,
   - a remapped `ids_part` that maps global expert indices in the part's range to local indices and maps all other experts to a dummy index,
   - a masked `weights_part` (routing weights zeroed for experts outside the part's range) so dummy/remote rows contribute exactly zero.
   The per-part outputs are summed to produce the layer's `moe_out`. Because the FFN is applied per selected-expert row and remote rows are zeroed, summing the per-part final outputs is mathematically identical to the unpartitioned graph.
3. **Vendor agnosticism:** no backend kernel changes. Each part is a normal tensor on a normal backend buffer; the existing scheduler handles cross-backend adds. The same design works for CPU, SYCL, CUDA, or any other backend combination.

**Warm/cold compatibility:** the uniform `LLAMA_EXPERT_SPLIT=K` path is only an M1 validation scaffold. The long-term goal remains routing-calibrated warm/cold placement (hot experts on GPU, cold experts on CPU). The loader/graph substrate is range-based: each part has a name, byte offset, expert offset, and buffer type. A later placement file can therefore replace the uniform K loop with arbitrary per-layer contiguous ranges and devices. Scattered hot experts can be emitted as multiple small contiguous ranges; if part-count overhead becomes a problem, we can add gathered parts or expert reordering, but the current contiguous-range mechanism is not a dead end for warm/cold placement.

**Planned milestones:**
- M1: hardcoded 2-way expert split (GPU0/GPU1) for Qwen3.8-Flash-Next to prove correctness/perf.
- M2: CLI/file plumbing for arbitrary per-layer expert ranges and backends.
- M3: out-of-tree trace-to-placement generator that emits the placement file.
- M4: validation + research.md update.

**Risks:**
- Loader partitioning must handle mmap, lazy read, scales/biases, and the gate_up fused layout correctly.
- The graph transform duplicates the MoE FFN subgraph K times, increasing graph size and scheduler overhead.
- Cross-backend adds for the per-part outputs may add latency; the win depends on hot-expert residency.

## Work Log & Resume Context
_State: PER-EXPERT M1, `-ot` PART-OVERRIDE HOOK, AND TRACE-TO-`-ot` GENERATOR DONE ON `us-otgen-expert-ot` 2026-10-01. Q2_K_XL K=2 passes parity/perf; Q3_K_XL K=8 capacity passes; Q3-trace K=8 hot-50 `-ot` placement beats uniform K=8 (`9.4 pp / 4.2 tg` vs `5.8 pp / 3.4 tg`). Core pushed to `nilo85` at `d49a538df`. Global expert-index tiering set aside as ghost (M6 deferred). Persona/workload traces VALIDATED as a real, robust axis (M7: within-persona 9x chance vs between 2.9x). Next: M8 decision (per-persona `-ot` files vs union) + M5 local-only placement + A/B._

### 2026-10-01 — Design + branch strategy (agreed with user)
- **Goal:** replace static layer-uniform `-ot` placement with workload-calibrated expert placement. Trace per-layer MoE routing over a representative pass, rank expert hotness, solve a VRAM-budget knapsack split proportional to per-device bandwidth, emit an `-ot` pattern file.
- **Trace source (core hook):** the MoE router's top-k selection is `selected_experts` (`src/llama-graph.cpp:2167`, `ggml_argsort_top_k`), named `ffn_moe_topk` (`:2170`). It is i32 `[n_expert_used, n_tokens]` (each column = a token's chosen experts). The `graph_get_cb` callback (`src/llama-context.cpp:2609`) already receives every named tensor as `(ubatch, tensor, name, il)`, so the hook keys on `name == "ffn_moe_topk"` — model-agnostic (works for any arch that calls `build_moe_ffn`).
- **Read-back:** after `graph_compute` (`:2599`, async), read each captured tensor with `ggml_backend_tensor_get(backend, tensor, ptr, 0, nbytes)` (same API as the logits path `:2674`); backend comes from `tensor->view.buffer`. Aggregate per-layer expert hit-counts, append one line per MoE layer per eval step to FILE, clear.
- **Trace format (v1):** one line per MoE layer per eval step: `il=<layer> <expert>:<count> ...` (only experts that were hit). Small, grep-friendly, feeds the generator.
- **Branch strategy (user):** keep the core hook a **scoped, master-based** change on `us-otgen-router-ot` (easy to review, upstream-able); validate in a **new 4-way combo** `us-otgen-29030-29245-28243` (merge into the existing 3-way combo). Generator stays out-of-tree in `b70_opt`.
- **Model:** Qwen3.8-Flash-Next (cached) per user.
- **Plumbing (confirmed 2026-10-01):** two-struct split. Public `llama_context_params` (`include/llama.h:365`, C-compatible) gets `const char * dump_routing`; internal `llama_cparams` (`src/llama-cparams.h:10`, C++) gets `std::string dump_routing`. Files: `include/llama.h` (field + default `llama_context_default_params`), `src/llama-cparams.h` (field), `src/llama-context.cpp` (normalize ~`:141` copies `params.dump_routing` into `cparams`; `graph_get_cb` `:2609` captures `ffn_moe_topk`; new `routing_trace_flush()` called after `graph_compute` `:1456`), `src/llama-context.h` (members: `std::ofstream routing_trace`, `mutable std::vector<std::pair<int,ggml_tensor*>> routing_captures`, method decl), `common/common.h` (`common_params.dump_routing`), `common/common.cpp:1657` (`common_context_params_to_llama` sets `.c_str()`), `common/arg.cpp:2756` (`--dump-routing` handler).

### 2026-10-01 — Hook implemented (branch `us-otgen-router-ot`, at master `19e28a277`)
- **Implemented** the full `--dump-routing FILE` hook across 7 files (75 insertions, 0 deletions): `include/llama.h` (+`const char * dump_routing`), `src/llama-cparams.h` (+`std::string dump_routing`, `#include <string>`), `src/llama-context.h` (+`<fstream>`/`<utility>`, `routing_trace` ofstream, `mutable routing_captures`, `routing_trace_flush()` decl), `src/llama-context.cpp` (normalize copy, default-params `nullptr`, `graph_get_cb` capture on `name=="ffn_moe_topk"`, `routing_captures.clear()` on graph reset `:1421`, `routing_trace_flush()` after compute `:1465` + full impl), `common/common.h`/`common.cpp`/`arg.cpp` (CLI flag `--dump-routing`/`LLAMA_ARG_DUMP_ROUTING`).
- **Key correctness point discovered:** the graph is REUSED across decode steps (`process_ubatch` `:1407` `res->can_reuse`), so `model.build_graph` (and thus `graph_get_cb`) runs only on rebuild, but the `ffn_moe_topk` node values are recomputed every step. Hence: capture the node pointer once at build (mutable, since `graph_get_cb()` is const), clear captures on `res->reset()` (old nodes go stale), and read back the CURRENT node values every step in `routing_trace_flush()` after `graph_compute`.
- **Read-back:** 4-arg `ggml_backend_tensor_get(tensor, ptr, 0, nbytes)` (convenience overload, `ggml-backend.cpp:350`; derives backend from the tensor) — matches the codebase's own read-back pattern (`llama-context.cpp:2774`). Sync get blocks until the async compute lands, so values are current.
- **Compile check (fast):** `icpx -fsyntax-only` on all 3 modified `.cpp` (`llama-context.cpp`, `common.cpp`, `arg.cpp`) inside `oneapi-toolkit:2026.1.1-devel` with mounted source — all EXIT=0. Full SYCL image build still pending (long pole).
- **Next:** full SYCL build of `us-otgen-router-ot` -> run `llama-cli --dump-routing` on Qwen3.8-Flash-Next to confirm a trace is emitted -> then out-of-tree generator.

### 2026-10-01 — Trace hook validated; two read-back bugs fixed
- **Podman SYCL visibility:** the `otgen-fresh` container needed the same GPU access pattern as the production 27B service: `-e ZES_ENABLE_SYSMAN=1` and `--device=/dev/dri` (or a specific `/dev/dri/renderD*` node). With that, the test container saw the B70 GPUs.
- **Non-interactive CLI:** this `llama-cli` build uses `-st --no-display-prompt`; `-no-cnv` is not a valid flag.
- **First trace was wrong:** `--dump-routing` emitted lines, but the expert IDs were huge and, reinterpreted as float32 bit patterns, looked like routing probabilities.
- **Fix 1 (strided view read):** `ffn_moe_topk` is a non-contiguous `ggml_view_4d` over the full `ffn_moe_argsort` tensor. The original flush read one raw `ggml_nbytes` block, which is the stride bounding box, not the logical top-k rows. `routing_trace_flush()` now reads each logical row with `ggml_backend_tensor_get(t, buf, i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3], n0*sizeof(int32_t))`.
- **Fix 2 (tensor lifetime):** even with strided reads, the values were still stale because the `ffn_moe_argsort` buffer is an intermediate tensor that the allocator can free/reuse before `routing_trace_flush()` runs after `graph_compute`. The hook now appends a duplicate `ffn_moe_topk` VIEW node to the built graph when `--dump-routing` is set. The VIEW op is a no-op, but its presence at the end of the graph increments the allocator's child/view use-counts for `ffn_moe_argsort`, keeping the argsort buffer valid until after compute.
- **Valid trace:** rebuilt `localhost/llama.cpp:otgen-fresh`, stopped `podman-llama-cpp-qwen3.8-27b.service`, ran the Q3_K_XL trace test, restarted the 27B service. Output: 48 MoE layers, 3264 `il=` lines, expert IDs `0..511`, no out-of-range IDs. This confirms Qwen3.8-Flash-Next has 512 routed experts per layer (48 x 512 = 24,576 total) and `n_expert_used=10` for this run.
- **`-ot` scope discovery:** current `-ot/--override-tensor` maps tensor-name regexes to backend buffer types for whole tensors (e.g. `blk.N.ffn_(gate|up|down)_exps.weight`). It does not select individual expert slices inside a fused expert tensor. The generator must therefore either (a) emit per-layer whole-expert-tensor placement, or (b) require a new per-expert tensor/layout/core mechanism for true per-expert pinning.
- **Next:** decide generator scope under current `-ot` semantics; if per-expert pinning is required, identify the smallest core/GGUF change that exposes per-expert tensor names or slice overrides.

### 2026-10-01 — Per-expert placement direction chosen
- User chose **full CPU+GPU expert placement** with a strong preference for a **vendor-agnostic** implementation, even if it remains a local fork / reference for maintainers.
- Investigated existing mechanisms:
  - `-ot` is whole-tensor only.
  - SYCL `mul_mat_id` asserts `!ggml_backend_buffer_is_sycl_split(src0->buffer)`; same-backend split buffers are not a path to per-expert placement.
  - `LLAMA_SPLIT_MODE_TENSOR` is disabled for `QWEN4EXP` and splits hidden axes, not the expert axis.
- Drafted the "static expert partitioning + partial MoE subgraphs" design (see section above). It avoids backend kernel changes by partitioning the fused expert tensor into separate tensors at load time and building one masked MoE subgraph per partition in `build_moe_ffn()`.
- **Next:** create a llama.cpp branch for the per-expert core work and start with a hardcoded 2-way expert split prototype on Qwen3.8-Flash-Next.

### 2026-10-01 — Per-expert M1 code draft on `us-otgen-expert-ot`
- Created branch `us-otgen-expert-ot` from `us-otgen-router-ot` (which already has the validated `--dump-routing` hook).
- Loader: added `llama_model_loader::expert_part_spec` and `n_part_tensors`; `create_tensor()` now has a synthetic-part path that builds a tensor from a requested shape, names it from the part spec, forces a buffer type, and registers a synthetic weight as a byte-slice of the original GGUF tensor without incrementing `n_created`.
- Graph: extended `build_moe_ffn()` with optional per-part expert weight/scale vectors and per-part expert offsets. When parts are present, it builds one masked MoE FFN subgraph per part (masked `ids_part`/`weights_part`, local expert remap, per-part gate/up/down + activation) and sums the per-part outputs. The normal single-tensor path now uses the same `sum_expert_rows()` helper as the partitioned path.
- Model: Qwen4exp `load_tensors` now supports `LLAMA_EXPERT_SPLIT=K` (M1: `K=2`): it skips the original fused expert tensors, creates K byte-sliced parts for `ffn_down_exps` and `ffn_gate_up_exps`, assigns part 0 to the layer's current device and part 1 to another GPU (or CPU fallback), and stores the part vectors/offsets in `llama_layer`. `build_layer_ffn` passes those part vectors into `build_moe_ffn()`.
- **Compile check:** added a fast oneAPI container compile path to `host-info.md` (§4 Path D). Used it to configure a non-SYCL CMake build in `/tmp/opencode/llama-build` and build the `llama` target with `icx`/`icpx`; it completed successfully, so the loader/graph/Qwen4exp changes compile across the current `llama` source set.
- Fixed follow-up compile issues found by the container build:
  - extended the no-bias `build_moe_ffn` overload as well as the full overload, because Qwen4exp calls the no-bias wrapper;
  - replaced the missing `ggml_cmp`/`GGML_OP_*` mask logic with a `ggml_step()`-based range mask;
  - moved GPU/CPU buffer-type selection for expert parts into `llama_model_base::get_expert_split_buft()` because `llama_model::impl` is incomplete outside `llama-model.cpp`.
- **Not yet done:** SYCL build, correctness validation, performance validation, CLI/file plumbing, trace-to-placement generator, final `research.md` update, branch push.
- **Next:** build the mounted SYCL build dir (Path E) and run a Qwen3.8-Flash-Next greedy parity test with `LLAMA_EXPERT_SPLIT=3` and mmap enabled.

### 2026-10-01 — Switched iteration to mounted SYCL build dir
- User correctly pointed out that full `podman build` image rebuilds are wrong for normal run-between-changes work. The image bakes binaries into `/app`; for iteration we should use an existing SYCL-capable image only as the toolchain/runtime container and build/run from a mounted build dir.
- Added `host-info.md` §4 Path E: configure/build `/home/niklas/sycl-build-otgen-expert` inside `localhost/llama.cpp:us-otgen-expert-ot` with mounted source, then run `/build/bin/llama-cli` from the same mounted build dir with GPU devices.
- First Path E attempt failed because the llama.cpp podman image has `ENTRYPOINT ["/app/tools.sh"]`; the mounted-build command must use `--entrypoint bash`. Fixed in `host-info.md` and restarted the mounted SYCL build.
- Mounted SYCL build for `llama-cli` succeeded in `/home/niklas/sycl-build-otgen-expert`; `/build/bin/llama-cli --version` runs from the mounted build dir.
- Stopped `podman-llama-cpp-qwen3.8-27b.service` for a bounded GPU validation run with `LLAMA_EXPERT_SPLIT=3`, mmap enabled, `--split-mode layer --tensor-split 50,50`, `-ngl 48`, `--ctx-size 4096`, `-n 32`, greedy. Log: `/tmp/opencode/otgen-expert-split-k3.log`.
- First K=3 run crashed in the partitioned `build_moe_ffn()` path at `ggml_reshape_3d(... selected_experts ...)`: `selected_experts` is a non-contiguous top-k view, so it must be `ggml_cont()` before reshape. The same block also used element offsets where `ggml_view_1d()` expects byte offsets, and it viewed `e1 == n_expert` out of bounds.
- Fixed the partitioned mask/ids block to use a contiguous selected-experts tensor, byte-scaled views, an arange of length `n_expert + 1` for upper bounds, f32 subtraction/cast for local ids, and a contiguous 2D offset tensor. Incremental mounted SYCL rebuild succeeded.
- `-hf` emitted `HTTPS is not supported` in this container because the image build did not enable OpenSSL/BoringSSL; it still used the local cache for the model. For cleaner tests, prefer the direct cached GGUF path under `/root/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/.../UD-Q3_K_XL/...`.
- Direct cached Q3_K_XL K=3 run reached GPU tensor allocation but failed with Level Zero `OUT_OF_DEVICE_MEMORY` while initializing a SYCL tensor buffer. This means K=3 still places too much expert weight on GPU for this model/rig, not that the graph transform crashed.
- Fixed `get_expert_split_buft()` so part assignments are not modulo-cycled: part 0 goes to the layer device, part 1 goes to the first remote GPU, part 2 goes to the second remote GPU only when the layer itself is on CPU, and all remaining parts go to CPU. This makes K=4 place GPU0/GPU1/CPU/CPU instead of accidentally putting the fourth part back on GPU0.
- Incremental mounted SYCL rebuild after the assignment fix succeeded. Next test should use `LLAMA_EXPERT_SPLIT=4` (or a smaller quant if K=4 still OOMs).
- K=4 also OOMed during GPU tensor allocation. Important capacity insight: with the current placement rule, every GPU layer has a local GPU part and a remote GPU part, so each GPU holds `2/K` of all routed experts, not `1/K`. For two GPUs, K=3 puts about 2/3 of experts on each GPU and K=4 puts about 1/2 on each GPU. To get about 1/3 or 1/4 of experts per GPU, use K=6 or K=8.
- K=8 Q3_K_XL still OOMed during GPU tensor allocation. Verbose log confirms the original fused expert tensors are skipped (`model has unused tensor ...ffn_down_exps.weight`), so this is not the old full-fused-tensor path. The remaining GPU pressure is likely non-expert weights plus the local/remote expert parts; K=8 alone is not enough for Q3_K_XL at `-ngl 48`.
- Use `tail -f --pid=$PID` with `timeout` for background GPU tests; it exits when the process dies and avoids manual polling.
- Next validation should either use Q2_K_XL with K=8 to prove the graph transform, or reduce Q3_K_XL GPU scope (`-ngl` smaller / larger K) until it fits.
- Fixed `llama_model_base::get_expert_split_buft()` to assign distinct candidate backends by part index: local device, then other GPUs, then CPU. The previous version sent both part 1 and part 2 to the same remote GPU, so `LLAMA_EXPERT_SPLIT=3` would not create a CPU partition.
- **Next:** run the Path E mounted SYCL build, then stop the 27B service for a bounded GPU parity/perf test with `LLAMA_EXPERT_SPLIT=3`, then restart the 27B service.

### 2026-10-01 - Q2 K=8 OOM, verbose part logging, two placement bugs found
- Ran Q2_K_XL K=8 dual-GPU with `-ngl 48`; it still OOMed during SYCL device buffer allocation.
- Added temporary `LLAMA_LOG_INFO` to the loader `part_spec` path to print each synthetic expert part name, buffer type, byte size, and offset, then rebuilt `llama-cli` in the mounted SYCL build dir.
- Verbose Q2_K_XL K=8 log shows:
  - original `ffn_down_exps.weight`, `ffn_gate_exps.weight`, and `ffn_up_exps.weight` tensors are skipped as unused, so the old full fused tensor path is not the OOM source;
  - parts intended for CPU were assigned `SYCL_Host` because `get_expert_split_buft()` fell back to `cpu_buft_list[0]`, which can be a GPU-associated host buffer;
  - `ffn_gate_up_exps.weight.partN` offsets are all zero because the tested unsloth UD GGUF has separate `ffn_gate_exps.weight` and `ffn_up_exps.weight` tensors, not a fused `ffn_gate_up_exps.weight` tensor.
- Fixed `llama_model_base::get_expert_split_buft()` to return the real CPU buffer type for CPU layers and for parts beyond the available remote GPUs, instead of using `cpu_buft_list[0]`.
- Refined the placement rule: CPU layers keep all expert parts on CPU. Remote-GPU expert parts are used only when the layer itself is on a GPU. This avoids a small `-ngl` single-GPU test from filling the one visible GPU with remote parts from many CPU layers.
- Updated the Qwen4exp expert-split loader path to create separate `ffn_gate_exps.weight.partN` and `ffn_up_exps.weight.partN` tensors when fused `ffn_gate_up_exps.weight` is absent; the graph already supports separate gate/up part vectors.
- 27B service topology: `podman-llama-cpp-qwen3.8-27b.service` uses `--device=/dev/dri/renderD129` (GPU0). Single-GPU dev tests can use the PCIe 4 attached B70 (`renderD130`) while the 27B service stays up; stop the 27B service only for tests that require both GPUs.
- Rebuilt the mounted SYCL `llama-cli` successfully after the CPU-buffer and separate gate/up fixes.
- Clarified design compatibility: the uniform `LLAMA_EXPERT_SPLIT=K` behavior is only a validation scaffold; the warm/cold goal can use the same range-based part/subgraph substrate with arbitrary per-layer ranges/devices from a later placement file.
- First single-GPU Q2_K_XL K=2 `-ngl 2` run on `renderD130` segfaulted. The temporary part log showed separate gate/up parts with non-zero offsets and CPU buffer type, but every part reported ~1.67 GB (F32-sized) instead of the expected Q2-sized slice.
- Root cause: the loader `part_spec` path used `gguf_find_tensor(metadata, ...)`, but for split GGUFs `metadata` is only the first shard. Tensors that live in later shards were not found, so the synthetic part fell back to `GGML_TYPE_F32` and allocated a huge buffer.
- Fixed the `part_spec` path to take the tensor type from the `weights_map` original tensor when available, with the old `gguf_find_tensor` path only as a fallback, and added the tensor type to the temporary part log.
- Second single-GPU run got past the F32 sizing bug but aborted in `ggml_new_object: not enough space in the context's memory pool`. The per-buft ggml context was sized from the current `n_part_tensors` value, which is still small when the first part tensor creates the context, so later part tensors exhaust the metadata pool.
- Fixed `ctx_for_buft()` to reserve a larger metadata pool once expert-part creation is in use (`part_spec.name` non-empty or `n_part_tensors > 0`), matching the existing virtual-model margin pattern.
- Third run still hit the same metadata-pool abort because the CPU ggml context had already been created for ordinary CPU tensors before the first expert part set `part_spec.name`. The fix must reserve part-tensor capacity before tensor loading starts.
- Added `llama_model_loader::reserve_part_tensors()` and made `llama_model_base::load_tensors()` reserve an upper bound of `n_layer_all*3*LLAMA_EXPERT_SPLIT` synthetic tensors before `load_arch_tensors()` runs.
- **Next:** rebuild, rerun the single-GPU Q2_K_XL K=2 `-ngl 2` validation on `renderD130`, then stop 27B and run dual-GPU Q2_K_XL.

### 2026-10-01 - Single-GPU Q2 K=2 validation passed
- Rebuilt the mounted SYCL `llama-cli` after the reserve-part-tensors fix.
- Ran Q2_K_XL K=2 `-ngl 2` on `renderD130` with `-c 2048`, `-n 16`, `-np 1`, mmap, greedy, while the 27B service remained up on `renderD129`.
- Result: exit 0, coherent output (`A GPU (Graphics Processing Unit) is a specialized electronic circuit designed to rapidly manipulate`, stopped by the 16-token limit), prompt eval 6.20 t/s, generation 4.22 t/s.
- The part log now shows correct quantized types and sizes: down parts are `iq4_nl` at ~235.9 MB each, gate/up parts are `iq2_xs`/`iq3_xxs` at the expected per-part sizes, offsets are non-zero for part 1, and GPU-layer part 0 is on `SYCL0` while the remaining parts are on CPU.
- The earlier `2/K` capacity note applied to the old placement rule where CPU layers could send remote parts to GPUs. With the refined rule (CPU layers keep all parts on CPU; GPU layers use local + remote GPU parts), a balanced all-GPU dual-GPU split puts about `1/K` of routed experts on each GPU.
- **Next:** stop `podman-llama-cpp-qwen3.8-27b.service`, run a bounded dual-GPU Q2_K_XL validation with `-ngl 48`, then restart the 27B service as soon as the test window ends.

### 2026-10-01 - Dual-GPU Q2 K=2 validation passed
- Stopped `podman-llama-cpp-qwen3.8-27b.service`, ran Q2_K_XL K=2 with both B70 devices, `-ngl 48`, `--split-mode layer --tensor-split 50,50`, mmap, `-c 4096`, `-n 32`, greedy, then restarted the 27B service.
- Result: exit 0, coherent output, prompt 25.60 t/s, generation 16.51 t/s.
- Memory breakdown after context allocation: SYCL0 ~23.5 GiB used, SYCL1 ~23.5 GiB used, Host ~28.9 GiB used. No Level Zero OOM and no ggml context-pool abort.
- Part log confirms the intended placement: CPU layer 0 keeps both expert parts on CPU; GPU layers split part 0 to the owning GPU and part 1 to the other GPU.
- This proves the M1 per-expert split path is functional for Q2_K_XL on the 2x32GB rig. The next useful step is parity/quality against the unsplit path, then a larger validation with Q3_K_XL or a performance-focused bounded test.

### 2026-10-01 - Single-GPU Q2 K=2 parity passed
- Ran the same Q2_K_XL `-ngl 2` prompt with `-n 32`, greedy, seed 1, on `renderD130` while the 27B service stayed up: once without `LLAMA_EXPERT_SPLIT` and once with `LLAMA_EXPERT_SPLIT=2`.
- Both runs produced the same 32-token output: `A GPU (Graphics Processing Unit) is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation of images in a frame buffer intended for output`.
- Timings: no-split 6.64 pp / 5.36 tg; K=2 split 13.70 pp / 6.32 tg. The split is slightly faster in this small single-GPU config because one expert half moves to the GPU for the GPU-resident layer.
- This is the first correctness parity result for the per-expert split path.
- Pushed `us-otgen-expert-ot` to `nilo85` at commit `6a9e94689` after changing the temporary expert-part log to debug level and cleaning the graph signature indentation.
- **Next:** run a bounded dual-GPU parity/perf comparison or move to Q3_K_XL / routing-aware placement.

### 2026-10-01 - Dual-GPU Q2 K=2 parity/perf passed
- Ran a bounded dual-GPU Q2_K_XL comparison while `podman-llama-cpp-qwen3.8-27b.service` was stopped.
- Baseline no-split path used `-cmoe` to keep experts on CPU:
  - log: `/tmp/opencode/otgen-expert-split-parity-dual-gpu-q2-ns-cmoe.log`
  - output: `A GPU is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the`
  - timings: `19.8 t/s` prompt, `13.3 t/s` generation.
- Split path used `LLAMA_EXPERT_SPLIT=2` and omitted `-cmoe`:
  - log: `/tmp/opencode/otgen-expert-split-parity-dual-gpu-q2-k2-split.log`
  - output matched the baseline text exactly.
  - timings: `31.1 t/s` prompt, `18.1 t/s` generation.
- Common flags: `-ngl 48`, `--split-mode layer`, `--tensor-split 50,50`, `--fit off`, `--ctx-size 4096`, `-n 32`, `-np 1`, `--temp 0`, `-s 1`, `-st`, `--no-display-prompt`, `--jinja`, `--reasoning off`, `-fa on`, `-ctk q4_0`, `-ctv q4_0`, `--load-mode mmap`.
- Restarted `podman-llama-cpp-qwen3.8-27b.service` after the dual-GPU test window.
- Conclusion: the M1 per-expert split path is correct for the Q2 greedy prefix and faster than the `-cmoe` CPU-expert baseline in this bounded dual-GPU config.

### 2026-10-01 - Q3 K=8 capacity passed, perf poor
- Ran Q3_K_XL with `LLAMA_EXPERT_SPLIT=8`, dual-GPU `-ngl 48`, `--split-mode layer`, `--tensor-split 50,50`, `--fit off`, mmap, `-c 4096`, `-n 32`, greedy seed 1, while the 27B service was stopped.
- Log: `/tmp/opencode/otgen-expert-split-q3-k8-dual-gpu.log`.
- Result: exit 0, coherent output, prompt `5.8 t/s`, generation `3.4 t/s`.
- Restarted `podman-llama-cpp-qwen3.8-27b.service` after the test.
- Conclusion: uniform K=8 proves the Q3 capacity path can load and run on this rig, but it is not a practical Q3 performance config. The next useful step is routing-aware placement with smaller hot-expert GPU ranges instead of uniform all-GPU-layer K-way splitting.
- **Next:** move to M2 placement-file plumbing or the out-of-tree trace-to-placement generator.

### 2026-10-01 - `-ot` can now target synthetic expert parts
- Added a small core hook: `llama_model_base::get_expert_split_buft()` checks `tensor_buft_overrides` for the synthetic part name before falling back to the uniform `LLAMA_EXPERT_SPLIT` placement rule.
- Qwen4exp now passes the synthetic part name (`...partN`) when selecting the expert-part buffer type.
- Built `llama-cli` in `/home/niklas/sycl-build-otgen-expert`.
- Single-GPU Q2_K_XL test on `renderD130` with `LLAMA_EXPERT_SPLIT=2` and `-ot 'blk.1.ffn_down_exps.weight.part0=CPU'` plus matching gate/up overrides moved layer 1 part 0 to CPU in the debug log and still generated coherent text.
- Pushed `us-otgen-expert-ot` to `nilo85` at `d49a538df`.
- This makes the existing `-ot` mechanism usable by the out-of-tree trace-to-placement generator for uniform K-way expert parts.

### 2026-10-01 - Trace-to-`-ot` generator implemented and first Q3 win
- Added out-of-tree generator: `/home/niklas/b70_opt/generate_expert_ot.py`.
- It parses `--dump-routing` traces, aggregates per-layer expert hits, ranks contiguous `.partN` expert ranges, and emits `-ot` patterns for synthetic expert parts.
- Generated a Q2 routing trace on `renderD130` while the 27B service stayed up:
  - trace: `/tmp/opencode/routing-trace-q2.txt`
  - prompt: LRU cache coding question, 128 generated tokens, 48 MoE layers, 6288 trace data lines.
- Generated placement:
  - file: `/tmp/opencode/placement-q2-k8-hot50.ot`
  - shell helper: `/tmp/opencode/placement-q2-k8-hot50-args.sh`
  - `K=8`, `hot-fraction=0.5`, `prefer-local`, `SYCL0/SYCL1=50/50`, separate gate/up layout.
- Q2_K_XL dual-GPU with generated K=8 hot-50 placement:
  - log: `/tmp/opencode/otgen-placement-q2-k8-hot50-dual-gpu.log`
  - `8.5 pp / 4.6 tg`.
  - This is slower than uniform K=2 (`31.1 pp / 18.1 tg`) because Q2 can keep all experts on GPU; the generated K=8 placement is mainly useful when all experts do not fit.
- Q3_K_XL dual-GPU with the same generated K=8 hot-50 placement:
  - log: `/tmp/opencode/otgen-placement-q3-k8-hot50-dual-gpu.log`
  - `7.2 pp / 3.9 tg`.
  - This beats the uniform K=8 Q3 capacity run (`5.8 pp / 3.4 tg`) by about `24% pp` and `15% tg` in this bounded prompt.
- Restarted `podman-llama-cpp-qwen3.8-27b.service` after the dual-GPU test window.
- Generated a Q3 routing trace on `renderD130` while the 27B service stayed up:
  - trace: `/tmp/opencode/routing-trace-q3.txt`
  - prompt: LRU cache coding question, 64 generated tokens, 48 MoE layers, 3216 trace data lines.
- Q3_K_XL dual-GPU with Q3-trace K=8 hot-50 placement:
  - log: `/tmp/opencode/otgen-placement-q3trace-k8-hot50-dual-gpu.log`
  - `9.4 pp / 4.2 tg`.
  - This beats the Q2-trace placement (`7.2 pp / 3.9 tg`) and uniform K=8 (`5.8 pp / 3.4 tg`), so the trace-to-`-ot` path is working and model-specific traces matter.
- Restarted `podman-llama-cpp-qwen3.8-27b.service` after the dual-GPU test window.
- **Next:** implement M5 (local-only placement) and A/B it; then compare against a practical Q3 CPU-expert baseline rather than uniform K=8 only.

### 2026-10-01 - Global expert-index tiering checked: mostly a ghost
- User proposed a new split axis: put the globally hottest experts' all-layer weights on GPU1, moderate on GPU2, cold on CPU, so the hottest compute never leaves GPU1.
- Ran a cheap, no-GPU check on the traces we already had (`/tmp/opencode/check_hotness_consistency.py` over the Q2+Q3 routing traces):
  - Per-layer top-set overlap is ~chance: top-10% mean pairwise Jaccard `0.062` vs chance `0.052` (1.19x); top-25% `0.152` vs `0.143` (1.06x). Hot experts are per-layer-scattered, not index-consistent across layers.
  - There is a mild global skew (global top-10% holds `21.3%` of routing mass, ~2x uniform), but those experts sit at mean per-layer rank ~119/512 and land in a layer's top tier only ~16% of the time. In a layer split, GPU1 only runs layers 24-47, so globally-hot weights in layers 0-23 would sit on GPU1 but be used by GPU0, reintroducing the cross-device traffic we want to remove.
- Verdict: the premise (index-consistent hotness) does not hold; the small real skew is not placeable without fighting the layer boundary. Logged as deferred M6. Per-layer-local (M5) stays the policy.

### 2026-10-01 - Persona/workload traces: real, robust signal (M7 validated)
- User hypothesis: different "personas" (coder vs reporter vs tool-caller) route to different experts, so traces should use varied persona prompts and placement could be per-persona.
- Design (to defeat one-prompt noise): 3 medium prompts per persona x 4 personas = 12 Q3 traces, single-GPU `-cmoe -ngl 2`, same greedy seed, 27B stayed up. Runner `/tmp/opencode/run_persona_traces.sh`, analyzer `/tmp/opencode/check_persona_overlap.py`, log `/tmp/opencode/otgen-persona-traces.log`.
- GPU behavior during runs (user-asked): 95% util but 87W/18% eff and only ~1.9GiB VRAM is the expected signature of a CPU-bound `-cmoe` run (2 sidecar layers on GPU, ~85GB experts on CPU, GPU mostly stalling). Trace validity unaffected.
- Result (per-layer top-set Jaccard): within-persona `0.470` (9.0x chance) vs between-persona `0.152` (2.9x) at top-10%; `0.568` (4.0x) vs `0.302` (2.1x) at top-25%. Within >> between => persona effect is real and stable, not noise.
- Unions: within-persona ~1.6x K (a persona is stable); cross-persona ~4.2x K at top-10% (~42% of a layer's experts), ~2.8x at top-25% (~69%, at/over the Q3 capacity wall) => a full 4-persona union likely OOMs; per-persona is the VRAM-safe path.
- Caveat: toolcaller outputs are short JSON (~20-35 traced steps vs ~131 for prose personas) so its hot set is noisier, but the within/between contrast still holds.
- **Next:** M8 decision (per-persona files vs union) based on the user's real workload mix; then feed the chosen persona's traces into the M5 local-only generator.
