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
- [ ] M1: env-driven K-way expert split for Qwen3.8-Flash-Next; validate greedy parity + perf. On this rig use `LLAMA_EXPERT_SPLIT=3` (GPU/GPU/CPU) for capacity; a 2-way all-GPU split does not fit the ~90GB Q3_K_XL model.
- [ ] M2: CLI/file plumbing for arbitrary per-layer expert ranges and backends.
- [ ] M3: out-of-tree trace-to-placement generator that emits the placement file.
- [ ] M4: validation + research.md update; push branches.
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
_State: PER-EXPERT M1 CODE DRAFT ON `us-otgen-expert-ot` 2026-10-01; loader, graph, and Qwen4exp wiring are in place. Normal iteration now uses a mounted SYCL build dir (host-info.md Path E) instead of rebuilding the podman image. CPU compile check passed; mounted SYCL build + GPU validation pending._

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
- Fixed `llama_model_base::get_expert_split_buft()` to assign distinct candidate backends by part index: local device, then other GPUs, then CPU. The previous version sent both part 1 and part 2 to the same remote GPU, so `LLAMA_EXPERT_SPLIT=3` would not create a CPU partition.
- **Next:** run the Path E mounted SYCL build, then stop the 27B service for a bounded GPU parity/perf test with `LLAMA_EXPERT_SPLIT=3`, then restart the 27B service.
