# us_otgen - Router-aware MoE expert placement (portable heatmap core mode + trace/-ot generator fallback)

**Type:** Core MoE-heatmap mode (two flags + placement seam) on the local fork branch; merge script and trace/-ot generator out-of-tree (zero fork risk).
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
- [ ] M5 (local-only placement) SUPERSEDED (2026-10-02): policy moves into the core - M10's heatmap placement is local-only by construction (hot -> owning GPU, cold -> CPU, never remote). A/B vs remote-spill folds into M11.
- [ ] M6 (DEFERRED, low priority): global expert-index tiering (hottest experts' all-layer weights on GPU1, moderate GPU2, cold CPU). Checked 2026-10-01 against our traces: per-layer top-set overlap is ~chance (1.06-1.19x), so hotness is per-layer-scattered, not index-consistent; the mild global skew (top-10% holds ~21% mass) can't be placed without fighting the layer split. Revisit only if a future model shows index-consistent hotness or we drop the layer split.
- [x] M7 (workload/persona-adaptive traces) VALIDATED 2026-10-01: 3 prompts x 4 personas (coder/reporter/toolcaller/chat), Q3, medium prompts, single-GPU. Per-layer top-set: within-persona Jaccard `0.470` (9.0x chance) vs between-persona `0.152` (2.9x) at top-10%; `0.568` (4.0x) vs `0.302` (2.1x) at top-25%. Personas are a real, robust placement axis (not one-prompt noise). Within-persona union ~1.6x K (stable); cross-persona union ~4.2x K at top-10% (~42% of a layer's experts) and ~2.8x at top-25% (~69%, at/over the Q3 capacity wall). Toolcaller traces thinner (short JSON, ~20-35 steps) -> noisier, but contrast holds.
- [x] M8 (decided 2026-10-01): user's real mix is tool-caller + coder + reasoning (not reporter/chat). 3-persona cross-union = 179 experts (35% of a layer) at top-10%, 321 (63%) at top-25%. 35% is under the 50% that already loaded on Q3 (K=8 hot-50), so a single union-default file fits at a tight hot-fraction; 63% is over the wall. Plan: ship a union-default (3 personas) + per-persona files, switched per session (static-but-calibrated "dynamic"; runtime migration stays out of scope). Caveat: scattered union-hot vs contiguous parts -> higher K for cleaner fit (K-vs-overhead knob).
- [ ] M9 (build) SUPERSEDED (2026-10-02): the union-default idea survives as M11's merge-script output (sum of per-workload heatmaps), but the deliverable is the core heatmap mode, not per-rig `-ot` files (see "MoE heatmap design"). `generate_expert_ot.py` remains a fallback/validation proxy.
- [x] M10 (Phase 1) DONE 2026-10-02 (pushed `bfa57f78f`+`be86ddb99`, parser fix `61e67a78b`): core MoE-heatmap mode - `--moe-heatmap FILE`, opt-in `--moe-heatmap-dump FILE` (normalized sorted exit dump), `--moe-heatmap-fraction F`. Calibration validated (dump overhead negligible, 9 heatmaps valid). Placement details (gating via `LLAMA_EXPERT_SPLIT`, aggregate-score ranking) SUPERSEDED by M14's fraction-only enabler + order-not-score loop.
- [x] M11 DONE 2026-10-02: out-of-tree `merge_moe_heatmaps.py` (normalize -> equal-weight avg by default, optional `--weights`; output sum-1 per layer = valid input). Calibration workflow validated end-to-end: 9 per-workload dumps -> `hm-mix.txt` -> clean dual-GPU A/B beat uniform K=8 and remote-spill `-ot` placement (heatmap frac 0.5 `16.1/4.6`, frac 0.375 `14.1/4.2` vs uniform `10.4/3.6`; see work log).
- [ ] M12 (only if M10 measurement demands): Phase 2 CPU-side counting under a CMake compile flag (production builds compile it out; runtime dump arg still gates it). Fallback only - a CPU-side heatmap would be placement-dependent and worse for cross-workload merges.
- [x] M13: generalize per-expert split + heatmap placement to other MoE archs. DONE 2026-10-02: extracted the per-expert part-split orchestration into shared `llama_model_base::create_expert_split_tensors(layer, il, n_embd, n_ff_exp, n_expert)` (K<=1 -> full tensors, K>1 -> `*_parts`+offsets, returns whether split). The rest of the machinery was already arch-agnostic (`create_expert_part`, `get_expert_split_buft`, `load_moe_heatmap`, `llama_layer` `*_parts`, `build_moe_ffn` `*_parts`/offsets params); generalizing an arch = call the method in `load_arch_tensors` + pass the `_parts` to `build_moe_ffn`. **qwen4exp** now calls it (regression passed); **gemma4** (first target, same splittable 3D layout) is wired (compile+code validated; runtime A/B future, no local model). More archs = same 2-line pattern.
- [ ] M14 (ACTIVE): single mechanism - fraction-only enabler + order-not-score placement. Remove `--dump-routing` and `LLAMA_EXPERT_SPLIT`; `--moe-heatmap-fraction` (explicitly set) is the sole split enabler; K derived internally (`clamp(round(1/F),2,32)`); no heatmap -> on-the-fly identity seed ordering via a swappable `make_seed_order` fn (same loop as file placement); core reads file **order only** (scores kept in-format for offline mixing); GPU = hottest `clamp(round(K*F),0,K)` parts local-only, rest CPU; `-ot` still wins. Design LOCKED with user 2026-10-02 (see "M14 design" section + user scenarios). Then gemma4 runtime A/B (UD-Q4_K_XL + MTP Q8_0 now cached, snapshot `c099eb48`) with the new flags.
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

## MoE heatmap design (2026-10-01) - portable heatmap + core placement + out-of-tree merge

**Why this replaces the `-ot` mapping as the primary mechanism:** the generator's `-ot` file bakes the *placement* (which part -> which device) into the file, so it is rig-specific and must be regenerated per rig. Expert hotness is a property of *model + workload*, not the rig. So keep a portable **heatmap** (per-layer expert scores) as the file, and let llama.cpp (which knows the rig: GPU count, which GPU owns each layer, VRAM) compute the placement from it. The `-ot` mapping generator (M3/M9) becomes a fallback/validation proxy, not the deliverable.

**One concept, two flags, one format:**
- `--moe-heatmap FILE` (input): load a heatmap and drive placement - rank each layer's K parts by aggregate score, hot (top `--moe-heatmap-fraction`) -> the layer's local GPU, cold -> CPU. Extends the existing `-ot`/expert-part seam (`get_expert_split_buft`); a manual `-ot` still wins.
- `--moe-heatmap-dump FILE` (opt-in): collect per-expert usage over the run and dump a heatmap to FILE **on exit**. Only active when passed, so production runs pay zero cost.
- `--moe-heatmap-fraction F` (default 0.5): the hot/cold cut.
- **Format (input == output, portable, rig-agnostic):** `il=<layer> <expert>:<score> ...` sorted by score desc. **Score is a relative value** (per-layer normalized to sum to 1, written as float), so files from sessions of different sizes are comparable, a short session is not drowned out by a long one, and a dump's output is a valid placement input (round-trip). Placement only ranks within a layer, so it is invariant to this scaling.

**The full-heatmap payoff (why the read-back, not CPU-side-only):** routing is placement-independent (the router picks experts from weights + hidden state, not device), so each *independent* workload run yields a complete, placement-independent heatmap. An out-of-tree **merge script** then combines N workload heatmaps (coding, prose, java, go, ...) as an **equal-weight average of the normalized distributions** (each session contributes equally regardless of length) into the "perfect fit" heatmap for the user's actual load mix. A CPU-side-only count would be placement-dependent and incomplete, so it is a fallback, not the goal.

**Staged (de-risk the cost):**
- **Phase 1 (now):** placement + full read-back dump, reusing the existing `ffn_moe_topk` capture. The read-back is a per-step GPU sync (48 tiny tensors/step); for this CPU-bound model it is likely ~1-2%/step (small data; the GPU often stalls, so losing async pipelining costs little), and it is opt-in so production is untouched. Measure the real cost on a calibration run.
- **Phase 2 (only if Phase 1 is too slow for long production-prompt calibration runs):** count in the CPU-side expert kernel. The CPU part already receives its expert ids (transferred for the compute), so counting them adds no extra sync; it is gated by a **CMake compile flag** (a production build compiles the counter/dump out -> 0 kernel impact) AND still needs the runtime `--moe-heatmap-dump` arg to activate. Invasive, so last resort.

**Out-of-tree (zero fork risk):** the merge script (normalize each input layer to sum 1, then weighted average -> merged, sum 1 per layer; equal weight by default) + file versioning. The core never merges.

**Upstream surface stays tiny:** the read-back/dump is a natural extension of our `--dump-routing` trace hook (the upstream candidate). The placement-from-heatmap + expert split stay local.

**Workflow:** full-CPU + `--moe-heatmap-dump hm-<workload>.txt` per workload -> merge -> `--moe-heatmap hm-mix.txt` (drives placement) for production.

## M14 design (2026-10-02) - single mechanism: fraction-only enabler, order-not-score placement (user directive, LOCKED)

Supersedes these M10-era details: the `LLAMA_EXPERT_SPLIT=K` env var, score-based part ranking, and the old uniform no-heatmap placement rule (part 0 local GPU / part 1+ remote-GPU round-robin). Everything below is user-driven ("I think its critical we only have a single mechanism", "enabler should only and explicitly be --moe-heatmap-fraction").

**One enabler, one concept:**
- `--moe-heatmap-fraction F` is the **only** thing that enables the expert split. Not set -> no split, model loads exactly as today (behavior default unchanged). F is the literal per-layer GPU expert share: `N_gpu = round(F * n_expert)`; `F=0` -> all experts on CPU, `F=0.1` -> 10%, `F=1` -> everything on GPU.
- `--moe-heatmap FILE` does NOT enable the split (no fraction -> warn + ignore, same shape as the old `LLAMA_EXPERT_SPLIT` warning). It refines placement *given* the fraction.
- `--moe-heatmap-dump FILE` does NOT enable the split either; it stays opt-in, zero-cost when absent, and works on any run (routing capture keys on the `ffn_moe_topk` graph node, which exists split or unsplit; routing is placement-independent).
- The user never sees K. **K is derived internally from F: `K = clamp(round(1/F), 2, 32)`** (F=0.5->2, 0.25->4, 0.1->10, 0.075->14). The fraction's placement resolution just follows F.

**A heatmap is an ORDERING, not a score table (the unification):**
- With a file: the ordering is the file's order (dump/merge output is written sorted hottest-first).
- Without a file: llama.cpp generates a **seed ordering on the fly** in one small function (identity `0,1,...,n_expert-1` today; swappable later for random/heuristic without touching the placement loop - the "tweak the seed later" hook).
- **The core reads only the order and ignores score values.** Scores stay in the file format solely so out-of-tree tools (merge script) can average/mix heatmaps offline. This drops the old aggregate-score ranking from `load_moe_heatmap`.

**One placement loop (identical for seed and file):**
1. Ordering per layer (file order, or seed if no file).
2. `N_gpu = round(F * n_expert)`; the first `N_gpu` experts in the ordering are the **hot set**.
3. K-way even split into contiguous index ranges (byte-slice parts, unchanged machinery: `create_expert_split_tensors` / `create_expert_part`).
4. Each part's hot count = number of its experts in the hot set; sort parts by hot count desc; GPU gets the hottest `n_hot = clamp(round(K*F), 0, K)` parts -> the layer's **local GPU only** (remote-GPU spill removed, per the M8/local-only policy), rest -> CPU. CPU-owned layers keep all parts on CPU (unchanged).
5. `-ot` overrides still win over everything (unchanged).

Seed case = degenerate identity ordering: hot set is experts `0..N_gpu-1`, so the first part(s) land on GPU - "just make it run" initial phase. Heatmap case = same loop with a smarts ordering, improved by dumping a real heatmap later.

**Removals (provenance confirmed ours: 0 hits in base `19e28a277` and 0 in master):**
- `--dump-routing` + `dump_routing` (llama.h, llama-cparams.h, llama-context.h/.cpp incl. `routing_trace` ofstream + per-step trace lines, common.h/.cpp/arg.cpp). Superseded technical debt: the `--moe-heatmap-dump` normalized heatmap replaced the raw per-step trace; `generate_expert_ot.py` (trace consumer) becomes history/fallback only.
- `LLAMA_EXPERT_SPLIT` env var (llama-model.cpp gating + `create_expert_split_tensors` reads the model param instead; comments in llama-model.h, gemma4.cpp, common/arg.cpp help text).

### User scenarios (documented per user requirement)

Heatmap artifacts live in `/home/niklas/b70_opt/heatmaps/` (portable, per-model).

1. **Initial seed (no heatmap yet - make it run):**
   `llama-cli -m model.gguf --moe-heatmap-fraction 0.5 ...`
   Identity seed ordering -> first 50% of each layer's experts (by index) on its local GPU, rest on CPU. Lower F until VRAM fits. Nothing else required.
2. **Use an existing heatmap (production run):**
   `llama-cli -m qwen.gguf --moe-heatmap-fraction 0.5 --moe-heatmap heatmaps/qwen3.8-flash-next.txt ...`
   Placement now follows hottest-first file order. Same F semantics; only the ordering changes.
3. **Tune / capture the heatmap for your workload:**
   `llama-cli -m model.gguf --moe-heatmap-fraction 0.5 --moe-heatmap-dump /tmp/hm-coder.txt -p "<representative workload>" -n 512 --temp 0 ...`
   Dumped on exit (sorted, normalized, placement-independent). Next run takes it as `--moe-heatmap` input (or feed it to the merge script). VRAM tuning is always just the fraction knob (F down = fewer experts on GPU).
4. **Mix heatmaps (out-of-tree `merge_moe_heatmaps.py`) - equal balance:**
   `./merge_moe_heatmaps.py hm-coder.txt hm-toolcaller.txt hm-reasoning.txt -o hm-mix.txt`
   Each input layer normalized to sum 1, then equal-weight average (every session counts equally regardless of length), output normalized + sorted desc -> directly valid `--moe-heatmap` input.
   **Weighted per input:**
   `./merge_moe_heatmaps.py hm-coder.txt hm-toolcaller.txt hm-prose.txt --weights 2,1,0.5 -o hm-coder-heavy.txt`
   Weights tilt the mix toward the workloads you care about most.

## Work Log & Resume Context
_State: PER-EXPERT M1, `-ot` PART-OVERRIDE HOOK, AND TRACE-TO-`-ot` GENERATOR DONE ON `us-otgen-expert-ot` 2026-10-01. Q2_K_XL K=2 passes parity/perf; Q3_K_XL K=8 capacity passes; Q3-trace K=8 hot-50 `-ot` placement beats uniform K=8 (`9.4 pp / 4.2 tg` vs `5.8 pp / 3.4 tg`). Core pushed to `nilo85` at `d49a538df`. Global expert-index tiering set aside as ghost (M6 deferred). Persona/workload traces VALIDATED (M7). M8 DECIDED: user's real mix is tool-caller + coder + reasoning; 3-persona union = 35% of a layer at top-10% (fits under the 50% that already loaded on Q3) -> ship a union-default + per-persona files, switched per session (static-but-calibrated; runtime migration out of scope). PLE n-gram table (~51B params, SSD-streamed) is the dominant per-token cost (host-info §3); our expert placement is a complementary, smaller lever. M10 CODE DONE on `us-otgen-expert-ot` (two flags + fraction knob; input placement local-only by construction; opt-in full dump on exit; see "MoE heatmap design" section) - then a parser off-by-one bug was found+fixed and scores switched to relative (per-layer normalized float, round-trip). Calibration validated: dump overhead negligible, 9 heatmaps valid. M11 merge script done (equal-weight avg of normalized). M5/M9 superseded by the pivot (2026-10-02). M13 added (last): generalize split+placement to other MoE archs, gemma4 first. Clean dual-GPU A/B re-run PASSED (heatmap frac 0.5 `16.1/4.6`, frac 0.375 `14.1/4.2`, both beat uniform K=8 `10.4/3.6`, same cycle); parser fix + relative-score redesign + merge script + docs pushed to `nilo85` at `61e67a78b`. M13 (last) DONE: per-expert split orchestration extracted into shared `llama_model_base::create_expert_split_tensors`; qwen4exp now calls it (regression passed: loads+generates, heatmap `4/8 parts/layer` applied); gemma4 (first target) wired to it (compile+code validated, no local model for runtime A/B); M13 pushed to `nilo85` at `b41b9c816`. gemma-4-26B-A4B-it UD-Q4_K_XL (16GB) + MTP Q8_0 (441MB) NOW CACHED (snapshot `c099eb48`) for the gemma4 runtime A/B. M14 design LOCKED with user 2026-10-02 (single mechanism: fraction-only enabler, seed-vs-file unified order-not-score placement loop, K derived internally `clamp(round(1/F),2,32)`, cut `--dump-routing` + `LLAMA_EXPERT_SPLIT`, core reads order and ignores scores, scores kept only for out-of-tree mixing); provenance verified ours (0 hits in base + master); docs written first per user rule; implementation ACTIVE. Heatmap artifacts now live in `/home/niklas/b70_opt/heatmaps/` (qwen3.8-flash-next.txt = the old hm-mix). M14 CODE DONE + tested: qwen ALL PASS (base/seed/file/verbose identical text; hmonly OOM expected=feature off); GEMMA SPLIT BUG found+diagnosed (split path nulls ffn_down_exps -> generic scale pass gate dead -> 30 per-expert .scale tensors never created -> count 658 vs 628); Fix A LANDED+verified (skip 3 scale originals + populate ffn_*_exps_s_parts in the preset; gemma base/seed/file/verbose all load, scale parts created+placed following weight parts, seed text==base, file text coherent; qwen regression text-identical; gemma heatmap saved to b70_opt/heatmaps). M14 + gemma4 scale fix committed+pushed to nilo85 at 340f6abf2. PLAN B FINAL LOCKED 2026-10-02: TENSOR_EXPERT flag design (supersedes the builder; see final-design entry). GRAPH SIDE SETTLED SAME DAY = B+ (keep weight args for model control; 9 part-args -> 1 `const llama_layer*` arg; grovemoe/deepseek4 audit proved full layer-replacement unsafe; the 9 part-args are a FORK addition NOT in master so B+ = +1 arg vs master vs +9 if kept - user re-confirmed B+ 2026-10-02 after first flinching at the layer as deep coupling; layer arg is a KNOWN HACK to revisit later). B+ IMPLEMENTATION LANDED 2026-10-02 (loader TENSOR_EXPERT byte-slice branch + model-base drain/scale-gates/retire-old-helpers + graph B+ layer arg + model files TENSOR_EXPERT + layer arg); one graph-builder bug found+fixed (local derivation must only use NON-EMPTY part vectors - a fused-gate_up layer leaves ffn_gate_exps_parts empty and a separate-gate+up layer leaves ffn_gate_up_exps_parts empty, so an unconditional `&layer->...` pointer tripped the size assert; now gated on `.empty()`); M14 suite ALL PASS (qwen base/seed/file/verbose EXIT=0 + identical text; qwen-hmonly EXIT=134 OOM=feature off; gemma all 4 EXIT=0; gemma seed text == base, file coherent; gemma -v scale parts SYCL0 offs=0 / CPU offs=256) - behavior-preserving re-plumbing confirmed. LANDED + pushed to nilo85 at 683779ead (tree clean)._

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

### 2026-10-01 - M8 decided: user's real mix -> union-default + per-persona, no runtime migration
- User's real personas: tool caller, coder, logical thinking/reasoning (not reporter/chat). Asked whether the end plan is "dynamic"; agreed framing = static-but-calibrated (per-persona files + union default, switched per session), runtime expert migration out of scope.
- Generated 3 reasoning traces (medium logic prompts, Q3, single-GPU `-cmoe -ngl 2`, 27B up): `/tmp/opencode/persona-reasoning-1.txt` / `-2` / `-3`, runner `/tmp/opencode/run_reasoning_traces.sh`, log `/tmp/opencode/otgen-reasoning-traces.log`.
- 3-persona (coder+toolcaller+reasoning) analysis: top-10% within `0.471` (9.0x) vs between `0.167` (3.2x); top-25% within `0.546` (3.8x) vs between `0.310` (2.2x). Cross-union: `179` experts (35% of a layer) at top-10%, `321` (63%) at top-25%.
- Capacity read: 35% is under the 50% that already loaded on Q3 (K=8 hot-50), so a single union-default file fits at a tight hot-fraction; 63% is over the wall. Scattered union-hot vs contiguous parts -> higher K for a cleaner fit (K-vs-overhead knob).
- **M8 decided:** ship a union-default (3 personas) + per-persona files, switched per session. **M9 (build):** emit the union-default `-ot` from the 3-persona traces using the M5 local-only policy; A/B vs uniform K=8 and a single-persona file; tune K + hot-fraction.

### 2026-10-01 - PLE n-gram table is the dominant per-token cost; expert placement is complementary
- User FYI (already in host-info §3): the PLE / n-gram table (~51B params, 29-36 GB single tensor, `TENSOR_READ_LAZY` at qwen4exp.cpp:194) is lazy-streamed from SSD, and the dominant per-token cost is NVMe row-gather for PLE, not GPU math. Our `--load-mode mmap` is primarily there to stream that table.
- Implication for us_otgen: our `-ot` expert placement moves expert parts between GPU/CPU but does NOT touch the PLE table. It is a real but smaller lever; the bigger PLE wins are the batched-gather/prefetch PRs (#29030, already in the validation combo) and us_plecache (hot-row LFU). Set tg expectations accordingly.
- Q4_K_XL (now downloaded): will run via mmap (PLE streaming) + cold-expert CPU + hot-expert GPU, but is bounded by SSD (PLE) + CPU (experts) -> expect slower than Q3.
- Transfer: the 3-persona adaptivity measured for experts (within 9x chance vs between 2.9x) likely also applies to PLE rows (different personas hash to different hot n-gram rows) -> reusable input for us_plecache.

### 2026-10-02 - Design pivot: portable MoE heatmap replaces per-rig `-ot` mapping (M10-M12)
- Design locked with the user in iteration (section "MoE heatmap design" above): the deliverable is a portable, sorted `il=<layer> <expert>:<score>` heatmap + core placement, NOT a generated per-rig `-ot` file. Two-args rationale (user): the placement input costs nothing at runtime, but dumping needs the per-step read-back sync -> it must be opt-in (`--moe-heatmap-dump`), so production pays zero.
- User challenged the GPU read-back cost ("severe GPU bottleneck"); challenge accepted: ~2KB x 48 layers/step, a few ms/step vs ~250ms decode steps on this CPU-bound model (~1-2%), and opt-in only -> measure on a real calibration run (M10) before building Phase 2.
- User proposed CPU-side kernel counting as the cheaper spot (expert ids are already host-resident for the CPU-part compute -> no extra sync); kept as Phase 2/M12 under a CMake compile flag (production build compiles it out; runtime arg still required to activate), fallback only because a CPU-only heatmap is placement-dependent and cannot be merged across runs.
- User's payoff insight for the full read-back: routing is placement-independent, so independent workload runs (java vs go vs prose) each dump a complete heatmap; the out-of-tree merge script sums them into the user's mix heatmap. M8's union-default becomes exactly this merge output.
- Docs written before any code, per user rule.
- **Next:** implement M10 on `us-otgen-expert-ot`: flag plumbing (llama.h/cparams/common/arg), model-side heatmap parse + per-layer hot-part precompute + `get_expert_split_buft` branch, context-side accumulate + on-exit dump; then container syntax check, mounted SYCL build, and a single-GPU calibration run to measure dump overhead.

### 2026-10-02 - M10 implemented: core MoE heatmap mode (flags + placement + dump)
- Implemented the locked design across 9 files (201 insertions):
  - `include/llama.h`: `llama_model_params` +`moe_heatmap`/`moe_heatmap_fraction` (placement is a model-load concern); `llama_context_params` +`moe_heatmap_dump` (runtime).
  - `src/llama-model.cpp`: `load_moe_heatmap()` parses `il=<layer> <expert>:<score>` (sums duplicate entries), ranks each layer's K contiguous parts by aggregate score (same part-range math as the qwen4exp loader), marks the top `n_part*fraction` parts hot; called from the `LLAMA_EXPERT_SPLIT` block in `load_tensors` (warns + ignores if no split). `get_expert_split_buft()`: `-ot` still wins, then heatmap (hot -> layer's local GPU, cold -> real CPU), then the old uniform rule.
  - `src/llama-context.cpp`: capture/keep-alive/flush now gated on `dump_routing || moe_heatmap_dump`; `routing_trace_flush()` also accumulates per-layer totals; the destructor dumps a sorted, placement-independent heatmap.
  - `common/`: `--moe-heatmap`, `--moe-heatmap-dump`, `--moe-heatmap-fraction` (string->`std::stof`, [0,1]; the established float-arg pattern, user-confirmed).
- Pushed to `nilo85`: `bfa57f78f` (feature), `be86ddb99` (keep-alive fix, below).
- Fast syntax check (oneAPI container, `icpx -fsyntax-only`): all 4 modified .cpp OK. Incremental mounted SYCL build OK.
- **First calibration bug found:** the keep-alive VIEW node (keeps the `ffn_moe_argsort` buffer alive until after compute) was only added for `--dump-routing`; `--moe-heatmap-dump` alone read freed/reused buffers -> expert IDs came out as float32 bit patterns (~0.09-0.10 routing probs), all scores 1. Fixed the gate to the same OR condition (`be86ddb99`); calibration rerunning.
- The pre-fix batch still proves the plumbing: 9/9 runs completed, 9 heatmap files emitted (49 lines each = header + 48 layers), dump-on-exit works, no crashes with the new flags.
- **Calibration rerun (post-fix) PASSED:** all 9 heatmap files valid (expert IDs 0..511, real token counts, sorted desc; e.g. coder-1 il=0 top = `93:45 134:32 0:25 ...`). Read-back dump overhead is negligible: same-prompt tg 4.3 t/s baseline vs 4.3 t/s with dump (single GPU, -cmoe, Q3). Merged 9 -> `hm-mix.txt` via new out-of-tree `merge_moe_heatmaps.py` (b70_opt; sums N heatmaps, optional weights, sorted desc = valid `--moe-heatmap` input): 48 layers, 20216 entries; 83% of the 24576 experts touched at least once across the mix.
- **Dual-GPU A/B (first pass):** uniform K=8 `8.4 pp / 3.4 tg`; heatmap frac 0.5 `12.0 pp / 4.4 tg`; frac 0.375 `4.7 pp / 3.2 tg`. 27B stopped for the cycle, restarted after (verified active). **SUPERSEDED** - this ran with the buggy parser + raw-count heatmap (below), so the numbers are not a clean validation; a re-run is pending.

### 2026-10-02 - Parser bug found + scores made relative (normalized float) + M13 added
- **Parser off-by-one bug (real):** `load_moe_heatmap` tokenized wrong - `find_first_not_of(" \t", colon+1)` lands *on the score digits* (they sit right after the colon), so every subsequent "expert" was actually the previous score. Symptom: out-of-range "experts" 567/526/657 (those are scores) and all mass collapsing into part 0, so every layer ranked hot parts `0 1 2 3`. Found via a `-v` smoke run (all layers `hot parts = 0 1 2 3`) + a standalone parse replica compiled in the container. Fixed: skip the layer number to the first token, and advance past the score via `find_first_of(" \t", colon+1)`.
- **Scores made relative (user-driven):** raw invoke counts are not comparable across sessions of different sizes, and a long session would dominate a merged heatmap. Now the **dump normalizes each layer's scores to sum to 1** (a routing distribution) and writes them as **float** (`std::fixed`/`setprecision(6)`); the parser reads them with `atof` (handles both int and float, so old raw files still parse) and accumulates in `double`. Placement only ranks within a layer, so it is invariant to the scaling; the win is comparability + round-trip (dump output is a valid `--moe-heatmap` input).
- **Merge script** (`merge_moe_heatmaps.py`) now normalizes each input layer to sum 1 then takes a **weighted average** (equal weight by default) -> output sums to 1 per layer = valid input. Equal weight (not token-proportional) so a short session of a use case the user cares about is not disqualified by having fewer total tokens.
- **No unit test added (decided with user):** the codebase has a good lightweight harness (`tests/testing.h`, 52 tests) but the parse/rank logic lives in `llama_model_base`/`llama_context` methods; `src/` headers are `PRIVATE` to the `llama` lib, so a test can't cleanly include them without moving model-specific logic into `common/` (category mismatch) or fiddling include paths. Not worth it - validating via the standalone parse replica + the A/B instead.
- **M13 added (last milestone):** the approach does NOT work on all MoE models today - the DUMP is broad (shared MoE builder emits `ffn_moe_topk`) but the PLACEMENT (per-expert part split) exists only in `qwen4exp.cpp`. M13 = port the split to other MoE archs, first target **gemma4** (same splittable 3D expert layout; nice models worth CPU-offloading), extracting a shared interface rather than duplicating.
- **Clean dual-GPU A/B re-run (fixed parser + normalized heatmap, same cycle):** A uniform K=8 `10.4 pp / 3.6 tg`; B heatmap frac 0.5 `16.1 pp / 4.6 tg` (4/8 parts on local GPU); C heatmap frac 0.375 `14.1 pp / 4.2 tg` (3/8 parts). Heatmap placement beats uniform on both pp and tg. Real binary confirms `MoE heatmap from ... 4/8 parts per layer` and 0/96 hot-parts lines collapsed to `0 1 2 3` (bug gone). The buggy run had C at `4.7/3.2` (worse than uniform) because the bad parse dumped all mass into parts 0-2; fixed, C now beats A. Absolute numbers run high vs the old `5.8/3.4` uniform ref (thermal variance) - the in-cycle A<B<C ordering is the clean result. B also beats the Q3-trace `-ot` hot-50 ref (`9.4/4.2`). 27B stopped for the cycle, restarted (active).
 - **Next:** commit+push the parser fix + relative-score redesign + merge script + docs to `nilo85`. DONE: pushed `61e67a78b` (src/llama-context.cpp, src/llama-model.cpp, src/llama-model.h).

### 2026-10-02 - M13: shared expert-split interface extracted + gemma4 wired
 - **Shared interface (the M13 core):** extracted the qwen4exp-only per-expert part-split orchestration into a new `llama_model_base::create_expert_split_tensors(layer, il, n_embd, n_ff_exp, n_expert)` (decl in llama-model.h, def in llama-model.cpp). It reads `LLAMA_EXPERT_SPLIT`; K<=1 -> full tensors (via `create_tensor_gate_up_exps`, which already handles the fused `gate_up` vs separate `gate`+`up` layout) and returns false; K>1 -> skips the full tensors, builds the `*_parts` + `ffn_expert_part_offsets` (same byte-offset/stride math as before, via `create_expert_part` + `get_expert_split_buft`), nulls the full tensors, returns true. The rest of the split machinery was ALREADY shared/arch-agnostic (`create_expert_part`, `get_expert_split_buft`, `load_moe_heatmap` in `llama_model_base`; `*_parts`/offsets in `llama_layer`; `build_moe_ffn` `*_parts`/offsets params with nullptr defaults; base `load_tensors` reads `LLAMA_EXPERT_SPLIT` + calls `load_moe_heatmap` for every arch) - only the tensor-creation orchestration + the graph `_parts` wiring were per-arch. So generalizing = call the new method in `load_arch_tensors` + pass the `_parts` to `build_moe_ffn` in the graph.
 - **qwen4exp:** replaced the ~75-line inline split block with the one-line `create_expert_split_tensors(...)` call (verbatim move, `ml.`->`ml->`).
 - **gemma4 (first target):** replaced the manual `gate_up`->`gate`+`up` fallback + `down` creation with the shared call, and appended `selected_experts_in=nullptr` + the 8 `*_parts` + `ffn_expert_part_offsets` args to its `build_moe_ffn` (arg order matches qwen4exp; `probs_in` stays `logits` as before). Non-split behavior is byte-for-byte preserved (`create_tensor_gate_up_exps` flags=0 == the old manual fallback).
 - **Build:** incremental SYCL clean (0 errors).
 - **qwen4exp regression (single GPU, 27B stayed up):** `LLAMA_EXPERT_SPLIT=8` + `--moe-heatmap hm-mix.txt` frac 0.5 loads + generates (`3.6 pp / 2.1 tg`); `-v` confirms `MoE heatmap from ... 48 layers x 512 experts, 4/8 parts per layer on local GPU` + per-layer `*_part` creation (CPU buft for CPU layers). Refactor is behavior-preserving.
 - **gemma4 validation:** compile + code-validated only (no gemma4 GGUF in the local HF cache); wiring matches the working qwen4exp pattern. Runtime A/B for gemma4 is future (needs a model).
 - **Next:** commit+push M13 (shared interface + qwen4exp refactor + gemma4 wiring) to `nilo85`. DONE: pushed `b41b9c816`.

### 2026-10-02 - gemma4 model cached + M14 single-mechanism design locked (docs first)
- gemma-4-26B-A4B-it (gemma4 arch, 26B total / 4B active MoE) downloaded by user: `unsloth/gemma-4-26B-A4B-it-GGUF` UD-Q4_K_XL (16GB) + assistant-downloaded `MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf` (441MB), snapshot `c099eb48e663fd284577b04978a94ffccb261841`, no partials. Runtime test now unblocked.
- User design review of the M10-M13 surface found TWO enabling paths (`LLAMA_EXPERT_SPLIT=K` env + flags) and TWO dump formats (`--dump-routing` raw trace vs `--moe-heatmap-dump` normalized). Verdict: "its critical we only have a single mechanism"; `--dump-routing` is "technical debt and should be removed"; `LLAMA_EXPERT_SPLIT` is opaque ("what is that even???").
- Design iteration (assistant proposals corrected by user, final LOCKED):
  1. Enabler = ONLY an explicitly-set `--moe-heatmap-fraction F`. File and dump never enable the split (file-without-fraction warns + ignores). Rationale (user): the fraction is the only knob expressing *how much* to offload; letting the file enable it would offload an implicit 0.5 the user never chose.
  2. F is the literal per-layer GPU expert share: `N_gpu = round(F*n_expert)` (0 = all CPU, 0.1 = 10%, 0.075 = 7.5%).
  3. K = internal, derived: `clamp(round(1/F), 2, 32)`; never user-visible (assistant's fixed-K=8 proposal rejected as arbitrary).
  4. Unification (user: "No, its exactly the same!!"): seed vs heatmap are ONE loop over an *ordering* - no file = identity seed order (small swappable `make_seed_order` fn - hook to try random/other seeds later); file = file order. Placement: first `N_gpu` in the ordering = hot set; K-way even split; rank parts by hot count; hottest `clamp(round(K*F),0,K)` parts -> local GPU (remote-GPU spill gone); rest CPU; `-ot` still wins.
  5. File semantics (user): "The heatmap file should still be sorted! so llama only cares about the order they come in, the score is JUST there so we can mix heatmaps" -> core parser keeps expert order, ignores score values; scores remain in the format for the merge script only.
  6. Four user scenarios to document (done above): initial seed / use existing heatmap / tune heatmap (+dump capture) / merge script mixing with equal and weighted examples.
- Provenance check before cutting (user rule: never remove master features): `git grep` base `19e28a277` + `master` = 0 hits for `dump_routing`, `LLAMA_EXPERT_SPLIT`, `moe_heatmap`; all ours (`30b6b6275`+`e2482ec81` dump-routing; `6a9e94689`/`bfa57f78f`/`b41b9c816` split; `bfa57f78f`/`be86ddb99`/`61e67a78b` heatmap). Safe.
- `merge_moe_heatmaps.py` unchanged by M14 (still normalizes + weighted-average; output stays sorted -> still order-meaningful for the core).
- Heatmap home: `/home/niklas/b70_opt/heatmaps/` (user: "save the heatmaps for all our models in the same folder as the research"); `qwen3.8-flash-next.txt` saved from `hm-mix.txt`; gemma4 heatmap to be dumped + saved here during the M14 test.
- **Next:** implement M14 on `us-otgen-expert-ot` (cut dump-routing + env var; model-param-driven split with sentinel fraction; `make_seed_order`; order-only parse; K from F; unified hot-set loop), build, then gemma4 A/B (unsplit vs seed vs qwen-mix N/A -> dumped gemma heatmap; parity check) + qwen4exp regression with the new flags, heatmaps saved to b70_opt/heatmaps, commit+push nilo85.
- Handover prompt written for post-compaction resume: `/home/niklas/b70_opt/handover.md` (self-contained: locked design, code checklist w/ anchors, Path E build cmd, test plan, env rules, finish steps). Regenerate it whenever state moves.

### 2026-10-02 - M14 implementation + first full test suite (qwen pass, gemma split BUG found)
- M14 code DONE on `us-otgen-expert-ot` (10 modified files, uncommitted, built clean): `--moe-heatmap-fraction` is the sole enabler (sentinel -1 = unset in common params -> cparams); `--moe-heatmap`/`--moe-heatmap-dump` never enable; `--dump-routing` + `LLAMA_EXPERT_SPLIT` cut; `setup_moe_split` reads `params.moe_heatmap_fraction`, derives `K=clamp(round(1/F),2,32)`, seed = `make_seed_order` identity when no file; core parser keeps per-layer expert ORDER, ignores scores; unified hot-set loop (`moe_hot_parts[il][p]` -> `get_expert_split_buft`: local GPU else CPU).
- Test suite v2 (`/tmp/opencode/test_m14.sh`, run ~3 min warm-cache): qwen F=0.25 (K=4): base/seed/file/verbose ALL load+generate, generated text IDENTICAL across the four (clean-text diff after stripping spinner/telemetry - earlier "MISMATCH" was a false positive from t/s stats); baseline `-ngl 999 -cmoe` kept dense placement matched. qwen-hmonly (heatmap, no fraction): EXIT=134 OOM - EXPECTED, no split -> all experts on GPU; also confirmed the warn+ignore path is silently skipped (setup_moe_split only runs when fraction set - the WARN at llama-model.cpp:1630 only fires inside it; acceptable: fraction unset = feature off).
- **GEMMA SPLIT BUG (blocker):** gemma-base (unsplit, F=0.5 absent) + dump EXIT=0; gemma seed/file/verbose (F=0.5, K=2) FAIL at load: `done_getting_tensors: wrong number of tensors; expected 658, got 628` (exactly -30 = 30 MoE layers).
- **Root cause (fully diagnosed):** gemma4 UD-Q4_K_XL stores a per-expert F32 scale as SEPARATE tensor `blk.N.ffn_down_exps.scale` ne={n_expert} (Q2_K weight + IQ1_S gate_up - "UD" mixed quant puts per-expert scale outside the block). The full-tensor scales are created by a GENERIC scale pass in llama-model.cpp (load pass after load_arch_tensors): `if (!layer.ffn_down_exps_s && layer.ffn_down_exps) create_tensor(FFN_DOWN_EXPS,"scale",{n_expert},TENSOR_NOT_REQUIRED)` (llama-model.cpp:1687-1692). The SPLIT path nulls `layer.ffn_down_exps` -> the gate never fires -> the 30 file scale tensors are never created -> count fails. qwen passes because Q3_K keeps scales in-block (no separate .scale tensors in its GGUF).
- Loader mechanics verified: `n_tensors` snapshot at construction = file count (weights_map.size(), llama-model-loader.cpp:718); TENSOR_SKIP'd tensors that exist in file DO increment n_created (line 1184); parts increment n_part_tensors, NOT n_created; parts get file-mapped via `orig_w` lookup (line 1298) and registered in weights_map under the part name (line 1339) - so a SKIP-then-part on the scale works (original SKIP'd = counted; scale part gets correct file slice via orig_w->offs + e0*stride).
- Graph side ALREADY READY: `build_moe_ffn` parts branch passes `(*down_exps_s_parts)[p]` etc. to `build_lora_mm_id` (llama-graph.cpp:2257/2263/2266/2280) which applies `w_s` as per-expert row-scale via get_rows(ids) (line 1561-1567, `w_s->ne[0]`=experts-in-part); ids_part are LOCAL (0..n_e-1) -> a contiguous 1D scale slice per part is exactly right. `llama_layer` has `ffn_*_exps_s_parts` members (llama-model.h:354-357); gemma4 already passes them. `get_expert_split_buft` keys on (il, part_idx) -> scale part lands on same buft as its weight part.
- **FIX A (agreed, unblock now):** in `create_expert_split_tensors` K>1 path: SKIP all four `("scale", {n_expert})` originals (NOT_REQUIRED|SKIP -> counted only if in file), fetch scale metas/strides (`nb[0]` of 1D scale), clear + populate the four `*_exps_s_parts` vectors in the parts loop (`create_expert_part(tn(...,"scale",il), {n_e}, 0, "<scale>.part<p>", e0*scale_stride, get_expert_split_buft(il,p,...))`). Only the down scale exists for gemma4; others no-op. qwen4exp unchanged (no scale tensors -> metas null -> zero new parts). Context size safe: loader adds `n_layer*256` slack when parts used (llama-model-loader.cpp:1132).
- **USER RATIONALE (design principle, verbatim intent):** "If the new shared code is too opinionated, can it be refactored to support per model logic? key is we want to minimize effort to make sure all moe models can be warm/cold offloaded to cpu. Not necessarily force a new super central way of loading models."
- **Agreed direction (PLAN B, MANDATORY NEXT STEP after A) - REFINED to a BUILDER PATTERN (user's design idea):** the feature is already model-agnostic where it matters - heatmap load / moe_hot_parts / get_expert_split_buft / part mechanism (create_expert_part, part_spec, offsets) / build_moe_ffn part consumption stay central, DO NOT touch. The OPINIONATED part is only the tensor-layout logic in create_expert_split_tensors. User: "a builder pattern ... gives each model full control, but actual placement is controlled by the builder". Agreed shape: a small builder (member of llama_model_base so it reaches get_expert_split_buft/moe_hot_parts) with `add(tn, suffix, ne_full, out_parts&)` (model registers WHAT: which expert tensors, fused-vs-separate, scales, and which layer.*_parts vector they fill) + `build()` (builder owns the HOW: SKIP originals, K-way split, offsets, hot->local-GPU/cold->CPU placement via get_expert_split_buft, writes parts into the model-provided vectors). A thin standard-MoE preset feeds the builder the usual tensors (down + gate_up/gate+up + scales) so common models stay a one-liner; unusual models drive the builder directly from their load_arch_tensors - per-model quirks never require core edits. Inside build() the generalization: expert dim = LAST logical dim, per-expert stride = nb[last], part ne = full ne with last dim -> n_e (covers 3D weights and 1D scales). Keep the builder MINIMAL (plain struct, add+build; no fluent chaining, no spec objects, no templates) per llama.cpp simplicity. Fix A is intentionally implemented inside the preset so B is a mostly-mechanical extraction.
- Also this session: llama-cli REPL EOF infinite-loop bug found (unrelated) -> user story `us_cli_repl_eof.md`; suite now pipes `echo /exit |` as workaround.
- **Next:** land Fix A -> rebuild -> re-run gemma trio (seed/file/verbose) + gemma parity vs base + save gemma heatmap to b70_opt/heatmaps/ -> docs + commit/push nilo85 -> THEN PLAN B refactor (primitive + preset) as separate commit.

### 2026-10-02 - Fix A landed: gemma4 split scale handling (blocker cleared)
- Implemented in `llama_model_base::create_expert_split_tensors` (src/llama-model.cpp) K>1 path, mirroring the generic scale pass (gate/down/up, NOT gate_up - the generic pass at :1684-1692 never creates a gate_up scale, so the split path stays a faithful split of the unsplit path):
  1. SKIP the 3 `("scale",{n_expert})` originals with `TENSOR_NOT_REQUIRED | TENSOR_SKIP` (counted if in file -> fixes the 658-vs-628 count; no-op if absent).
  2. Fetch scale metas via `get_tensor_meta`; per-expert byte stride = `nb[0]` (1D F32 scale).
  3. Clear the 3 `ffn_*_exps_s_parts` vectors.
  4. In the parts loop, for each present scale create a 1D `{n_e}` part at `e0*stride` via `create_expert_part` + `get_expert_split_buft(il,p,...)` -> same buft as the weight part.
- Loader semantics verified: TENSOR_SKIP (llama-model-loader.cpp:1179) increments n_created + returns nullptr but keeps the weights_map entry (with offs) so a part maps a slice of the SKIP'd original (part path :1298/:1338). Context safe via the `n_layer*256` part slack (:1133).
- Build: SYCL clean (0 errors/warnings).
- Test suite (single GPU renderD130, 27B stayed up on renderD129): qwen base/seed/file/verbose ALL EXIT=0, clean generated text IDENTICAL across the three (raw diff "MISMATCH" is the spinner false positive; awk-extracted answer text matches byte-for-byte). qwen-hmonly EXIT=134 OOM = expected (no fraction = feature off). **gemma base/seed/file/verbose ALL EXIT=0** (was failing 658-vs-628). gemma seed text == base text (identical); gemma file text differs but is coherent (placement-dependent FP branching, NOT a scale bug - output is sensible, not garbage).
- `-v` verbose gemma run confirms the fix end-to-end: `setup_moe_split: MoE split (seed order): 30 layers x 128 experts, 1/2 parts per layer on local GPU`; scale parts created `blk.N.ffn_down_exps.scale.part0 type=f32 buft=SYCL0 bytes=256 offs=0` / `part1 buft=CPU offs=256` (64 experts x 4B each, following the weight parts' GPU/CPU placement). Weight parts split 50/50 as before.
- gemma heatmap dumped + saved to `/home/niklas/b70_opt/heatmaps/gemma-4-26b-a4b.txt` (30 layers, sorted, sums to 1/layer).
- **Next:** DONE - committed + pushed to `nilo85` at `340f6abf2` (M14 single-mechanism + gemma4 scale fix, 10 files). REMAINING: PLAN B refactor (primitive + preset) as a separate commit.

### 2026-10-02 - PLAN B design refined: builder pattern (user's idea, agreed) [SUPERSEDED same day by the TENSOR_EXPERT flag design - see next entry; kept as design history]
- After Fix A landed, user proposed: "I wonder if a builder pattern would not play nice, that gives each model full control, but actual placement is controlled by the builder". Agreed - it expresses the separation better than a bare per-tensor primitive function.
- Agreed concrete shape (see PLAN B paragraph in the Fix A entry for the full version):
  - Builder = small struct, member of `llama_model_base` (needs reach to `get_expert_split_buft` + `moe_hot_parts`). API: `add(tn, suffix, ne_full, out_parts&)` + `build()`.
  - Model = full control of the WHAT: which expert tensors (names/suffixes), fused-vs-separate choice, which scales, and which `layer.*_parts` vector each fills. Lives in the model's `load_arch_tensors` (where that layout knowledge already exists for the K<=1 path).
  - Builder = owns the HOW: SKIP each original (NOT_REQUIRED|SKIP), K-way even split, part offsets `p*(n_expert/K)`, placement via `get_expert_split_buft(il,p,name)` (hot->local GPU, cold->CPU), writes parts into the model-provided vectors.
  - Standard-MoE preset = thin helper registering down + (gate_up | gate+up) + down/gate/up scales, then build() -> common models keep a one-liner.
  - Inside build(): expert dim = last logical dim; stride = `nb[ne_full.size()-1]`; part ne = ne_full with last element -> n_e. One code path for 3D weights and 1D scales.
  - MINIMAL: plain struct, add+build only. No fluent chaining, no spec objects, no virtuals/templates (llama.cpp simplicity rule).
- Implementation gotchas captured in handover (create_expert_part needs a vector-ne overload or conversion; get_tensor_meta takes const char*; K<=1 path stays byte-for-byte; reserve_part_tensors unchanged - n_layer*256 context slack covers scale parts).
- **Next:** implement the builder refactor on `us-otgen-expert-ot`, rebuild (Path E), re-run full suite (qwen 4x EXIT=0 + identical text, hmonly OOM, gemma 4x EXIT=0, seed==base, file coherent) + one `-v` gemma run (expect `scale.part0 ... SYCL0 offs=0` / `part1 ... CPU offs=256`), then docs + commit/push nilo85.

### 2026-10-02 - PLAN B FINAL design: TENSOR_EXPERT flag (user's idea; supersedes the builder)
- Design thread (all user-driven, verbatim intent):
  1. User: "should we have expert split 'builder' or is there another more central aspect in llama we can extend? i.e if there is a pattern to add the experts, maybe we just need to make the hot/cold split there, and model does not really need to be aware at all?"
  2. Upstream comparison (merge-base origin/master @ 19e28a277): the model files carry exactly TWO breaks: (a) load_arch_tensors routed expert creation through create_expert_split_tensors, (b) build_moe_ffn call passes 9 *_parts args (nullptr-defaulted, other models unaffected).
  3. First pivot: a central post-pass beside the generic scale pass (load side 100% upstream). REJECTED by user: "if rig has more ram, it might not be mmapped! also, in case of not alot of ram, we dont want GPU tensors to also take system ram!" - orphaning is real memory, not theoretical: the full expert tensors, once created, live in the ggml context forever; in non-mmap the per-buft buffer is sized from context tensors (llama-model.cpp:1880 ggml_backend_alloc_ctx_tensors_from_buft), so dead fused copies = ~70% of an MoE's weights, plus host staging spikes for GPU-placed tensors (loader:1799).
  4. Re-considered builder (SKIP = full tensor never enters context -> clean in all load modes). Then user: "alternativly, can we extend the current core pattern, to work with what we need? i.e. is it just a matter of sending a boolean if its offloadable?" -> YES, via the existing create_tensor flags bitfield.
  5. User: "for b flag TENSOR_EXPERT, would that not be a better fit?" -> YES. FACT vs POLICY: a model knows unconditionally "this is an expert tensor"; whether to split today is runtime policy (moe_split_k from --moe-heatmap-fraction) the model must not know. TENSOR_EXPERT marks the property (like TENSOR_READ_LAZY), the core applies the consequence. TENSOR_EXPERT_SPLIT would leak policy into the contract.
  6. Graph side: user: "sending the parts sounds like deep coupling, could we solve it by exposing whatever missing info, with a primitive, example, if we miss the layer, maybe it makes sense to make the layer info part of the graph model?" Precedent exists: build_qkv ALREADY takes `const llama_layer &` (llama-graph.h:1080; models call `build_qkv(model.layers[il], cur, ...)`).
  7. Full layer-based build_moe_ffn (replacing the weight args) audited against ALL 57 files / 69 call sites: exactly 2 are load-bearing and CANNOT be expressed via the layer: grovemoe.cpp:153 (second MoE block per layer uses the ffn_*_chexps field group at a different n_expert = n_chunk_expert) and deepseek4.cpp:1295 (bias chosen conditionally at graph time: ffn_exp_probs_b / ffn_exp_probs_b_vl / nullptr + computed selected_experts). A pure layer replacement would silently break both -> user's control concern confirmed.
  8. GRAPH SIDE LOCKED = B+ (user-approved): keep the weight/bias/n_expert args (that is the model control); REPLACE the 9 *_parts args with ONE trailing `const llama_layer *` arg (nullptr default); builder uses the layer's parts when non-empty, else the full-tensor args. qwen4exp/gemma4 pass the layer (1 arg); the other 55 models untouched. Tech debt: minimal - the 9-arg debt is REMOVED; weight args stay pinned by the quirky models, so B+ is the end state, not a stepping stone.
  9. USER DECISION (2026-10-02, final): user first flinched at the layer arg ("this is DEEP COUPLING - the signature is all primitives, now you send the whole layer") and asked to KEEP the 9 *_parts args and only add the layer (minimal additive), then "if there is a better primitive to send instead of layer do that". On checking origin/master: the 9 *_parts args are a FORK addition (master's build_moe_ffn ends at selected_experts_in; 0 hits for *_parts/expert_part_offsets in master's graph). So KEEPING them = +9 args vs master; B+ (drop them, add the layer) = +1 arg vs master. User: "lets go as originally planned then! scratch my latest decision about the 9 param" -> B+ re-confirmed as the minimal-diff-against-master choice. The layer arg is a KNOWN HACK (deep coupling to the whole layer; builder only reads ffn_*_exps_parts + ffn_expert_part_offsets) to be revisited later - do NOT "fix" it now.
- LOCKED DESIGN:
  - `TENSOR_EXPERT`: new bit in llama_model_loader create_tensor flags; mirrored as const int on llama_model_base.
  - `setup_moe_split` ARMS the loader: enabled, K, per-layer hot set + expert order. Placement logic (hot -> owning local GPU, cold -> CPU, honoring tensor_buft_overrides) moves INTO the loader (replaces model-base get_expert_split_buft). reserve_part_tensors stays.
  - Loader create_tensor branch on TENSOR_EXPERT: tensor absent + NOT_REQUIRED -> silent no-op (must short-circuit BEFORE the skip path's warn/count logic); split off -> normal create; split on -> SKIP full (n_created++, weights_map entry kept), create K byte-slice parts (expert dim = last logical dim; stride = nb[last]) placed per the hot rule, return nullptr for the full tensor, stash parts in a loader map keyed by original tensor name.
  - llama_model_base after load_arch_tensors and BEFORE the generic scale pass: DRAIN the stash into layer.ffn_*_exps_parts + ffn_expert_part_offsets (standard-MoE name -> vector preset; fused-vs-separate probed via get_tensor_meta). Drain before scale pass so the pass sees filled parts.
  - Generic scale pass: gates on full tensors only (`&& layer.ffn_down_exps`) are DEAD under split (that was the Fix A bug) -> gates become `(layer.ffn_down_exps || !layer.ffn_down_exps_parts.empty())`; per-expert scale creates tagged TENSOR_EXPERT -> loader slices scales into *_s_parts. This SUBSUMES Fix A (its bespoke skip-and-rebuild gets retired).
  - Model files: git-revert to origin/master, then (load) expert creates pass TENSOR_EXPERT (qwen4exp via create_tensor_gate_up_exps' existing flags arg + down create; gemma4 gate_up create with TENSOR_EXPERT|TENSOR_NOT_REQUIRED, presence-based fallback gate/up, down) and (graph) the one layer arg. Final diff vs master = flag tokens + 1 graph arg. create_expert_split_tensors / create_expert_part / get_expert_split_buft retired from model base; placement moves to the loader (needs per-layer local buft handed over by setup_moe_split, since dev_layer lives in the model base).
  - Graph side = B+: build_moe_ffn (both overloads) drops the 9 *_parts args, gains one trailing `const llama_layer * layer = nullptr`; consumption reads parts/offsets from the layer when provided and non-empty, else uses the full-tensor args. llm_graph_context already references llama_layer (build_qkv), so no new dependency.
  - Model graph call sites: qwen4exp/gemma4 append the layer pointer (`&l` / `&model.layers[il]`) to the build_moe_ffn call - ONE extra arg, the only graph-side diff vs master.
- SEMANTICS LANDMINE (must handle in impl): under TENSOR_EXPERT a nullptr no longer means absent (split full tensors return null BY DESIGN). Null-based presence fallbacks break: create_tensor_gate_up_exps internal fallback AND gemma4.cpp's inline `if (layer.ffn_gate_up_exps == nullptr)` separate-gate/up fallback would then create required-but-absent tensors -> throw. Fix: key those fallbacks on meta presence (get_tensor_meta), not null.
- Audit vs all drawbacks raised across the design iterations:
  - FIXED: non-mmap dead buffer space + staging spike (full never created); weights_map-erase trick unnecessary; builder's whole-call divergence -> one flag token on upstream calls; layout knowledge centralized (one standard-MoE drain preset, not per-model helpers); Fix A subsumed by central scale slicing.
  - ACCEPTED, UNFIXABLE: build_moe_ffn keeps the weight/bias/n_expert args (grovemoe chexps + deepseek4 conditional bias need them - audit item 7); model files carry flag tokens + 1 layer arg (+ presence-based fallback tweak in gemma4/helper).
  - NEW small costs: loader stash map + drain pass; internal dynamic-ne construction in the loader branch (parts ne is runtime-computed, not initializer_list-friendly); per-layer local buft handed loader by setup_moe_split.
  - REMOVED vs the pre-B+ plan: the 9 *_parts args (replaced by the layer arg).
- Behavior parity: resulting tensors/placements/counting are IDENTICAL to the validated 340f6abf2 split; the M14 suite must reproduce the same matrix (qwen base/seed/file/verbose EXIT=0 + identical text; qwen-hmonly EXIT=134; gemma all 4 EXIT=0; gemma seed text == base, file coherent; gemma -v shows scale parts SYCL0 offs=0 / CPU offs=256).
- Next: implement per this entry (handover.md was consumed + deleted at session start; its build/test commands and landmines remain valid - test script /tmp/opencode/test_m14.sh, heatmap /tmp/opencode/hm-qwen.txt).

### 2026-10-02 - B+ implementation LANDED (TENSOR_EXPERT flag + graph layer arg; M14 suite ALL PASS)
- Implemented per the LOCKED DESIGN above, on `us-otgen-expert-ot`:
  - Loader (`llama-model-loader.{h,cpp}`): `TENSOR_EXPERT = 1 << 6` flag + arm state + `expert_parts` stash; `create_tensor` TENSOR_EXPERT branch (absent+NOT_REQUIRED -> silent no-op; split on -> SKIP full, create K byte-slice parts, expert dim = last logical dim, stride = nb[last], place per hot rule, stash by original name, return nullptr); `expert_split_buft` placement (hot -> owning local GPU, cold -> CPU).
  - Model base (`llama-model.{h,cpp}`): `const int TENSOR_EXPERT` mirror; `setup_moe_split` arms the loader; `drain_expert_parts` (after load_arch_tensors, before the generic scale pass) moves the stash into `layer.ffn_*_exps_parts` + `ffn_expert_part_offsets`; generic scale-pass gates become `(layer.ffn_down_exps || !layer.ffn_down_exps_parts.empty())` (the full-tensor-only gates were DEAD under split - the Fix A bug) and per-expert scales are created tagged TENSOR_EXPERT so the loader slices them into `*_s_parts`; `create_expert_split_tensors`/`create_expert_part`/`get_expert_split_buft` retired; `create_tensor_gate_up_exps` fallback keyed on `get_tensor_meta` presence (not null - a split gate_up returns null by design).
  - Graph (`llama-graph.{h,cpp}`) = B+: both `build_moe_ffn` overloads drop the 9 `*_parts`/`expert_part_offsets` args, gain ONE trailing `const llama_layer * layer = nullptr`; the implementation derives the part pointers from the layer.
  - Model files: `qwen4exp.cpp` + `gemma4.cpp` reverted to origin/master then minimally patched (load: expert creates pass TENSOR_EXPERT via `create_tensor_gate_up_exps` + down create; graph: append `nullptr` (selected_experts_in) + `&model.layers[il]`). Final diff vs master = flag tokens + 1 graph arg per model.
- BUG found+fixed (first test run): the graph builder's local derivation set ALL part pointers to `&layer->...` unconditionally (non-null). A fused-gate_up layer (qwen) leaves `ffn_gate_exps_parts`/`ffn_up_exps_parts` empty, and a separate-gate+up layer (gemma) leaves `ffn_gate_up_exps_parts` empty - so the non-null-but-empty pointer tripped `GGML_ASSERT(gate_up_exps_parts->size() == n_parts)` (0 != K) and aborted every split run at graph build. FIX: gate each derived pointer on `!layer->ffn_*_exps_parts.empty()` (and the offsets on `!layer->ffn_expert_part_offsets.empty()`) so a layer exposes only the part vectors it actually has; the fused-vs-separate assert (`gate_up || (gate && up)`) then holds by construction.
- Minor: `drain_expert_parts` lambda param was typed `LLM_TENSOR` (a typo) - the enum is `llm_tensor` (llama-arch.h:441); fixed.
- Build: SYCL clean (0 errors).
- M14 suite (`/tmp/opencode/test_m14.sh`, single GPU renderD130, 27B held on renderD129): qwen base/seed/file/verbose ALL EXIT=0, answer text byte-identical across the three; qwen-hmonly EXIT=134 OOM = expected (no fraction -> feature off, all experts on GPU). gemma base/seed/file/verbose ALL EXIT=0; gemma seed text == base (identical); gemma file text differs but coherent (placement-dependent FP branching, not a bug). gemma `-v` confirms end-to-end: `setup_moe_split: MoE split (seed order): 30 layers x 128 experts, 1/2 parts per layer on local GPU`; `blk.N.ffn_down_exps.scale.part0 buft=SYCL0 offs=0` / `part1 buft=CPU offs=256` (64 experts x 4B, scale parts follow the weight parts' GPU/CPU split).
- Behavior parity: the re-plumbed split reproduces the validated 340f6abf2 matrix exactly - the TENSOR_EXPERT flag path is behavior-preserving.
- **Landed + pushed** to `nilo85` at `683779ead` (8 files, +228/-328 vs 340f6abf2). Tree clean.
