# us_otgen — Router-aware `-ot` pattern generator (offline tool + small core hook)

**Type:** Most is out-of-tree tooling (zero fork risk); small optional core addition to make traces first-class.
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
