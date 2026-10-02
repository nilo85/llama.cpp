#!/usr/bin/env python3
"""DEPRECATED (2026-10-02): its --dump-routing input was removed in M14 (single-mechanism
MoE heatmap design). Kept as history/fallback for old traces only; use
--moe-heatmap-fraction + --moe-heatmap / --moe-heatmap-dump instead.

Generate -ot patterns for synthetic per-expert .partN tensors from routing traces.

The trace format is the one emitted by llama-cli --dump-routing:
  il=<layer> <expert>:<count> ...

This script aggregates expert hits per layer, splits the expert axis into K
contiguous parts, ranks the parts by hotness, and emits one
`<tensor-name-pattern>=<buffer-type>` line per part/tensor.
"""

import argparse
import re
import sys
from collections import defaultdict


def parse_trace(path, counts):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("il="):
                continue
            parts = line[3:].split()
            if not parts:
                continue
            try:
                il = int(parts[0])
            except ValueError:
                continue
            while il >= len(counts):
                counts.append(defaultdict(int))
            for tok in parts[1:]:
                if ":" not in tok:
                    continue
                e_s, c_s = tok.rsplit(":", 1)
                try:
                    e = int(e_s)
                    c = int(c_s)
                except ValueError:
                    continue
                counts[il][e] += c


def proportional_targets(total, weights):
    if total <= 0:
        return [0] * len(weights)
    if sum(weights) <= 0:
        return [0] * len(weights)
    targets = [int(total * w / sum(weights)) for w in weights]
    i = 0
    while sum(targets) < total:
        targets[i % len(targets)] += 1
        i += 1
    while sum(targets) > total:
        for i in range(len(targets)):
            if targets[i] > 0:
                targets[i] -= 1
                break
        else:
            break
    return targets


def assign_parts(scores, k, hot_count, gpus, weights, local_idx, prefer_local, local_only=False):
    assign = {p: "CPU" for p in range(k)}
    if hot_count <= 0:
        return assign

    order = sorted(range(k), key=lambda p: (-scores[p], p))
    top = order[:hot_count]

    if local_only:
        # M5: hot parts stay on the layer's owning GPU, everything else CPU
        for p in top:
            assign[p] = gpus[local_idx]
        return assign

    if prefer_local and len(gpus) > 1:
        local_target = proportional_targets(hot_count, weights)[local_idx]
        local_target = max(0, min(local_target, len(top)))
        for p in top[:local_target]:
            assign[p] = gpus[local_idx]
        rem = top[local_target:]
        rem_count = len(rem)
        other_idx = [i for i in range(len(gpus)) if i != local_idx]
        other_w = [weights[i] for i in other_idx]
        if rem_count > 0 and sum(other_w) > 0:
            targets = proportional_targets(rem_count, other_w)
            idx = 0
            for p in rem:
                while idx < len(targets) and targets[idx] <= 0:
                    idx += 1
                if idx >= len(targets):
                    break
                assign[p] = gpus[other_idx[idx]]
                targets[idx] -= 1
        elif rem_count > 0:
            for p in rem:
                assign[p] = gpus[local_idx]
    else:
        targets = proportional_targets(hot_count, weights)
        idx = 0
        for p in top:
            while idx < len(targets) and targets[idx] <= 0:
                idx += 1
            if idx >= len(targets):
                break
            assign[p] = gpus[idx]
            targets[idx] -= 1

    return assign


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", action="append", required=True, help="routing trace file, may be given multiple times")
    ap.add_argument("--n-layers", type=int, default=48)
    ap.add_argument("--n-expert", type=int, default=512)
    ap.add_argument("--k", type=int, default=8, help="number of contiguous expert parts per layer")
    ap.add_argument("--layout", choices=["separate", "fused"], default="separate")
    ap.add_argument("--gpus", default="SYCL0,SYCL1", help="comma-separated GPU buffer types")
    ap.add_argument("--cpu", default="CPU")
    ap.add_argument("--hot-fraction", type=float, default=0.5, help="fraction of parts per layer placed on GPUs")
    ap.add_argument("--local-split", default="50,50", help="comma-separated weights for the GPU list")
    ap.add_argument("--prefer-local", action="store_true", help="give the layer's local GPU the hottest parts first")
    ap.add_argument("--local-only", action="store_true", help="M5: hot parts stay on the layer's owning GPU, everything else CPU (never the remote GPU)")
    ap.add_argument("--out", default="placement.ot")
    ap.add_argument("--shell-out", default="", help="also write a bash file defining OT_ARGS")
    args = ap.parse_args()

    if args.n_expert % args.k != 0:
        sys.exit(f"error: --n-expert {args.n_expert} is not divisible by --k {args.k}")

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    weights = [int(x) for x in args.local_split.split(",") if x.strip() != ""]
    if len(weights) != len(gpus):
        weights = [1] * len(gpus)

    counts = [defaultdict(int) for _ in range(args.n_layers)]
    for trace in args.trace:
        parse_trace(trace, counts)
    while len(counts) < args.n_layers:
        counts.append(defaultdict(int))

    hot_count = int(args.k * args.hot_fraction)
    hot_count = max(0, min(hot_count, args.k))
    part_size = args.n_expert // args.k

    lines = []
    for il in range(args.n_layers):
        hot = [0] * args.n_expert
        for e, c in counts[il].items():
            if 0 <= e < args.n_expert:
                hot[e] += c

        if not counts[il]:
            assign = {p: args.cpu for p in range(args.k)}
        else:
            scores = []
            for p in range(args.k):
                scores.append(sum(hot[p * part_size:(p + 1) * part_size]))
            local_idx = min(il * len(gpus) // args.n_layers, len(gpus) - 1) if gpus else 0
            assign = assign_parts(scores, args.k, hot_count, gpus, weights, local_idx, args.prefer_local, args.local_only)

        for p in range(args.k):
            dev = assign[p]
            if args.layout == "separate":
                names = [
                    f"blk.{il}.ffn_down_exps.weight.part{p}",
                    f"blk.{il}.ffn_gate_exps.weight.part{p}",
                    f"blk.{il}.ffn_up_exps.weight.part{p}",
                ]
            else:
                names = [
                    f"blk.{il}.ffn_down_exps.weight.part{p}",
                    f"blk.{il}.ffn_gate_up_exps.weight.part{p}",
                ]
            for name in names:
                lines.append(f"{re.escape(name)}={dev}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if args.shell_out:
        with open(args.shell_out, "w", encoding="utf-8") as f:
            f.write("# generated by generate_expert_ot.py\n")
            f.write("OT_ARGS=()\n")
            f.write(f'while IFS= read -r line; do\n')
            f.write('  case "$line" in ""|\\#*) continue ;; esac\n')
            f.write('  OT_ARGS+=(-ot "$line")\n')
            f.write(f'done < {args.out}\n')

    print(f"wrote {args.out} with {len(lines)} overrides", file=sys.stderr)
    if args.shell_out:
        print(f"wrote {args.shell_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
