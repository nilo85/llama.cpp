#!/usr/bin/env python3
"""Merge MoE heatmap files (core --moe-heatmap-dump output) into one.

Format (input == output): `il=<layer> <expert>:<score> ...`, `#` comments skipped.
Each input layer is normalized to sum to 1 first, so files from sessions of different
sizes are comparable and a short session is not drowned out by a long one. The layers
are then combined as a weighted average (default: equal weight per file). The output is
also normalized (sums to 1 per layer) and is a valid --moe-heatmap input: placement
ranks each layer's K parts by aggregate score.
"""

import argparse
import sys
from collections import defaultdict


def parse_heatmap(path):
    totals = defaultdict(lambda: defaultdict(float))
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or not line.startswith("il="):
                continue
            parts = line.split(None, 1)
            il = int(parts[0][3:])
            if len(parts) < 2:
                continue
            for tok in parts[1].split():
                if ":" not in tok:
                    continue
                expert, score = tok.split(":", 1)
                totals[il][int(expert)] += float(score)
    return totals


def normalize(totals):
    # scale each layer so its scores sum to 1 (no-op if already normalized)
    norm = defaultdict(lambda: defaultdict(float))
    for il, experts in totals.items():
        s = sum(experts.values())
        if s <= 0:
            continue
        for expert, score in experts.items():
            norm[il][expert] = score / s
    return norm


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help="heatmap files to merge")
    ap.add_argument("-o", "--output", default="merged-heatmap.txt", help="merged output file")
    ap.add_argument("--weights", default="", help="comma-separated per-file weights (default: all 1)")
    args = ap.parse_args()

    weights = [1.0] * len(args.files)
    if args.weights:
        weights = [float(w) for w in args.weights.split(",")]
        if len(weights) != len(args.files):
            sys.exit(f"error: {len(args.files)} files but {len(weights)} weights")

    total_weight = sum(weights)
    merged = defaultdict(lambda: defaultdict(float))
    for path, w in zip(args.files, weights):
        norm = normalize(parse_heatmap(path))
        for il, experts in norm.items():
            for expert, score in experts.items():
                merged[il][expert] += w * score

    with open(args.output, "w") as f:
        f.write("# merged MoE heatmap: il=<layer> <expert>:<score> (weighted avg of normalized inputs, sum 1 per layer, sorted desc)\n")
        for il in sorted(merged):
            if not merged[il]:
                continue
            # divide by total weight -> weighted average, sums to 1 per layer
            items = sorted(((e, s / total_weight) for e, s in merged[il].items()), key=lambda kv: kv[1], reverse=True)
            f.write(f"il={il} " + " ".join(f"{e}:{s:.6f}" for e, s in items) + "\n")

    n_experts = sum(len(v) for v in merged.values())
    print(f"wrote {args.output}: {len(merged)} layers, {n_experts} expert entries")


if __name__ == "__main__":
    main()
