"""Aggregate multi-seed eval CSVs into a paper-ready table with mean ± std.

Reads logs/multiseed_<model>_<dist>_seed<N>/baseline.csv for each
(model, dist, seed) combination and computes:
  - BR_teacher composite (goal × (1-coll) × (1-fall)) — mean ± std across seeds
  - Best-fixed composite (max across all B0/B1/B2 configs) — mean ± std
  - Δ = BR - best_fixed — does adaptive beat hand-tuned?
  - Per-axis breakdown (safety, completion, efficiency, margin)

Usage (after pulling logs to laptop):
  rsync -avz --include='*/' --include='*.csv' --exclude='*' \\
    'chrisliang@130.64.84.163:~/Desktop/safety-go2/IsaacLab/logs/multiseed_*' \\
    ./data_from_lab/
  python3 scripts/aggregate_multiseed.py
"""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parent.parent / "data_from_lab"
MODELS = ["v13_1", "v13_2"]
DISTS = ["trainmatch", "ood", "stressor"]
SEEDS = [42, 123, 7]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_csv_rows(path: Path):
    if not path.exists():
        return None
    return list(csv.DictReader(open(path)))


def composite(r):
    g = f(r["goal_reach_rate"]) or 0
    c = f(r["collision_rate_actual"]) or 0
    fa = f(r["fall_rate"]) or 0
    return g * (1 - c) * (1 - fa)


def safety(r):
    c = f(r["collision_rate_actual"]) or 0
    fa = f(r["fall_rate"]) or 0
    return 1 - max(c, fa)


def completion(r):
    return f(r["goal_reach_rate"]) or 0


def efficiency(r):
    return f(r.get("path_efficiency", "")) or 0


def margin(r):
    return f(r.get("mean_h", "")) or 0


def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def aggregate():
    print(f"\n{'='*78}")
    print(f"MULTI-SEED HEADLINE (BR vs best-fixed composite)")
    print(f"{'='*78}")
    print(f"{'model':>8s}  {'dist':>11s}  {'BR_μ':>6s} {'BR_σ':>5s}    "
          f"{'fix_μ':>6s} {'fix_σ':>5s}    {'Δ_μ':>6s} {'Δ_σ':>5s}    n_seeds")

    for model in MODELS:
        for dist in DISTS:
            br_vals = []
            best_fixed_vals = []
            diff_vals = []
            n_seeds = 0
            for seed in SEEDS:
                csv_path = DATA_ROOT / f"multiseed_{model}_{dist}_seed{seed}" / "baseline.csv"
                rows = load_csv_rows(csv_path)
                if rows is None:
                    continue
                br_row = next((r for r in rows if r["mode"] == "BR"), None)
                if br_row is None:
                    continue
                br_c = composite(br_row)
                fixed_cs = [composite(r) for r in rows if r["mode"] != "BR"]
                if not fixed_cs:
                    continue
                best_fixed = max(fixed_cs)
                br_vals.append(br_c)
                best_fixed_vals.append(best_fixed)
                diff_vals.append(br_c - best_fixed)
                n_seeds += 1
            br_m, br_s = mean_std(br_vals)
            fx_m, fx_s = mean_std(best_fixed_vals)
            d_m, d_s = mean_std(diff_vals)
            win_marker = ""
            if not math.isnan(d_m):
                if d_m > 0 and d_m > 2 * d_s:
                    win_marker = "  ← WIN (>2σ)"
                elif d_m > 0:
                    win_marker = "  ← lean win"
                elif d_m < 0 and abs(d_m) > 2 * d_s:
                    win_marker = "  ← LOSS"
            print(f"{model:>8s}  {dist:>11s}  "
                  f"{br_m:6.3f} {br_s:5.3f}    "
                  f"{fx_m:6.3f} {fx_s:5.3f}    "
                  f"{d_m:+6.3f} {d_s:5.3f}    {n_seeds}{win_marker}")

    print(f"\n{'='*78}")
    print(f"PER-AXIS BREAKDOWN (BR_teacher only, multi-seed mean ± std)")
    print(f"{'='*78}")
    print(f"{'model':>8s}  {'dist':>11s}  "
          f"{'safety':>13s}  {'complete':>13s}  "
          f"{'efficiency':>13s}  {'margin':>13s}")

    for model in MODELS:
        for dist in DISTS:
            metrics = {
                "safety": [], "completion": [],
                "efficiency": [], "margin": [],
            }
            for seed in SEEDS:
                csv_path = DATA_ROOT / f"multiseed_{model}_{dist}_seed{seed}" / "baseline.csv"
                rows = load_csv_rows(csv_path)
                if rows is None:
                    continue
                br_row = next((r for r in rows if r["mode"] == "BR"), None)
                if br_row is None:
                    continue
                metrics["safety"].append(safety(br_row))
                metrics["completion"].append(completion(br_row))
                metrics["efficiency"].append(efficiency(br_row))
                metrics["margin"].append(margin(br_row))
            row = f"{model:>8s}  {dist:>11s}"
            for k in ("safety", "completion", "efficiency", "margin"):
                m, s = mean_std(metrics[k])
                row += f"  {m:6.3f}±{s:5.3f}"
            print(row)

    print(f"\n{'='*78}")
    print(f"PARETO CHECK — does BR achieve BETTER safety at SAME-OR-BETTER completion?")
    print(f"{'='*78}")
    for model in MODELS:
        for dist in DISTS:
            br_safety_vals, br_compl_vals = [], []
            fx_safety_vals, fx_compl_vals = [], []
            for seed in SEEDS:
                csv_path = DATA_ROOT / f"multiseed_{model}_{dist}_seed{seed}" / "baseline.csv"
                rows = load_csv_rows(csv_path)
                if rows is None: continue
                br_row = next((r for r in rows if r["mode"] == "BR"), None)
                if br_row is None: continue
                br_safety_vals.append(safety(br_row))
                br_compl_vals.append(completion(br_row))
                # best-fixed by composite
                fixed = [(r, composite(r)) for r in rows if r["mode"] != "BR"]
                if not fixed: continue
                best = max(fixed, key=lambda kv: kv[1])[0]
                fx_safety_vals.append(safety(best))
                fx_compl_vals.append(completion(best))
            if not br_safety_vals:
                continue
            br_s, _ = mean_std(br_safety_vals)
            br_c, _ = mean_std(br_compl_vals)
            fx_s, _ = mean_std(fx_safety_vals)
            fx_c, _ = mean_std(fx_compl_vals)
            verdict = ""
            if br_s > fx_s and br_c >= fx_c - 0.02:
                verdict = "  ← Pareto-dominant"
            elif br_s > fx_s:
                verdict = "  ← safer (cost some completion)"
            elif br_c > fx_c and br_s >= fx_s - 0.02:
                verdict = "  ← faster (similar safety)"
            print(f"{model:>8s}  {dist:>11s}  "
                  f"BR=(safety {br_s:.3f}, complete {br_c:.3f})  "
                  f"vs fixed=({fx_s:.3f}, {fx_c:.3f}){verdict}")


if __name__ == "__main__":
    aggregate()
