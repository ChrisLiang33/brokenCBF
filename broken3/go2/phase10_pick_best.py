"""For each (scene, baseline family ∈ {B0, B1, B2}) pick the best config
by MEAN safe_reach across the 6 DR cells of that scene. Prints a table
and writes phase10_best_baselines.csv.

Also prints the V2-arch table for the same metric so the comparison is
side-by-side. The bar V2 archs need to beat is the "B2 best" row per
scene (and B1 if B1 happens to win the scene).

Run:
    python3 phase10_pick_best.py --eval_dir phase10_outputs/eval_results
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict


def family_of(label: str) -> str | None:
    """Recover family from policy_label or filename."""
    if label.startswith("baseline_B0"): return "B0"
    if label.startswith("baseline_B1"): return "B1"
    if label.startswith("baseline_B2"): return "B2"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out_csv", default="phase10_best_baselines.csv")
    args = ap.parse_args()

    # collect: (scene, family) -> list of (label, mean_safe_reach, mean_coll, cells)
    by_sf = defaultdict(list)
    # V2-arch: (scene, arch) -> (label, mean_safe_reach, ...)
    v2 = defaultdict(list)
    for name in sorted(os.listdir(args.eval_dir)):
        if not (name.startswith("eval_") and name.endswith(".json")):
            continue
        path = os.path.join(args.eval_dir, name)
        try:
            d = json.load(open(path))
        except Exception as e:
            print(f"[skip] {name}: {e}", file=sys.stderr)
            continue
        cells = d.get("cells", [])
        if not cells:
            continue
        scene = d.get("scene", "?")
        label = d.get("policy", "")
        sr_mean = sum(c["safe_reach"] for c in cells) / len(cells)
        coll_mean = sum(c["collision_rate"] for c in cells) / len(cells)
        ttg_vals = [c["time_to_goal_mean"] for c in cells
                    if c["time_to_goal_mean"] == c["time_to_goal_mean"]]  # not NaN
        ttg_mean = sum(ttg_vals) / max(len(ttg_vals), 1) if ttg_vals else float("nan")
        int_mean = sum(c["intervention_mean"] for c in cells) / len(cells)
        entry = {
            "label": label, "scene": scene,
            "safe_reach_mean": sr_mean,
            "collision_mean": coll_mean,
            "ttg_mean": ttg_mean,
            "int_mean": int_mean,
            "cells": cells,
        }
        fam = family_of(label)
        if fam is not None:
            by_sf[(scene, fam)].append(entry)
        elif label.startswith("V2"):
            arch = d.get("arch", label.split("_int")[0])
            v2[(scene, arch)].append(entry)

    # === best per (scene, family) ===
    print("=" * 90)
    print("BEST PER FAMILY (averaged over the scene's 6 DR cells)")
    print("=" * 90)
    print(f"{'scene':<10} {'family':<4} {'best config':<40} {'safe_reach':>10}  {'coll':>6}  {'ttg':>7}")
    best_rows = []
    for (scene, fam), candidates in sorted(by_sf.items()):
        candidates.sort(key=lambda e: -e["safe_reach_mean"])
        best = candidates[0]
        cfg = best["label"].replace("baseline_", "")
        print(f"{scene:<10} {fam:<4} {cfg:<40} {best['safe_reach_mean']:>10.3f}  "
              f"{best['collision_mean']:>6.3f}  {best['ttg_mean']:>7.1f}")
        best_rows.append({
            "scene": scene, "family": fam, "config": cfg,
            "safe_reach_mean": best["safe_reach_mean"],
            "collision_mean": best["collision_mean"],
            "ttg_mean": best["ttg_mean"],
            "int_mean": best["int_mean"],
        })

    # === V2 archs ===
    print()
    print("=" * 90)
    print("V2 ARCHS (averaged over the scene's 6 DR cells)")
    print("=" * 90)
    print(f"{'scene':<10} {'arch':<16} {'label':<32} {'safe_reach':>10}  {'coll':>6}  {'ttg':>7}")
    for (scene, arch), entries in sorted(v2.items()):
        for e in entries:
            print(f"{scene:<10} {arch:<16} {e['label']:<32} {e['safe_reach_mean']:>10.3f}  "
                  f"{e['collision_mean']:>6.3f}  {e['ttg_mean']:>7.1f}")

    # === head-to-head: V2 best vs B2 best per scene ===
    print()
    print("=" * 90)
    print("HEAD-TO-HEAD: V2 best vs B2 best per scene")
    print("=" * 90)
    by_scene_v2 = defaultdict(list)
    for (scene, arch), entries in v2.items():
        for e in entries:
            by_scene_v2[scene].append((arch, e))
    by_scene_b2 = {row["scene"]: row for row in best_rows if row["family"] == "B2"}
    for scene in sorted(by_scene_v2.keys()):
        b2 = by_scene_b2.get(scene)
        v2_best = max(by_scene_v2[scene], key=lambda t: t[1]["safe_reach_mean"])
        v2_label = f"{v2_best[0]} ({v2_best[1]['label']})"
        v2_sr = v2_best[1]["safe_reach_mean"]
        if b2 is None:
            print(f"{scene:<10}  V2-best: {v2_label}  sr={v2_sr:.3f}  |  B2: (no data)")
            continue
        b2_sr = b2["safe_reach_mean"]
        delta = v2_sr - b2_sr
        verdict = "✓ V2 wins" if delta > 0 else "✗ B2 wins"
        print(f"{scene:<10}  V2-best: {v2_label}  sr={v2_sr:.3f}  |  "
              f"B2-best: {b2['config']}  sr={b2_sr:.3f}  |  Δ={delta:+.3f}  {verdict}")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()) if best_rows else
                           ["scene", "family", "config", "safe_reach_mean",
                            "collision_mean", "ttg_mean", "int_mean"])
        w.writeheader()
        for r in best_rows:
            w.writerow(r)
    print(f"\n[picker] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
