"""Full Phase 10 / V2 breakdown:
  - For each scene, prints a pivot table: rows = DR cells, cols = methods.
  - For each scene, prints a per-method-mean table (sortable summary).
  - For each scene, head-to-head per-DR-cell vs B2-best.

Reads every eval_*.json under --eval_dir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--metric", default="safe_reach",
                    choices=["safe_reach", "reach_rate", "collision_rate",
                             "fall_rate", "timeout_rate", "stuck_rate",
                             "time_to_goal_mean", "intervention_mean",
                             "phi_mean", "alpha_mean"])
    args = ap.parse_args()

    # by_scene[scene] = list of {label, cells, family}
    by_scene = defaultdict(list)
    for name in sorted(os.listdir(args.eval_dir)):
        if not (name.startswith("eval_") and name.endswith(".json")):
            continue
        path = os.path.join(args.eval_dir, name)
        try:
            d = json.load(open(path))
        except Exception:
            continue
        cells = d.get("cells", [])
        if not cells:
            continue
        scene = d.get("scene", "?")
        label = d.get("policy", "")
        if label.startswith("baseline_B0"):
            fam = "B0"
        elif label.startswith("baseline_B1"):
            fam = "B1"
        elif label.startswith("baseline_B2"):
            fam = "B2"
        elif label.startswith("V2"):
            fam = "V2"
        else:
            fam = "?"
        by_scene[scene].append({
            "label": label,
            "fam": fam,
            "cells": {(c["dr_axis"], c["dr_value"]): c for c in cells},
        })

    # determine scene order (E1, E2, E3, E4 if present)
    scene_order = sorted(by_scene.keys(), key=lambda s: (s != "E1Gap", s != "E2Slalom",
                                                         s != "E3Wall", s != "E4Field", s))

    METRIC = args.metric
    METRIC_FMT = ".3f" if METRIC != "time_to_goal_mean" and METRIC != "intervention_mean" else ".1f"

    for scene in scene_order:
        entries = by_scene[scene]
        if not entries:
            continue
        # all DR cells appearing in this scene (sorted by axis then value)
        all_cells = set()
        for e in entries:
            all_cells |= set(e["cells"].keys())
        cell_list = sorted(all_cells, key=lambda k: (k[0], k[1]))
        # for each entry, mean over DR cells (for the right sort column)
        for e in entries:
            vals = [e["cells"][c][METRIC] for c in cell_list if c in e["cells"]
                    and e["cells"][c][METRIC] == e["cells"][c][METRIC]]
            e["mean"] = sum(vals)/max(len(vals), 1) if vals else float("nan")
        # sort: V2 first by descending mean, then B2 by descending, then B1, B0
        fam_order = {"V2": 0, "B2": 1, "B1": 2, "B0": 3, "?": 4}
        entries.sort(key=lambda e: (fam_order[e["fam"]], -e["mean"]))

        print("\n" + "=" * 110)
        print(f"  SCENE: {scene}    metric: {METRIC}")
        print("=" * 110)
        # header: label + each cell
        cell_hdr = " ".join(f"{ax[:3]}={v:<5.2f}" for ax, v in cell_list)
        print(f"  {'method':<45} {'mean':>7}   {cell_hdr}")
        print("  " + "-" * 108)
        for e in entries:
            row = " ".join(
                (f"{e['cells'][c][METRIC]:<8{METRIC_FMT}}"
                 if c in e["cells"] and e["cells"][c][METRIC] == e["cells"][c][METRIC]
                 else f"{'—':<8}")
                for c in cell_list
            )
            print(f"  {e['label']:<45} {e['mean']:>7{METRIC_FMT}}   {row}")

    # ===== aggregate per-scene per-family rank =====
    print("\n" + "=" * 110)
    print(f"  AGGREGATE: best per family per scene  (metric: {METRIC})")
    print("=" * 110)
    print(f"  {'scene':<10}  {'V2-best':<45} {'V2-sr':>8}  {'B2-best':<40} {'B2-sr':>8}  delta")
    print("  " + "-" * 108)
    for scene in scene_order:
        entries = by_scene.get(scene, [])
        if not entries:
            continue
        best_per_fam = {}
        for e in entries:
            if e["fam"] not in best_per_fam or e["mean"] > best_per_fam[e["fam"]]["mean"]:
                best_per_fam[e["fam"]] = e
        v2 = best_per_fam.get("V2")
        b2 = best_per_fam.get("B2")
        if v2 and b2:
            delta = v2["mean"] - b2["mean"]
            mark = "✓ V2" if delta > 0 else "✗ B2"
            print(f"  {scene:<10}  {v2['label']:<45} {v2['mean']:>8{METRIC_FMT}}  "
                  f"{b2['label']:<40} {b2['mean']:>8{METRIC_FMT}}  {delta:+.3f} {mark}")
        elif v2:
            print(f"  {scene:<10}  {v2['label']:<45} {v2['mean']:>8{METRIC_FMT}}  (no B2)")


if __name__ == "__main__":
    main()
