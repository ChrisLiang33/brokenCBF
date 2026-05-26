#!/usr/bin/env python3
"""Pull joint_actual numbers from each iteration's collision-split CSV.

joint_actual = (1 - collision_rate_actual) * (1 - fall_rate) * goal_reach_rate

For each iteration: print BR teacher row + best fixed baseline row (ranked
by joint_actual).
"""
import csv
import os
import sys

LOG_DIR = os.path.expanduser("~/Desktop/safety-go2/IsaacLab/logs")
ITERS = [
    ("push",      "PUSH (3-param with `a` frozen)"),
    ("push_a",    "PUSH_A (release `a`)"),
    ("push_a_c",  "PUSH_A_C (release `a` + `c`)"),
    ("phitax",    "PHITAX (+ φ over-inflation tax)"),
    ("tiltdr",    "TILTDR (+ tilt -10 + symm perception DR)"),
    ("aclamp",    "ACLAMP (α range clamped to [0.5, 3.0])"),
]


def joint_actual(row):
    try:
        ca = float(row.get("collision_rate_actual") or 0)
        fa = float(row.get("fall_rate") or 0)
        gr = float(row.get("goal_reach_rate") or 0)
        return (1 - ca) * (1 - fa) * gr
    except Exception:
        return 0.0


def fnum(row, key):
    try:
        return float(row.get(key) or 0)
    except Exception:
        return 0.0


print(f"{'iter':<10} {'mode':<28} {'joint_act':>10}  {'col_act':>8}  "
      f"{'col_perc':>9}  {'fall':>6}  {'goal':>6}")
print("-" * 90)

for name, label in ITERS:
    csv_path = os.path.join(LOG_DIR, f"baseline_eval_{name}_collsplit/baseline.csv")
    if not os.path.exists(csv_path):
        print(f"{name:<10} MISSING ({csv_path})")
        continue
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    br = next((r for r in rows if r["mode"] == "BR"), None)
    fixed = [r for r in rows if r["mode"] in ("B0", "B1", "B2")]
    fixed_sorted = sorted(fixed, key=lambda r: -joint_actual(r))

    if br:
        ja = joint_actual(br)
        print(f"{name:<10} {'BR_teacher':<28} {ja:>10.3f}  "
              f"{fnum(br, 'collision_rate_actual'):>8.3f}  "
              f"{fnum(br, 'collision_rate'):>9.3f}  "
              f"{fnum(br, 'fall_rate'):>6.3f}  "
              f"{fnum(br, 'goal_reach_rate'):>6.3f}")
    if fixed_sorted:
        b = fixed_sorted[0]
        ja = joint_actual(b)
        print(f"{'':<10} {'best fixed: '+b['name']:<28} {ja:>10.3f}  "
              f"{fnum(b, 'collision_rate_actual'):>8.3f}  "
              f"{fnum(b, 'collision_rate'):>9.3f}  "
              f"{fnum(b, 'fall_rate'):>6.3f}  "
              f"{fnum(b, 'goal_reach_rate'):>6.3f}")
    # Gap line
    if br and fixed_sorted:
        gap = joint_actual(br) - joint_actual(fixed_sorted[0])
        sign = "+" if gap >= 0 else "−"
        print(f"{'':<10} {'gap (BR − best)':<28} {sign}{abs(gap)*100:>9.1f} pp")
    print()
