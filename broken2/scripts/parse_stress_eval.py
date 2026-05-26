#!/usr/bin/env python3
"""Per-axis stress eval analysis.

Pulls joint_actual numbers from each stress axis CSV produced by
stress_eval_sweep.sh and prints an attribution table:

  axis           BR joint_act   best fixed joint_act   gap (pp)   ...
  ─────          ────────────   ──────────────────     ────────
  narrow         X              Y                       Δ
  friction       X              Y                       Δ
  com            X              Y                       Δ
  ...
  wide-all       X              Y                       Δ   (from baseline_eval_<label>_indist)

joint_actual = (1 - collision_rate_actual) * (1 - fall_rate) * goal_reach_rate.

Interpretation:
  gap > 0 → axis demands adaptation; teacher adapts here.
  gap ≈ 0 → axis isn't getting adaptive signal in training.
  gap < 0 → teacher misallocates adaptation on this axis.

If sum(per-axis gaps) < wide-all gap, the interaction between axes is
nonlinear (multiple axes jointly require adaptation).

Usage:
  python3 parse_stress_eval.py --label defl
  python3 parse_stress_eval.py --label omnidefl
  python3 parse_stress_eval.py --label defl --indist_dir logs/baseline_eval_wk3defl_indist
"""

import argparse
import csv
import os
import sys

REPO_LOGS = os.path.expanduser("~/Desktop/safety-go2/IsaacLab/logs")

AXES = [
    "narrow",
    "friction",
    "com",
    "sigma_act",
    "radius_error",
    "push",
]


def fnum(row, key):
    try:
        return float(row.get(key) or 0)
    except Exception:
        return 0.0


def joint_actual(row):
    return (
        (1 - fnum(row, "collision_rate_actual"))
        * (1 - fnum(row, "fall_rate"))
        * fnum(row, "goal_reach_rate")
    )


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def best_fixed(rows):
    fixed = [r for r in rows if r["mode"] in ("B0", "B1", "B2")]
    if not fixed:
        return None
    return max(fixed, key=joint_actual)


def br_row(rows):
    br = [r for r in rows if r["mode"] == "BR"]
    return br[0] if br else None


def analyse_one(axis, csv_path):
    if not os.path.exists(csv_path):
        return {"axis": axis, "status": "MISSING", "path": csv_path}
    rows = load_csv(csv_path)
    br = br_row(rows)
    best = best_fixed(rows)
    if not br or not best:
        return {"axis": axis, "status": "INCOMPLETE", "path": csv_path}
    ja_br = joint_actual(br)
    ja_fix = joint_actual(best)
    return {
        "axis": axis,
        "status": "OK",
        "path": csv_path,
        "br_ja": ja_br,
        "br_col_a": fnum(br, "collision_rate_actual"),
        "br_fall": fnum(br, "fall_rate"),
        "br_goal": fnum(br, "goal_reach_rate"),
        "br_alpha": fnum(br, "avg_cbf_alpha_mean"),
        "br_phi": fnum(br, "avg_cbf_phi_mean"),
        "br_a": fnum(br, "avg_cbf_a_mean"),
        "br_c": fnum(br, "avg_cbf_c_mean"),
        "br_defl": fnum(br, "avg_deflection_mean"),
        "fix_ja": ja_fix,
        "fix_name": best["name"],
        "fix_alpha": fnum(best, "alpha"),
        "fix_phi": fnum(best, "phi"),
        "gap_pp": (ja_br - ja_fix) * 100.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True,
                   help="Label used by stress_eval_sweep.sh (e.g. 'defl').")
    p.add_argument("--indist_dir",
                   default=None,
                   help="Override path to wide-everything anchor CSV directory "
                        "(default: <REPO_LOGS>/baseline_eval_wk3<label>_indist).")
    args = p.parse_args()

    base_dir = os.path.join(REPO_LOGS, f"stress_eval_{args.label}")
    indist_dir = args.indist_dir or os.path.join(
        REPO_LOGS, f"baseline_eval_wk3{args.label}_indist"
    )

    rows = []
    for axis in AXES:
        rows.append(analyse_one(axis, os.path.join(base_dir, axis, "baseline.csv")))
    rows.append(analyse_one("wide-all", os.path.join(indist_dir, "baseline.csv")))

    # Print attribution table.
    print(f"\nLabel: {args.label}")
    print(f"Base dir: {base_dir}")
    print(f"Wide-all anchor: {indist_dir}")
    print()
    hdr = ("axis", "BR j_act", "fix j_act", "gap pp", "fix", "BR α", "BR φ",
           "BR a", "BR c", "BR defl", "col_a", "fall", "goal")
    print(f"{hdr[0]:<13} {hdr[1]:>8} {hdr[2]:>9} {hdr[3]:>7}  "
          f"{hdr[4]:<24} "
          f"{hdr[5]:>5} {hdr[6]:>5} {hdr[7]:>5} {hdr[8]:>6} {hdr[9]:>7}  "
          f"{hdr[10]:>5} {hdr[11]:>5} {hdr[12]:>5}")
    print("─" * 130)
    isolated_gap_sum = 0.0
    wide_all_gap = None
    for r in rows:
        if r["status"] != "OK":
            print(f"{r['axis']:<13} [{r['status']}] {r['path']}")
            continue
        if r["axis"] not in ("narrow", "wide-all"):
            isolated_gap_sum += r["gap_pp"]
        if r["axis"] == "wide-all":
            wide_all_gap = r["gap_pp"]
        print(
            f"{r['axis']:<13} {r['br_ja']:>8.3f} {r['fix_ja']:>9.3f} "
            f"{r['gap_pp']:>+6.1f}  "
            f"{r['fix_name'][:24]:<24} "
            f"{r['br_alpha']:>5.2f} {r['br_phi']:>5.2f} {r['br_a']:>5.2f} "
            f"{r['br_c']:>+6.2f} {r['br_defl']:>7.3f}  "
            f"{r['br_col_a']:>5.3f} {r['br_fall']:>5.3f} {r['br_goal']:>5.3f}"
        )

    print()
    print(f"Sum of isolated per-axis gaps (excl. narrow, wide-all): "
          f"{isolated_gap_sum:+.1f} pp")
    if wide_all_gap is not None:
        print(f"Wide-all (combined) gap:                                 "
              f"{wide_all_gap:+.1f} pp")
        nonlinearity = wide_all_gap - isolated_gap_sum
        sign = "interactive (nonlinear)" if nonlinearity > 1.0 else (
            "antagonistic" if nonlinearity < -1.0 else "linear-additive")
        print(f"Combined − sum(isolated):                                "
              f"{nonlinearity:+.1f} pp  ({sign})")


if __name__ == "__main__":
    main()
