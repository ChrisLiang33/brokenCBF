#!/usr/bin/env python3
"""One-shot results aggregator for v3.0e / v3.0f / etc.

Reads the eval CSVs (indist + HeavyCOM) and the α-correlation JSON, applies
the locked decision criterion, and prints a clean verdict + summary.

Locked criterion (state-conditional adaptation test):
  PASS:  Pearson(α, friction) > 0.20  OR  Pearson(α, |com_offset|) > 0.20
         AND BR combined beats best-of-B0-sweep by ≥3pp on ≥1 task
  AMBIG: any DR-feature corr > 0.20 but combined metric ties / loses
  FAIL:  all DR-feature corrs < 0.10  AND  BR loses on both tasks

Usage:
  # On lab box (defaults look at v3.0e files):
  python3 ~/Desktop/safety-go2/scripts/aggregate_v3_results.py

  # Pick a different version:
  python3 ~/Desktop/safety-go2/scripts/aggregate_v3_results.py --version v3_0f

  # Or point at specific files:
  python3 ~/Desktop/safety-go2/scripts/aggregate_v3_results.py \\
    --indist_csv path/to/indist.csv \\
    --heavy_csv path/to/heavy.csv \\
    --corr_json path/to/alpha_corr.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _find_br_row(rows: list[dict]) -> dict | None:
    for r in rows:
        if r.get("mode", "").upper() == "BR":
            return r
    return None


def _find_best_b0_combined(rows: list[dict]) -> tuple[dict | None, float]:
    """Return (row with best combined among B0 modes, that combined value)."""
    best_row = None
    best_combined = float("inf")
    for r in rows:
        if r.get("mode", "").upper() != "B0":
            continue
        c = _to_float(r.get("fall_rate", "0")) + _to_float(r.get("stuck_rate", "0"))
        if c < best_combined:
            best_combined = c
            best_row = r
    return best_row, best_combined


def _combined(row: dict) -> float:
    return _to_float(row.get("fall_rate", "0")) + _to_float(row.get("stuck_rate", "0"))


def print_table(headers, rows, col_pad=2):
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    pad = " " * col_pad
    sep_line = "─" * (sum(widths) + col_pad * (len(headers) - 1) + 2)
    print(f"  {sep_line}")
    print("  " + pad.join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print(f"  {sep_line}")
    for r in rows:
        print("  " + pad.join(str(v).ljust(w) for v, w in zip(r, widths)))
    print(f"  {sep_line}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v3_0e",
                        help="Tag for default file paths (v3_0e, v3_0f, ...)")
    parser.add_argument("--indist_csv", default=None)
    parser.add_argument("--heavy_csv", default=None)
    parser.add_argument("--corr_json", default=None)
    parser.add_argument("--logs_root", default="logs",
                        help="Root dir for eval outputs (default: logs)")
    parser.add_argument("--pp_threshold", type=float, default=0.03,
                        help="Combined-metric improvement threshold for PASS (default 0.03 = 3pp)")
    parser.add_argument("--corr_strong", type=float, default=0.20)
    parser.add_argument("--corr_weak", type=float, default=0.10)
    args = parser.parse_args()

    base = Path(args.logs_root)
    v = args.version
    indist_csv = Path(args.indist_csv) if args.indist_csv else base / f"baseline_eval_{v}_indist" / "baseline.csv"
    heavy_csv = Path(args.heavy_csv) if args.heavy_csv else base / f"baseline_eval_{v}_HeavyCOM" / "baseline.csv"
    corr_json = Path(args.corr_json) if args.corr_json else Path(f"diagnose_alpha_corr_{v}.json")

    missing = [p for p in (indist_csv, heavy_csv, corr_json) if not p.exists()]
    if missing:
        print("ERROR: missing files:", *[f"  {m}" for m in missing], sep="\n")
        return 1

    indist_rows = _read_csv(indist_csv)
    heavy_rows = _read_csv(heavy_csv)
    with open(corr_json) as f:
        corr = json.load(f)

    print("=" * 72)
    print(f"  Results aggregation — {v}")
    print("=" * 72)
    print(f"  indist CSV : {indist_csv}")
    print(f"  HeavyCOM CSV : {heavy_csv}")
    print(f"  α-corr JSON : {corr_json}")
    print()

    # ── 1. α distribution check ──
    indist_br = _find_br_row(indist_rows)
    heavy_br = _find_br_row(heavy_rows)
    if not indist_br or not heavy_br:
        print("ERROR: BR row missing from one of the CSVs")
        return 1

    a_mean_i = _to_float(indist_br.get("avg_cbf_alpha_mean", "0"))
    a_std_i = _to_float(indist_br.get("avg_cbf_alpha_std", "0"))
    a_mean_h = _to_float(heavy_br.get("avg_cbf_alpha_mean", "0"))
    a_std_h = _to_float(heavy_br.get("avg_cbf_alpha_std", "0"))
    dmean = abs(a_mean_i - a_mean_h)
    dstd = abs(a_std_i - a_std_h)

    print("§1 α distribution across tasks (cross-task signal):")
    print_table(
        ["", "indist", "HeavyCOM", "Δ"],
        [
            ["alpha_mean", f"{a_mean_i:.3f}", f"{a_mean_h:.3f}", f"{dmean:.3f}"],
            ["alpha_std", f"{a_std_i:.3f}", f"{a_std_h:.3f}", f"{dstd:.3f}"],
        ],
    )
    print()

    # ── 2. Within-task α correlation with DR features ──
    corrs = corr.get("correlations_with_alpha", {})
    dr_features = ["friction", "base_mass", "base_height",
                   "com_x", "com_y", "com_z",
                   "|com_offset|", "|applied_force|", "|applied_torque|"]
    cbf_features = ["h", "slack", "Lgh_norm_sq", "Lgh_dot_udes", "|tracking_err|"]

    print("§2 Within-task α correlations (state-conditioning signal):")
    rows = []
    for feat in dr_features + cbf_features:
        if feat not in corrs:
            continue
        r = corrs[feat]
        ar = abs(r)
        if ar > args.corr_strong:
            tag = "  ←← STRONG"
        elif ar > args.corr_weak:
            tag = "  ← weak"
        else:
            tag = ""
        kind = "DR" if feat in dr_features else "CBF"
        rows.append([kind, feat, f"{r:+.3f}", tag])
    print_table(["kind", "feature", "Pearson(α, ·)", ""], rows)
    print()

    # ── 3. Combined metric vs best fixed α ──
    print("§3 Combined metric (BR vs best fixed-α):")
    indist_b0_row, indist_b0_combined = _find_best_b0_combined(indist_rows)
    heavy_b0_row, heavy_b0_combined = _find_best_b0_combined(heavy_rows)

    indist_br_combined = _combined(indist_br)
    heavy_br_combined = _combined(heavy_br)

    indist_delta_pp = (indist_b0_combined - indist_br_combined) * 100  # positive = BR better
    heavy_delta_pp = (heavy_b0_combined - heavy_br_combined) * 100

    print_table(
        ["task", "best fixed α", "B0 combined", "BR combined", "Δ (pp, BR-better)"],
        [
            ["indist",
             indist_b0_row.get("alpha", "?") if indist_b0_row else "?",
             f"{indist_b0_combined:.4f}",
             f"{indist_br_combined:.4f}",
             f"{indist_delta_pp:+.2f}"],
            ["HeavyCOM",
             heavy_b0_row.get("alpha", "?") if heavy_b0_row else "?",
             f"{heavy_b0_combined:.4f}",
             f"{heavy_br_combined:.4f}",
             f"{heavy_delta_pp:+.2f}"],
        ],
    )
    print()

    # ── 4. Side metrics (efficiency, goal-reach) ──
    print("§4 Side metrics (BR only):")
    keys = ["goal_reach_rate", "mean_time_to_goal", "path_efficiency",
            "mean_v_along_cmd", "mean_dist_traveled", "avg_deflection_mean",
            "avg_qp_active_rate"]
    rows = []
    for k in keys:
        vi = _to_float(indist_br.get(k, "")) if k in indist_br else float("nan")
        vh = _to_float(heavy_br.get(k, "")) if k in heavy_br else float("nan")
        rows.append([k, f"{vi:.4f}", f"{vh:.4f}"])
    print_table(["metric", "indist", "HeavyCOM"], rows)
    print()

    # ── 5. Verdict ──
    dr_strong = any(
        feat in corrs and abs(corrs[feat]) > args.corr_strong
        for feat in dr_features
    )
    dr_max = max((abs(corrs[f]) for f in dr_features if f in corrs), default=0.0)
    combined_win = (indist_delta_pp >= args.pp_threshold * 100) or (heavy_delta_pp >= args.pp_threshold * 100)
    cross_task_signal = (dmean >= 0.3) or (dstd >= 0.3)

    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    if dr_strong and combined_win:
        verdict = "PASS"
        note = "DR-feature state-conditioning emerged AND BR beats best-fixed."
    elif dr_strong and not combined_win:
        verdict = "AMBIG"
        note = "State-conditioning visible (DR corr > 0.20) but combined metric ties/loses. Tune."
    elif (dr_max > args.corr_weak) and not combined_win:
        verdict = "WEAK"
        note = f"Some DR correlation (max |r|={dr_max:.3f}) but below STRONG threshold. Marginal signal."
    else:
        verdict = "FAIL"
        note = "No meaningful DR-feature correlation. Policy ignores env class. Escalate (curriculum, oracle teacher)."

    print(f"  {verdict}")
    print(f"  {note}")
    print()
    print(f"  Tests:")
    print(f"    DR-feature corr > {args.corr_strong}                 : {'YES' if dr_strong else 'no'} (max DR |r| = {dr_max:.3f})")
    print(f"    BR combined beats best-fixed by ≥{args.pp_threshold*100:.0f}pp : {'YES' if combined_win else 'no'} (best Δ = {max(indist_delta_pp, heavy_delta_pp):+.2f}pp)")
    print(f"    Cross-task Δα_mean or Δα_std ≥ 0.3       : {'YES' if cross_task_signal else 'no'} (Δμ={dmean:.3f}, Δσ={dstd:.3f})")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
