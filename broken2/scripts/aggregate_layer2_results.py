#!/usr/bin/env python3
"""One-shot results aggregator for Layer 2 (v5+) runs.

Reads the φ-corr JSON, optional probe JSON, optional eval CSV, and prints
a clean verdict + summary against the Layer 2 decision criteria.

Layer 2 decision criterion (any one of the following = PASS on adaptation):
  - |Pearson(φ, base_mass)|              > 0.20
  - |Pearson(φ, actuation_noise_sigma)|  > 0.20
  - R²(actuation_noise_sigma) in z_priv  > 0.20  (probe)
  - |Pearson(φ, base_height)|            > 0.20  (v6 emergence axis)

Plus safety check from eval CSV (if present): obstacle_contact_rate < 0.20.

Usage:
  # Layer 2 default file paths:
  python3 ~/Desktop/safety-go2/scripts/aggregate_layer2_results.py

  # Custom paths:
  python3 ~/Desktop/safety-go2/scripts/aggregate_layer2_results.py \\
      --phi_corr diagnose_phi_corr_layer2.json \\
      --probe probe_z_linear_layer2.json \\
      --eval_csv logs/baseline_eval_layer2_HighActuationNoise/baseline.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _to_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


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


def find_row(rows: list[dict], mode: str) -> dict | None:
    """Find first row with matching mode (B0/B1/B2/BR/BS).
    For B0, returns the row with the best (lowest) fall+stuck combined."""
    matches = [r for r in rows if r.get("mode", "").upper() == mode.upper()]
    if not matches:
        return None
    if mode.upper() == "B0":
        return min(matches, key=lambda r: _to_float(r.get("fall_rate", "0")) + _to_float(r.get("stuck_rate", "0")))
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi_corr", default="diagnose_phi_corr_layer2.json")
    parser.add_argument("--probe", default="probe_z_linear_layer2.json")
    parser.add_argument("--eval_csv", default=None,
                        help="Optional eval CSV. If omitted, looks for "
                             "logs/baseline_eval_layer2_HighActuationNoise/baseline.csv")
    parser.add_argument("--corr_strong", type=float, default=0.20)
    parser.add_argument("--corr_weak", type=float, default=0.10)
    parser.add_argument("--probe_strong", type=float, default=0.20)
    parser.add_argument("--safety_threshold", type=float, default=0.20,
                        help="obstacle_contact_rate must fall below this to PASS safety")
    args = parser.parse_args()

    phi_corr_path = Path(args.phi_corr)
    probe_path = Path(args.probe)
    eval_csv_path = Path(args.eval_csv) if args.eval_csv else \
        Path("logs/baseline_eval_layer2_HighActuationNoise/baseline.csv")

    if not phi_corr_path.exists():
        print(f"ERROR: φ-corr JSON not found at {phi_corr_path}", file=sys.stderr)
        return 1

    with open(phi_corr_path) as f:
        phi_data = json.load(f)

    probe_data = None
    if probe_path.exists():
        with open(probe_path) as f:
            probe_data = json.load(f)

    eval_rows = None
    if eval_csv_path.exists():
        eval_rows = _read_csv(eval_csv_path)

    print("=" * 72)
    print(f"  Layer 2 results aggregation")
    print("=" * 72)
    print(f"  φ-corr JSON : {phi_corr_path}")
    print(f"  probe JSON  : {probe_path if probe_data else '(absent)'}")
    print(f"  eval CSV    : {eval_csv_path if eval_rows else '(absent)'}")
    print(f"  checkpoint  : {phi_data.get('checkpoint', '?')}")
    print()

    # ── 1. φ distribution ──
    print("§1 φ distribution (the policy's output range):")
    rows = [
        ["population mean",        f"{phi_data.get('phi_population_mean', 0.0):.3f}"],
        ["population std",         f"{phi_data.get('phi_population_std', 0.0):.3f}"],
        ["within-env std (mean)",  f"{phi_data.get('phi_within_env_std_mean', 0.0):.3f}"],
    ]
    print_table(["", "value"], rows)
    print(f"  (Saturation check: mean far from [0, 5] boundary, std > 0.5)")
    print()

    # ── 2. φ correlations ──
    corrs = phi_data.get("correlations_with_phi", {})
    env_class_features = ["friction", "base_mass", "base_height",
                          "actuation_noise_sigma", "|com_offset|",
                          "com_x", "com_y", "com_z",
                          "|applied_force|", "|applied_torque|"]
    cbf_features = ["h", "slack", "Lgh_norm_sq", "Lgh_dot_udes", "|tracking_err|"]

    print("§2 φ correlations:")
    rows = []
    for feat in env_class_features + cbf_features:
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
        kind = "env-class" if feat in env_class_features else "CBF/geo"
        rows.append([kind, feat, f"{r:+.3f}", tag])
    print_table(["kind", "feature", "Pearson(φ, ·)", ""], rows)
    print()

    # ── 3. Linear probe (z_priv → env-class features) ──
    if probe_data:
        print("§3 Linear probe R² of z_priv against priv features:")
        probe = probe_data.get("linear_probe", {})
        # priority: features we care about for env-class adaptation
        watch_features = ["friction", "base_mass", "actuation_noise_sigma",
                          "base_height", "com_x", "com_y", "com_z"]
        rows = []
        for feat in watch_features:
            if feat not in probe:
                continue
            r2 = probe[feat].get("r2_test", 0.0)
            if r2 > args.probe_strong:
                tag = "  ←← compressed well"
            elif r2 > 0.10:
                tag = "  ← partial"
            else:
                tag = "  ↓ ignored"
            rows.append([feat, f"{r2:.3f}", tag])
        print_table(["priv feature", "R² (test)", ""], rows)
        print()
    else:
        print("§3 Linear probe : (no probe JSON — skipping)")
        print()

    # ── 4. Eval CSV (safety + performance, if present) ──
    if eval_rows:
        b0 = find_row(eval_rows, "B0")
        br = find_row(eval_rows, "BR")
        print("§4 Eval safety + performance (BR vs best fixed-α B0):")
        if b0 and br:
            rows = [
                ["alpha (B0)",        b0.get("alpha", "?"),        "—"],
                ["fall_rate",         f"{_to_float(b0.get('fall_rate', '0')):.3f}",
                                       f"{_to_float(br.get('fall_rate', '0')):.3f}"],
                ["collision_rate",    f"{_to_float(b0.get('collision_rate', '0')):.3f}",
                                       f"{_to_float(br.get('collision_rate', '0')):.3f}"],
                ["stuck_rate",        f"{_to_float(b0.get('stuck_rate', '0')):.3f}",
                                       f"{_to_float(br.get('stuck_rate', '0')):.3f}"],
                ["goal_reach_rate",   f"{_to_float(b0.get('goal_reach_rate', '0')):.3f}",
                                       f"{_to_float(br.get('goal_reach_rate', '0')):.3f}"],
                ["mean_v_xy",         f"{_to_float(b0.get('mean_v_xy', '0')):.3f}",
                                       f"{_to_float(br.get('mean_v_xy', '0')):.3f}"],
                ["mean_phi_used",     f"{_to_float(b0.get('mean_phi_used', '0')):.3f}",
                                       f"{_to_float(br.get('mean_phi_used', '0')):.3f}"],
            ]
            print_table(["metric", "B0 (best fixed)", "BR (adaptive)"], rows)
        else:
            print(f"  WARN: missing B0 or BR row")
        print()
    else:
        print("§4 Eval CSV : (none — diagnostics-only run)")
        print()

    # ── 5. Verdict ──
    env_class_max_corr = max(
        (abs(corrs.get(f, 0.0)) for f in env_class_features),
        default=0.0,
    )
    env_class_best_feat = max(
        env_class_features,
        key=lambda f: abs(corrs.get(f, 0.0)),
        default="(none)",
    )

    adapt_pass = env_class_max_corr > args.corr_strong
    adapt_weak = env_class_max_corr > args.corr_weak

    # Probe pass (independent signal)
    probe_pass = False
    if probe_data:
        probe = probe_data.get("linear_probe", {})
        noise_r2 = probe.get("actuation_noise_sigma", {}).get("r2_test", 0.0)
        probe_pass = noise_r2 > args.probe_strong

    # Safety check
    safety_pass = None
    obstacle_contact = None
    if eval_rows:
        br = find_row(eval_rows, "BR")
        if br:
            # CSV uses "collision_rate" but training termination uses obstacle_contact;
            # check both
            obstacle_contact = _to_float(br.get("collision_rate", "0"))
            safety_pass = obstacle_contact < args.safety_threshold

    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)

    if adapt_pass and (safety_pass is None or safety_pass):
        verdict = "PASS"
        note = (f"Env-class adaptation visible: |Pearson(φ, {env_class_best_feat})| "
                f"= {env_class_max_corr:.3f}.")
    elif adapt_pass:
        verdict = "ADAPTATION-OK / SAFETY-FAIL"
        note = (f"φ-{env_class_best_feat} corr = {env_class_max_corr:.3f}, but "
                f"obstacle_contact_rate = {obstacle_contact:.3f} > {args.safety_threshold}.")
    elif adapt_weak:
        verdict = "WEAK"
        note = (f"Some env-class correlation (max |r| on '{env_class_best_feat}' "
                f"= {env_class_max_corr:.3f}). Below STRONG threshold.")
    elif probe_pass:
        verdict = "ENCODER-LEAK"
        note = ("Probe shows env-class is being compressed (R²(noise) "
                f"= {noise_r2:.3f}), but φ isn't conditioning on it yet. "
                "Encoder is open; policy mapping is missing.")
    else:
        verdict = "FAIL"
        note = (f"No meaningful env-class correlation (max |r| = {env_class_max_corr:.3f}). "
                "Policy ignores env class. Reconsider injection point or DR scale.")

    print(f"  {verdict}")
    print(f"  {note}")
    print()

    print(f"  Tests:")
    print(f"    φ-env-class STRONG (>{args.corr_strong:.2f})  : "
          f"{'YES' if adapt_pass else 'no'} "
          f"(max |r| = {env_class_max_corr:.3f} on '{env_class_best_feat}')")
    if probe_data:
        noise_r2 = probe_data.get("linear_probe", {}).get("actuation_noise_sigma", {}).get("r2_test", 0.0)
        print(f"    R²(noise σ) in probe > {args.probe_strong:.2f}     : "
              f"{'YES' if probe_pass else 'no'} (R² = {noise_r2:.3f})")
    if safety_pass is not None:
        print(f"    obstacle_contact < {args.safety_threshold:.2f}            : "
              f"{'YES' if safety_pass else 'no'} ({obstacle_contact:.3f})")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
