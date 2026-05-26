#!/usr/bin/env python3
"""Extract per-iter training metrics from an rsl_rl training log to CSV.

Targets the standard rsl_rl print format used by Isaac Lab. Pulls out
the rewards, the new REWARD-2 terms (base_contact_penalty, stuck), the
locomotion tracking metrics, and the termination rates per iter.

Usage:
  python3 extract_training_summary.py <log_file> [output_csv]
  # default output: <log>.training_summary.csv

After v2.9 training, run on the lab box:
  python3 ~/Desktop/safety-go2/scripts/extract_training_summary.py \\
    ~/Desktop/safety-go2/IsaacLab/logs/train_and_eval_v29.log
"""

import csv
import re
import sys
from pathlib import Path


ITER_RE = re.compile(r"Learning iteration\s+(\d+)/")

METRICS = [
    ("mean_reward",            re.compile(r"Mean reward:\s*([-\d.eE+]+)")),
    ("mean_ep_len",            re.compile(r"Mean episode length:\s*([-\d.eE+]+)")),
    ("action_std",             re.compile(r"Mean action std:\s*([-\d.eE+]+)")),
    ("r_collision",            re.compile(r"Episode_Reward/collision:\s*([-\d.eE+]+)")),
    ("r_base_contact_penalty", re.compile(r"Episode_Reward/base_contact_penalty:\s*([-\d.eE+]+)")),
    ("r_stuck",                re.compile(r"Episode_Reward/stuck:\s*([-\d.eE+]+)")),
    ("r_u_safe_deviation",     re.compile(r"Episode_Reward/u_safe_deviation:\s*([-\d.eE+]+)")),
    ("r_proximity",            re.compile(r"Episode_Reward/proximity:\s*([-\d.eE+]+)")),
    ("r_action_rate",          re.compile(r"Episode_Reward/action_rate:\s*([-\d.eE+]+)")),
    ("r_infeasibility",        re.compile(r"Episode_Reward/infeasibility:\s*([-\d.eE+]+)")),
    ("error_vel_xy",           re.compile(r"Metrics/base_velocity/error_vel_xy:\s*([-\d.eE+]+)")),
    ("error_vel_yaw",          re.compile(r"Metrics/base_velocity/error_vel_yaw:\s*([-\d.eE+]+)")),
    ("term_time_out",          re.compile(r"Episode_Termination/time_out:\s*([-\d.eE+]+)")),
    ("term_base_contact",      re.compile(r"Episode_Termination/base_contact:\s*([-\d.eE+]+)")),
    ("term_obstacle_contact",  re.compile(r"Episode_Termination/obstacle_contact:\s*([-\d.eE+]+)")),
    # CBF health stats (added v2.11 prep, 2026-05-07). Surfaced from
    # cbf_go2_env.py:_cbf_filter via extras["log"]. Permissive regex —
    # matches bare key regardless of rsl_rl prefix (Mean / Episode_Info / etc).
    ("cbf_alpha_mean",         re.compile(r"\bcbf_alpha_mean:\s*([-\d.eE+]+)")),
    ("cbf_alpha_std",          re.compile(r"\bcbf_alpha_std:\s*([-\d.eE+]+)")),
    ("cbf_phi_mean",           re.compile(r"\bcbf_phi_mean:\s*([-\d.eE+]+)")),
    ("cbf_phi_std",            re.compile(r"\bcbf_phi_std:\s*([-\d.eE+]+)")),
    ("cbf_a_mean",             re.compile(r"\bcbf_a_mean:\s*([-\d.eE+]+)")),
    ("cbf_a_std",              re.compile(r"\bcbf_a_std:\s*([-\d.eE+]+)")),
    ("cbf_c_mean",             re.compile(r"\bcbf_c_mean:\s*([-\d.eE+]+)")),
    ("cbf_c_std",              re.compile(r"\bcbf_c_std:\s*([-\d.eE+]+)")),
    ("h_min",                  re.compile(r"\bh_min:\s*([-\d.eE+]+)")),
    ("h_mean",                 re.compile(r"\bh_mean:\s*([-\d.eE+]+)")),
    ("qp_active_rate",         re.compile(r"\bqp_active_rate:\s*([-\d.eE+]+)")),
    ("u_safe_clamp_rate",      re.compile(r"\bu_safe_clamp_rate:\s*([-\d.eE+]+)")),
]


def parse_log(path):
    """Yield one dict of per-iter metrics for each iteration block."""
    cur = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = ITER_RE.search(line)
            if m:
                if cur is not None:
                    yield cur
                cur = {"iter": int(m.group(1))}
                continue
            if cur is None:
                continue
            for col, regex in METRICS:
                if col in cur:
                    continue
                m = regex.search(line)
                if m:
                    cur[col] = float(m.group(1))
                    break
        if cur is not None:
            yield cur


def main():
    if len(sys.argv) < 2:
        print("Usage: extract_training_summary.py <log> [out.csv]", file=sys.stderr)
        sys.exit(2)
    log_path = Path(sys.argv[1])
    out_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else log_path.with_suffix(".training_summary.csv")
    )

    rows = list(parse_log(log_path))
    if not rows:
        print(f"No iterations found in {log_path}", file=sys.stderr)
        sys.exit(1)

    cols = ["iter"] + [c for c, _ in METRICS]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})

    print(f"Wrote {len(rows)} iterations to {out_path}")

    # Quick summary: first, middle, last iters
    if len(rows) >= 3:
        idxs = [0, len(rows) // 2, len(rows) - 1]
        print("\nSpot check (first / middle / last iters):")
        iter_strs = [f"{rows[i]['iter']:>10d}" for i in idxs]
        print(f"  iter:                   {' '.join(iter_strs)}")
        for col, _ in METRICS:
            vals = []
            for i in idxs:
                v = rows[i].get(col)
                vals.append(f"{v:>10.4f}" if v is not None else f"{'':>10s}")
            print(f"  {col:<24s}{' '.join(vals)}")


if __name__ == "__main__":
    main()
