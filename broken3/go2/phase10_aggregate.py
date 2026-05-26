"""Aggregate Phase 10 / V2 per-(policy, scene) JSONs into one flat CSV.

Walks `--eval_dir`, parses every `eval_*.json` written by
phase10_eval_unified.py, and writes one row per (policy, scene, dr_cell)
to `--out`. Columns are the union of the per-cell metrics plus the
identifying fields.

No Isaac dependency -- pure stdlib so it runs on the head node.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for name in sorted(os.listdir(args.eval_dir)):
        if not name.startswith("eval_") or not name.endswith(".json"):
            continue
        path = os.path.join(args.eval_dir, name)
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception as e:
            print(f"[aggregate] skip {name}: {e}", file=sys.stderr)
            continue
        base = {
            "policy": d.get("policy", ""),
            "arch": d.get("arch", ""),
            "scene": d.get("scene", ""),
            "dr_axis": d.get("dr_axis", ""),
        }
        for cell in d.get("cells", []):
            row = dict(base)
            row.update(cell)
            rows.append(row)

    if not rows:
        print(f"[aggregate] no eval JSONs found in {args.eval_dir}", file=sys.stderr)
        sys.exit(1)

    # union of keys, sorted for determinism but with identity fields first
    identity = ["policy", "arch", "scene", "dr_axis", "dr_value"]
    other = sorted({k for r in rows for k in r if k not in identity})
    fields = identity + other
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[aggregate] wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
