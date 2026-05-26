"""Consolidated post-pipeline summary. Reads each stage's saved outputs
and prints a single comparison table: BR (learned teacher) vs B2 (best
non-degenerate baseline) vs deployed (student-substituted).
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent

TEACHER_CSV = ROOT / "phase5_teacher_outputs" / "phase5_learned_eval.csv"
TEACHER_TXT = ROOT / "phase5_teacher_outputs" / "phase5_teacher_summary.txt"
PER_CHANNEL_JSON = ROOT / "phase5_per_channel_outputs" / "phase5_per_channel_summary.json"
STUDENT_JSON = ROOT / "phase5_student_outputs" / "student_summary.json"
DEPLOY_JSON = ROOT / "phase5_deploy_outputs" / "phase5_deploy_summary.json"
DEPLOY_CSV = ROOT / "phase5_deploy_outputs" / "phase5_deploy_eval.csv"
BASELINE_JSON = ROOT / "phase5_baseline_outputs" / "phase5_baselines_summary.json"


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _f(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def aggregate(rows: list[dict], coll_key="collision_rate",
              reach_key="reach_rate", int_key="intervention_mean"):
    """worst_coll, worst_reach, mean_int across a per-d row list."""
    if not rows:
        return None
    return {
        "worst_coll": max(_f(r[coll_key]) for r in rows),
        "worst_reach": min(_f(r[reach_key]) for r in rows),
        "mean_int": sum(_f(r[int_key]) for r in rows) / len(rows),
    }


def main():
    print()
    print("=" * 90)
    print("  PHASE 5  --  FINAL SUMMARY")
    print("=" * 90)

    # ---- Teacher (BR) eval ----
    print("\n[TEACHER]")
    teacher_rows = _load_csv(TEACHER_CSV)
    if not teacher_rows:
        print(f"  WARN  missing {TEACHER_CSV}")
    else:
        print(f"  {'d (N)':>6}  {'coll':>5}  {'reach':>5}  {'fall':>5}  "
              f"{'stuck':>5}  {'int':>6}  {'phi_mean':>9}  {'alpha_mean':>10}  "
              f"{'jitter':>7}")
        for r in teacher_rows:
            print(f"  {_f(r['disturbance_force']):>6.1f}  "
                  f"{_f(r['collision_rate']):>5.2f}  {_f(r['reach_rate']):>5.2f}  "
                  f"{_f(r['fall_rate']):>5.2f}  {_f(r.get('stuck_rate', 0)):>5.2f}  "
                  f"{_f(r['intervention_mean']):>6.0f}  "
                  f"{_f(r['phi_mean']):>+9.3f}  {_f(r['alpha_mean']):>10.3f}  "
                  f"{_f(r.get('jitter_mean', 0)):>7.3f}")
        teacher_agg = aggregate(teacher_rows)
        print(f"  AGG  worst_coll={teacher_agg['worst_coll']:.2f}  "
              f"worst_reach={teacher_agg['worst_reach']:.2f}  "
              f"mean_int={teacher_agg['mean_int']:.0f}")

    # ---- Per-channel response ----
    print("\n[PER-CHANNEL RESPONSE]  (does the teacher use each priv channel?)")
    pc = _load_json(PER_CHANNEL_JSON)
    if pc is None:
        print(f"  WARN  missing {PER_CHANNEL_JSON}")
    else:
        for s in pc.get("summary", []):
            tag = "DRIVES" if (s["drives_phi"] or s["drives_alpha"]) else "ignored"
            print(f"  {s['channel']:>16}:  phi_span={s['phi_span_pct']:>5.1f}%  "
                  f"alpha_span={s['alpha_span_pct']:>5.1f}%  -> {tag}")
        print(f"  policy actually uses: {pc.get('drivers', [])}")

    # ---- Student R^2 ----
    print("\n[STUDENT]  (phi(history) -> priv recovery)")
    st = _load_json(STUDENT_JSON)
    if st is None:
        print(f"  WARN  missing {STUDENT_JSON}")
    else:
        for name, r2 in zip(st.get("priv_names", []), st.get("r2_per_channel", [])):
            print(f"  {name:>16}:  R^2 = {r2:+.3f}")
        print(f"  mean R^2 across {len(st.get('r2_per_channel', []))} channels: "
              f"{st.get('mean_r2', float('nan')):+.3f}")

    # ---- Deployment (student-substituted) ----
    print("\n[DEPLOYMENT]  (substitute priv with student(history))")
    deploy = _load_json(DEPLOY_JSON)
    deploy_rows = _load_csv(DEPLOY_CSV)
    if deploy is None or not deploy_rows:
        print(f"  WARN  missing deployment outputs")
    else:
        print(f"  {'d (N)':>6}  {'coll':>5}  {'reach':>5}  {'fall':>5}  "
              f"{'int':>6}  {'pred_err':>8}")
        for r in deploy_rows:
            print(f"  {_f(r['disturbance_force']):>6.1f}  "
                  f"{_f(r['collision_rate']):>5.2f}  {_f(r['reach_rate']):>5.2f}  "
                  f"{_f(r['fall_rate']):>5.2f}  "
                  f"{_f(r['intervention_mean']):>6.0f}  "
                  f"{_f(r.get('dist_pred_err', 0)):>8.2f}")
        deploy_agg = aggregate(deploy_rows)
        print(f"  AGG  worst_coll={deploy_agg['worst_coll']:.2f}  "
              f"worst_reach={deploy_agg['worst_reach']:.2f}  "
              f"mean_int={deploy_agg['mean_int']:.0f}")
        print(f"  retention vs teacher: phi {100*deploy['phi_span']/max(deploy['teacher_phi_span'], 1e-9):.0f}%  "
              f"alpha {100*deploy['alpha_span']/max(deploy['teacher_alpha_span'], 1e-9):.0f}%")
        print(f"  verdict: {deploy.get('verdict', '?')}")

    # ---- Comparison to baselines ----
    print("\n[COMPARISON TO BASELINES]  (BR + DEPLOYED vs B0 / B1 / B2)")
    bl = _load_json(BASELINE_JSON)
    if bl is None:
        print(f"  WARN  missing {BASELINE_JSON} -- run phase5_baselines.py")
    else:
        print(f"  {'family':>14}  {'config':>40}  {'wcoll':>6}  {'wreach':>7}  {'meanint':>8}")
        for s in bl:
            cfg = s.get("config", "?")[:40]
            print(f"  {s['baseline']:>14}  {cfg:>40}  "
                  f"{s['worst_coll']:>6.2f}  {s['worst_reach']:>7.2f}  "
                  f"{s['mean_int']:>8.0f}")
        # add BR + deployed rows
        if teacher_rows:
            ta = aggregate(teacher_rows)
            print(f"  {'BR (teacher)':>14}  {'priv visible (privileged)':>40}  "
                  f"{ta['worst_coll']:>6.2f}  {ta['worst_reach']:>7.2f}  "
                  f"{ta['mean_int']:>8.0f}")
        if deploy_rows:
            da = aggregate(deploy_rows)
            print(f"  {'BR (deployed)':>14}  {'priv = student(history)':>40}  "
                  f"{da['worst_coll']:>6.2f}  {da['worst_reach']:>7.2f}  "
                  f"{da['mean_int']:>8.0f}")

        # Headline verdict: did BR beat the strongest non-degenerate baseline?
        # B2 is usually the legitimate strongest (B1 often wins via 0% reach).
        b2 = next((s for s in bl if s["baseline"] == "B2"), None)
        if b2 and teacher_rows:
            ta = aggregate(teacher_rows)
            beat_safety = ta["worst_coll"] < b2["worst_coll"]
            beat_reach = ta["worst_reach"] >= b2["worst_reach"]
            beat_eff = ta["mean_int"] <= b2["mean_int"]
            wins = sum([beat_safety, beat_reach, beat_eff])
            print()
            print(f"  BR (teacher) vs B2 (TISSf state-decay):")
            print(f"    safety:  BR {ta['worst_coll']:.2f} vs B2 {b2['worst_coll']:.2f}  "
                  f"{'WIN' if beat_safety else 'lose'}")
            print(f"    reach:   BR {ta['worst_reach']:.2f} vs B2 {b2['worst_reach']:.2f}  "
                  f"{'WIN' if beat_reach else 'lose'}")
            print(f"    int:     BR {ta['mean_int']:.0f} vs B2 {b2['mean_int']:.0f}  "
                  f"{'WIN' if beat_eff else 'lose'}")
            print(f"    BR beats B2 on {wins}/3 axes")

    print()
    print("=" * 90)
    print("  Outputs in phase5_*_outputs/. Per-stage logs in phase5_pipeline_logs/.")
    print("=" * 90)


if __name__ == "__main__":
    main()
