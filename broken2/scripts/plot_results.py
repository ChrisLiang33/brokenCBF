"""Generate paper-grade plots from baseline.csv files for a given version.

Usage (from project root):
    python3 scripts/plot_results.py v215     # v2.15 results
    python3 scripts/plot_results.py v212     # v2.12 results (for comparison)

Reads:  IsaacLab/logs/baseline_eval_<version>_<task>/baseline.csv
Writes: IsaacLab/logs/<version>_plots/*.png

Four plots:
  1. <ver>_pareto.png         per-task 2-axis scatter (stuck × fall)
  2. <ver>_bfx.png            Bf-X degradation bar chart
  3. <ver>_margins.png        BR-vs-best-baseline margin per task
  4. <ver>_2axis_split.png    safety axis | performance axis side-by-side bars

Headless-safe (uses Agg backend).
"""
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_ROOT = Path("IsaacLab/logs")
TASKS = [
    "indist", "DensePack", "Slippery", "HighDisturbance",
    "HeavyCOM", "FastObstacles", "RealisticCompound", "NoisyPerception",
    "HighActuationNoise", "RadiusError",
]
BFX_SPECS = [
    # (dir-tag,                          label,                      slot)
    ("bfalpha_indist",                  "Bf-α / in-dist",           "α"),
    ("bfalpha_DensePack",               "Bf-α / DensePack",          "α"),
    ("bfphi_HighActuationNoise",        "Bf-φ / HighActNoise",       "φ"),
    ("bfphi_HighDist",                  "Bf-φ / HighDist (legacy)",  "φ"),  # for v2.12 backcompat
    ("bfa_NoisyPerception",             "Bf-a / NoisyPerc",          "a"),
    ("bfc_HeavyCOM",                    "Bf-c / HeavyCOM",           "c"),
    ("bfc_RadiusError",                 "Bf-c / RadiusError",        "c"),
    ("bfc_FastObs",                     "Bf-c / FastObs (legacy)",   "c"),  # for v2.12 backcompat
]
SLOT_COLOR = {"α": "#d62728", "φ": "#ff7f0e", "a": "#2ca02c", "c": "#1f77b4"}
MODE_COLOR = {"B0": "#7f7f7f", "B1": "#bcbd22", "B2": "#ff9900", "BR": "#1f77b4"}


def load_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def best_per_mode(rows):
    """For each `mode`, pick the (config-row) with the lowest combined fall+stuck."""
    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)
    out = {}
    for m, rs in by_mode.items():
        triples = []
        for r in rs:
            try:
                f, s = float(r["fall_rate"]), float(r["stuck_rate"])
                triples.append((f, s, f + s, r))
            except (ValueError, KeyError):
                continue
        if not triples:
            continue
        out[m] = min(triples, key=lambda t: t[2])
    return out


def best_baseline_mode(modemap):
    pool = {m: modemap[m] for m in ("B0", "B1", "B2") if m in modemap}
    if not pool:
        return None
    return min(pool, key=lambda m: pool[m][2])


# ────────────────────────────────────────────────────────────────────
# Plot 1: per-task 2-axis Pareto (stuck × fall)
# ────────────────────────────────────────────────────────────────────
def plot_2axis_pareto(version, outpath):
    fig, axes = plt.subplots(2, 5, figsize=(20, 9), sharex=True, sharey=True)
    axes = axes.flat
    any_data = False
    for ax, tag in zip(axes, TASKS):
        rows = load_csv(LOG_ROOT / f"baseline_eval_{version}_{tag}" / "baseline.csv")
        if rows is None:
            ax.set_title(f"{tag} (missing)", fontsize=9, color="gray")
            ax.tick_params(labelbottom=False, labelleft=False)
            continue
        any_data = True
        mm = best_per_mode(rows)
        for m, (f, s, _, _) in mm.items():
            color = MODE_COLOR.get(m, "purple")
            marker = "*" if m == "BR" else "o"
            size = 220 if m == "BR" else 110
            ax.scatter(
                s, f, s=size, color=color, marker=marker, label=m,
                edgecolor="black", linewidth=0.7, zorder=4 if m == "BR" else 3,
            )
        ax.set_title(tag, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    fig.text(0.5, 0.04, "stuck rate  (lower = more responsive)", ha="center", fontsize=12)
    fig.text(0.04, 0.5, "fall rate  (lower = safer)", va="center", rotation="vertical", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99), ncol=4)
    plt.suptitle(f"2-axis Pareto — {version}  (lower-left = best)", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0.05, 0.05, 1.0, 0.96])
    plt.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close()
    return any_data


# ────────────────────────────────────────────────────────────────────
# Plot 2: Bf-X degradation bars
# ────────────────────────────────────────────────────────────────────
def plot_bfx_bars(version, outpath):
    labels, degrs, colors = [], [], []
    for tag, label, slot in BFX_SPECS:
        rows = load_csv(LOG_ROOT / f"baseline_eval_{version}_{tag}" / "baseline.csv")
        if rows is None:
            continue
        mm = best_per_mode(rows)
        br = mm.get("BR")
        bfx_key = next((k for k in mm if k.startswith("Bf-")), None)
        if br is None or bfx_key is None:
            continue
        bfx = mm[bfx_key]
        labels.append(label)
        degrs.append((bfx[2] - br[2]) * 100.0)
        colors.append(SLOT_COLOR[slot])
    if not labels:
        return False

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.55 * len(labels) + 1.5)))
    y_pos = list(range(len(labels)))
    bars = ax.barh(y_pos, degrs, color=colors, edgecolor="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.axvline(3, color="green", linestyle="--", alpha=0.4, linewidth=1.0)
    ax.axvline(-1, color="red", linestyle="--", alpha=0.4, linewidth=1.0)
    ax.text(3, len(labels) - 0.3, " load-bearing →", color="green", fontsize=9)
    ax.text(-1, len(labels) - 0.3, "← inversion", color="red", fontsize=9, ha="right")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("degradation when slot clamped to mean (pp combined; higher = more load-bearing)")
    ax.set_title(f"Bf-X ablations — {version}\nper-slot load-bearing test on theory-aligned axes")
    ax.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars, degrs):
        ax.text(
            val + (0.6 if val >= 0 else -0.6),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}",
            ha="left" if val >= 0 else "right", va="center", fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close()
    return True


# ────────────────────────────────────────────────────────────────────
# Plot 3: BR margin per task (single bar chart)
# ────────────────────────────────────────────────────────────────────
def plot_margin_per_task(version, outpath):
    labels, margins = [], []
    for tag in TASKS:
        rows = load_csv(LOG_ROOT / f"baseline_eval_{version}_{tag}" / "baseline.csv")
        if rows is None:
            continue
        mm = best_per_mode(rows)
        bm = best_baseline_mode(mm)
        br = mm.get("BR")
        if bm is None or br is None:
            continue
        labels.append(tag)
        margins.append((mm[bm][2] - br[2]) * 100.0)
    if not labels:
        return False

    colors = [
        "tab:green" if m >= 1.0 else "tab:red" if m <= -1.0 else "tab:gray"
        for m in margins
    ]
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * len(labels) + 1.5)))
    y_pos = list(range(len(labels)))
    bars = ax.barh(y_pos, margins, color=colors, edgecolor="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("BR margin vs best baseline  (pp combined; positive = WIN)")
    ax.set_title(f"BR vs best-baseline per task — {version}")
    ax.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars, margins):
        ax.text(
            val + (0.6 if val >= 0 else -0.6),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}",
            ha="left" if val >= 0 else "right", va="center", fontsize=10,
        )
    avg = sum(margins) / len(margins)
    wins = sum(1 for m in margins if m >= 1.0)
    losses = sum(1 for m in margins if m <= -1.0)
    ties = len(margins) - wins - losses
    ax.text(
        0.99, 0.04,
        f"avg: {avg:+.2f} pp  |  {wins}W / {ties}T / {losses}L",
        transform=ax.transAxes, fontsize=11, weight="bold", ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close()
    return True


# ────────────────────────────────────────────────────────────────────
# Plot 4: 2-axis split (safety bars | performance bars)
# ────────────────────────────────────────────────────────────────────
def plot_2axis_split(version, outpath):
    labels, safety_m, perf_m = [], [], []
    for tag in TASKS:
        rows = load_csv(LOG_ROOT / f"baseline_eval_{version}_{tag}" / "baseline.csv")
        if rows is None:
            continue
        mm = best_per_mode(rows)
        baselines = {m: mm[m] for m in ("B0", "B1", "B2") if m in mm}
        br = mm.get("BR")
        if not baselines or br is None:
            continue
        # Best baseline per axis (independent picks for safety vs perf):
        best_safety_b = min(baselines.values(), key=lambda t: t[0])
        best_perf_b   = min(baselines.values(), key=lambda t: t[1])
        labels.append(tag)
        safety_m.append((best_safety_b[0] - br[0]) * 100.0)
        perf_m.append((best_perf_b[1] - br[1]) * 100.0)
    if not labels:
        return False

    fig, (ax_s, ax_p) = plt.subplots(1, 2, figsize=(14, max(5, 0.45 * len(labels) + 1.5)), sharey=True)
    y_pos = list(range(len(labels)))
    s_colors = ["tab:green" if m >= 0 else "tab:red" for m in safety_m]
    p_colors = ["tab:green" if m >= 0 else "tab:red" for m in perf_m]

    ax_s.barh(y_pos, safety_m, color=s_colors, edgecolor="black", linewidth=0.6)
    ax_s.set_yticks(y_pos); ax_s.set_yticklabels(labels)
    ax_s.invert_yaxis()
    ax_s.set_xlabel("BR safety margin (pp; +ve = fewer falls than baseline)")
    ax_s.set_title("Safety axis (fall rate)")
    ax_s.axvline(0, color="black", linewidth=0.7)
    ax_s.grid(True, axis="x", alpha=0.3)
    for i, val in enumerate(safety_m):
        ax_s.text(
            val + (0.4 if val >= 0 else -0.4), i,
            f"{val:+.1f}",
            ha="left" if val >= 0 else "right", va="center", fontsize=9,
        )

    ax_p.barh(y_pos, perf_m, color=p_colors, edgecolor="black", linewidth=0.6)
    ax_p.set_xlabel("BR perf margin (pp; +ve = less stuck than baseline)")
    ax_p.set_title("Performance axis (stuck rate)")
    ax_p.axvline(0, color="black", linewidth=0.7)
    ax_p.grid(True, axis="x", alpha=0.3)
    for i, val in enumerate(perf_m):
        ax_p.text(
            val + (0.4 if val >= 0 else -0.4), i,
            f"{val:+.1f}",
            ha="left" if val >= 0 else "right", va="center", fontsize=9,
        )

    if safety_m and perf_m:
        s_avg = sum(safety_m) / len(safety_m)
        p_avg = sum(perf_m) / len(perf_m)
        ax_s.text(
            0.99, 0.04, f"avg safety: {s_avg:+.2f} pp",
            transform=ax_s.transAxes, fontsize=10, weight="bold", ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
        )
        ax_p.text(
            0.99, 0.04, f"avg perf:   {p_avg:+.2f} pp",
            transform=ax_p.transAxes, fontsize=10, weight="bold", ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
        )

    plt.suptitle(f"2-axis breakdown — {version}", fontsize=13)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close()
    return True


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "v215"
    out_dir = LOG_ROOT / f"{version}_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[plot_results] generating plots for {version}")
    print(f"  reading from: {LOG_ROOT}/baseline_eval_{version}_*/baseline.csv")
    print(f"  writing to:   {out_dir}/\n")

    plots = [
        ("pareto",       plot_2axis_pareto,    f"{version}_pareto.png"),
        ("bfx",          plot_bfx_bars,        f"{version}_bfx.png"),
        ("margins",      plot_margin_per_task, f"{version}_margins.png"),
        ("2axis_split",  plot_2axis_split,     f"{version}_2axis_split.png"),
    ]
    n_ok = 0
    for name, fn, fname in plots:
        outpath = out_dir / fname
        try:
            ok = fn(version, outpath)
            mark = "✓" if ok else "(no data)"
            print(f"  {mark}  {name:<14s}  →  {outpath}")
            if ok:
                n_ok += 1
        except Exception as e:
            print(f"  ✗  {name:<14s}  ERROR: {e}")

    print(f"\nDone. {n_ok}/{len(plots)} plots written to {out_dir}/")


if __name__ == "__main__":
    main()
