"""Render V8 results + diagnostics into a single PNG for slides."""
import csv
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1] / "data_from_lab" / "wk3tight8"
OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "v8_results.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def safety_score(row):
    g = float(row["goal_reach_rate"])
    f = float(row["fall_rate"])
    s = float(row["stuck_rate"])
    c = float(row["collision_rate_actual"])
    return g * (1 - f) * (1 - s) * (1 - c)


def load_eval(path):
    rows = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            row["safety_score"] = safety_score(row)
            rows.append(row)
    return rows


indist = load_eval(ROOT / "indist.csv")
trainmatch = load_eval(ROOT / "trainmatch.csv")
phi = json.loads((ROOT / "diagnose_phi_corr_wk3tight8.json").read_text())
alpha = json.loads((ROOT / "diagnose_alpha_corr_wk3tight8.json").read_text())
zprobe = json.loads((ROOT / "probe_z_linear_wk3tight8.json").read_text())


def sort_rows(rows):
    return sorted(rows, key=lambda r: r["safety_score"], reverse=True)


def short_name(r):
    if r["mode"] == "BR":
        return "BR (V8 teacher)"
    if r["mode"] == "B0":
        return f"B0 α={float(r['alpha']):.1f}"
    if r["mode"] == "B1":
        return f"B1 α={float(r['alpha']):.1f}, φ={float(r['phi']):.1f}"
    if r["mode"] == "B2":
        return f"B2 α={float(r['alpha']):.1f}, λ={float(r['lambda']):.0f}"
    return r["name"]


fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle(
    "V8 (PHIWIN_TIGHTCOR_V8, 2026-05-19, 2500 iters) — per-step φ, no lock, "
    "SHIELD c-comp at −0.10",
    fontsize=15, fontweight="bold", y=0.995,
)


def bar_eval(ax, rows, title, n=9):
    top = sort_rows(rows)[:n]
    names = [short_name(r) for r in top]
    vals = [r["safety_score"] for r in top]
    colors = ["#d62728" if r["mode"] == "BR" else "#1f77b4" for r in top]
    y = np.arange(len(names))[::-1]
    bars = ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0.4, 0.85)
    ax.set_xlabel("safety_score = goal × (1−fall)(1−stuck)(1−collision_actual)",
                  fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9)


bar_eval(axes[0, 0], trainmatch,
         "Trainmatch eval (deploy-realistic) — BR ties oracle best fixed")
bar_eval(axes[0, 1], indist,
         "In-dist eval — BR mid-pack; low-α B0 wins where deploy ≈ train")


# Lock comparison panel — V5 locked vs V8 no-lock
ax = axes[0, 2]
metrics = ["φ within-env\nstd", "φ_mean", "|Pearson(φ, σ_act)|",
           "BR safety_score\n(trainmatch)"]
v5 = [0.25, 2.39, 0.00, 0.691]
v8 = [phi["phi_within_env_std_mean"],
      phi["phi_population_mean"],
      abs(phi["correlations_with_phi"]["actuation_noise_sigma"]),
      sort_rows(trainmatch)[0]["safety_score"] if sort_rows(trainmatch)[0]["mode"] == "BR"
      else next(r for r in trainmatch if r["mode"] == "BR")["safety_score"]]
x = np.arange(len(metrics))
w = 0.38
ax.bar(x - w/2, v5, w, label="V5 (locked)", color="#888888", edgecolor="black")
ax.bar(x + w/2, v8, w, label="V8 (no-lock)", color="#d62728", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=9)
ax.set_title("Lock removal effect (V5 → V8)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle=":", alpha=0.5)
for i, (a, b) in enumerate(zip(v5, v8)):
    ax.text(i - w/2, a + 0.03, f"{a:.2f}", ha="center", fontsize=8)
    ax.text(i + w/2, b + 0.03, f"{b:.2f}", ha="center", fontsize=8)


# z_priv probe R²
ax = axes[1, 0]
probe = zprobe["linear_probe"]
order = ["base_height", "actuation_noise_sigma", "com_x", "com_z", "com_y",
         "tracking_x", "tracking_z", "tracking_y",
         "torque_x", "force_x", "force_y", "torque_z",
         "base_mass", "force_z", "friction", "torque_y"]
labels = [k.replace("_", " ") for k in order]
vals = [probe[k]["r2_test"] for k in order]
colors = []
for k in order:
    v = probe[k]["r2_test"]
    if k in ("actuation_noise_sigma", "com_x", "com_y", "com_z") and v > 0.4:
        colors.append("#2ca02c")  # was blind before, now visible
    elif v > 0.4:
        colors.append("#1f77b4")
    else:
        colors.append("#888888")
y = np.arange(len(labels))[::-1]
ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.5)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlim(0, 1.0)
ax.set_xlabel("R² (linear probe on z_priv → priv feature)", fontsize=9)
ax.set_title("z_priv encoder health  (green = previously blind, now decoded)",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", linestyle=":", alpha=0.5)
for i, v in enumerate(vals):
    ax.text(v + 0.01, y[i], f"{v:.2f}", va="center", fontsize=8)


def head_corr_panel(ax, head_label, corr_dict, within_ep, color):
    items = [(k, v) for k, v in corr_dict.items()
             if k not in ("|applied_force|", "|applied_torque|", "com_x", "com_z")]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    items = items[:6]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    y = np.arange(len(names))[::-1]
    bar_colors = [color if v > 0 else "#ff7f0e" for v in vals]
    ax.barh(y, vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(-0.5, 0.6)
    ax.set_xlabel("Pearson correlation (between-env)", fontsize=9)
    ax.set_title(f"What {head_label} attends to (between-env)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for i, v in enumerate(vals):
        ax.text(v + (0.01 if v >= 0 else -0.01), y[i], f"{v:+.2f}",
                va="center",
                ha="left" if v >= 0 else "right", fontsize=8)

    # within-episode footnote
    we_lines = []
    for sig, val in within_ep.items():
        we_lines.append(f"  within-ep Pearson({head_label}_t, {sig}) = {val:+.2f}")
    ax.text(0.02, -0.32, "\n".join(we_lines),
            transform=ax.transAxes, fontsize=9, family="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff7e6",
                      edgecolor="#cc9933", linewidth=0.6))


head_corr_panel(
    axes[1, 1], "φ", phi["correlations_with_phi"],
    {"h_t": phi["within_episode_pearson"]["phi_vs_h"]["mean"],
     "||L_g h||²_t": phi["within_episode_pearson"]["phi_vs_Lgh_norm_sq"]["mean"]},
    color="#d62728",
)
head_corr_panel(
    axes[1, 2], "α", alpha["correlations_with_alpha"],
    {"|tracking_err|": alpha["within_episode_pearson_with_alpha"]["|tracking_err|"]["mean"],
     "h_t": alpha["within_episode_pearson_with_alpha"]["h"]["mean"],
     "|base_ang_vel|": alpha["within_episode_pearson_with_alpha"]["|base_ang_vel|"]["mean"]},
    color="#1f77b4",
)

plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
print(f"Wrote {OUT}")
