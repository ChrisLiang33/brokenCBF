"""Render full V8 report as self-contained HTML with multi-axis visualizations."""
import csv
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1] / "data_from_lab" / "wk3tight8"
OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "v8_results.html"


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


indist = load(ROOT / "indist.csv")
trainmatch = load(ROOT / "trainmatch.csv")
ood = load(ROOT / "ood.csv")
phi = json.loads((ROOT / "diagnose_phi_corr_wk3tight8.json").read_text())
alpha = json.loads((ROOT / "diagnose_alpha_corr_wk3tight8.json").read_text())
zprobe = json.loads((ROOT / "probe_z_linear_wk3tight8.json").read_text())
shared = np.load(ROOT / "diagnose_shared_signal_wk3tight8.npz", allow_pickle=True)


def short_name(r):
    if r["mode"] == "BR":
        return "BR (V8 teacher)"
    a = float(r["alpha"])
    if r["mode"] == "B0":
        return f"B0 α={a:.1f}"
    if r["mode"] == "B1":
        return f"B1 α={a:.1f}, φ={float(r['phi']):.1f}"
    if r["mode"] == "B2":
        return f"B2 α={a:.1f}, λ={float(r['lambda']):.0f}"
    return r["name"]


AXES = [
    ("Completion", "goal_reach_rate", "higher", 1.0,
     "Fraction of episodes reaching the goal in time."),
    ("Collision (actual)", "collision_rate_actual", "lower", 1.0,
     "Fraction of episodes with a TRUE collision (post hoc, ground-truth wall contact). "
     "Distinct from perceived collisions from synthetic LiDAR clusters."),
    ("Fall rate", "fall_rate", "lower", 1.0,
     "Fraction of episodes where the robot's base tips past the safe roll/pitch threshold."),
    ("Path efficiency", "path_efficiency", "higher", 1.0,
     "Final displacement / total path length. Higher = less wandering."),
    ("Time to goal (s)", "mean_time_to_goal", "lower", 260.0,
     "Mean wall-clock seconds to reach the goal for successful episodes."),
    ("Deflection ‖u_safe − u_nominal‖", "avg_deflection_mean", "lower", 0.8,
     "Average magnitude of the safety filter's correction to the teleop command. "
     "Lower = the CBF intervenes less, smoother control."),
]


# ── Eval panel: one sub-panel per axis ─────────────────────────────
def panel_axis(rows, axis_label, key, direction, vmax, desc):
    items = sorted([(r, float(r[key])) for r in rows],
                   key=lambda kv: kv[1], reverse=(direction == "higher"))
    top = items[:8]
    if not any(r["mode"] == "BR" for r, _ in top):
        br_item = next(it for it in items if it[0]["mode"] == "BR")
        top = top[:7] + [br_item]
    bar_color = "#1f77b4" if direction == "higher" else "#ff7f0e"
    rows_html = []
    for r, v in top:
        width = min(98.0, 100.0 * v / vmax)
        is_br = r["mode"] == "BR"
        cls = "row br" if is_br else "row"
        color = "#d62728" if is_br else bar_color
        rows_html.append(
            f'<div class="{cls}"><span class="label">{short_name(r)}{" ★" if is_br else ""}</span>'
            f'<div class="barwrap"><div class="bar" style="width:{width:.1f}%;background:{color};"></div></div>'
            f'<span class="val">{v:.3f}</span></div>'
        )
    arrow = "↑ higher = better" if direction == "higher" else "↓ lower = better"
    arrow_color = "#1f77b4" if direction == "higher" else "#ff7f0e"
    return (
        f'<div class="subpanel"><div class="subhead">{axis_label} '
        f'<span style="color:{arrow_color};font-size:11px;">({arrow})</span></div>'
        f'<div class="subdesc">{desc}</div>'
        f'{"".join(rows_html)}</div>'
    )


def eval_panel(rows, title, subtitle):
    sub = [panel_axis(rows, lbl, k, d, vmax, desc) for (lbl, k, d, vmax, desc) in AXES]
    return (
        f'<div class="panel wide"><h2>{title} <small>{subtitle}</small></h2>'
        f'<div class="subgrid">{"".join(sub)}</div></div>'
    )


# ── BR rank summary table ──────────────────────────────────────────
def br_rank_table(rows, eval_label):
    rows_html = []
    for lbl, k, d, _, _ in AXES:
        vals = sorted([(short_name(r), float(r[k])) for r in rows],
                      key=lambda kv: kv[1], reverse=(d == "higher"))
        br_val = next(v for n, v in vals if "BR" in n)
        br_rank = next(i for i, (n, _) in enumerate(vals) if "BR" in n) + 1
        best_name, best_val = vals[0]
        is_top = "BR" in best_name
        cls = "rank-cell top" if is_top else ("rank-cell" if br_rank <= 4 else "rank-cell bad")
        rows_html.append(
            f'<tr><td>{lbl}</td><td><b>{br_val:.3f}</b></td>'
            f'<td class="{cls}">{br_rank} / {len(vals)}</td>'
            f"<td>{best_val:.3f} ({best_name})</td></tr>"
        )
    return (
        f'<div class="ranktable"><div class="subhead">{eval_label}</div>'
        f'<table><thead><tr><th>Axis</th><th>BR value</th><th>BR rank</th><th>Best (config)</th></tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody></table></div>'
    )


# ── Lock comparison panel ──────────────────────────────────────────
def lock_panel():
    metrics = [
        ("φ within-env std", 0.25, 1.15, 1.5,
         "Per-episode std of φ — how much φ moves through a single rollout. "
         "V5's lock pinned this near zero; V8 lets it vary freely."),
        ("φ population mean", 2.39, 1.79, 3.0,
         "Average φ across all envs. Lower in V8 because SHIELD c-comp at −0.10 (vs −0.05 in V5) "
         "cancels the synthetic-LiDAR inflation, so the policy doesn't over-hedge."),
        ("|Pearson(φ, σ_act)| between-env", 0.00, 0.20, 0.5,
         "Does the policy raise φ when actuation noise is higher? V5 said no (lock masked it). V8 says yes (mild)."),
        ("BR safety_score on trainmatch", 0.691, 0.768, 1.0,
         "Collapsed scalar shown only for V5/V8 comparison. Real comparison is multi-axis above."),
    ]
    rows = []
    for lbl, v5, v8, vmax, desc in metrics:
        rows.append(
            f'<div class="pair-row"><div class="pair-label">{lbl}</div>'
            f'<div class="pair-desc">{desc}</div>'
            f'<div class="bars"><span class="pair-name">V5</span>'
            f'<div class="pair-barwrap"><div class="pair-bar v5" style="width:{100*v5/vmax:.1f}%"></div></div>'
            f'<span class="pair-val">{v5:.2f}</span></div>'
            f'<div class="bars"><span class="pair-name">V8</span>'
            f'<div class="pair-barwrap"><div class="pair-bar v8" style="width:{100*v8/vmax:.1f}%"></div></div>'
            f'<span class="pair-val">{v8:.2f}</span></div></div>'
        )
    return (
        '<div class="panel"><h2>Lock removal effect (V5 → V8)</h2>'
        + "".join(rows)
        + '<div class="legend">'
          '<span class="swatch" style="background:#888;"></span>V5 (windowed lock @ 5s)'
          '<span class="swatch" style="background:#d62728;"></span>V8 (no lock, per-step φ)</div></div>'
    )


# ── z_priv probe panel ─────────────────────────────────────────────
def zprobe_panel():
    items = sorted(zprobe["linear_probe"].items(),
                   key=lambda kv: kv[1]["r2_test"], reverse=True)
    bars = []
    for k, info in items:
        r2 = info["r2_test"]
        color = ("#2ca02c" if k in ("actuation_noise_sigma", "com_x", "com_y", "com_z")
                 and r2 > 0.4
                 else ("#1f77b4" if r2 > 0.4 else "#888"))
        label = k.replace("_", " ").replace("noise sigma", "noise σ")
        bars.append(
            f'<div class="row"><span class="label">{label}</span>'
            f'<div class="barwrap"><div class="bar" style="width:{100*r2:.1f}%;background:{color};"></div></div>'
            f'<span class="val">{r2:.2f}</span></div>'
        )
    return (
        '<div class="panel"><h2>z_priv linear-probe R²_test '
        '<small>green = previously blind (V16 R²&lt;0.05), now decoded</small></h2>'
        '<div class="subdesc">Train a linear map z_priv → each priv feature, measure R². '
        'Tells us what the privileged encoder is bottlenecking.</div>'
        + "".join(bars) + "</div>"
    )


# ── Correlation panels ─────────────────────────────────────────────
def corr_panel(title, corr_dict, within_lines, accent, intro):
    items = [(k, v) for k, v in corr_dict.items()
             if k not in ("|applied_force|", "|applied_torque|")]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    items = items[:6]
    rows = []
    for k, v in items:
        width = min(50.0, abs(v) / 0.5 * 50)
        side = "pos" if v > 0 else "neg"
        if v > 0:
            style = f"left:50%;width:{width:.1f}%;background:{accent};"
        else:
            style = f"right:50%;width:{width:.1f}%;background:#ff7f0e;"
        sign = "+" if v > 0 else "−"
        rows.append(
            f'<div class="corr-row"><span class="label">{k}</span>'
            f'<div class="corr-axis"><div class="corr-bar {side}" style="{style}"></div></div>'
            f'<span class="val">{sign}{abs(v):.2f}</span></div>'
        )
    return (
        f'<div class="panel"><h2>{title}</h2>'
        f'<div class="subdesc">{intro}</div>'
        + "".join(rows)
        + '<div class="axis"><span></span><div class="ticks">'
          '<span class="tick" style="left:0%">−0.5</span>'
          '<span class="tick" style="left:25%">−0.25</span>'
          '<span class="tick" style="left:50%">0</span>'
          '<span class="tick" style="left:75%">+0.25</span>'
          '<span class="tick" style="left:100%">+0.5</span></div><span></span></div>'
          f'<div class="note">{"<br>".join(within_lines)}</div></div>'
    )


phi_within = [
    "<b>Within-episode (Pearson φ_t vs signal_t):</b>",
    f"φ_t vs h_t = <b>+{phi['within_episode_pearson']['phi_vs_h']['mean']:.2f}</b> "
    f"<i>(opposite of TISSf shape — TISSf says φ ↑ when h ↓; V8 says φ ↑ when h ↑. V5 was −0.10 under lock.)</i>",
    f"φ_t vs ‖L_g h‖²_t = {phi['within_episode_pearson']['phi_vs_Lgh_norm_sq']['mean']:+.2f}",
]
alpha_within = [
    "<b>Within-episode (Pearson α_t vs signal_t):</b>",
    f"α_t vs h_t = <b>{alpha['within_episode_pearson_with_alpha']['h']['mean']:+.2f}</b>"
    f" &nbsp;&nbsp; α_t vs |tracking_err|_t = <b>{alpha['within_episode_pearson_with_alpha']['|tracking_err|']['mean']:+.2f}</b>",
    f"α_t vs |base_ang_vel|_t = {alpha['within_episode_pearson_with_alpha']['|base_ang_vel|']['mean']:+.2f}"
    f" &nbsp;&nbsp; α_t vs σ_act = {alpha['within_episode_pearson_with_alpha']['actuation_noise_sigma']['mean']:+.2f}",
]


# ── CSS ────────────────────────────────────────────────────────────
CSS = """
:root {
  --br: #d62728; --fixed: #1f77b4; --neg: #ff7f0e; --grey: #888;
  --green: #2ca02c; --bg: #fafafa; --panel: #fff; --border: #d8d8d8;
  --text: #222; --muted: #666; --note-bg: #fff7e6; --note-border: #cc9933;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--text);
       font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 0; padding: 28px 32px; font-size: 14px; line-height: 1.5;
       max-width: 1400px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
h1 small { color: var(--muted); font-weight: normal; font-size: 13px; }
h2.section { font-size: 16px; margin: 28px 0 10px; padding-bottom: 4px;
             border-bottom: 2px solid var(--border); color: #234; }
h2.section .num { color: var(--muted); margin-right: 6px; }
.sub { color: var(--muted); margin-bottom: 14px; font-size: 13px; }
p { margin: 8px 0; }
code { background: #eee; padding: 1px 5px; border-radius: 3px;
       font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.summary { background: #eef6ff; border: 1px solid #99c2ff; border-radius: 6px;
           padding: 12px 16px; margin: 8px 0 16px; font-size: 13px; }
.summary b { color: #0b3d91; }
.summary ul { margin: 4px 0 4px 18px; padding: 0; }
.summary li { margin: 3px 0; }
.grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
@media (max-width: 1100px) { .grid { grid-template-columns: 1fr; } }
.panel { background: var(--panel); border: 1px solid var(--border);
         border-radius: 6px; padding: 14px 18px 16px; }
.panel.wide { grid-column: 1 / -1; }
.panel h2 { font-size: 14px; margin: 0 0 8px;
            border-bottom: 1px solid #eee; padding-bottom: 5px; }
.panel h2 small { color: var(--muted); font-weight: normal; font-size: 12px; margin-left: 6px; }
.subgrid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
@media (max-width: 1100px) { .subgrid { grid-template-columns: 1fr; } }
.subpanel { border: 1px solid #eee; border-radius: 4px; padding: 8px 10px; background: #fcfcfc; }
.subhead { font-size: 12px; font-weight: 600; margin-bottom: 4px; color: var(--text); }
.subdesc { font-size: 11px; color: var(--muted); margin-bottom: 8px; line-height: 1.3; }
.row { display: grid; grid-template-columns: 140px 1fr 52px; align-items: center;
       gap: 7px; margin: 3px 0; font-size: 12px; }
.row .label { text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
              font-size: 11px; }
.row.br .label { font-weight: bold; color: var(--br); }
.row .barwrap { background: #f0f0f0; height: 13px; border-radius: 2px; overflow: hidden; }
.row .bar { height: 100%; }
.row .val { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }
.row.br .val { font-weight: bold; color: var(--br); }
.corr-row { display: grid; grid-template-columns: 150px 1fr 56px; align-items: center;
            gap: 8px; margin: 3px 0; font-size: 13px; }
.corr-row .label { text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.corr-row .val { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }
.corr-axis { position: relative; height: 16px; }
.corr-axis::before { content: ""; position: absolute; top: 0; bottom: 0; left: 50%;
                     width: 1px; background: #888; }
.corr-bar { position: absolute; top: 2px; bottom: 2px; border-radius: 2px; }
.axis { display: grid; grid-template-columns: 150px 1fr 56px; gap: 8px;
        color: var(--muted); font-size: 10px; margin-top: 4px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.axis .ticks { position: relative; height: 11px; }
.axis .tick { position: absolute; transform: translateX(-50%); }
.pair-row { margin: 10px 0; }
.pair-row .pair-label { font-size: 12px; font-weight: 600; margin-bottom: 2px;
                        font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.pair-row .pair-desc { font-size: 11px; color: var(--muted); margin-bottom: 4px;
                       line-height: 1.3; }
.pair-row .bars { display: grid; grid-template-columns: 36px 1fr 50px;
                  gap: 6px; align-items: center; margin: 2px 0; }
.pair-row .pair-name { font-size: 11px; color: var(--muted); text-align: right;
                       font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.pair-row .pair-barwrap { background: #f0f0f0; height: 12px; border-radius: 2px; overflow: hidden; }
.pair-row .pair-bar { height: 100%; }
.pair-row .v5 { background: var(--grey); }
.pair-row .v8 { background: var(--br); }
.pair-row .pair-val { font-size: 11px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.legend { font-size: 11px; color: var(--muted); margin-top: 8px;
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.legend .swatch { display: inline-block; width: 10px; height: 10px;
                  vertical-align: middle; margin: 0 4px 0 12px; border: 1px solid #888; }
.legend .swatch:first-child { margin-left: 0; }
.note { background: var(--note-bg); border: 1px solid var(--note-border);
        border-radius: 4px; padding: 7px 10px; margin-top: 10px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 11px; color: #553300; }
.ranktable { background: #fff; border: 1px solid var(--border); border-radius: 6px;
             padding: 10px 14px; }
.ranktable .subhead { margin-bottom: 6px; font-size: 13px; color: #234; }
.ranktable table { width: 100%; border-collapse: collapse; font-size: 12px; }
.ranktable th, .ranktable td { padding: 5px 8px; text-align: left;
                               border-bottom: 1px solid #eee; }
.ranktable th { background: #f5f5f5; font-weight: 600; }
.ranktable .rank-cell { font-weight: bold; }
.ranktable .rank-cell.top { color: var(--green); }
.ranktable .rank-cell.bad { color: var(--neg); }
.caveat { background: #fff8f8; border: 1px solid #f5b5b5; border-radius: 4px;
          padding: 8px 12px; margin: 8px 0; }
.caveat b { color: #a02020; }
.config { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
          background: #f5f5f5; padding: 10px 12px; border-radius: 4px;
          border: 1px solid #ddd; line-height: 1.6; }
.config .key { color: #234; }
.toc { background: #f5f5f5; border-left: 3px solid #234; padding: 8px 14px;
       margin-bottom: 16px; font-size: 12px; }
.toc a { color: #234; text-decoration: none; margin-right: 14px; }
.toc a:hover { text-decoration: underline; }
"""

# ── Final HTML assembly ────────────────────────────────────────────
html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>V8 full report — PHIWIN_TIGHTCOR_V8, 2026-05-19</title>
<style>{CSS}</style></head><body>

<h1>V8 — PHIWIN_TIGHTCOR_V8 <small>2026-05-19 · 2500 iters · 4096 envs · per-step φ · no lock · SHIELD c-comp at −0.10</small></h1>
<div class="sub">Full report on the adaptive teacher run that closed the lock question. Multi-axis Pareto evaluation, head diagnostics, encoder probe, and lock-removal comparison vs V5 baseline.</div>

<div class="toc">
  <b>Sections:</b>
  <a href="#tldr">TL;DR</a>
  <a href="#what">What V8 is</a>
  <a href="#eval">Multi-axis eval</a>
  <a href="#rank">BR rank summary</a>
  <a href="#heads">Head diagnostics</a>
  <a href="#encoder">Encoder probe</a>
  <a href="#lock">Lock removal</a>
  <a href="#shared">Shared signal</a>
  <a href="#caveats">Caveats</a>
  <a href="#verdict">Verdict</a>
</div>

<h2 class="section" id="tldr"><span class="num">1.</span>TL;DR</h2>
<div class="summary">
  <ul>
    <li><b>BR matches oracle best fixed on trainmatch</b> (tied at safety_score ≈ 0.77, BR is the only 100% completion config).</li>
    <li><b>BR LOSES on OOD</b> — rank 8/16 by safety_score, dominated by B0 α=2 (0.722 vs BR 0.626). V16-era reversal pattern is back. Fall rate 3× higher than B0, collision rate 8.7pp higher.</li>
    <li><b>BR loses in-dist</b> too (rank 4) — low-hedge fixed wins where deploy ≈ train.</li>
    <li><b>The φ lock was unnecessary AND harmful</b> — V8 fixed it (within-env std 4.6×, σ_act sensitivity unmasked).</li>
    <li><b>z_priv encoder is healthy</b> — σ_act R²=0.58, COM R²~0.52. Bottleneck shifted from encoder to head.</li>
    <li><b>Shared-signal analysis (partial regression) found:</b>
      <ul>
        <li><i>tracking_err</i> and <i>base_height</i> dominate BOTH heads (β≈+0.28 and ≈+0.35 respectively). Heads partially coupled at the env level.</li>
        <li>φ DOES have independent σ_act loading (β=−0.17, t=−3.0). Theory-correct, but small effect.</li>
        <li>Within-episode, α reads tracking_err AND h_t; φ reads h_t only. Heads ARE differentiated step-to-step.</li>
        <li><i>friction, mass, COM</i> all have β ≈ 0 — encoder sees them, heads ignore them. The bottleneck is at the head, not z_priv.</li>
      </ul></li>
    <li><b>Honest read on the paper:</b> trainmatch win is real but narrow. OOD loss means the "adaptive teacher generalizes to deploy" story is NOT supported. V9 — strip tracking_err + base_height from priv — is the necessary next experiment.</li>
  </ul>
</div>

<h2 class="section" id="what"><span class="num">2.</span>What V8 is</h2>
<p>V8 is the SHIELD-corrected, per-step-φ training of the adaptive CBF teacher. Single-variable derivation from V5:</p>
<div class="config">
<span class="key">V5 → V7:</span> c clamp range changed from (−0.05, −0.05) to (−0.10, −0.10).<br>
&nbsp;&nbsp;&nbsp;&nbsp;Reason: SHIELD synthetic-LiDAR clustering inflates every perceived obstacle by +0.10m.<br>
&nbsp;&nbsp;&nbsp;&nbsp;c clamp = −0.10 fully cancels this; V5's −0.05 only undid half.<br>
<span class="key">V7 → V8:</span> phi_lock_mode = "windowed" → "per_step".<br>
&nbsp;&nbsp;&nbsp;&nbsp;V7 still locked φ for 250-step (5s) windows. V8 lets φ vary every control step.<br>
&nbsp;&nbsp;&nbsp;&nbsp;Open question Test 1 confirmed: lock was hurting deploy. V8 verifies from-scratch training.<br>
<span class="key">Everything else from V5:</span> α head adaptive, α_param_range = (0.5, 3.0), φ_param_range = (0.5, 2.0),<br>
&nbsp;&nbsp;&nbsp;&nbsp;within-episode σ_act regime resampled at 5s windows, freeze_alpha_value = None,<br>
&nbsp;&nbsp;&nbsp;&nbsp;split priv-MLP + grid-CNN encoder, RMA teacher-student, PPO.<br>
<span class="key">Train length:</span> 2500 iters, 4096 envs, ~3.3h on lab box RTX 5090.<br>
<span class="key">Ckpt:</span> logs/rsl_rl/cbf_go2_teacher_rma/2026-05-19_14-18-46/model_2499.pt
</div>

<h2 class="section" id="eval"><span class="num">3.</span>Multi-axis evaluation</h2>
<p>Three eval distributions:</p>
<ul>
  <li><b>Trainmatch</b> (deploy-realistic, DR matches training): tests robustness to the deploy distribution we trained for. Includes tight corridors, full σ_act range, c=−0.10, adversarial QP noise.</li>
  <li><b>In-dist</b> (DR matches training, training-task obstacle scatter): a pure sanity check that the policy works in the regime it was trained on.</li>
  <li><b>OOD</b> (deploy-realistic, NOT matched to training): default obstacle scatter (no tight corridors), narrow σ_act range, c=−0.05 (SHIELD perception bias UNcompensated), no adversarial QP noise. This is the closest thing we have to "novel deploy distribution".</li>
</ul>
<p>Six axes per eval. The single safety_score collapse used earlier hides trade-offs — these panels show them directly. BR (red) vs top 8 fixed baselines per axis. ★ marks BR.</p>

<div class="grid">
  {eval_panel(trainmatch, "Trainmatch eval (DEPLOY-REALISTIC, matches training)",
              "BR is rank 1 in completion; pays cost on path/deflection — ties best fixed on safety_score")}
</div>
<div class="grid" style="margin-top:16px;">
  {eval_panel(indist, "In-dist eval (DEPLOY = TRAIN)",
              "BR is rank 1 in completion; low-α B0 dominates efficiency axes")}
</div>
<div class="grid" style="margin-top:16px;">
  {eval_panel(ood, "OOD eval (DEPLOY ≠ TRAIN, c=−0.05 uncompensated)",
              "BR rank 8/16 by safety_score — adaptive teacher LOSES on OOD")}
</div>

<h2 class="section" id="rank"><span class="num">4.</span>BR rank summary across 3 evals</h2>
<div class="grid">
  {br_rank_table(trainmatch, "Trainmatch eval (deploy ≈ train)")}
  {br_rank_table(indist, "In-dist eval (deploy = train)")}
</div>
<div class="grid" style="margin-top:16px;">
  {br_rank_table(ood, "OOD eval (deploy ≠ train)")}
  <div class="panel"><h2>Summary: where adaptation pays off</h2>
    <table style="width:100%; border-collapse:collapse; font-size:12px; margin-top:6px;">
      <thead><tr style="background:#f5f5f5;"><th style="padding:5px;text-align:left">Eval</th><th style="padding:5px;text-align:right">BR safety_score</th><th style="padding:5px;text-align:right">Best fixed</th><th style="padding:5px;text-align:left">Verdict</th></tr></thead>
      <tbody>
        <tr><td style="padding:5px">Trainmatch</td><td style="padding:5px;text-align:right;font-weight:bold">0.768</td><td style="padding:5px;text-align:right">0.769 (B2 α=2 λ=3)</td><td style="padding:5px;color:#1f77b4">tied with oracle</td></tr>
        <tr><td style="padding:5px">In-dist</td><td style="padding:5px;text-align:right">0.614</td><td style="padding:5px;text-align:right;font-weight:bold">0.710 (B0 α=0.5)</td><td style="padding:5px;color:#a02020">BR loses (−9.6pp)</td></tr>
        <tr><td style="padding:5px">OOD</td><td style="padding:5px;text-align:right">0.626</td><td style="padding:5px;text-align:right;font-weight:bold">0.722 (B0 α=2)</td><td style="padding:5px;color:#a02020">BR loses (−9.6pp)</td></tr>
      </tbody>
    </table>
    <p style="font-size:12px; margin-top:10px; color:#444;">
      BR only wins where the eval distribution matches what it was trained for.
      On both in-dist and OOD, fixed-α B0 baselines win. The adaptive teacher
      <b>does not yet generalize across deploy distributions</b> — same pattern
      as V16-era results.</p>
  </div>
</div>

<h2 class="section" id="heads"><span class="num">5.</span>Adaptation head diagnostics</h2>
<p>What signals are α and φ actually attending to? Bars show between-env Pearson
correlation; orange box at the bottom of each panel shows the within-episode correlations
(per-step within a single rollout).</p>

<div class="grid">
  {corr_panel("What φ attends to", phi["correlations_with_phi"], phi_within, "#d62728",
              "φ controls the actuation-uncertainty margin: φ‖L_g h‖² in the CBF constraint. "
              "Theory (Kolathaya ISSf / TISSf) says φ should rise with actuation noise σ_act and with proximity to barrier (h ↓).")}
  {corr_panel("What α attends to", alpha["correlations_with_alpha"], alpha_within, "#1f77b4",
              "α controls the CBF recovery rate: ḣ ≥ −α(h − c). Theory: low α when tracking is poor (don't push to the limit); "
              "high α when tracking is clean.")}
</div>

<div class="caveat" style="margin-top:14px;">
  <b>Surprise:</b> Both heads' strongest between-env signal is <code>|tracking_err|</code>
  (+0.35 for φ, +0.36 for α). σ_act — φ's textbook signal — is third for φ and basically
  zero for α. φ's within-episode shape is the wrong sign for TISSf. Interpretation: both
  heads learned to use the same "things are bad / things are fine" proxy (tracking error
  is the easiest within-episode signal that survives LLN washout over 800 steps).
</div>

<h2 class="section" id="encoder"><span class="num">6.</span>z_priv encoder probe</h2>
<p>Train a linear regression z_priv → priv feature, measure test R². Tells us
whether the priv encoder bottleneck is starving any head of necessary information.</p>
<div class="grid">
  {zprobe_panel()}
  <div class="panel"><h2>Comparison vs V16 baseline</h2>
  <div class="subdesc">Earlier diagnostic memory noted: "z_priv blind to COM &amp; σ_actuation (R²&lt;0.05)". V8 numbers:</div>
  <table style="width:100%; border-collapse:collapse; font-size:12px; margin-top:6px;">
    <thead><tr style="background:#f5f5f5;"><th style="padding:5px;text-align:left">Feature</th><th style="padding:5px;text-align:right">V16-era</th><th style="padding:5px;text-align:right">V8</th><th style="padding:5px;text-align:left">Status</th></tr></thead>
    <tbody>
      <tr><td style="padding:5px">actuation_noise_σ</td><td style="padding:5px;text-align:right;color:#888">&lt;0.05</td><td style="padding:5px;text-align:right;font-weight:bold">0.58</td><td style="padding:5px;color:#2ca02c">DECODED</td></tr>
      <tr><td style="padding:5px">com_x</td><td style="padding:5px;text-align:right;color:#888">&lt;0.05</td><td style="padding:5px;text-align:right;font-weight:bold">0.55</td><td style="padding:5px;color:#2ca02c">DECODED</td></tr>
      <tr><td style="padding:5px">com_y</td><td style="padding:5px;text-align:right;color:#888">&lt;0.05</td><td style="padding:5px;text-align:right;font-weight:bold">0.52</td><td style="padding:5px;color:#2ca02c">DECODED</td></tr>
      <tr><td style="padding:5px">com_z</td><td style="padding:5px;text-align:right;color:#888">&lt;0.05</td><td style="padding:5px;text-align:right;font-weight:bold">0.52</td><td style="padding:5px;color:#2ca02c">DECODED</td></tr>
      <tr><td style="padding:5px">base_height</td><td style="padding:5px;text-align:right">&mdash;</td><td style="padding:5px;text-align:right;font-weight:bold">0.85</td><td style="padding:5px;color:#1f77b4">strong</td></tr>
      <tr><td style="padding:5px">friction</td><td style="padding:5px;text-align:right">&mdash;</td><td style="padding:5px;text-align:right;font-weight:bold">0.07</td><td style="padding:5px;color:#a02020">still weak</td></tr>
    </tbody></table>
  <p style="font-size:12px; margin-top:10px; color:#444;">The COM + σ_act unlocking lines up with V5→V8 changes
  (c-comp + within-episode σ_act + lock removal). Friction remains weak — z reads friction through gait-effect proxies, not directly.</p>
  </div>
</div>

<h2 class="section" id="lock"><span class="num">7.</span>Lock removal effect (V5 → V8)</h2>
<p>The φ output lock held φ constant for 250-step (5s) windows during training, on the
hypothesis that per-step φ would be too noisy for credit assignment. Test 1 (eval-only
lock removal on the V5 checkpoint) suggested the lock was hurting deploy. V8 trains
without the lock from scratch.</p>
<div class="grid">
  {lock_panel()}
  <div class="panel"><h2>Test 1 recap (eval-only lock removal on V5 ckpt)</h2>
  <div class="subdesc">Before training V8 from scratch, we ran the same V5 ckpt through two eval pipelines:
  one with windowed lock (training-matched), one with per-step φ.</div>
  <table style="width:100%; border-collapse:collapse; font-size:12px;">
    <thead><tr style="background:#f5f5f5;"><th style="padding:5px;text-align:left">Metric</th><th style="padding:5px;text-align:right">V5 ckpt + lock eval</th><th style="padding:5px;text-align:right">V5 ckpt + no-lock eval</th><th style="padding:5px;text-align:right">V8 ckpt + no-lock eval</th></tr></thead>
    <tbody>
      <tr><td style="padding:5px">BR goal_reach</td><td style="padding:5px;text-align:right">0.923</td><td style="padding:5px;text-align:right">0.985</td><td style="padding:5px;text-align:right;font-weight:bold">1.000</td></tr>
      <tr><td style="padding:5px">BR φ_mean</td><td style="padding:5px;text-align:right">3.16</td><td style="padding:5px;text-align:right">1.71</td><td style="padding:5px;text-align:right;font-weight:bold">2.43</td></tr>
      <tr><td style="padding:5px">BR safety_score</td><td style="padding:5px;text-align:right">~0.69</td><td style="padding:5px;text-align:right">~0.69</td><td style="padding:5px;text-align:right;font-weight:bold">0.77</td></tr>
    </tbody></table>
  <p style="font-size:12px; margin-top:10px; color:#444;">Reading: lock removal alone at eval recovered most of the gap (Test 1).
  V8 training-without-lock pushes past that, by giving the policy gradient signal it couldn't see when φ was held constant.</p>
  </div>
</div>

<h2 class="section" id="shared"><span class="num">8.</span>Shared-signal analysis (partial regression)</h2>
<p>Caveat #4 from the earlier writeup asked: are α and φ heads really specialized,
or are they both reading the same "tracking_err is high" lump-sum symptom?
Method: multivariate regression of each head on standardized priv features.
Standardized β with |t| &gt; 2 = independent effect after controlling for everything else.</p>

<div class="grid">
  <div class="panel"><h2>α head — partial β (after controlling for other priv features)</h2>
    <div class="subdesc">Marginal Pearson masks confounding; partial β is the head's
    independent sensitivity to each feature.</div>
    <div class="row"><span class="label">base_height</span><div class="barwrap"><div class="bar" style="width:43%;background:#1f77b4;"></div></div><span class="val">+0.427 ★</span></div>
    <div class="row"><span class="label">tracking_err</span><div class="barwrap"><div class="bar" style="width:27%;background:#1f77b4;"></div></div><span class="val">+0.271 ★</span></div>
    <div class="row"><span class="label">base_ang_vel</span><div class="barwrap"><div class="bar" style="width:13.7%;background:#ff7f0e;"></div></div><span class="val">−0.137</span></div>
    <div class="row"><span class="label">σ_act</span><div class="barwrap"><div class="bar" style="width:9.5%;background:#ff7f0e;"></div></div><span class="val">−0.095</span></div>
    <div class="row"><span class="label">com_norm</span><div class="barwrap"><div class="bar" style="width:5.2%;background:#888;"></div></div><span class="val">−0.052</span></div>
    <div class="row"><span class="label">base_mass</span><div class="barwrap"><div class="bar" style="width:5.1%;background:#888;"></div></div><span class="val">+0.051</span></div>
    <div class="row"><span class="label">friction</span><div class="barwrap"><div class="bar" style="width:4.4%;background:#888;"></div></div><span class="val">−0.044</span></div>
    <div class="note">★ = independent effect (|t| &gt; 2). α reads base_height and
    tracking_err as primary signals. σ_act, friction, mass, COM all ignored.</div>
  </div>

  <div class="panel"><h2>φ head — partial β (after controlling for other priv features)</h2>
    <div class="subdesc">Same regression, φ as target. Compare to α: shared base_height
    and tracking_err loadings, but φ has its own σ_act sensitivity.</div>
    <div class="row"><span class="label">base_height</span><div class="barwrap"><div class="bar" style="width:31.4%;background:#d62728;"></div></div><span class="val">+0.314 ★</span></div>
    <div class="row"><span class="label">tracking_err</span><div class="barwrap"><div class="bar" style="width:28.2%;background:#d62728;"></div></div><span class="val">+0.282 ★</span></div>
    <div class="row"><span class="label">σ_act</span><div class="barwrap"><div class="bar" style="width:16.7%;background:#ff7f0e;"></div></div><span class="val">−0.167 ★</span></div>
    <div class="row"><span class="label">base_ang_vel</span><div class="barwrap"><div class="bar" style="width:11.9%;background:#ff7f0e;"></div></div><span class="val">−0.119</span></div>
    <div class="row"><span class="label">friction</span><div class="barwrap"><div class="bar" style="width:7.1%;background:#888;"></div></div><span class="val">−0.071</span></div>
    <div class="row"><span class="label">base_mass</span><div class="barwrap"><div class="bar" style="width:6.4%;background:#888;"></div></div><span class="val">+0.064</span></div>
    <div class="row"><span class="label">com_norm</span><div class="barwrap"><div class="bar" style="width:0.5%;background:#888;"></div></div><span class="val">+0.005</span></div>
    <div class="note">★ = independent effect (|t| &gt; 2). φ has independent σ_act
    loading (β=−0.17, t=−3.0). <b>Theory-correct direction (Kolathaya ISSf).</b>
    Small but real.</div>
  </div>
</div>

<div class="panel" style="margin-top:14px;">
  <h2>Head specialization: where α and φ differ</h2>
  <table style="width:100%; border-collapse:collapse; font-size:12px;">
    <thead><tr style="background:#f5f5f5;">
      <th style="padding:5px;text-align:left">Feature</th>
      <th style="padding:5px;text-align:right">α partial β</th>
      <th style="padding:5px;text-align:right">φ partial β</th>
      <th style="padding:5px;text-align:left">Read</th>
    </tr></thead>
    <tbody>
      <tr><td style="padding:5px">tracking_err_norm</td><td style="padding:5px;text-align:right">+0.27</td><td style="padding:5px;text-align:right">+0.28</td><td style="padding:5px;color:#a02020">COUPLED (~identical)</td></tr>
      <tr><td style="padding:5px">base_height</td><td style="padding:5px;text-align:right">+0.43</td><td style="padding:5px;text-align:right">+0.31</td><td style="padding:5px;color:#a02020">shared, α uses more</td></tr>
      <tr><td style="padding:5px">σ_act</td><td style="padding:5px;text-align:right">−0.10</td><td style="padding:5px;text-align:right;font-weight:bold">−0.17 ★</td><td style="padding:5px;color:#1f77b4">φ-specific (TISSf-aligned)</td></tr>
      <tr><td style="padding:5px">base_ang_vel</td><td style="padding:5px;text-align:right">−0.14</td><td style="padding:5px;text-align:right">−0.12</td><td style="padding:5px;color:#888">shared, both small</td></tr>
      <tr><td style="padding:5px">friction</td><td style="padding:5px;text-align:right">−0.04</td><td style="padding:5px;text-align:right">−0.07</td><td style="padding:5px;color:#888">neither uses</td></tr>
      <tr><td style="padding:5px">base_mass</td><td style="padding:5px;text-align:right">+0.05</td><td style="padding:5px;text-align:right">+0.06</td><td style="padding:5px;color:#888">neither uses</td></tr>
      <tr><td style="padding:5px">com_norm</td><td style="padding:5px;text-align:right">−0.05</td><td style="padding:5px;text-align:right">+0.005</td><td style="padding:5px;color:#888">neither uses</td></tr>
    </tbody>
  </table>
  <p style="font-size:12px; margin-top:8px; color:#444;">
    <b>Within-episode partial β</b> (regress head_t on [tracking_err_t, h_t] within
    each env, mean across envs):<br>
    α: tracking_err_t = +0.13, h_t = +0.29 — <i>α reads BOTH</i><br>
    φ: tracking_err_t = +0.03, h_t = +0.19 — <i>φ reads h only</i><br>
    Heads ARE differentiated step-to-step, even though they share signals between envs.
  </p>
  <p style="font-size:12px; margin-top:8px; color:#444;">
    <b>Feature ↔ feature correlations between envs are all |r| &lt; 0.18.</b>
    tracking_err is NOT acting as a lump-sum proxy for σ_act / friction / mass / COM
    in this rollout. The +0.27/+0.28 marginal Pearson is a genuine direct signal.
  </p>
</div>

<h2 class="section" id="caveats"><span class="num">9.</span>Open caveats</h2>
<div class="caveat">
  <b>1. The adaptive teacher does NOT yet generalize across deploy distributions.</b>
  BR matches oracle best fixed only on trainmatch. On both in-dist and OOD, fixed-α
  B0 baselines beat BR by ~10pp safety_score. Same pattern as V16-era results.
</div>
<div class="caveat">
  <b>2. Heads partially coupled at env level, differentiated within-episode.</b>
  Both heads gate strongly off tracking_err (β≈+0.28) and base_height (α: +0.43, φ: +0.31).
  Only φ has independent σ_act sensitivity (β=−0.17, t=−3.0). Within-episode, α reads
  tracking_err AND h_t; φ reads h_t only. So the heads ARE specialized at the temporal
  level but share strong env-class signals.
</div>
<div class="caveat">
  <b>3. friction, base_mass, COM all have β ≈ 0 for both heads</b> — despite z_priv
  encoding COM at R² ≈ 0.52. The bottleneck is at the HEAD, not the encoder. The
  policy could read COM/friction/mass from z, but doesn't. Why: the reward landscape
  doesn't reward differentiating these axes (within current DR/eval setup).
</div>
<div class="caveat">
  <b>4. φ goes anti-TISSf within episode.</b> Within-ep Pearson(φ_t, h_t) = +0.16
  (theory says negative). φ raises when h is high, opposite of textbook TISSf shape.
  But within-ep partial β of φ_t on h_t is +0.19 (positive but small). Mechanism is
  context-aware margin, not TISSf-shape regression.
</div>
<div class="caveat">
  <b>5. Friction stays opaque to z_priv (R² 0.07).</b> Friction is observable only through
  gait-effect proxies; encoder doesn't extract it cleanly. Doesn't matter for current
  V8 since neither head reads friction anyway, but limits the ceiling of any
  friction-curriculum experiment.
</div>

<h2 class="section" id="verdict"><span class="num">10.</span>Verdict and next steps</h2>
<p><b>V8 is a partial win, not the new baseline.</b> The framework fixes (SHIELD c-comp,
lock removal, encoder unblocking) are real and should stay. But the OOD reversal
shows the adaptive teacher hasn't yet learned a deploy-generalizing strategy. The
trainmatch win is what you'd expect from a policy tuned to one specific eval
distribution.</p>

<p><b>The specific failure mode:</b> both heads gate primarily off tracking_err and
base_height. tracking_err is partially observable (cmd − measured_base_vel); base_height
is partially observable (IMU + height sensor). Neither is "privileged" in the RMA sense.
On OOD, these signals are present but generated by different causes, so the heads fire
their trained mappings on out-of-domain inputs.</p>

<p><b>V9 hypothesis:</b> strip tracking_err AND base_height from the priv obs. Force
the heads to read σ_act, friction, mass, COM (true privileged info) via z_priv. The
φ head already shows independent σ_act loading — V9 would test if α can develop
analogous specialization on a different priv axis, and whether either head can
generalize across deploy distributions when forced to depend on truly privileged
signals.</p>

<p><b>Next steps:</b></p>
<ul>
  <li>V9: priv obs = [friction, base_mass, σ_act, COM]. Drop tracking_err, base_height,
  base_ang_vel, applied_force/torque from priv layer. Retrain V8 setup with this priv.</li>
  <li>Update professor with V8 + OOD result + shared-signal analysis. Honest framing.</li>
  <li>Backlog: friction-curriculum experiment, c/a/b head adaptation. Lower priority now
  given V9 is the natural next step.</li>
  <li>Hold off on hardware deploy until V9 settles whether priv design matters.</li>
</ul>

<div style="margin-top:24px; color:var(--muted); font-size:11px; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; border-top:1px solid var(--border); padding-top:8px;">
  Generated by scripts/render_v8_html.py · Data: ~/Desktop/safety-go2/data_from_lab/wk3tight8/<br>
  Population statistics — φ: mean={phi['phi_population_mean']:.2f}, pop std={phi['phi_population_std']:.2f}, within-env std={phi['phi_within_env_std_mean']:.2f};
  α: mean={alpha['alpha_population_mean']:.2f}, pop std={alpha['alpha_population_std']:.2f}, within-env std={alpha['alpha_within_env_std_mean']:.2f}
</div>

</body></html>
"""

OUT.write_text(html)
print(f"Wrote {OUT}")
