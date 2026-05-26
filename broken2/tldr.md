# TLDR — 2026-05-12

## Goal

RL-learned state-conditional CBF parameters (α, φ, a, b, c) for Go2.
Win = adaptive policy beats best fixed-α on combined fall+stuck by ≥3pp on ≥1 task.

## Layer 1 (α-only) — exhausted

| Run | What changed | Verdict |
|---|---|---|
| v3.0a-d | Monolithic CNN encoder | FAIL: encoder collapse — grid path starved dyn path for gradient |
| v3.0e | Pretrain+freeze priv encoder | WEAK: α saturated at 5.0 (std=0) |
| v3.0f | + analytical aux loss + v_along_cmd | WEAK, worse: α=5 again, lost v3.0e's state-reactivity |
| v3.1 | RMA split encoders + drop v_along_cmd + AUX 0.005 + no cbf_state in policy | WEAK but big: α=4.0, std=1.5 (saturation broken). Geometry-reactive (h r=+0.58), flat on DR (friction r=+0.008). |
| v3.2 | + 45% adversarial + grazing 28° + speed 1.2 + AUX=0 + priv_fov FOV-gated CBF | MARGINAL: α μ=3.05 σ=1.96, qp_active doubled to 0.23-0.26, HeavyCOM combined +3.01pp (edge of significance). DR corrs still <0.10. |
| v3.2.1 | + z_priv 8→16 (capacity test) | FAIL: friction r=0.034 (unchanged), HeavyCOM win didn't replicate (-0.15pp). Capacity not the bottleneck. |

## What Layer 1 told us

- Architecture (RMA split + AUX=0 + adversarial data) **broke saturation**. α now varies with what the policy can perceive.
- α is **geometry-reactive** (slack r=0.84, h r=0.71, ‖L_g h‖² r=-0.66) — NOT env-class adaptive.
- DR features stay at r<0.10 regardless of encoder capacity (8 vs 16 priv dims). Friction simply isn't action-relevant when the only knob is α, because α controls *aggressiveness* given a margin; it doesn't trade off against slip risk in any obvious way.
- v3.2's HeavyCOM +3.01pp wasn't load-bearing — disappeared under a re-train.

**Conclusion:** with α as the only released parameter, this is the ceiling. Need another knob whose physics ties to a DR axis.

## Layer 2 (next) — release φ + actuation-noise DR

φ multiplies ‖L_g h‖² in the CBF rhs (Kolathaya 2018 ISSf term). It directly compensates for input disturbances — exactly what actuation noise *is*. Hypothesis: with φ released and a wider actuation-noise DR axis, φ correlates with the noise level (z_priv now has actuation_noise_sigma to encode).

- Releases φ (was frozen at 0). QP wiring already in place.
- Adds `actuation_noise_sigma` to teacher priv obs (priv_dim 15→16).
- Widens actuation noise DR: σ_max 0.10 → 0.20.
- Keeps Layer 1's adversarial planner + RMA arch + AUX=0.
- New OOD eval: `Isaac-CBF-Go2-RMA-HighActuationNoise-v0` (σ_max=0.30).

## Decision criterion (same as Layer 1)

- **PASS:** any DR-feature |Pearson(φ, ·)| > 0.20 AND BR combined beats best-fixed by ≥3pp on ≥1 task
- **AMBIG:** DR corr visible but combined ties — tune
- **WEAK:** DR corr 0.10-0.20 — marginal
- **FAIL:** all DR corrs <0.10

## Tools

- `scripts/diagnose_z.py` — Z health
- `scripts/probe_z_linear.py` — per-feature R² of Z → priv (now expects 16 priv dims; use `--priv_dim 15` for v3.x evals)
- `scripts/diagnose_alpha_corr.py` — within-distribution α correlation
- `scripts/aggregate_v3_results.py --version <vN>` — one-shot verdict

## Next step

Sync Layer 2 files from Mac, launch on lab in fresh tmux.

```bash
# from Mac
rsync -av ~/Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/ \
  chrisliang@130.64.84.163:Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/
rsync -av ~/Desktop/safety-go2/scripts/train_and_eval_layer2.sh ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  chrisliang@130.64.84.163:Desktop/safety-go2/scripts/

# on lab
tmux new -s layer2
cd ~/Desktop/safety-go2/IsaacLab
~/Desktop/safety-go2/scripts/train_and_eval_layer2.sh 2>&1 | tee logs/train_and_eval_layer2.log
```

If Layer 2 fails too, escalate: Layer 3 (release `a` + perception-bias DR), or pivot to oracle teacher / curriculum.
