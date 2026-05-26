# Per-axis stress eval protocol

Goal: empirically attribute the adaptive-teacher win to specific DR axes. For each axis, narrow every *other* axis to its mean / zero and widen *one* axis to a stress range. Compare the BR teacher to the best fixed baseline (or a representative B1) on that single-axis-wide distribution. The gap on each axis tells us which DR axis is actually generating the adaptive signal.

Decision rule per axis:
- **Gap > 0 pp (BR > fixed):** axis genuinely demands adaptation; the teacher is adapting on this axis.
- **Gap ≈ 0 pp:** axis isn't getting adaptive signal in training (or fixed handles it fine).
- **Gap < 0 pp:** teacher is *worse* than fixed on this axis — adaptation is misallocated or the reward shaping pushed it into a bad basin for this axis.

This replaces theory-driven reward iteration with empirical-driven DR iteration. If an axis gap is flat, we widen its training DR. If an axis gap is negative, we investigate and possibly widen DR or drop the reward shaping that's miscalibrating that axis.

## Stress configs

All inherit from `CbfGo2EnvCfg_LAYER3_PUSH_A_C` (so the same priv obs, planner mix, reward stack, perception_mode, push event structure). The NARROW base collapses every DR knob to mean / zero. Each axis variant inherits from NARROW and widens one knob.

### `CbfGo2EnvCfg_STRESS_NARROW` — reference (all DR collapsed)

Expected: BR ≈ fixed. No axis demands adaptation.

| axis | NARROW setting |
|---|---|
| friction (static, dynamic) | (0.7, 0.7), (0.6, 0.6) — single point |
| add_base_mass | zero range |
| add_base_com | zero range (no COM offset DR) |
| applied force/torque | disabled (event removed) |
| push event | disabled (`events.push_robot = None`) |
| actuation_noise_sigma_max | 0.0 |
| obstacle_radius_perception_error_max | 0.0 |
| obstacle_radius_perception_error_range | None (defaults to (0, 0)) |
| obstacle_position_noise_sigma_max | 0.0 |

### `CbfGo2EnvCfg_STRESS_FRICTION` — friction wide, others narrow

Tests: does α adapt to friction? The locked-best teacher had `Pearson(α, base_height) = −0.66`, suggesting α adapts to *something*, but base_height isn't friction. Friction adaptiveness should still register here if friction-DR width drives any of the +1.6 pp gap.

| change vs NARROW |
|---|
| friction (static_friction_range) → (0.15, 1.50) |
| friction (dynamic_friction_range) → (0.10, 1.30) |

(Same wide range as the existing "Near-OOD" config.)

### `CbfGo2EnvCfg_STRESS_COM` — COM offset wide, others narrow

Tests: does α adapt to COM offset? Strongest correlation observed in OMNI diagnostic (α–base_height = −0.66). Expect *largest gap* here if base_height is the dominant DR axis.

| change vs NARROW |
|---|
| add_base_com → wide range (need to audit Isaac Lab default and match HEAVY_COM range) |

### `CbfGo2EnvCfg_STRESS_SIGMA_ACT` — actuation noise wide, others narrow

Tests: does φ adapt to actuation noise (per Kolathaya 2018 ISSf)? Theory predicts φ* ∝ 1/(2σ²). If φ is genuinely adaptive, gap should grow with σ_act range.

| change vs NARROW |
|---|
| actuation_noise_sigma_max → 0.20 (2× training) |

### `CbfGo2EnvCfg_STRESS_RADIUS_ERROR` — perception error wide, others narrow

Tests: does c adapt to perception bias? Theory predicts c < 0 when perceived R is over-estimated (shifts h-boundary outward to compensate).

| change vs NARROW |
|---|
| obstacle_radius_perception_error_range → (−0.30, +0.30) |

### `CbfGo2EnvCfg_STRESS_PUSH` — push disturbance enabled, others narrow

Tests: does `a` adapt to bounded disturbance (Dean 2019 additive slack)?

| change vs NARROW |
|---|
| events.push_robot → re-enable with ±1 m/s every 5-10s (copy from LAYER3_PUSH) |

## Eval pipeline

For each candidate teacher (locked-best, locked-best+DEFL, omni, omni+DEFL — whichever survive the sweep), run:

```
for axis in [NARROW, FRICTION, COM, SIGMA_ACT, RADIUS_ERROR, PUSH]:
    eval_baseline.py \
        --task Isaac-CBF-Go2-RMA-Stress-${axis}-v0 \
        --num_envs 64 --steps_per_config 1000 \
        --modes B0,B1,B2,BR \
        --checkpoint <ckpt> \
        --output_dir logs/stress_eval/<teacher>_${axis}/
```

Per axis, compute:
- `joint_actual_BR`
- `joint_actual_best_fixed`
- `gap = BR − best_fixed` (in pp)

Plot a table:

```
                    NARROW   FRICTION  COM    SIGMA   RADIUS   PUSH
locked-best         X        X         X      X       X        X
locked-best+DEFL    X        X         X      X       X        X
omni                X        X         X      X       X        X
omni+DEFL           X        X         X      X       X        X
```

Each cell = gap (BR − best fixed), pp. The shape of the table tells us:
- Which axis is the dominant adaptive driver (largest gap).
- Whether the deflection penalty preserves the adaptive win across axes or trades some for others.
- Whether omni (truth perception) recovers any axis the locked-best doesn't.

## Wall-time

Per axis eval: ~3-5 min (64 envs × 1000 steps × 8 baseline modes including BR).
6 axes × 4 teachers = 24 evals × ~5 min = **~2 hours**.

Can run after the deflection sweep completes. No retraining needed.

## What this does NOT test

- **Combinations of axes.** Real deploy is everything wide simultaneously. The per-axis stress isolates *attribution*; the wide-everything eval (= current training distribution) measures *combined* gap. Both matter; the per-axis is the diagnostic.
- **OOD on each axis (beyond training range).** We're testing in-distribution width contributions. OOD generalization is a separate concern (already covered by HeavyCOM, HighActuationNoise, etc.).
- **The reverse counterfactual.** This doesn't tell us "if we widen friction DR more during training, would the gap grow?" That requires retraining.

## Implementation order

1. Audit Isaac Lab Go2 default DR events (add_base_mass, add_base_com, applied force/torque) — need exact param names to override.
2. Add `CbfGo2EnvCfg_STRESS_NARROW` config + verify all DR is collapsed via a smoke test (priv obs values should be near-zero variance across envs).
3. Add 5 axis-wide configs inheriting from NARROW.
4. Register 6 gym tasks.
5. Add eval sweep script `scripts/stress_eval_sweep.sh` that takes a teacher checkpoint and runs all 6 axes sequentially.
6. Add analysis script `scripts/parse_stress_eval.py` that pulls gaps from each CSV and prints the attribution table.

Implementation cost: ~2 hours coding, runnable after deflection sweep finishes.
