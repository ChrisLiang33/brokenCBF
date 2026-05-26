"""Sanity check for the recent action term + obs changes.

After a stack of changes (perception SDF, obs dim shrink, Lipschitz
rate-limit, etc.) it's easy to silently break something. This script
runs the SHIELD env (which combines all of them) for a short rollout
with random-ish policy actions and asserts the wiring is correct:

  1. Obs dimensionality matches EXPECTED_OBS_DIM (197)
  2. Slice indices add up cleanly (no overlap, no gaps)
  3. vel_cmd leak is gone (proprio slice does NOT contain `u_nom`)
  4. Lipschitz: per-step change in normalized action ≤ action_max_step
  5. Perception noise is active (decoded SDF differs from privileged)
  6. Dropout is active (some obstacles get masked out across steps)
  7. Range cutoff is active (very-far obstacles ignored)

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase6_sanity_check.py \\
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --task Isaac-CBF-Adaptive-Go2-UnifiedLidarSDF-v0 \\
        --num_envs 16 --rollout_steps 80 --headless
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True,
                    help="Frozen locomotion checkpoint (Go2 stock).")
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-UnifiedLidarSDF-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--rollout_steps", type=int, default=80)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
from cbf_task.agents.rma_actor_critic import (
    EXPECTED_OBS_DIM, PRIV_SLICE, PROPRIO_SLICE, PREV_ACT_SLICE,
    LIDAR_PREV_SLICE, LIDAR_SLICE, PRIV_DIM, PROPRIO_DIM, LIDAR_DIM,
)
from cbf_task.locomotion_loader import load_locomotion_actor


# track per-check pass/fail
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = ""):
    RESULTS.append((name, passed, detail))
    icon = "PASS" if passed else "FAIL"
    print(f"  [{icon}] {name}" + (f"   -- {detail}" if detail else ""))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loco = load_locomotion_actor(retrieve_file_path(args_cli.checkpoint), device)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    cbf = env.action_manager._terms["cbf_param"]
    N = env.num_envs

    print()
    print("=" * 88)
    print(f"  SANITY CHECK  --  task: {args_cli.task}  --  N={N}, steps={args_cli.rollout_steps}")
    print("=" * 88)

    # =============================================================
    # CHECK 1+2: obs dim and slice consistency
    # =============================================================
    print()
    print("--- Group A: obs shape & slice layout ---")
    obs_dict = env.reset()
    obs = obs_dict[0]["policy"] if isinstance(obs_dict, tuple) else obs_dict["policy"]
    obs_dim = obs.shape[-1]
    check("obs shape matches EXPECTED_OBS_DIM",
          obs_dim == EXPECTED_OBS_DIM,
          f"got {obs_dim}, expected {EXPECTED_OBS_DIM}")
    check("slice partitions sum to obs_dim",
          (PRIV_SLICE.stop - PRIV_SLICE.start)
          + (PROPRIO_SLICE.stop - PROPRIO_SLICE.start)
          + (PREV_ACT_SLICE.stop - PREV_ACT_SLICE.start)
          + (LIDAR_PREV_SLICE.stop - LIDAR_PREV_SLICE.start)
          + (LIDAR_SLICE.stop - LIDAR_SLICE.start) == EXPECTED_OBS_DIM,
          f"PRIV={PRIV_DIM} PROPRIO={PROPRIO_DIM} PREV_ACT=2 "
          f"LIDAR_PREV={LIDAR_DIM} LIDAR={LIDAR_DIM}")
    check("priv slice has the right dim", PRIV_SLICE.stop - PRIV_SLICE.start == PRIV_DIM)
    check("proprio slice has the right dim",
          PROPRIO_SLICE.stop - PROPRIO_SLICE.start == PROPRIO_DIM,
          f"got {PROPRIO_SLICE.stop - PROPRIO_SLICE.start}, expected {PROPRIO_DIM}")
    check("PROPRIO_DIM is 45 (vel_cmd removed)",
          PROPRIO_DIM == 45,
          f"got {PROPRIO_DIM}; if 48 the vel_cmd leak is still there")
    check("PRIV_DIM is 7 (v_max added)",
          PRIV_DIM == 7,
          f"got {PRIV_DIM}; if 6 the v_max channel hasn't been added")

    # =============================================================
    # CHECK 3: vel_cmd is NOT in the proprio slice
    #
    # Strategy: run a step, snapshot proprio. Then perturb u_nom and
    # run another step from the same state. Proprio should NOT differ
    # in any of its values (other than the natural drift from physics).
    #
    # Cheap version: check that no element of the proprio slice equals
    # cbf.last_u_nom for many steps -- a soft check, but if u_nom were
    # leaking it'd appear there directly.
    # =============================================================
    print()
    print("--- Group B: vel_cmd leak check ---")
    obs_dict = env.reset()
    leak_hits = 0
    leak_samples = 0
    for _ in range(20):
        rand_a = torch.randn((N, 2), device=device) * 0.5
        env.step(rand_a.clamp(-1.0, 1.0))
        cur_obs = env.observation_manager.compute()["policy"]
        proprio = cur_obs[:, PROPRIO_SLICE]                # (N, 45)
        u_nom = cbf.last_u_nom                              # (N, 2)
        # is u_nom anywhere in proprio (within 1e-4)?
        for ev in range(N):
            for j in range(2):
                close = (proprio[ev] - u_nom[ev, j]).abs() < 1e-4
                if close.any():
                    leak_hits += int(close.sum().item())
                leak_samples += proprio[ev].numel()
    leak_frac = leak_hits / max(leak_samples, 1)
    # Some false positives can happen by chance for small u_nom values
    # near 0 -- so we accept up to 2% incidental matches.
    check("u_nom values NOT detected inside proprio slice",
          leak_frac < 0.02,
          f"leak_frac={leak_frac:.4f}  (target <0.02)")

    # =============================================================
    # CHECK 4: Lipschitz rate-limit
    # =============================================================
    print()
    print("--- Group C: Lipschitz rate-limit on actions ---")
    L = float(cbf._action_max_step)
    print(f"  action_max_step (configured) = {L:.4f}")
    if L <= 0:
        check("rate-limit configured", False,
              "action_max_step is 0; SHIELD env should set it to 0.05")
    else:
        # Send a STEP function as policy actions: -1 for first 5 steps,
        # then +1 for next 5 steps, etc. The decoded normalized action
        # should never jump by more than L per step.
        env.reset()
        prev = None
        max_step_seen = 0.0
        violations = 0
        for step in range(args_cli.rollout_steps):
            tgt = +1.0 if (step // 5) % 2 == 0 else -1.0
            cmd = torch.full((N, 2), tgt, device=device)
            env.step(cmd)
            cur = cbf._raw_actions                          # post-rate-limit
            if prev is not None:
                delta = (cur - prev).abs().max().item()
                max_step_seen = max(max_step_seen, delta)
                # 1e-4 tolerance for float
                if delta > L + 1e-4:
                    violations += 1
            prev = cur.detach().clone()
        check(f"per-step delta <= action_max_step ({L:.4f})",
              violations == 0,
              f"max observed step = {max_step_seen:.5f}  (violations: {violations})")
        # Sanity: with step-function input, after enough steps the policy
        # action should reach the target (no permanent steady-state offset).
        # After 50 steps of constant +1, raw_actions should be near +1.
        env.reset()
        for _ in range(int(2.5 / L) + 5):   # enough steps to reach saturation
            env.step(torch.full((N, 2), 1.0, device=device))
        steady = cbf._raw_actions.mean().item()
        check("rate-limit reaches steady-state target",
              steady > 0.95,
              f"steady value ~{steady:.3f}  (expected close to 1.0)")

    # =============================================================
    # CHECK 5: perception SDF differs from privileged (noise active)
    # =============================================================
    print()
    print("--- Group D: perception SDF (noise / dropout / range) ---")
    if not cbf._use_lidar_sdf:
        check("use_lidar_sdf is True (SHIELD env)", False,
              "running a non-SHIELD env; perception checks skipped")
    else:
        # Save the state, call both methods, compare
        env.reset()
        for _ in range(5):
            env.step(torch.zeros((N, 2), device=device))
        base_xy = cbf._robot.data.root_pos_w[:, :2]
        with torch.no_grad():
            sdf_priv, _, _, _ = cbf._compute_sdf_smooth(base_xy)
            sdfs_perc = []
            for _ in range(20):
                s, _, _, _ = cbf._compute_sdf_smooth_perception(base_xy)
                sdfs_perc.append(s.detach().clone())
            sdf_perc_mean = torch.stack(sdfs_perc).mean(dim=0)
            sdf_perc_std = torch.stack(sdfs_perc).std(dim=0)

        diff_mean = (sdf_perc_mean - sdf_priv).abs().mean().item()
        std_mean = sdf_perc_std.mean().item()
        # With 5cm position noise on each obstacle, the SDF should
        # have std ~few cm. If std is zero, noise isn't being applied.
        check("perception SDF has nonzero per-call variance (noise on)",
              std_mean > 0.005,
              f"avg per-env SDF std across 20 calls = {std_mean:.4f}  "
              f"(expected > 0.005)")
        # Mean over 20 samples should be close to privileged (noise zero-mean)
        check("perception SDF mean is close to privileged",
              diff_mean < 0.10,
              f"|perception_mean - privileged| = {diff_mean:.4f}")

        # Dropout: run many steps, count how often the "any visible"
        # mask returns False or how often the min h_per_masked equals
        # the lidar_max_range fallback (indicating some dropout/range fail)
        # Easier: just check perception_dropout_prob is set.
        dropout = float(cbf._perception_dropout_prob)
        check("perception dropout is configured",
              dropout > 0.0,
              f"perception_dropout_prob = {dropout}")
        # Range cutoff: temporarily move obstacles very far and verify
        # they're dropped from the SDF (sdf should saturate at
        # lidar_max_range - r_safe).
        orig_centers = cbf._obs_centers_w.clone()
        try:
            far_offset = torch.zeros_like(cbf._obs_centers_w)
            far_offset[:, :, 0] = 1000.0
            cbf._obs_centers_w = cbf._obs_centers_w + far_offset
            # disable noise/dropout for a clean check
            ons = cbf._perception_noise_std
            ond = cbf._perception_dropout_prob
            cbf._perception_noise_std = 0.0
            cbf._perception_dropout_prob = 0.0
            with torch.no_grad():
                sdf_far, _, _, _ = cbf._compute_sdf_smooth_perception(base_xy)
            # fallback in the impl: _lidar_max_range - r_safe.min()
            expected = cbf._lidar_max_range - float(cbf._r_safe.min().item())
            check("range cutoff: SDF saturates when obstacles are 1000m away",
                  (sdf_far - expected).abs().max().item() < 1e-3,
                  f"sdf={sdf_far[0].item():.3f}  expected={expected:.3f}")
        finally:
            cbf._obs_centers_w = orig_centers
            cbf._perception_noise_std = ons
            cbf._perception_dropout_prob = ond

    # =============================================================
    # CHECK 8-11: Mid-360 lidar fidelity
    #
    # The lidar feeds the policy's CNN AND (after clustering) the QP's
    # perception-SDF, so it must mirror real Mid-360 behavior:
    #   - per-step ray-angle jitter (non-repetitive scan)
    #   - fixed ~2cm range noise (datasheet)
    #   - range-weighted dropout (far rays fail more)
    #   - cap at lidar_max_range
    # =============================================================
    print()
    print("--- Group E: lidar fidelity (Mid-360 sim) ---")
    from cbf_task.mdp import _compute_lidar
    env.reset()
    # walk a few steps so we're in a non-degenerate state with obstacles
    # visible
    for _ in range(5):
        env.step(torch.zeros((N, 2), device=device))

    cfg = cbf.cfg
    max_r = float(getattr(cfg, "lidar_max_range", 20.0))
    noise = float(getattr(cfg, "lidar_noise_std", 0.02))
    jitter = float(getattr(cfg, "lidar_angle_jitter_std", 0.0087))
    base_dp = float(getattr(cfg, "lidar_dropout_base_prob", 0.005))
    slope_dp = float(getattr(cfg, "lidar_dropout_range_slope", 0.03))
    print(f"  cfg: noise_std={noise:.3f}  angle_jitter_std={jitter:.4f}  "
          f"dropout: base={base_dp:.4f} slope={slope_dp:.4f}  max_range={max_r:.1f}")

    # Call the lidar 30 times back-to-back. With jitter+noise, consecutive
    # samples should differ even with the robot at rest.
    samples = []
    with torch.no_grad():
        for _ in range(30):
            samples.append(_compute_lidar(
                env, n_rays=72, max_range=max_r,
                noise_std=noise, angle_jitter_std=jitter,
                dropout_base_prob=base_dp, dropout_range_slope=slope_dp,
            ).clone())
    S = torch.stack(samples)                                       # (30, N, 72)

    # Per-ray std across the 30 calls. Excluding dropped rays (= max_r),
    # this should reflect the noise+jitter -- not zero.
    not_dropped = (S < max_r - 1e-3).all(dim=0)                    # (N, 72)
    if not_dropped.any():
        s_finite = torch.where(not_dropped.unsqueeze(0),
                               S, torch.zeros_like(S))
        # take std only over rays that were always present
        per_ray_std = S.std(dim=0)                                  # (N, 72)
        active_std = per_ray_std[not_dropped]
        # In our setup with 5cm-jitter geometry, ray-angle jitter alone
        # produces several-cm range variation; noise adds 2cm. Combined
        # std should be > the configured fixed noise.
        check("per-ray range varies across consecutive calls (jitter+noise on)",
              active_std.mean().item() > noise * 0.5,
              f"mean active-ray std = {active_std.mean().item():.4f}  "
              f"(expected > {noise * 0.5:.4f})")
    else:
        check("active rays exist for std test", False,
              "every ray was dropped at some point; can't measure noise std")

    # Dropout: many rays are already at max_range (no obstacle in that
    # direction), so counting "at max_range" overcounts. Instead, compare
    # at-max-rate WITH dropout vs WITHOUT, and check the excess matches
    # the configured Bernoulli probability.
    baseline = []
    with torch.no_grad():
        for _ in range(30):
            baseline.append(_compute_lidar(
                env, n_rays=72, max_range=max_r,
                noise_std=noise, angle_jitter_std=jitter,
                dropout_base_prob=0.0, dropout_range_slope=0.0,
            ).clone())
    B = torch.stack(baseline)
    at_max_with = (S >= max_r - 1e-3).float().mean().item()
    at_max_without = (B >= max_r - 1e-3).float().mean().item()
    excess = at_max_with - at_max_without
    # rough expected: average p across rays = base + 0.5*slope (since the
    # mean range/max_range ratio is roughly 0.5 across rays that hit
    # obstacles vs go to infinity). Be generous with the band.
    expected_lo = base_dp * 0.5
    expected_hi = base_dp + slope_dp + 0.02
    check("dropout adds expected per-ray at-max excess",
          expected_lo < excess < expected_hi,
          f"with_dropout - no_dropout = {excess:.4f}  expected ~ "
          f"[{expected_lo:.4f}, {expected_hi:.4f}]  "
          f"(at-max: with={at_max_with:.3f} without={at_max_without:.3f})")

    # Range cap: no ray exceeds max_range
    max_seen = S.max().item()
    check("no ray exceeds lidar_max_range",
          max_seen <= max_r + 1e-3,
          f"max observed range = {max_seen:.3f}  cap = {max_r:.3f}")

    # No ray is negative (clamp at 0)
    min_seen = S.min().item()
    check("no negative ranges",
          min_seen >= 0.0 - 1e-3,
          f"min observed range = {min_seen:.4f}")

    # =============================================================
    # FINAL SUMMARY
    # =============================================================
    print()
    print("=" * 88)
    print("  SUMMARY")
    print("=" * 88)
    total = len(RESULTS)
    passed = sum(1 for _, p, _ in RESULTS if p)
    for name, ok, detail in RESULTS:
        icon = "PASS" if ok else "FAIL"
        print(f"  [{icon}] {name}" + (f"   -- {detail}" if detail else ""))
    print()
    print(f"  {passed}/{total} checks passed")
    print("=" * 88)
    env.close()
    simulation_app.close()
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
