"""RMA teacher PPO training. Mirrors phase2_train.py but:
- task is `Isaac-CBF-Adaptive-Go2-RMA-v0` (priv obs surfaces 4 channels)
- imports `cbf_task.agents.rma_actor_critic` so `RMAMLPModel` is
  registered in `rsl_rl.models` before the runner builds.
- evaluation still computes the per-step (phi, alpha) diagnostic so we
  can immediately see whether the teacher actually adapts vs d.

GATE FIRST: run `phase5_fingerprint_gate.py` and confirm R^2 >= 0.5 for
each priv channel BEFORE launching this training. If any channel is
weak, the teacher will silently ignore that part of z.

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase5_train_teacher.py \\
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --num_envs 64 --max_iterations 1000 --headless
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-RMA-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--out_dir", default="phase5_teacher_outputs")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--policy_checkpoint", default=None,
                    help="If set, load these weights and skip training "
                         "(useful for eval-only on a saved teacher).")
parser.add_argument("--eval_disturbances", type=float, nargs="+",
                    default=[0.0, 15.0, 30.0, 45.0])
parser.add_argument("--eval_eps_per_cell", type=int, default=256,
                    help="Capped at num_envs. 256 gives ~±6pp CI on binary "
                         "collision rate (vs ~±12pp at 64).")
parser.add_argument("--eval_max_steps", type=int, default=1250)
parser.add_argument("--diag_interval", type=int, default=50,
                    help="Iterations between in-training CBF diagnostics. "
                         "Set to 0 to disable.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import csv
import importlib.metadata as metadata
import json
import time

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
# importing this module auto-registers RMAMLPModel in rsl_rl.models so the
# runner cfg's `class_name="RMAMLPModel"` resolves
from cbf_task.agents import rma_actor_critic  # noqa: F401
# Belt-and-suspenders: also import the RMA-classic model so it's
# registered in rsl_rl.models when training the RMAStatic task. The
# qualified `class_name="module:Class"` should resolve via importlib
# without this, but the explicit import matches the existing pattern
# for RMAMLPModel.
from cbf_task.agents import rma_classic_actor_critic  # noqa: F401
from cbf_task.agents import rma_history_actor_critic  # noqa: F401
from cbf_task.locomotion_loader import load_locomotion_actor


TASK = args_cli.task


def _fmt_hms(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def _emit_training_diag(env_wrapped, runner, iter_idx: int,
                          total_iters: int | None = None,
                          wall_t0: float | None = None) -> None:
    """Print a one-line CBF/policy health snapshot during training.

    Reads the action term's per-step caches that rsl_rl does NOT already
    log (phi, alpha, intervention, action jitter, policy std). Per-
    episode termination rates (collision/reach/fall/timeout) are already
    in rsl_rl's per-iter table -- we do NOT duplicate those here because
    the sticky `episode_*_any` flags are within-episode cumulative, not
    per-episode rates, and reading them at a random mid-rollout moment
    over-reports by 10-30x.

    Signatures we WARN on (each is a known-bad mode):
      - action_std < 0.05  -> policy collapsed (deterministic)
      - action_std > 1.5   -> policy degenerate (near-random)
      - phi/alpha mean within 1% of bound -> pegged at bound
      - intervention ~ 0   -> QP never firing (lidar/SDF dead?)
      - jitter_mean > 0.20 -> Lipschitz overrun (check action_max_step)
    """
    try:
        env = env_wrapped.unwrapped
        cbf = env.action_manager._terms["cbf_param"]
        phi = cbf.last_phi.detach().float()
        alpha = cbf.last_alpha.detach().float()
        intervention = cbf.last_intervention.detach().float()
        action_jitter = cbf.last_action_jitter.detach().float()
        action_std = float(runner.alg.get_policy().output_std.mean().item())
    except Exception as exc:
        print(f"[diag iter={iter_idx:5d}] diagnostic snapshot failed: {exc}")
        return

    phi_lo, phi_hi = float(cbf._phi_lo), float(cbf._phi_hi)
    alpha_lo, alpha_hi = float(cbf._alpha_lo), float(cbf._alpha_hi)
    phi_w, alpha_w = max(phi_hi - phi_lo, 1e-9), max(alpha_hi - alpha_lo, 1e-9)
    phi_mean = float(phi.mean().item())
    phi_std = float(phi.std().item())
    alpha_mean = float(alpha.mean().item())
    alpha_std = float(alpha.std().item())
    int_mean = float(intervention.mean().item())
    jit_mean = float(action_jitter.mean().item())

    eta_str = ""
    if total_iters is not None and wall_t0 is not None and iter_idx > 0:
        elapsed = time.time() - wall_t0
        rate = iter_idx / max(elapsed, 1e-6)
        remaining = max(total_iters - iter_idx, 0)
        eta_sec = remaining / max(rate, 1e-6)
        eta_str = (f"  elapsed={_fmt_hms(elapsed)} "
                   f"ETA(total)={_fmt_hms(eta_sec)} "
                   f"@ {rate:.2f} iter/s")

    print(f"[diag iter={iter_idx:5d}/{total_iters or '?'}] "
          f"phi={phi_mean:+.3f}±{phi_std:.2f} [{phi_lo:+.1f},{phi_hi:+.1f}]  "
          f"alpha={alpha_mean:.2f}±{alpha_std:.2f} [{alpha_lo:.1f},{alpha_hi:.1f}]  "
          f"int={int_mean:.3f}  jit={jit_mean:.3f}  a_std={action_std:.3f}"
          f"{eta_str}")

    warnings = []
    if action_std < 0.05:
        warnings.append(f"action_std={action_std:.3f} < 0.05 (policy collapsed)")
    if action_std > 1.5:
        warnings.append(f"action_std={action_std:.3f} > 1.5 (policy degenerate)")
    if abs(phi_mean - phi_hi) < 0.01 * phi_w:
        warnings.append(f"phi pegged at HI ({phi_mean:.3f} vs hi={phi_hi:.3f})")
    if abs(phi_mean - phi_lo) < 0.01 * phi_w:
        warnings.append(f"phi pegged at LO ({phi_mean:.3f} vs lo={phi_lo:.3f})")
    if abs(alpha_mean - alpha_hi) < 0.01 * alpha_w:
        warnings.append(f"alpha pegged at HI ({alpha_mean:.3f} vs hi={alpha_hi:.3f})")
    if abs(alpha_mean - alpha_lo) < 0.01 * alpha_w:
        warnings.append(f"alpha pegged at LO ({alpha_mean:.3f} vs lo={alpha_lo:.3f})")
    if int_mean < 1e-4:
        warnings.append("intervention ~0 (QP never firing; lidar/SDF dead?)")
    if jit_mean > 0.20:
        warnings.append(f"jitter_mean={jit_mean:.3f} > 0.20 (Lipschitz overrun? check action_max_step)")
    for w in warnings:
        print(f"[diag iter={iter_idx:5d}]   [WARN] {w}")


def _run_training_with_diag(env_wrapped, runner, total_iters: int,
                             diag_interval: int) -> None:
    """Wrap runner.learn() in a chunked loop so we can emit per-chunk
    diagnostics WITHOUT modifying rsl_rl.

    Note on cosmetic index collision: rsl_rl sets
    `current_learning_iteration = it` (the LAST iter index, not iter+1)
    inside its train loop, so the next .learn(N) call's printed iteration
    labels OVERLAP the previous chunk's last iter by 1 (e.g. "iter 49/99"
    appears after "iter 49/50"). This is purely a label collision -- the
    underlying training does N distinct PPO updates per chunk, no
    iterations are lost or duplicated. Verified by inspection of rsl_rl
    5.0.1's OnPolicyRunner.learn.
    """
    if diag_interval <= 0:
        runner.learn(num_learning_iterations=total_iters,
                     init_at_random_ep_len=False)
        return
    done = 0
    t0 = time.time()
    while done < total_iters:
        chunk = min(diag_interval, total_iters - done)
        runner.learn(num_learning_iterations=chunk,
                     init_at_random_ep_len=False)
        done += chunk
        _emit_training_diag(env_wrapped, runner, done,
                              total_iters=total_iters, wall_t0=t0)


def eval_teacher_at_disturbance(env_wrapped, runner, disturbance_mag,
                                 eval_steps, n_eps, device):
    """Roll out the trained teacher at a fixed disturbance and collect:
    collision/reach/fall rates, mean intervention, and per-step (phi, alpha)
    distribution stats.
    """
    cbf = env_wrapped.unwrapped.action_manager._terms["cbf_param"]
    cbf._disturbance_force_lo = float(disturbance_mag)
    cbf._disturbance_force_hi = float(disturbance_mag)
    policy = runner.get_inference_policy(device=device)

    # threshold for "in unsafe zone" -- match phase5_baselines.py
    UNSAFE_THR = 0.2
    N = env_wrapped.unwrapped.num_envs
    min_h = torch.full((N,), float("inf"), device=device)
    intervention_sum = torch.zeros(N, device=device)
    unsafe_steps = torch.zeros(N, device=device)
    goal_step = torch.full((N,), -1, device=device, dtype=torch.long)
    env_wrapped.unwrapped.reset()
    cbf.episode_reach_any.zero_()
    cbf.episode_collide_any.zero_()
    cbf.episode_fall_any.zero_()
    cbf.episode_stuck_any.zero_()
    obs = env_wrapped.get_observations()
    phi_hist = []
    alpha_hist = []
    jitter_hist = []
    for step in range(eval_steps):
        action = policy(obs)
        obs, _, _, _ = env_wrapped.step(action)
        min_h = torch.minimum(min_h, cbf.last_h_realized)
        intervention_sum = intervention_sum + cbf.last_intervention
        unsafe_steps = unsafe_steps + (cbf.last_h_realized < UNSAFE_THR).float()
        newly_reached = cbf.episode_reach_any & (goal_step == -1)
        goal_step[newly_reached] = step
        phi_hist.append(cbf.last_phi.detach().clone())
        alpha_hist.append(cbf.last_alpha.detach().clone())
        jitter_hist.append(cbf.last_action_jitter.detach().clone())
    n_eps = min(n_eps, N)
    sel = slice(0, n_eps)
    phi_all = torch.stack(phi_hist, dim=0)[:, sel].flatten()
    alpha_all = torch.stack(alpha_hist, dim=0)[:, sel].flatten()
    jitter_all = torch.stack(jitter_hist, dim=0)[:, sel].flatten()
    reached_mask = goal_step[sel] >= 0
    time_to_goal_mean = (float(goal_step[sel][reached_mask].float().mean().item())
                         if reached_mask.any() else float("nan"))
    return {
        "n": int(n_eps),
        "collision_rate": float(cbf.episode_collide_any[sel].float().mean().item()),
        "reach_rate": float(cbf.episode_reach_any[sel].float().mean().item()),
        "fall_rate": float(cbf.episode_fall_any[sel].float().mean().item()),
        "stuck_rate": float(cbf.episode_stuck_any[sel].float().mean().item()),
        "min_h_mean": float(min_h[sel].clamp(min=-10.0).mean().item()),
        "intervention_mean": float(intervention_sum[sel].mean().item()),
        # NEW: mirror phase5_baselines.py for apples-to-apples comparison
        "time_in_unsafe_frac": float((unsafe_steps[sel] / eval_steps).mean().item()),
        "time_to_goal_mean": time_to_goal_mean,
        "phi_mean": float(phi_all.mean().item()),
        "phi_std": float(phi_all.std().item()),
        "alpha_mean": float(alpha_all.mean().item()),
        "alpha_std": float(alpha_all.std().item()),
        # action smoothness diagnostic: |Δphi|/phi_width + |Δalpha|/
        # alpha_width per step. Healthy values: well under 0.1 means the
        # action shifts by < 10% of bound width step-to-step. > 0.3
        # likely too jittery for locomotion to track.
        "jitter_mean": float(jitter_all.mean().item()),
        "jitter_p95": float(jitter_all.quantile(0.95).item()),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args_cli.out_dir, exist_ok=True)

    loco = load_locomotion_actor(retrieve_file_path(args_cli.checkpoint), device)
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco
    env = gym.make(TASK, cfg=env_cfg)

    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    agent_cfg.max_iterations = int(args_cli.max_iterations)
    agent_cfg.seed = int(args_cli.seed)

    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    log_dir = os.path.join(os.path.abspath(args_cli.out_dir), "rsl_rl")
    os.makedirs(log_dir, exist_ok=True)

    if args_cli.policy_checkpoint is not None:
        agent_cfg.max_iterations = 0
        runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(),
                                log_dir=None, device=device)
        ckpt = retrieve_file_path(args_cli.policy_checkpoint)
        print(f"[teacher] eval-only mode: loading -> {ckpt}")
        runner.load(ckpt)
        train_secs = 0.0
    else:
        print(f"[teacher] training PPO for {agent_cfg.max_iterations} iterations "
              f"({args_cli.num_envs} envs) ...")
        runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(),
                                log_dir=log_dir, device=device)
        t0 = time.time()
        _run_training_with_diag(env_wrapped, runner,
                                 total_iters=agent_cfg.max_iterations,
                                 diag_interval=int(args_cli.diag_interval))
        train_secs = time.time() - t0
        runner.save(os.path.join(log_dir, "model_final.pt"))
        print(f"[teacher] saved -> {log_dir}/model_final.pt  ({train_secs:.0f}s)")

    # eval at the disturbance sweep, with (phi, alpha) diagnostic
    print()
    print("[teacher] running eval ...")
    learned_rows = []
    with torch.inference_mode():
        for d in args_cli.eval_disturbances:
            m = eval_teacher_at_disturbance(
                env_wrapped, runner, d,
                args_cli.eval_max_steps, args_cli.eval_eps_per_cell, device,
            )
            learned_rows.append({"disturbance_force": float(d), **m})
            print(f"    d={d:>5.1f}N  coll={m['collision_rate']:.2f}  "
                  f"reach={m['reach_rate']:.2f}  fall={m['fall_rate']:.2f}  "
                  f"stuck={m['stuck_rate']:.2f}  "
                  f"int={m['intervention_mean']:.0f}")
            print(f"      policy outputs: phi mean={m['phi_mean']:+.3f} "
                  f"std={m['phi_std']:.3f}  alpha mean={m['alpha_mean']:.2f} "
                  f"std={m['alpha_std']:.2f}")
            print(f"      action smoothness: jitter mean={m['jitter_mean']:.3f}  "
                  f"p95={m['jitter_p95']:.3f}  "
                  f"({'OK' if m['jitter_mean'] < 0.10 else 'noisy' if m['jitter_mean'] < 0.30 else 'JITTERY -- locomotion will struggle'})")

    with open(os.path.join(args_cli.out_dir, "phase5_learned_eval.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(learned_rows[0].keys()))
        w.writeheader()
        w.writerows(learned_rows)

    # the key question for the RMA pivot: does the teacher adapt phi to d?
    phi_range = (max(L["phi_mean"] for L in learned_rows)
                 - min(L["phi_mean"] for L in learned_rows))
    alpha_range = (max(L["alpha_mean"] for L in learned_rows)
                   - min(L["alpha_mean"] for L in learned_rows))
    at = env_cfg.actions.cbf_param
    phi_width = at.phi_bounds[1] - at.phi_bounds[0]
    alpha_width = at.alpha_bounds[1] - at.alpha_bounds[0]
    adapts = (phi_range > 0.25 * phi_width) or (alpha_range > 0.25 * alpha_width)

    print()
    print("=" * 80)
    print("  Phase 5 (RMA teacher) -- does the policy actually adapt to disturbance?")
    print("=" * 80)
    print(f"  phi_mean spans {phi_range:.3f} ({100*phi_range/phi_width:.1f}% of "
          f"[{at.phi_bounds[0]}, {at.phi_bounds[1]}])")
    print(f"  alpha_mean spans {alpha_range:.3f} ({100*alpha_range/alpha_width:.1f}% "
          f"of [{at.alpha_bounds[0]}, {at.alpha_bounds[1]}])")
    verdict = "PASS -- teacher adapts via z" if adapts else "REVIEW -- teacher flat"
    print(f"  verdict: {verdict}")

    txt_path = os.path.join(args_cli.out_dir, "phase5_teacher_summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"Task: {args_cli.task}\n")
        f.write(f"Iterations: {args_cli.max_iterations}\n")
        f.write(f"Train seconds: {train_secs:.0f}\n")
        f.write(f"Verdict: {verdict}\n\n")
        f.write(f"{'d (N)':>6}  {'coll':>5}  {'reach':>5}  {'fall':>5}  "
                f"{'int':>6}  {'phi_mean':>9}  {'phi_std':>7}  "
                f"{'alpha_mean':>10}  {'alpha_std':>9}\n")
        for L in learned_rows:
            f.write(f"{L['disturbance_force']:>6.1f}  "
                    f"{L['collision_rate']:>5.2f}  {L['reach_rate']:>5.2f}  "
                    f"{L['fall_rate']:>5.2f}  {L['intervention_mean']:>6.0f}  "
                    f"{L['phi_mean']:>+9.3f}  {L['phi_std']:>7.3f}  "
                    f"{L['alpha_mean']:>10.3f}  {L['alpha_std']:>9.3f}\n")
        f.write(f"\nphi_range across d:   {phi_range:.3f} "
                f"({100*phi_range/phi_width:.1f}% of bound width)\n")
        f.write(f"alpha_range across d: {alpha_range:.3f} "
                f"({100*alpha_range/alpha_width:.1f}% of bound width)\n")
    print(f"\n  saved -> {log_dir}/, phase5_learned_eval.csv, phase5_teacher_summary.txt")

    env.close()
    simulation_app.close()
    sys.exit(0 if adapts else 1)


if __name__ == "__main__":
    main()
