"""Phase 10 / V2 unified eval script.

Evaluates ONE policy on ONE scene, sweeping a primary DR axis at 3
settings. Run by the overnight orchestrator across the cartesian
product of (policy, scene). Per-scene DR axis default:

    E1Gap    -> friction          (φ test -- traction matters in tight gap)
    E2Slalom -> v_max             (α test -- speed/braking on a weave)
    E3Wall   -> motor_strength    (φ test -- power matters at the gap-edge)
    E4Field  -> v_max             (α test -- braking under density)

Headline metric per cell:
    safe_reach = reach_rate * (1 - collision_rate)
Plus time_to_goal (mean over reached eps), mean_intervention, jitter.

Each (scene, DR_setting) cell rolls out 512 envs × 1250 steps and
aggregates across 512 episodes (per user spec).

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase10_eval_unified.py \\
        --policy_dir phase10_outputs/V2Full_int0 \\
        --scene E1Gap \\
        --num_envs 512 --eval_steps 1250 --eps_per_cell 512 \\
        --headless

Or for a fixed (phi, alpha) baseline (no policy load):
    ... --baseline_phi 0.0 --baseline_alpha 2.5
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy_dir", default=None,
                    help="Trained-policy dir (contains rsl_rl/model_final.pt "
                         "and manifest.txt). Mutually exclusive with --baseline_*.")
parser.add_argument("--checkpoint", required=True,
                    help="Locomotion policy checkpoint (frozen).")
parser.add_argument("--scene", required=True,
                    choices=["E1Gap", "E2Slalom", "E3Wall", "E4Field"])
parser.add_argument("--dr_axis", default=None,
                    choices=[None, "friction", "motor_strength", "v_max",
                             "disturbance", "mass", "com_offset",
                             "actuation_noise"],
                    help="Single-axis sweep override (back-compat). Use "
                         "--dr_sweeps for the multi-axis path.")
parser.add_argument("--dr_values", type=float, nargs="+", default=None,
                    help="Sweep values for --dr_axis. Default per axis.")
parser.add_argument("--dr_sweeps", nargs="+", default=None,
                    help="Multiple axis sweeps in one invocation. Format: "
                         "'axis:v1,v2,v3'. Example: "
                         "--dr_sweeps friction:0.3,0.6,1.0 v_max:1.0,1.5,2.0  "
                         "All other DR axes are pinned to nominal so the "
                         "swept axis is the only variable per cell.")
parser.add_argument("--obs_arch_override", default=None,
                    choices=[None, "V2Full", "V2NoPriv", "V2NoProprio",
                             "V2RMAClassic", "V2History"],
                    help="Force a particular obs layout (for baselines that "
                         "have no manifest). Ignored if policy_dir is set.")
parser.add_argument("--baseline_type", default=None,
                    choices=[None, "B0", "B1", "B2"],
                    help="B0: const alpha, phi=0 (Exponential CBF, Ames 2017). "
                         "B1: const (phi, alpha) (ECBF + ISSf). "
                         "B2: alpha const, phi(h)=(1/eps0)*exp(-lam*h) "
                         "(TISSf-CBF, Cohen 2024 / Molnar 2023). "
                         "Setting this enables baseline mode -- ignores "
                         "--policy_dir. For B0/B1 you can equivalently use "
                         "--baseline_phi / --baseline_alpha (back-compat).")
parser.add_argument("--baseline_phi", type=float, default=None)
parser.add_argument("--baseline_alpha", type=float, default=None)
parser.add_argument("--baseline_eps0", type=float, default=None,
                    help="B2 only: 1/eps0 = max phi near obstacle.")
parser.add_argument("--baseline_lam", type=float, default=None,
                    help="B2 only: decay rate of phi as h grows.")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--eval_steps", type=int, default=1250)
parser.add_argument("--eps_per_cell", type=int, default=512)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", action="store_true", default=False,
                    help="Record one full episode of env 0 at eval start.")
parser.add_argument("--video_length", type=int, default=1250,
                    help="Length (env.step calls) of the eval video. "
                         "Default 1250 = one full 25s episode at 50Hz.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import importlib.metadata as metadata
import json

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
from cbf_task.agents import rma_actor_critic           # noqa: F401
from cbf_task.agents import rma_classic_actor_critic   # noqa: F401
from cbf_task.agents import rma_history_actor_critic   # noqa: F401
from cbf_task.locomotion_loader import load_locomotion_actor


# Per-scene defaults: list of (axis, values) sweeps. EACH scene now
# sweeps BOTH a φ-aligned axis (friction or motor) AND the α-aligned
# axis (v_max), so adaptation can be tested along both directions per
# scene. 6 cells per scene (2 axes × 3 values).
SCENE_DEFAULTS = {
    "E1Gap":    [("friction",       [0.3, 0.6, 1.0]),
                 ("v_max",          [1.0, 1.5, 2.0])],
    "E2Slalom": [("motor_strength", [0.7, 1.0, 1.3]),
                 ("v_max",          [1.0, 1.5, 2.0])],
    "E3Wall":   [("friction",       [0.3, 0.6, 1.0]),
                 ("v_max",          [1.0, 1.5, 2.0])],
    "E4Field":  [("motor_strength", [0.7, 1.0, 1.3]),
                 ("v_max",          [1.0, 1.5, 2.0])],
}

# Map DR axis name -> action-term attribute pair (lo, hi).
DR_ATTR = {
    "friction":         ("_friction_lo", "_friction_hi"),
    "motor_strength":   ("_motor_strength_lo", "_motor_strength_hi"),
    "v_max":            ("_v_max_lo", "_v_max_hi"),
    "disturbance":      ("_disturbance_force_lo", "_disturbance_force_hi"),
    "mass":             ("_base_mass_lo", "_base_mass_hi"),
    "com_offset":       ("_com_offset_lo", "_com_offset_hi"),
    "actuation_noise":  ("_actuation_noise_std_lo", "_actuation_noise_std_hi"),
}

# Nominal (default) values per axis. Used to PIN non-swept channels so
# each cell isolates the effect of the swept axis without noise from
# other DR (which would otherwise still be randomized over the V2
# training ranges).
NOMINAL_DR = {
    "friction":         0.6,
    "motor_strength":   1.0,
    "v_max":            1.5,
    "disturbance":      0.0,
    "mass":             0.0,
    "com_offset":       0.0,
    "actuation_noise":  0.0,
}

UNSAFE_THR = 0.2


def _read_manifest(policy_dir: str) -> dict:
    """Tiny key=value manifest reader (matches phase10_train_unified)."""
    path = os.path.join(policy_dir, "manifest.txt")
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _scene_task_id(scene: str, arch: str) -> str:
    if arch in {"V2Full", "V2NoPriv", "V2NoProprio"}:
        return f"Isaac-CBF-Adaptive-Go2-V2Eval-{scene}-v0"
    if arch == "V2RMAClassic":
        return f"Isaac-CBF-Adaptive-Go2-V2Eval-{scene}-RMAClassic-v0"
    if arch == "V2History":
        return f"Isaac-CBF-Adaptive-Go2-V2Eval-{scene}-History-v0"
    raise ValueError(f"unknown arch {arch}")


def _set_dr_axis(cbf, axis: str, value: float) -> None:
    lo_attr, hi_attr = DR_ATTR[axis]
    setattr(cbf, lo_attr, float(value))
    setattr(cbf, hi_attr, float(value))


def _pin_all_dr_to_nominal(cbf) -> None:
    """Pin every DR axis to its nominal value. Per-axis sweeps then
    re-set only the swept axis -- so each cell varies exactly one
    physical parameter, and the (φ, α) response measured per cell is
    cleanly attributable to that axis."""
    for axis, nom in NOMINAL_DR.items():
        _set_dr_axis(cbf, axis, nom)


def _baseline_action(baseline_spec, cbf, device, N):
    """Build the per-step action tensor for a baseline (B0/B1/B2).

    baseline_spec is a dict with keys:
      type: "B0" / "B1" / "B2"
      phi, alpha: floats (B0/B1 only)
      alpha, eps0, lam: floats (B2)

    For B0/B1: returns a CONSTANT action tensor (computed once).
    For B2: returns a *function* (cbf -> action_tensor) since phi depends
    on per-step h_realized.
    """
    phi_lo, phi_hi = cbf._phi_lo, cbf._phi_hi
    alpha_lo, alpha_hi = cbf._alpha_lo, cbf._alpha_hi
    def encode(phi_t, alpha_t):
        a_phi   = 2.0*(phi_t   - phi_lo)/max(phi_hi-phi_lo, 1e-9) - 1.0
        a_alpha = 2.0*(alpha_t - alpha_lo)/max(alpha_hi-alpha_lo, 1e-9) - 1.0
        return torch.stack([a_phi, a_alpha], dim=-1)
    t = baseline_spec["type"]
    if t in ("B0", "B1"):
        phi_v = baseline_spec.get("phi", 0.0)
        alpha_v = baseline_spec["alpha"]
        phi_t = torch.full((N,), float(phi_v), device=device)
        alpha_t = torch.full((N,), float(alpha_v), device=device)
        action_const = encode(phi_t, alpha_t)
        return ("const", action_const)
    if t == "B2":
        alpha_v = baseline_spec["alpha"]
        eps0 = baseline_spec["eps0"]
        lam = baseline_spec["lam"]
        def state_fn():
            h = cbf.last_h_realized.clamp(min=0.0)
            phi_t = ((1.0 / eps0) * torch.exp(-lam * h)).clamp(min=phi_lo, max=phi_hi)
            alpha_t = torch.full((N,), float(alpha_v), device=device)
            return encode(phi_t, alpha_t)
        return ("state_fn", state_fn)
    raise ValueError(f"unknown baseline type {t}")


def _roll_out(env_wrapped, policy_or_baseline, eval_steps: int,
              n_eps: int, device, is_baseline: bool):
    """Roll out one cell with MUTUALLY EXCLUSIVE outcomes.

    Each env has exactly one bucket: {collide, fall, reach, timeout}.
    The bucket is LATCHED on the first terminal event of the cell, with
    priority collide > fall > reach when several flags fire on the same
    step (a safety violation should always be reported). After the
    initial latch, subsequent within-cell episodes for that env do NOT
    update the bucket. Result: reach + collide + fall + timeout = 1.0
    -- no more "80% reach / 80% collide" double-counting.

    stuck_rate is reported separately as the SUBSET of timeout envs that
    triggered the stuck flag (5s of continuous slow-not-at-goal) -- it
    is included in timeout, not in addition to it.

    Implementation: snapshot the sticky `episode_*_any` flags each step
    and detect deltas. The action-term resets per-env stuck counters on
    auto-reset but does NOT clear the sticky flags (those persist for
    the whole cell). We clear them once at cell start.
    """
    env = env_wrapped.unwrapped
    cbf = env.action_manager._terms["cbf_param"]
    N = env.num_envs
    n_eps = min(n_eps, N)
    sel = slice(0, n_eps)

    min_h = torch.full((N,), float("inf"), device=device)
    intervention_sum = torch.zeros(N, device=device)
    unsafe_steps = torch.zeros(N, device=device)

    env.reset()
    cbf.episode_reach_any.zero_()
    cbf.episode_collide_any.zero_()
    cbf.episode_fall_any.zero_()
    cbf.episode_stuck_any.zero_()

    # OUTCOME: 0=timeout/pending, 1=reach, 2=collide, 3=fall
    outcome = torch.zeros(N, dtype=torch.long, device=device)
    goal_step = torch.full((N,), -1, device=device, dtype=torch.long)
    reach_prev = cbf.episode_reach_any.clone()
    coll_prev = cbf.episode_collide_any.clone()
    fall_prev = cbf.episode_fall_any.clone()

    obs = env_wrapped.get_observations()
    phi_acc, alpha_acc, jitter_acc = [], [], []
    # baseline action setup -- constant for B0/B1, per-step fn for B2
    base_kind, base_payload = (None, None)
    if is_baseline:
        base_kind, base_payload = _baseline_action(policy_or_baseline, cbf, device, N)
    for step in range(eval_steps):
        if is_baseline:
            if base_kind == "const":
                action = base_payload
            else:  # state_fn (B2)
                action = base_payload()
        else:
            action = policy_or_baseline(obs)
        obs, _, _, _ = env_wrapped.step(action)
        min_h = torch.minimum(min_h, cbf.last_h_realized)
        intervention_sum = intervention_sum + cbf.last_intervention
        unsafe_steps = unsafe_steps + (cbf.last_h_realized < UNSAFE_THR).float()

        # detect FIRST-firings this step (sticky-flag deltas)
        reach_now = cbf.episode_reach_any
        coll_now = cbf.episode_collide_any
        fall_now = cbf.episode_fall_any
        reach_new = reach_now & (~reach_prev)
        coll_new = coll_now & (~coll_prev)
        fall_new = fall_now & (~fall_prev)
        # latch with priority collide > fall > reach
        pending = (outcome == 0)
        coll_latch = pending & coll_new
        outcome = torch.where(coll_latch, torch.full_like(outcome, 2), outcome)
        pending = pending & ~coll_latch
        fall_latch = pending & fall_new
        outcome = torch.where(fall_latch, torch.full_like(outcome, 3), outcome)
        pending = pending & ~fall_latch
        reach_latch = pending & reach_new
        outcome = torch.where(reach_latch, torch.full_like(outcome, 1), outcome)
        goal_step = torch.where(reach_latch & (goal_step == -1),
                                 torch.full_like(goal_step, step),
                                 goal_step)
        reach_prev = reach_now.clone()
        coll_prev = coll_now.clone()
        fall_prev = fall_now.clone()

        phi_acc.append(cbf.last_phi.detach().clone())
        alpha_acc.append(cbf.last_alpha.detach().clone())
        jitter_acc.append(cbf.last_action_jitter.detach().clone())

    phi_all = torch.stack(phi_acc, dim=0)[:, sel].flatten()
    alpha_all = torch.stack(alpha_acc, dim=0)[:, sel].flatten()
    jitter_all = torch.stack(jitter_acc, dim=0)[:, sel].flatten()
    o_sel = outcome[sel]
    reach_rate = float((o_sel == 1).float().mean().item())
    coll_rate = float((o_sel == 2).float().mean().item())
    fall_rate = float((o_sel == 3).float().mean().item())
    timeout_rate = float((o_sel == 0).float().mean().item())
    # stuck = subset of timeout (env tagged stuck flag and never terminated)
    stuck_rate = float(((o_sel == 0) & cbf.episode_stuck_any[sel]).float().mean().item())
    reached_mask = o_sel == 1
    ttg = (float(goal_step[sel][reached_mask].float().mean().item())
           if reached_mask.any() else float("nan"))
    return {
        "n": int(n_eps),
        # mutually exclusive outcomes -- these four sum to 1.0
        "reach_rate": reach_rate,
        "collision_rate": coll_rate,
        "fall_rate": fall_rate,
        "timeout_rate": timeout_rate,
        # subset-of-timeout diagnostic
        "stuck_rate": stuck_rate,
        # headline metric (== reach_rate now that buckets are exclusive)
        "safe_reach": reach_rate,
        "time_to_goal_mean": ttg,
        "min_h_mean": float(min_h[sel].clamp(min=-10.0).mean().item()),
        "intervention_mean": float(intervention_sum[sel].mean().item()),
        "time_in_unsafe_frac": float((unsafe_steps[sel] / eval_steps).mean().item()),
        "phi_mean": float(phi_all.mean().item()),
        "phi_std": float(phi_all.std().item()),
        "alpha_mean": float(alpha_all.mean().item()),
        "alpha_std": float(alpha_all.std().item()),
        "jitter_mean": float(jitter_all.mean().item()),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args_cli.out_dir, exist_ok=True)

    # ---- figure out arch + scene task id ----
    baseline_spec = None
    if args_cli.policy_dir is not None:
        manifest = _read_manifest(args_cli.policy_dir)
        arch = manifest["arch"]
        policy_label = os.path.basename(os.path.normpath(args_cli.policy_dir))
    else:
        # baseline mode -- build the baseline spec dict.
        arch = args_cli.obs_arch_override or "V2Full"
        bt = args_cli.baseline_type
        # back-compat: --baseline_phi + --baseline_alpha alone implies B1
        if bt is None and args_cli.baseline_phi is not None \
                and args_cli.baseline_alpha is not None:
            bt = "B1"
        if bt is None:
            raise ValueError("Need either --policy_dir or --baseline_type "
                             "(+ matching args).")
        if bt in ("B0", "B1"):
            if bt == "B0":
                phi = 0.0
                if args_cli.baseline_alpha is None:
                    raise ValueError("B0 needs --baseline_alpha")
                alpha = args_cli.baseline_alpha
            else:
                if args_cli.baseline_phi is None or args_cli.baseline_alpha is None:
                    raise ValueError("B1 needs --baseline_phi + --baseline_alpha")
                phi = args_cli.baseline_phi
                alpha = args_cli.baseline_alpha
            baseline_spec = {"type": bt, "phi": phi, "alpha": alpha}
            policy_label = f"baseline_{bt}_phi{phi}_alpha{alpha}"
        else:  # B2
            if (args_cli.baseline_alpha is None or args_cli.baseline_eps0 is None
                    or args_cli.baseline_lam is None):
                raise ValueError("B2 needs --baseline_alpha + --baseline_eps0 + --baseline_lam")
            baseline_spec = {
                "type": "B2",
                "alpha": args_cli.baseline_alpha,
                "eps0": args_cli.baseline_eps0,
                "lam": args_cli.baseline_lam,
            }
            policy_label = (f"baseline_B2_alpha{baseline_spec['alpha']}"
                            f"_eps{baseline_spec['eps0']}_lam{baseline_spec['lam']}")
    task_id = _scene_task_id(args_cli.scene, arch)
    print(f"[eval] policy={policy_label}  arch={arch}  scene={args_cli.scene}  task={task_id}")

    # ---- DR sweeps (list of (axis, values) pairs) ----
    if args_cli.dr_sweeps is not None:
        sweeps = []
        for spec in args_cli.dr_sweeps:
            axis, vals = spec.split(":")
            sweeps.append((axis, [float(v) for v in vals.split(",")]))
    elif args_cli.dr_axis is not None:
        vals = args_cli.dr_values if args_cli.dr_values is not None \
            else [NOMINAL_DR[args_cli.dr_axis]]
        sweeps = [(args_cli.dr_axis, vals)]
    else:
        sweeps = SCENE_DEFAULTS[args_cli.scene]
    n_cells = sum(len(v) for _, v in sweeps)
    print(f"[eval] DR sweeps ({n_cells} cells): "
          + "  ".join(f"{ax}={vs}" for ax, vs in sweeps))

    # ---- env ----
    loco = load_locomotion_actor(retrieve_file_path(args_cli.checkpoint), device)
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco
    # apply masking flags at eval (NoPriv / NoProprio policies trained
    # with them, so eval must match the training obs distribution)
    if arch == "V2NoPriv":
        env_cfg.actions.cbf_param.obs_mask_priv = True
    elif arch == "V2NoProprio":
        env_cfg.actions.cbf_param.obs_mask_proprio = True

    env = gym.make(task_id, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_dir = os.path.join(os.path.abspath(args_cli.out_dir), "videos")
        os.makedirs(video_dir, exist_ok=True)
        # Tag the video filename with policy + scene so the orchestrator's
        # output dir doesn't get a heap of identically-named files.
        name_prefix = f"{policy_label}_{args_cli.scene}"
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": name_prefix,
        }
        print(f"[eval] video recording ON  folder={video_dir}  "
              f"prefix={name_prefix}  length={args_cli.video_length}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    agent_cfg.max_iterations = 0   # eval-only

    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ---- load policy (or baseline) ----
    if args_cli.policy_dir is not None:
        runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(),
                                log_dir=None, device=device)
        ckpt_path = os.path.join(args_cli.policy_dir, "rsl_rl", "model_final.pt")
        ckpt = retrieve_file_path(ckpt_path)
        print(f"[eval] loading -> {ckpt}")
        runner.load(ckpt)
        policy_or_baseline = runner.get_inference_policy(device=device)
        is_baseline = False
    else:
        policy_or_baseline = baseline_spec
        is_baseline = True

    cbf = env_wrapped.unwrapped.action_manager._terms["cbf_param"]
    # cells: iterate all (axis, value) pairs from `sweeps`. Per cell,
    # pin every OTHER DR axis to nominal so the swept axis is the only
    # variable -- otherwise V2's full DR ranges would mask the per-axis
    # signal we're trying to measure.
    cells = []
    cell_i = 0
    with torch.inference_mode():
        for axis, values in sweeps:
            for v in values:
                cell_i += 1
                _pin_all_dr_to_nominal(cbf)
                _set_dr_axis(cbf, axis, v)
                print(f"[eval]   cell {cell_i}/{n_cells}  {axis}={v}  "
                      f"(others=nominal) ...")
                m = _roll_out(env_wrapped, policy_or_baseline,
                              args_cli.eval_steps, args_cli.eps_per_cell, device,
                              is_baseline)
                m["dr_axis"] = axis
                m["dr_value"] = float(v)
                cells.append(m)
                _total = (m['reach_rate'] + m['collision_rate']
                          + m['fall_rate'] + m['timeout_rate'])
                print(f"      reach={m['reach_rate']:.3f}  coll={m['collision_rate']:.3f}  "
                      f"fall={m['fall_rate']:.3f}  timeout={m['timeout_rate']:.3f}  "
                      f"(sum={_total:.3f}, stuck⊂timeout={m['stuck_rate']:.3f})  "
                      f"ttg={m['time_to_goal_mean']:.1f}  "
                      f"phi={m['phi_mean']:+.2f}±{m['phi_std']:.2f}  "
                      f"alpha={m['alpha_mean']:.2f}±{m['alpha_std']:.2f}")

    out_path = os.path.join(
        args_cli.out_dir,
        f"eval_{policy_label}_{args_cli.scene}.json")
    with open(out_path, "w") as f:
        json.dump({
            "policy": policy_label,
            "arch": arch,
            "scene": args_cli.scene,
            "sweeps": [{"axis": ax, "values": list(vs)} for ax, vs in sweeps],
            "num_envs": args_cli.num_envs,
            "eval_steps": args_cli.eval_steps,
            "eps_per_cell": args_cli.eps_per_cell,
            "cells": cells,
        }, f, indent=2)
    print(f"[eval] saved -> {out_path}")

    env.close()
    simulation_app.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
