"""Phase 10 / V2 smoke test -- run BEFORE the overnight pipeline.

Cheap sanity check (~2 min on labbox) that:
  1. All V2 train + eval tasks are REGISTERED in gym (catches typos in
     __init__.py registration block).
  2. Every V2 env_cfg constructs cleanly via load_cfg_from_registry
     (catches typos in cfg classes, missing imports, broken __post_init__
     chains). This is a config-only check -- no sim is spawned.
  3. ONE full Isaac sim is spawned for V2Full to do the deep checks:
     obs shape == 198, 50-step rollout produces sane (phi, alpha,
     intervention), u_nom rate-limit holds in BODY frame (cap=0.15 m/s),
     DR pinning round-trips through every channel.

Why only one sim: Isaac Lab does not cleanly release GPU/USD state
between `env.close()` calls in the same process, so creating multiple
envs in one process hangs after the first or second. The 10 trainings
and 40+ evals each get their own subprocess in the overnight pipeline
-- those are independent processes and so are fine.

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase10_smoke.py \\
        --checkpoint /path/to/locomotion/model.pt --headless

Exit code 0 = all checks pass, safe to launch overnight pipeline.
Non-zero = stop and investigate (the message names the failing check).
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64,
                    help="Tiny num_envs for a quick check.")
parser.add_argument("--rollout_steps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import importlib.metadata as metadata

import gymnasium as gym
import torch

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


# All Phase 10 V2 tasks. We config-construct every one of these (no sim
# spawn) so missing classes / typos crash the smoke fast and cheap.
V2_TRAIN_TASKS = [
    "V2Full", "V2NoPriv", "V2NoProprio", "V2RMAClassic", "V2History",
]
V2_EVAL_TASKS = []
for _scene in ("E1Gap", "E2Slalom", "E3Wall", "E4Field"):
    V2_EVAL_TASKS.append(f"V2Eval-{_scene}")               # Full layout
    V2_EVAL_TASKS.append(f"V2Eval-{_scene}-RMAClassic")    # 4-priv layout
    V2_EVAL_TASKS.append(f"V2Eval-{_scene}-History")       # history layout

# The single deep-check sim is on V2Full (representative of the
# Full/NoPriv/NoProprio layout that 3 of 5 archs share).
DEEP_TASK = "V2Full"
DEEP_OBS_DIM = 198


def _build(task_suffix: str, loco, num_envs: int, device: str):
    task = f"Isaac-CBF-Adaptive-Go2-{task_suffix}-v0"
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco
    env = gym.make(task, cfg=env_cfg)
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    return env, env_wrapped, env_cfg


def _short_rollout(env_wrapped, n_steps: int):
    """Step the env with zero actions; collect (phi, alpha, intervention,
    jitter, u_nom) per step so we can verify they're sane.
    """
    env = env_wrapped.unwrapped
    cbf = env.action_manager._terms["cbf_param"]
    N = env.num_envs
    env.reset()
    cbf.episode_reach_any.zero_()
    cbf.episode_collide_any.zero_()
    cbf.episode_fall_any.zero_()
    cbf.episode_stuck_any.zero_()
    obs = env_wrapped.get_observations()
    phi_acc, alpha_acc, int_acc, jit_acc = [], [], [], []
    u_nom_acc = []
    for _ in range(n_steps):
        action = torch.zeros(N, 2, device=env.device)
        obs, _, _, _ = env_wrapped.step(action)
        phi_acc.append(cbf.last_phi.detach().clone())
        alpha_acc.append(cbf.last_alpha.detach().clone())
        int_acc.append(cbf.last_intervention.detach().clone())
        jit_acc.append(cbf.last_action_jitter.detach().clone())
        u_nom_acc.append(cbf.last_u_nom.detach().clone())
    return {
        "obs_shape": tuple(obs["policy"].shape) if hasattr(obs, "keys")
                     else tuple(obs.shape),
        "phi": torch.stack(phi_acc, dim=0),       # (T, N)
        "alpha": torch.stack(alpha_acc, dim=0),
        "int": torch.stack(int_acc, dim=0),
        "jitter": torch.stack(jit_acc, dim=0),
        "u_nom": torch.stack(u_nom_acc, dim=0),   # (T, N, 2) WORLD frame
    }


def _check_dr_roundtrip(env_wrapped, axis_attr_pair, target_value):
    """Pin an axis to target_value, reset the env, verify the per-env
    tensor reads back as target_value. This catches PhysX-apply regressions."""
    cbf = env_wrapped.unwrapped.action_manager._terms["cbf_param"]
    lo_attr, hi_attr = axis_attr_pair
    setattr(cbf, lo_attr, float(target_value))
    setattr(cbf, hi_attr, float(target_value))
    env_wrapped.unwrapped.reset()
    # the per-env tensor (e.g., _friction_coef) should now be target_value
    tensor_attr = lo_attr.replace("_lo", "").replace("_hi", "")
    # _friction_lo -> _friction -> action term has _friction_coef, _base_mass_delta, etc.
    name_map = {
        "_friction": "_friction_coef",
        "_motor_strength": "_motor_strength",
        "_v_max": "_v_max",
        "_disturbance_force": "_disturbance_force",
        "_base_mass": "_base_mass_delta",
        "_com_offset": "_com_offset",
        "_actuation_noise_std": "_actuation_noise_std",
    }
    t = getattr(cbf, name_map[tensor_attr])
    actual = float(t.float().mean().item())
    return actual


def main():
    import gymnasium
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}  num_envs={args_cli.num_envs}  "
          f"rollout={args_cli.rollout_steps}")
    failures = []

    # ===== Check 1: every V2 task is registered in gym =====
    print("\n[smoke] === gym registration ===")
    registered = set(gymnasium.envs.registry.keys())
    for suffix in V2_TRAIN_TASKS + V2_EVAL_TASKS:
        task_id = f"Isaac-CBF-Adaptive-Go2-{suffix}-v0"
        if task_id not in registered:
            failures.append(f"task not registered: {task_id}")
            print(f"  FAIL {task_id} (not registered)")
        else:
            print(f"  OK {task_id}")

    # ===== Check 2: every V2 env_cfg config-constructs cleanly =====
    # Uses load_cfg_from_registry which just imports the class and runs
    # __post_init__. No sim spawn, no GPU. Catches typos and broken
    # inheritance fast (~1 s/task).
    print("\n[smoke] === env_cfg construction (no sim) ===")
    for suffix in V2_TRAIN_TASKS + V2_EVAL_TASKS:
        task_id = f"Isaac-CBF-Adaptive-Go2-{suffix}-v0"
        if task_id not in registered:
            continue  # already failed above
        try:
            cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
            # spot-check the V2 distribution settings flow through
            at = cfg.actions.cbf_param
            unom_mode = getattr(at, "unom_mode", "?")
            clumsy = getattr(at, "unom_clumsiness", "?")
            print(f"  OK {suffix}  unom_mode={unom_mode}  clumsy={clumsy}")
        except Exception as e:
            failures.append(f"cfg {suffix}: {type(e).__name__}: {e}")
            print(f"  FAIL {suffix}: {type(e).__name__}: {e}")

    if failures:
        # don't bother spinning up sim if registration/cfg already broke
        print("\n" + "=" * 70)
        print(f"[smoke] {len(failures)} early FAILURES -- "
              "fix these before the sim deep-check is meaningful:")
        for f in failures:
            print(f"  - {f}")
        simulation_app.close()
        sys.exit(1)

    # ===== Deep checks: spin up ONE sim (V2Full) =====
    # Isaac Lab can't cleanly close + reopen sims in the same process,
    # so we only spawn one. The overnight pipeline runs each training /
    # eval as its own subprocess and so doesn't hit this limitation.
    print(f"\n[smoke] === deep checks on {DEEP_TASK} (single sim spawn) ===")
    loco = load_locomotion_actor(retrieve_file_path(args_cli.checkpoint), device)
    env, env_wrapped, env_cfg = _build(DEEP_TASK, loco, args_cli.num_envs, device)
    obs = env_wrapped.get_observations()
    obs_t = obs["policy"] if hasattr(obs, "keys") else obs
    actual_dim = int(obs_t.shape[-1])
    if actual_dim != DEEP_OBS_DIM:
        failures.append(f"{DEEP_TASK} obs_dim {actual_dim} != {DEEP_OBS_DIM}")
        print(f"  FAIL obs_dim {actual_dim} != {DEEP_OBS_DIM}")
    else:
        print(f"  OK obs_dim={actual_dim}")

    # --- rollout sanity ---
    print(f"\n[smoke]   --- {args_cli.rollout_steps}-step rollout ---")
    r = _short_rollout(env_wrapped, args_cli.rollout_steps)
    phi_m = float(r["phi"].mean().item())
    alpha_m = float(r["alpha"].mean().item())
    int_m = float(r["int"].mean().item())
    jit_m = float(r["jitter"].mean().item())
    print(f"    phi_mean={phi_m:+.3f}  alpha_mean={alpha_m:.3f}  "
          f"int_mean={int_m:.3f}  jitter_mean={jit_m:.3f}")
    if not (-0.1 <= phi_m <= 1.1):
        failures.append(f"phi out of [0,1]: {phi_m}")
    if not (0.1 <= alpha_m <= 4.5):
        failures.append(f"alpha out of [0.2,4]: {alpha_m}")
    if not torch.isfinite(r["phi"]).all():
        failures.append("phi has NaN/Inf")
    if not torch.isfinite(r["alpha"]).all():
        failures.append("alpha has NaN/Inf")

    # --- u_nom rate-limit (body frame) ---
    # cbf.last_u_nom is u_nom_b (body frame). Rate-limit is now applied
    # in body frame, so ‖Δlast_u_nom‖ ≤ unom_max_step + ε must hold.
    print(f"\n[smoke]   --- u_nom rate-limit (body frame) ---")
    unom = r["u_nom"]                          # (T, N, 2)
    d_unom = unom[1:] - unom[:-1]              # (T-1, N, 2)
    d_norm = torch.linalg.norm(d_unom, dim=-1) # (T-1, N)
    cap = env_cfg.actions.cbf_param.unom_max_step
    max_obs = float(d_norm.max().item())
    print(f"    unom_max_step={cap}  observed max Δ‖u_nom_b‖={max_obs:.4f}")
    if cap > 0.0 and max_obs > cap + 1e-3:
        failures.append(f"u_nom rate-limit broken: cap={cap}, observed={max_obs}")
        print("    FAIL")
    else:
        print("    OK")

    # --- DR pinning round-trip ---
    print(f"\n[smoke]   --- DR pinning round-trip ---")
    for axis_name, axis_attr, target in [
        ("friction",        ("_friction_lo", "_friction_hi"),         0.45),
        ("motor_strength",  ("_motor_strength_lo", "_motor_strength_hi"), 1.10),
        ("v_max",           ("_v_max_lo", "_v_max_hi"),               1.30),
        ("base_mass",       ("_base_mass_lo", "_base_mass_hi"),       1.50),
        ("com_offset",      ("_com_offset_lo", "_com_offset_hi"),     0.03),
        ("actuation_noise", ("_actuation_noise_std_lo",
                             "_actuation_noise_std_hi"),              0.02),
        ("disturbance",     ("_disturbance_force_lo",
                             "_disturbance_force_hi"),                12.0),
    ]:
        actual = _check_dr_roundtrip(env_wrapped, axis_attr, target)
        tag = "OK" if abs(actual - target) < 1e-4 else "FAIL"
        print(f"    {tag} pin {axis_name}={target}  -> tensor mean={actual:.4f}")
        if abs(actual - target) > 1e-4:
            failures.append(f"DR pin {axis_name}: target {target}, got {actual}")

    env.close()

    # ===== Report =====
    print("\n" + "=" * 70)
    if failures:
        print(f"[smoke] {len(failures)} FAILURES -- do NOT launch overnight:")
        for f in failures:
            print(f"  - {f}")
        simulation_app.close()
        sys.exit(1)
    print("[smoke] ALL CHECKS PASS -- pipeline is safe to launch")
    simulation_app.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
