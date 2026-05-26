"""Phase 10 / V2 unified training script.

Trains one (architecture × intervention_cost) cell of the V2 unified
comparison. All 5 architectures share the SAME training distribution
(K=3 random obstacles, 7 DR channels, clumsy-human u_nom, V2 rewards).
The architecture is selected by `--task`; the intervention cost is set
by `--intervention_cost` (default 0.0; flip to -0.05 for the alt
variant).

The 10 trainings:
  V2Full        × {0.0, -0.05}
  V2NoPriv      × {0.0, -0.05}
  V2NoProprio   × {0.0, -0.05}
  V2RMAClassic  × {0.0, -0.05}
  V2History     × {0.0, -0.05}

Skips the per-disturbance sweep at the end -- that lived in
phase5_train_teacher.py as a quick sanity check, but on the V2
distribution it's not particularly informative. The proper eval is
`phase10_eval_unified.py` (scene × DR setting × policy).

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase10_train_unified.py \\
        --task Isaac-CBF-Adaptive-Go2-V2Full-v0 \\
        --intervention_cost 0.0 \\
        --checkpoint /path/to/locomotion/model.pt \\
        --num_envs 2048 --max_iterations 3000 \\
        --out_dir phase10_outputs/V2Full_int0 --headless
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True,
                    help="Locomotion policy checkpoint (frozen).")
parser.add_argument("--task", required=True,
                    help="One of Isaac-CBF-Adaptive-Go2-V2{Full,NoPriv,"
                         "NoProprio,RMAClassic,History}-v0")
parser.add_argument("--intervention_cost", type=float, default=0.0,
                    help="Reward weight on intervention penalty. "
                         "User-confirmed sweep: {0.0, -0.05}.")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--diag_interval", type=int, default=200,
                    help="Iterations between in-training diagnostics.")
parser.add_argument("--video", action="store_true", default=False,
                    help="Periodically record env-0 to mp4 during training.")
parser.add_argument("--video_interval", type=int, default=15000,
                    help="env.step interval between video recordings. "
                         "Default 15000 ~ every 5-10 min at 2048 envs, "
                         "num_steps_per_env=24.")
parser.add_argument("--video_length", type=int, default=1250,
                    help="Length (env.step calls) of each video. Default "
                         "1250 = one full 25s episode at 50Hz.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless
# Video recording requires a render context in headless mode.
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import importlib.metadata as metadata
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401  (registers V2 tasks)
from cbf_task.agents import rma_actor_critic           # noqa: F401
from cbf_task.agents import rma_classic_actor_critic   # noqa: F401
from cbf_task.agents import rma_history_actor_critic   # noqa: F401
from cbf_task.locomotion_loader import load_locomotion_actor


def _fmt_hms(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def _emit_diag(env_wrapped, runner, iter_idx, total_iters, wall_t0):
    """One-line CBF / policy health snapshot.

    Reads the action term caches (phi, alpha, intervention, jitter) and
    the policy's output_std. Pegging/collapse warnings only -- per-
    episode rates are in rsl_rl's own per-iter table.
    """
    try:
        env = env_wrapped.unwrapped
        cbf = env.action_manager._terms["cbf_param"]
        phi = cbf.last_phi.detach().float()
        alpha = cbf.last_alpha.detach().float()
        intervention = cbf.last_intervention.detach().float()
        jitter = cbf.last_action_jitter.detach().float()
        a_std = float(runner.alg.get_policy().output_std.mean().item())
    except Exception as exc:
        print(f"[diag iter={iter_idx:5d}] diagnostic snapshot failed: {exc}")
        return
    phi_lo, phi_hi = float(cbf._phi_lo), float(cbf._phi_hi)
    alpha_lo, alpha_hi = float(cbf._alpha_lo), float(cbf._alpha_hi)
    phi_w = max(phi_hi - phi_lo, 1e-9)
    alpha_w = max(alpha_hi - alpha_lo, 1e-9)
    pm, ps = float(phi.mean()), float(phi.std())
    am, asd = float(alpha.mean()), float(alpha.std())
    im, jm = float(intervention.mean()), float(jitter.mean())
    elapsed = time.time() - wall_t0
    rate = iter_idx / max(elapsed, 1e-6)
    eta = max(total_iters - iter_idx, 0) / max(rate, 1e-6)
    print(f"[diag iter={iter_idx:5d}/{total_iters}] "
          f"phi={pm:+.3f}±{ps:.2f}[{phi_lo:+.1f},{phi_hi:+.1f}]  "
          f"alpha={am:.2f}±{asd:.2f}[{alpha_lo:.1f},{alpha_hi:.1f}]  "
          f"int={im:.3f}  jit={jm:.3f}  a_std={a_std:.3f}  "
          f"elapsed={_fmt_hms(elapsed)} ETA={_fmt_hms(eta)} @ {rate:.2f} it/s",
          flush=True)
    # only the most useful pegging warnings (don't dump WARN for every
    # iteration -- the user asked for diagnostic discipline)
    if a_std < 0.05:
        print(f"[diag]   [WARN] action_std={a_std:.3f} (policy collapsed)")
    if abs(pm - phi_hi) < 0.01 * phi_w:
        print(f"[diag]   [WARN] phi pegged at HI ({pm:.3f} ~ {phi_hi:.3f})")
    if abs(pm - phi_lo) < 0.01 * phi_w:
        print(f"[diag]   [WARN] phi pegged at LO ({pm:.3f} ~ {phi_lo:.3f})")
    if abs(am - alpha_hi) < 0.01 * alpha_w:
        print(f"[diag]   [WARN] alpha pegged at HI ({am:.3f} ~ {alpha_hi:.3f})")
    if abs(am - alpha_lo) < 0.01 * alpha_w:
        print(f"[diag]   [WARN] alpha pegged at LO ({am:.3f} ~ {alpha_lo:.3f})")


def _run_chunked(env_wrapped, runner, total_iters, diag_interval):
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
        _emit_diag(env_wrapped, runner, done, total_iters, t0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args_cli.out_dir, exist_ok=True)

    loco = load_locomotion_actor(retrieve_file_path(args_cli.checkpoint), device)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco
    # intervention-cost variant: 0.0 vs -0.05
    env_cfg.rewards.intervention.weight = float(args_cli.intervention_cost)
    print(f"[train] task={args_cli.task}  intervention_cost={args_cli.intervention_cost}  "
          f"num_envs={args_cli.num_envs}  max_iter={args_cli.max_iterations}")

    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_dir = os.path.join(os.path.abspath(args_cli.out_dir), "videos", "train")
        os.makedirs(video_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[train] video recording ON  folder={video_dir}  "
              f"interval={args_cli.video_interval} steps  "
              f"length={args_cli.video_length} steps")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    agent_cfg.max_iterations = int(args_cli.max_iterations)
    agent_cfg.seed = int(args_cli.seed)
    agent_cfg.experiment_name = (f"phase10_{args_cli.task.split('-')[-2]}"
                                  f"_int{args_cli.intervention_cost}")

    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    log_dir = os.path.join(os.path.abspath(args_cli.out_dir), "rsl_rl")
    os.makedirs(log_dir, exist_ok=True)
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(),
                            log_dir=log_dir, device=device)
    t0 = time.time()
    _run_chunked(env_wrapped, runner, agent_cfg.max_iterations,
                 int(args_cli.diag_interval))
    train_secs = time.time() - t0
    final_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_path)
    print(f"[train] saved -> {final_path}  ({_fmt_hms(train_secs)})")

    # tiny manifest so the eval script can pick the right env / model
    arch_tag = args_cli.task.split("-")[-2]   # e.g. "V2Full"
    with open(os.path.join(args_cli.out_dir, "manifest.txt"), "w") as f:
        f.write(f"task={args_cli.task}\n")
        f.write(f"arch={arch_tag}\n")
        f.write(f"intervention_cost={args_cli.intervention_cost}\n")
        f.write(f"num_envs={args_cli.num_envs}\n")
        f.write(f"iterations={args_cli.max_iterations}\n")
        f.write(f"checkpoint={final_path}\n")
        f.write(f"train_seconds={train_secs:.0f}\n")
    env.close()
    simulation_app.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
