"""RMA student adaptation module -- predict priv from proprio history.

At deployment, the policy can't see the priv slice. The student learns
to predict the 7 privileged channels (disturbance, friction, mass_delta,
motor_strength, actuation_noise, com_offset, v_max) from a window of
`deployable_obs` (proprio + prev_loco_action; NO velocity_commands --
that leak was removed 2026-05-24). At deployment we substitute
`obs[..., 0:7]` with `student(history)` and the trained actor MLP runs
as-is.

Design choice: predict priv directly (7-dim), not the teacher's latent
z (8-dim). Predicting priv is architecture-independent (works whether
the teacher uses the branched encoder or a flat MLP), and the
substitution at deploy time is a clean obs-level slice replacement. The
trade-off is that we lose any compression the teacher's z-encoder might
have learned, but for 7 priv channels that's negligible.

Validated R^2 prior (Phase 1.5 fingerprint gate on the original 4
channels): R^2 >= 0.93 from a 20-step window of deployable_obs. The 3
NEW channels (actuation_noise, com_offset, v_max) have NOT been
fingerprint-gated -- if R^2 on v_max comes back low, the window may
need to be longer than 20 steps. v_max is the only validated
ALPHA-channel ([[alpha_channel_search]]), so its R^2 is what matters
most for the deployment story; the other two failed their gates and
the policy ignores them.

Run on labbox (teacher is the v7 SHIELD checkpoint):
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase5_train_student.py \\
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --policy_checkpoint phase6_shield_v7_teacher_outputs/rsl_rl/model_final.pt \\
        --num_envs 64 --collect_steps 2000 --headless

This script trains and saves the student weights + R^2 metadata. It
does NOT do the deployment-substitution evaluation -- that's a
separate step (TODO: fold into this script or use phase5_deploy_eval.py
once it's updated to 7-priv).
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--policy_checkpoint", required=True)
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-UnifiedLidarSDF-v0",
                    help="Must match the env the teacher was trained on. "
                         "Default is the SHIELD env (7 priv channels). "
                         "If you pass --policy_checkpoint for a v6-or-earlier "
                         "teacher trained on a different env, override this.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--collect_steps", type=int, default=2000,
                    help="Env steps to roll the teacher for collecting "
                         "(history, z) pairs. 64 envs * 2000 steps = ~128k "
                         "samples, plenty for an 8-d regression target.")
parser.add_argument("--window_len", type=int, default=20)
parser.add_argument("--student_epochs", type=int, default=300)
parser.add_argument("--student_hidden", type=int, nargs="+",
                    default=[256, 128])
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--out_dir", default="phase5_student_outputs")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import importlib.metadata as metadata
import json

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
from cbf_task.agents import rma_actor_critic  # noqa: F401 -- registers RMAMLPModel
from cbf_task.agents.rma_actor_critic import PRIV_SLICE, PRIV_DIM, PROPRIO_DIM
from cbf_task.locomotion_loader import load_locomotion_actor
from cbf_task.mdp import deployable_obs, priv_obs


# Order MUST match mdp.priv_obs (line 420-426): disturbance, friction,
# mass_delta, motor_strength, actuation_noise, com_offset, v_max.
# v_max is the only validated alpha-channel (phase6_vmax_gate); R^2 here
# is the headline metric for whether the student is deployment-ready.
PRIV_NAMES = ["disturbance", "friction", "mass_delta", "motor_strength",
              "actuation_noise", "com_offset", "v_max"]
# Channels the trained policy actually USES for adaptation. Annotated
# in the R^2 printout so a low R^2 on a non-validated channel doesn't
# trigger a false alarm. See [[alpha_channel_search]].
PRIV_VALIDATED = {"disturbance", "v_max"}


class StudentMLP(nn.Module):
    """phi(proprio_history) -> priv_hat. Flat MLP over the 20*48=960-d window."""

    def __init__(self, history_len: int, proprio_dim: int, priv_dim: int,
                 hidden: list[int]):
        super().__init__()
        in_dim = history_len * proprio_dim
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ELU()]
            last = h
        layers.append(nn.Linear(last, priv_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def collect_rollouts(env_wrapped, runner, num_envs, collect_steps,
                     window_len, device):
    """Drive the env with the trained teacher; per env, maintain a 20-step
    deployable_obs history buffer. After warmup, snapshot (history, priv)
    every step. Returns numpy X (N, window*48) and Y (N, 4).
    """
    policy = runner.get_inference_policy(device=device)

    env_wrapped.unwrapped.reset()
    obs = env_wrapped.get_observations()

    history_buf = torch.zeros(
        (num_envs, window_len, PROPRIO_DIM), device=device,
    )
    X_list, Y_list = [], []
    print(f"[student] collecting {collect_steps} rollout steps "
          f"({num_envs} envs in parallel) ...")
    for step in range(collect_steps):
        action = policy(obs)
        obs, _, _, _ = env_wrapped.step(action)

        # snapshot proprio AFTER the step (reflects the post-step state)
        unwr = env_wrapped.unwrapped
        proprio_now = deployable_obs(unwr)
        history_buf = torch.roll(history_buf, shifts=-1, dims=1)
        history_buf[:, -1, :] = proprio_now

        # warmup: skip until we've actually filled the buffer
        if step < window_len:
            continue

        # priv target directly from the env (7-dim raw priv values).
        # Predicting priv (not the teacher's latent z) is architecture-
        # independent and the substitution at deploy is a clean obs-
        # level slice replacement: obs[..., 0:7] = student(history).
        priv = priv_obs(unwr)                          # (N, 7)
        X_list.append(history_buf.detach().clone().reshape(num_envs, -1).cpu())
        Y_list.append(priv.detach().cpu())

        if (step + 1) % 200 == 0:
            print(f"    step {step+1:>4d}  samples={(step + 1 - window_len) * num_envs}")

    X = torch.cat(X_list, dim=0).numpy()
    Y = torch.cat(Y_list, dim=0).numpy()
    return X, Y


def per_dim_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-9)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args_cli.out_dir, exist_ok=True)

    # 1) build env + load teacher
    loco = load_locomotion_actor(retrieve_file_path(args_cli.checkpoint), device)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco
    env = gym.make(args_cli.task, cfg=env_cfg)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    agent_cfg.max_iterations = 0
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(),
                            log_dir=None, device=device)
    runner.load(retrieve_file_path(args_cli.policy_checkpoint))
    print(f"[student] teacher loaded -> {args_cli.policy_checkpoint}")

    # 2) collect data
    with torch.inference_mode():
        X, Y = collect_rollouts(env_wrapped, runner, args_cli.num_envs,
                                args_cli.collect_steps, args_cli.window_len,
                                device)
    print(f"[student] collected X={X.shape}  Y={Y.shape}")
    env.close()

    # 3) train/test split
    n = X.shape[0]
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    n_train = int(n * 0.85)
    tr, te = idx[:n_train], idx[n_train:]

    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)

    # 4) train student
    student = StudentMLP(
        history_len=args_cli.window_len,
        proprio_dim=PROPRIO_DIM,
        priv_dim=PRIV_DIM,
        hidden=list(args_cli.student_hidden),
    ).to(device)
    print(f"[student] params: {sum(p.numel() for p in student.parameters())}")

    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    bs = args_cli.batch_size

    tr_t = torch.tensor(tr, device=device, dtype=torch.long)
    te_t = torch.tensor(te, device=device, dtype=torch.long)

    print(f"[student] training {args_cli.student_epochs} epochs ...")
    for ep in range(args_cli.student_epochs):
        student.train()
        perm = torch.randperm(len(tr_t), device=device)
        total = 0.0
        for i in range(0, len(tr_t), bs):
            sel = tr_t[perm[i:i+bs]]
            yhat = student(Xt[sel])
            loss = loss_fn(yhat, Yt[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * sel.shape[0]
        if (ep + 1) % 25 == 0 or ep == 0:
            student.eval()
            with torch.no_grad():
                yhat_te = student(Xt[te_t])
                te_loss = loss_fn(yhat_te, Yt[te_t]).item()
            print(f"    epoch {ep+1:>3d}  train={total/len(tr_t):.4f}  "
                  f"test={te_loss:.4f}")

    # 5) eval -- per-dim R^2
    student.eval()
    with torch.no_grad():
        yhat_te = student(Xt[te_t]).cpu().numpy()
    r2 = per_dim_r2(Y[te], yhat_te)
    mean_r2 = float(np.mean(r2))

    print()
    print("=" * 78)
    print("  RMA STUDENT  --  R^2 per priv channel (test set)")
    print("  [*] = validated channel the policy actually uses for adaptation")
    print("=" * 78)
    for i, (name, v) in enumerate(zip(PRIV_NAMES, r2)):
        tag = " [*]" if name in PRIV_VALIDATED else "    "
        print(f"  {name:>16}{tag}:  R^2 = {v:+.3f}")
    # Validated-channel mean is the headline number: high R^2 on
    # disturbance/v_max means deployment substitution should preserve
    # adaptation behavior. R^2 on non-validated channels is informational
    # only (the policy ignores them).
    validated_r2 = [v for name, v in zip(PRIV_NAMES, r2) if name in PRIV_VALIDATED]
    mean_validated = float(np.mean(validated_r2)) if validated_r2 else float("nan")
    print()
    print(f"  mean R^2 across all {PRIV_DIM} channels: {mean_r2:+.3f}")
    print(f"  mean R^2 across validated channels only: {mean_validated:+.3f}  "
          f"<-- the deployment-relevant number")
    print("=" * 78)

    # 6) save
    ckpt_path = os.path.join(args_cli.out_dir, "student.pt")
    torch.save({
        "state_dict": student.state_dict(),
        "history_len": args_cli.window_len,
        "proprio_dim": PROPRIO_DIM,
        "priv_dim": PRIV_DIM,
        "priv_names": PRIV_NAMES,
        "hidden": list(args_cli.student_hidden),
        "r2_per_channel": r2.tolist(),
        "mean_r2": mean_r2,
    }, ckpt_path)
    with open(os.path.join(args_cli.out_dir, "student_summary.json"), "w") as f:
        json.dump({"priv_names": PRIV_NAMES,
                   "validated_channels": sorted(PRIV_VALIDATED),
                   "r2_per_channel": r2.tolist(),
                   "mean_r2": mean_r2,
                   "mean_r2_validated": mean_validated,
                   "n_samples": int(n),
                   "n_train": int(n_train),
                   "n_test": int(n - n_train),
                   "task": args_cli.task,
                   "teacher_ckpt": args_cli.policy_checkpoint}, f, indent=2)
    print(f"\n  saved -> {ckpt_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
