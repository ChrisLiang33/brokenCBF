"""Input attribution: of the actor's (phi, alpha) output variance,
how much is driven by z (compressed priv), proprio, and lidar?

Answers the two-step question:
  Q1. Do (phi, alpha) adapt AT ALL?
      -> std(phi), std(alpha) > some threshold means yes
  Q2. If they adapt, what's driving the adaptation?
      -> partial R^2: re-fit a ridge regression with each input group
         dropped, see which drop reduces R^2 the most. That input is
         contributing the most unique variance.

Why ridge regression: the input groups have very different
dimensionalities (z=8, proprio=45, lidar=144). Ordinary least squares
would let lidar's high dim absorb noise variance, inflating its raw
R^2. Ridge regularization (lambda * I) keeps the comparison fair.

Why linear: the actor MLP is non-linear, so we miss interactions. But
the FIRST-ORDER attribution (which input group has the biggest linear
contribution) is what the user actually cares about for the "does it
USE priv" question.

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase8_input_attribution.py \\
        --teacher_ckpt phase7_rma_static_teacher_outputs/rsl_rl/model_final.pt \\
        --locomotion_ckpt /home/.../model_299.pt \\
        --task Isaac-CBF-Adaptive-Go2-RMAStatic-v0 \\
        --headless
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher_ckpt", required=True)
parser.add_argument("--locomotion_ckpt", required=True)
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-RMAStatic-v0",
                    help="Env that the teacher was trained on. Slice "
                         "constants are auto-resolved from the model class.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rollout_steps", type=int, default=800,
                    help="More steps = better regression estimate.")
parser.add_argument("--ridge_alpha", type=float, default=1.0,
                    help="L2 regularization strength for the partial R^2.")
parser.add_argument("--out_dir", default="phase8_input_attribution_outputs")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

# ---------------------------------------------------------------------------
import json
import importlib.metadata as metadata

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
import cbf_task  # noqa
from cbf_task.agents import rma_actor_critic  # noqa: register RMAMLPModel
from cbf_task.agents import rma_classic_actor_critic  # noqa: register RMAClassicMLPModel
from cbf_task.locomotion_loader import load_locomotion_actor


def ridge_r2(X: np.ndarray, y: np.ndarray, lam: float) -> float:
    """R^2 of a ridge-regression fit of y on X. Adds a bias column.
    Numerically stable for d <= ~few thousand."""
    if X.shape[1] == 0:
        # no features: R^2 with mean predictor = 0
        return 0.0
    n, d = X.shape
    # center to absorb bias; equivalent to adding a bias column for R^2
    X_c = X - X.mean(axis=0, keepdims=True)
    y_c = y - y.mean()
    # ridge solution
    A = X_c.T @ X_c + lam * np.eye(d)
    w = np.linalg.solve(A, X_c.T @ y_c)
    y_pred = X_c @ w
    ss_res = float(((y_c - y_pred) ** 2).sum())
    ss_tot = float((y_c ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-9)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args_cli.out_dir, exist_ok=True)

    loco = load_locomotion_actor(retrieve_file_path(args_cli.locomotion_ckpt), device)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args_cli.seed
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco
    env = gym.make(args_cli.task, cfg=env_cfg)
    cbf = env.unwrapped.action_manager._terms["cbf_param"]

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(),
                             log_dir=None, device=device)
    runner.load(retrieve_file_path(args_cli.teacher_ckpt))
    policy = runner.get_inference_policy(device=device)

    # Detect model architecture and resolve slice attrs accordingly.
    # RMAMLPModel / RMAClassicMLPModel have a _BranchedMLP with
    # (_priv_slice, _propr_slice, _lp_slice, _ld_slice, z_enc).
    # RMAHistoryMLPModel has a _HistoryBranchedMLP with
    # (_ph_slice, _pact_slice, _lp_slice, _ld_slice, proprio_enc).
    actor = runner.alg.get_policy()
    branched = actor.mlp
    is_history = hasattr(branched, "_ph_slice")
    print(f"  model class: {type(actor).__name__}  "
          f"(history mode: {is_history})")

    # ----- rollout: collect per-input features + (phi, alpha) per step -----
    env_wrapped.unwrapped.reset()
    Z_buf, P_buf, L_buf, PHI_buf, A_buf = [], [], [], [], []
    print(f"  rolling {args_cli.rollout_steps} steps with N={args_cli.num_envs} ...")
    obs = env_wrapped.get_observations()

    def _as_tensor(o):
        """env_wrapped.get_observations() returns a TensorDict in
        rsl_rl 5.0.1's new-format pipeline. Extract the "policy" group
        tensor for slicing. If it's already a tensor, pass through."""
        if hasattr(o, "keys"):           # TensorDict / dict
            t = o["policy"]
        else:
            t = o
        return t.to(device)

    for step in range(args_cli.rollout_steps):
        obs_t = _as_tensor(obs)          # (N, obs_dim) tensor
        with torch.inference_mode():
            action = policy(obs)         # policy accepts TensorDict
            if is_history:
                # For RMAHistory: there's no raw priv. "z" slot becomes
                # the proprio CNN's compressed feature (proprio_feat,
                # 64-dim). Proprio raw = the full history flattened
                # (large). Lidar same as standard.
                proprio_hist = obs_t[..., branched._ph_slice]
                z = branched.proprio_enc(proprio_hist)        # (N, 64) = "proprio_feat"
                proprio = proprio_hist                          # the raw input
                lidar = torch.cat([obs_t[..., branched._lp_slice],
                                    obs_t[..., branched._ld_slice]], dim=-1)
            else:
                # Standard RMAMLPModel / RMAClassicMLPModel
                priv = obs_t[..., branched._priv_slice]
                z = branched.z_enc(priv)                       # (N, 8)
                proprio = obs_t[..., branched._propr_slice]
                lidar = torch.cat([obs_t[..., branched._lp_slice],
                                    obs_t[..., branched._ld_slice]], dim=-1)
        Z_buf.append(z.detach().cpu())
        P_buf.append(proprio.detach().cpu())
        L_buf.append(lidar.detach().cpu())
        PHI_buf.append(cbf.last_phi.detach().cpu())
        A_buf.append(cbf.last_alpha.detach().cpu())
        obs, _, _, _ = env_wrapped.step(action)

    Z = torch.stack(Z_buf, dim=0).flatten(0, 1).numpy()           # (N*T, 8)
    P = torch.stack(P_buf, dim=0).flatten(0, 1).numpy()           # (N*T, 45)
    L = torch.stack(L_buf, dim=0).flatten(0, 1).numpy()           # (N*T, 144)
    Y_phi   = torch.stack(PHI_buf, dim=0).flatten().numpy()       # (N*T,)
    Y_alpha = torch.stack(A_buf, dim=0).flatten().numpy()
    print(f"  collected samples: Z={Z.shape}  P={P.shape}  L={L.shape}  "
          f"Y_phi={Y_phi.shape}")

    # ----- Q1: does it adapt at all? -----
    print()
    print("=" * 96)
    print("  Q1  --  do (phi, alpha) adapt at all?")
    print("=" * 96)
    phi_std = float(Y_phi.std())
    alpha_std = float(Y_alpha.std())
    phi_range = float(Y_phi.max() - Y_phi.min())
    alpha_range = float(Y_alpha.max() - Y_alpha.min())
    phi_w = float(cbf._phi_hi - cbf._phi_lo)
    alpha_w = float(cbf._alpha_hi - cbf._alpha_lo)
    print(f"  phi:    std={phi_std:.3f}   range={phi_range:.3f} "
          f"({100*phi_range/phi_w:.1f}% of bound width)")
    print(f"  alpha:  std={alpha_std:.3f}   range={alpha_range:.3f} "
          f"({100*alpha_range/alpha_w:.1f}% of bound width)")
    phi_adapts = phi_range > 0.10 * phi_w
    alpha_adapts = alpha_range > 0.10 * alpha_w
    if phi_adapts and alpha_adapts:
        verdict_q1 = "BOTH adapt"
    elif phi_adapts:
        verdict_q1 = "phi adapts, alpha pegged"
    elif alpha_adapts:
        verdict_q1 = "alpha adapts, phi pegged"
    else:
        verdict_q1 = "NEITHER adapts (policy is constant)"
    print(f"  verdict: {verdict_q1}")

    # ----- Q2: what's driving the variance? -----
    print()
    print("=" * 96)
    print("  Q2  --  what drives the variance? (ridge R^2, lam="
          f"{args_cli.ridge_alpha})")
    if is_history:
        print("  NOTE: this is an RMAHistory model. The 'z' column is the "
              "CNN-compressed proprio_feat (64-dim); 'proprio' column is "
              "the raw 50-step history (2250-dim); 'lidar' is the lidar "
              "ring (144-dim). The compressed feat is what the actor's "
              "main MLP actually consumes from the history input.")
    print("=" * 96)
    X_full = np.concatenate([Z, P, L], axis=1)
    dims = {"z": Z.shape[1], "proprio": P.shape[1], "lidar": L.shape[1]}
    offsets = {"z": 0, "proprio": Z.shape[1],
               "lidar": Z.shape[1] + P.shape[1]}
    slices_ = {grp: slice(offsets[grp], offsets[grp] + dims[grp])
               for grp in ("z", "proprio", "lidar")}

    results = {}
    for out_name, y in (("phi", Y_phi), ("alpha", Y_alpha)):
        r2_full = ridge_r2(X_full, y, args_cli.ridge_alpha)
        per_group = {}
        for grp, sl in slices_.items():
            # marginal: only this group's features
            X_only = X_full[:, sl]
            r2_marg = ridge_r2(X_only, y, args_cli.ridge_alpha)
            # partial / unique: full minus this group
            keep = np.ones(X_full.shape[1], dtype=bool)
            keep[sl] = False
            X_drop = X_full[:, keep]
            r2_drop = ridge_r2(X_drop, y, args_cli.ridge_alpha)
            r2_unique = max(r2_full - r2_drop, 0.0)
            per_group[grp] = {"marginal": r2_marg, "unique": r2_unique}
        results[out_name] = {"r2_full": r2_full, "per_group": per_group}
        print()
        print(f"  {out_name}:  full R^2 = {r2_full:.3f}")
        print(f"    {'group':<10}  {'marginal R^2':>14}  {'unique R^2':>12}")
        for grp in ("z", "proprio", "lidar"):
            pg = per_group[grp]
            print(f"    {grp:<10}  {pg['marginal']:>14.3f}  {pg['unique']:>12.3f}")
        # narrative: rank by unique R^2
        ranked = sorted(per_group.items(),
                        key=lambda kv: -kv[1]["unique"])
        if r2_full < 0.05:
            narrative = ("output is approximately constant (R^2 ~ 0); "
                         "attribution is meaningless")
        else:
            top_grp, top = ranked[0]
            narrative = (f"{top_grp} contributes the most UNIQUE variance "
                         f"({top['unique']:.3f}); others may be correlated "
                         f"but {top_grp} is the dominant driver")
        print(f"    --> {narrative}")

    # ----- save -----
    summary = {
        "task": args_cli.task,
        "teacher_ckpt": args_cli.teacher_ckpt,
        "rollout_steps": args_cli.rollout_steps,
        "num_envs": args_cli.num_envs,
        "ridge_alpha": args_cli.ridge_alpha,
        "n_samples": int(Y_phi.shape[0]),
        "q1": {
            "phi_std": phi_std, "phi_range_pct": 100 * phi_range / phi_w,
            "alpha_std": alpha_std, "alpha_range_pct": 100 * alpha_range / alpha_w,
            "verdict": verdict_q1,
        },
        "q2": results,
    }
    out_path = os.path.join(args_cli.out_dir, "input_attribution.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"  saved -> {out_path}")
    print("=" * 96)

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
