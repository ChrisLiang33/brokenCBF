"""Encoder health diagnostic.

Tests four things on a trained teacher:
  (1) Architecture instantiated correctly -- is self.mlp actually
      `_BranchedMLP` with `_LidarCNN`, or did class_name resolution
      silently fall back to a vanilla `MLPModel.MLP`?
  (2) Encoder activations are alive -- for a batch of real rollout obs,
      compute per-dim std of z (z_enc output) and lidar_feat
      (lidar_enc output). Dead dim = std < 1e-3 = weight collapsed.
  (3) Encoder is sensitive to its own input -- vary the priv slice
      across its DR range with everything else fixed, measure how z
      changes; same for lidar with forward-ray distance.
  (4) Encoder output range is reasonable -- no saturation at extreme
      magnitudes that would suggest miscalibrated weights.

Run on labbox:
    cd ~/Desktop/cbf_rl_mvp/go2
    ~/IsaacLab/isaaclab.sh -p phase6_encoder_health.py \\
        --checkpoint /home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt \\
        --policy_checkpoint phase6_lidarcnn_teacher_outputs/rsl_rl/model_final.pt \\
        --task Isaac-CBF-Adaptive-Go2-RandObs-v0 \\
        --num_envs 256 --headless
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--policy_checkpoint", required=True)
parser.add_argument("--task", default="Isaac-CBF-Adaptive-Go2-RMA-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--n_rollout_steps", type=int, default=200)
parser.add_argument("--out_dir", default="phase6_encoder_health_outputs")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import importlib.metadata as metadata

import gymnasium as gym
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
from cbf_task.agents import rma_actor_critic  # noqa: F401
from cbf_task.agents.rma_actor_critic import (
    PRIV_SLICE, PROPRIO_SLICE, PREV_ACT_SLICE, LIDAR_SLICE, LIDAR_PREV_SLICE,
    EXPECTED_OBS_DIM, PRIV_DIM, LIDAR_DIM, _BranchedMLP, _LidarCNN,
)
from cbf_task.locomotion_loader import load_locomotion_actor


# ---------------------------------------------------------------------------
def section(title: str) -> None:
    print()
    print("=" * 88)
    print(f"  {title}")
    print("=" * 88)


def check_architecture(actor) -> dict:
    """Section 1: did the right model class get instantiated?"""
    section("1. ARCHITECTURE CHECK")
    mlp = actor.mlp
    print(f"  actor.__class__:        {type(actor).__name__}")
    print(f"  actor.mlp.__class__:    {type(mlp).__name__}")

    is_branched = isinstance(mlp, _BranchedMLP)
    is_lidar_cnn = (is_branched
                    and isinstance(getattr(mlp, "lidar_enc", None), _LidarCNN))
    is_z_enc = is_branched and hasattr(mlp, "z_enc")

    print(f"  _BranchedMLP active:    {is_branched}")
    print(f"  z_enc present:          {is_z_enc}")
    print(f"  _LidarCNN active:       {is_lidar_cnn}")

    if not is_branched:
        print("  ! FAIL -- model fell back to vanilla flat MLP. The")
        print("    runner cfg's class_name='RMAMLPModel' didn't resolve")
        print("    and rsl_rl built MLPModel instead. The CNN encoder")
        print("    you added is not being used during training.")
    else:
        print("  PASS -- branched encoder + lidar CNN are active.")
        # parameter counts per sub-encoder
        z_params = sum(p.numel() for p in mlp.z_enc.parameters())
        lidar_params = sum(p.numel() for p in mlp.lidar_enc.parameters())
        main_params = sum(p.numel() for p in mlp.main.parameters())
        print(f"  z_enc params:       {z_params}")
        print(f"  lidar_enc params:   {lidar_params}")
        print(f"  main MLP params:    {main_params}")
    return {"is_branched": is_branched, "is_lidar_cnn": is_lidar_cnn,
            "is_z_enc": is_z_enc}


def collect_obs_batch(env_wrapped, runner, n_steps, device) -> torch.Tensor:
    """Roll the teacher for N steps; return concatenated obs tensor."""
    env_wrapped.unwrapped.reset()
    obs = env_wrapped.get_observations()
    policy = runner.get_inference_policy(device=device)
    obs_list = []
    for _ in range(n_steps):
        action = policy(obs)
        obs, _, _, _ = env_wrapped.step(action)
        # obs is a TensorDict; extract policy group
        pol = obs["policy"] if hasattr(obs, "keys") else obs
        obs_list.append(pol.detach().clone())
    return torch.cat(obs_list, dim=0)        # (n_steps * num_envs, 198)


def check_activation_health(actor, mlp, all_obs) -> dict:
    """Section 2: encoder activations on real obs -- per-dim std, dead
    neurons, value ranges.
    """
    section("2. ACTIVATION HEALTH  (per-dim std, dead-neuron count)")
    # forward through obs_normalizer first (the encoder sees normalized obs)
    with torch.no_grad():
        normalized = actor.obs_normalizer(all_obs)
        priv = normalized[..., PRIV_SLICE]
        lidar_prev = normalized[..., LIDAR_PREV_SLICE]
        lidar = normalized[..., LIDAR_SLICE]
        z = mlp.z_enc(priv)                                                  # (B, z_dim)
        lidar_feat = mlp.lidar_enc(torch.cat([lidar_prev, lidar], dim=-1))   # (B, lidar_feat_dim)

    DEAD_STD_THR = 1e-3

    def report(name, out):
        std_per_dim = out.std(dim=0)
        n_dead = int((std_per_dim < DEAD_STD_THR).sum().item())
        print(f"  {name}:  shape {tuple(out.shape)}")
        print(f"    mean:       {out.mean().item():+.4f}")
        print(f"    std (overall): {out.std().item():.4f}")
        print(f"    min / max:  {out.min().item():+.3f} / {out.max().item():+.3f}")
        print(f"    std per dim:  min={std_per_dim.min().item():.4f}  "
              f"max={std_per_dim.max().item():.4f}  "
              f"mean={std_per_dim.mean().item():.4f}")
        print(f"    dead dims (std<{DEAD_STD_THR}):  {n_dead}/{out.shape[-1]}")
        return {"shape": list(out.shape), "mean": float(out.mean()),
                "std_overall": float(out.std()),
                "min": float(out.min()), "max": float(out.max()),
                "std_per_dim_min": float(std_per_dim.min()),
                "std_per_dim_max": float(std_per_dim.max()),
                "dead_dims": n_dead,
                "total_dims": int(out.shape[-1])}

    z_health = report("z_enc(priv) -> z", z)
    lidar_health = report("lidar_enc(lidar+delta) -> lidar_feat", lidar_feat)
    return {"z": z_health, "lidar_feat": lidar_health}


def check_input_sensitivity(actor, mlp, canonical_obs, cbf) -> dict:
    """Section 3: hold canonical obs, vary one encoder's input slice,
    measure how the output moves. Flat = encoder ignoring that input.
    """
    section("3. INPUT SENSITIVITY  (vary input slice, watch encoder output)")

    # --- z_enc: vary disturbance (priv[0]) from low to high ---
    print("  z_enc sensitivity to disturbance (priv[0]):")
    obs_baseline = canonical_obs.clone()
    z_outputs = []
    d_values = torch.linspace(0.0, 45.0, 6)
    for d in d_values:
        obs = obs_baseline.clone()
        obs[..., PRIV_SLICE.start] = d
        with torch.no_grad():
            norm = actor.obs_normalizer(obs)
            z = mlp.z_enc(norm[..., PRIV_SLICE])
        z_outputs.append(z[0].cpu().clone())
        print(f"    disturbance={float(d):>5.1f}  z[:4]={[f'{v:+.3f}' for v in z[0, :4].tolist()]}")
    z_stack = torch.stack(z_outputs, dim=0)
    z_span = (z_stack.max(dim=0).values - z_stack.min(dim=0).values).mean().item()
    print(f"    avg z dim span across disturbance: {z_span:.3f}  "
          f"({'OK -- encoder responding' if z_span > 0.05 else 'WEAK -- encoder flat'})")

    # --- lidar_enc: vary the forward-ray distance ---
    print("\n  lidar_enc sensitivity to forward-ray distance:")
    n_rays = LIDAR_DIM
    fwd_idx = n_rays // 2
    MAX_RANGE = 20.0
    lf_outputs = []
    dists = [0.3, 1.0, 3.0, 10.0, 20.0]
    for d in dists:
        obs = canonical_obs.clone()
        # set both lidar frames to the same configuration (all rays max,
        # forward ray at d). Zero temporal delta -- isolates the spatial
        # response of the CNN.
        obs[..., LIDAR_PREV_SLICE] = MAX_RANGE
        obs[..., LIDAR_PREV_SLICE.start + fwd_idx] = d
        obs[..., LIDAR_SLICE] = MAX_RANGE
        obs[..., LIDAR_SLICE.start + fwd_idx] = d
        with torch.no_grad():
            norm = actor.obs_normalizer(obs)
            lp = norm[..., LIDAR_PREV_SLICE]
            lr = norm[..., LIDAR_SLICE]
            lf = mlp.lidar_enc(torch.cat([lp, lr], dim=-1))
        lf_outputs.append(lf[0].cpu().clone())
        print(f"    fwd_dist={d:>5.1f}m  lidar_feat[:4]={[f'{v:+.3f}' for v in lf[0, :4].tolist()]}")
    lf_stack = torch.stack(lf_outputs, dim=0)
    lf_span = (lf_stack.max(dim=0).values - lf_stack.min(dim=0).values).mean().item()
    print(f"    avg lidar_feat dim span across distance: {lf_span:.3f}  "
          f"({'OK -- CNN responding' if lf_span > 0.05 else 'WEAK -- CNN flat'})")

    return {"z_span_across_disturbance": float(z_span),
            "lidar_feat_span_across_distance": float(lf_span)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args_cli.out_dir, exist_ok=True)

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
    actor = runner.alg.actor
    cbf = env_wrapped.unwrapped.action_manager._terms["cbf_param"]

    # =========================================================
    # 1. architecture check
    # =========================================================
    arch = check_architecture(actor)
    results = {"arch": arch}

    # Only run the rest if branched encoder is present
    if not arch["is_branched"]:
        print()
        print("Skipping activation/sensitivity checks -- vanilla MLP, no separate")
        print("encoder to probe. Fix the class_name resolution and retrain first.")
        env.close()
        simulation_app.close()
        sys.exit(1)

    mlp = actor.mlp

    # =========================================================
    # 2. activation health on real obs
    # =========================================================
    with torch.inference_mode():
        all_obs = collect_obs_batch(env_wrapped, runner,
                                     args_cli.n_rollout_steps, device)
    results["activation_health"] = check_activation_health(actor, mlp, all_obs)

    # =========================================================
    # 3. input-sensitivity probes
    # =========================================================
    # canonical obs = one real sample mid-rollout
    canonical_obs = all_obs[all_obs.shape[0] // 2:all_obs.shape[0] // 2 + 1]
    results["input_sensitivity"] = check_input_sensitivity(
        actor, mlp, canonical_obs, cbf,
    )

    # final verdict
    section("ENCODER HEALTH SUMMARY")
    z_dead = results["activation_health"]["z"]["dead_dims"]
    z_total = results["activation_health"]["z"]["total_dims"]
    lf_dead = results["activation_health"]["lidar_feat"]["dead_dims"]
    lf_total = results["activation_health"]["lidar_feat"]["total_dims"]
    z_span = results["input_sensitivity"]["z_span_across_disturbance"]
    lf_span = results["input_sensitivity"]["lidar_feat_span_across_distance"]
    print(f"  z_enc:      {z_total - z_dead}/{z_total} alive, "
          f"sensitivity span {z_span:.3f}")
    print(f"  lidar_enc:  {lf_total - lf_dead}/{lf_total} alive, "
          f"sensitivity span {lf_span:.3f}")
    all_ok = (z_dead == 0 and lf_dead == 0 and z_span > 0.05 and lf_span > 0.05)
    verdict = ("PASS -- encoders look healthy"
               if all_ok
               else "REVIEW -- one or more encoders are weak / collapsed")
    print(f"  verdict: {verdict}")
    results["verdict"] = verdict

    with open(os.path.join(args_cli.out_dir, "phase6_encoder_health.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved -> {args_cli.out_dir}/")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
