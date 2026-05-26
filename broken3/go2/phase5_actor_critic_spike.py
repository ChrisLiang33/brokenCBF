"""Step 4 of the RMA build: verify the branched-encoder model builds and
forward-passes against the RMA env's 43-dim obs.

What we check:
1. Env builds with `policy` obs shape (N, 43).
2. `RMAMLPModel` instantiates (as actor and as critic).
3. Forward passes return the right shapes:
   - actor.forward(obs)         -> (N, 2)  deterministic mean
   - actor.forward(obs, stochastic_output=True) -> (N, 2) sample
   - critic.forward(obs)        -> (N, 1)
4. Encoder shapes inside the branched MLP are right (z=8, lidar_feat=16).
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True if not hasattr(args_cli, "headless") else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import gymnasium as gym
import torch
from tensordict import TensorDict

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cbf_task  # noqa: F401
from cbf_task.locomotion_loader import load_locomotion_actor
from cbf_task.agents.rma_actor_critic import (
    RMAMLPModel,
    EXPECTED_OBS_DIM,
    PRIV_SLICE,
    STATE_SLICE,
    PREV_ACT_SLICE,
    LIDAR_SLICE,
)


TASK = "Isaac-CBF-Adaptive-Go2-RMA-v0"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loco = load_locomotion_actor(retrieve_file_path(args_cli.checkpoint), device)
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    env_cfg.actions.cbf_param.locomotion_policy_obj = loco

    print(f"[ac_spike] building {TASK} ({args_cli.num_envs} envs) ...")
    env = gym.make(TASK, cfg=env_cfg)
    obs_dict, _ = env.reset()

    print(f"[ac_spike] obs keys: {list(obs_dict.keys())}")
    pol = obs_dict["policy"]
    print(f"[ac_spike] policy obs shape: {tuple(pol.shape)}  "
          f"(expect (N, {EXPECTED_OBS_DIM}))")
    assert pol.shape == (args_cli.num_envs, EXPECTED_OBS_DIM)

    # show a sample row to confirm the slice contents make sense
    print(f"[ac_spike] env 0 slices:")
    print(f"    priv     : {pol[0, PRIV_SLICE].tolist()}")
    print(f"    state    : {[f'{v:+.2f}' for v in pol[0, STATE_SLICE].tolist()]}")
    print(f"    prev_act : {[f'{v:+.2f}' for v in pol[0, PREV_ACT_SLICE].tolist()]}")
    rays = pol[0, LIDAR_SLICE].tolist()
    print(f"    lidar    (min/max/median): {min(rays):.2f} / {max(rays):.2f} / "
          f"{sorted(rays)[len(rays)//2]:.2f}")

    # rsl_rl 5.0.1's MLPModel expects a TensorDict + obs_groups dict.
    # We mirror the runner's setup: actor and critic both read group "policy".
    td = TensorDict({"policy": pol}, batch_size=[args_cli.num_envs])
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    print("\n[ac_spike] instantiating actor + critic ...")
    actor = RMAMLPModel(
        obs=td,
        obs_groups=obs_groups,
        obs_set="actor",
        output_dim=2,
        hidden_dims=(128, 64),
        activation="elu",
        obs_normalization=True,
        # deterministic for the spike -- no distribution wrapping
        distribution_cfg=None,
    ).to(device)
    critic = RMAMLPModel(
        obs=td,
        obs_groups=obs_groups,
        obs_set="critic",
        output_dim=1,
        hidden_dims=(128, 64),
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
    ).to(device)
    print(f"[ac_spike] actor params:  {sum(p.numel() for p in actor.parameters())}")
    print(f"[ac_spike] critic params: {sum(p.numel() for p in critic.parameters())}")

    # forward passes
    with torch.no_grad():
        a_out = actor.forward(td)
        v_out = critic.forward(td)
    print(f"[ac_spike] actor.forward(td)  shape: {tuple(a_out.shape)}  "
          f"sample env 0: {[f'{v:+.3f}' for v in a_out[0].tolist()]}")
    print(f"[ac_spike] critic.forward(td) shape: {tuple(v_out.shape)}  "
          f"value  env 0: {v_out[0].item():+.3f}")

    # inspect the branched internals: hand-run encoders on a slice
    branched = actor.mlp                       # _BranchedMLP
    norm_obs = actor.obs_normalizer(pol)       # (N, 43) normalized
    z = branched.z_enc(norm_obs[:, PRIV_SLICE])
    lf = branched.lidar_enc(norm_obs[:, LIDAR_SLICE])
    print(f"[ac_spike] z_enc(priv)     -> shape {tuple(z.shape)}   (expect (N, 8))")
    print(f"[ac_spike] lidar_enc(lidar)-> shape {tuple(lf.shape)}  (expect (N, 16))")

    ok = (
        a_out.shape == (args_cli.num_envs, 2)
        and v_out.shape == (args_cli.num_envs, 1)
        and z.shape == (args_cli.num_envs, 8)
        and lf.shape == (args_cli.num_envs, 16)
    )
    print(f"\n[ac_spike] verdict: "
          f"{'PASS -- branched model plumbed correctly' if ok else 'REVIEW'}")

    env.close()
    simulation_app.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
