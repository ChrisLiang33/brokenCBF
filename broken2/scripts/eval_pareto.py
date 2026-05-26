"""Wk3.5c: hand-tuned vs trained teacher Pareto eval.

Runs multiple CBF configs against the same environment distribution and
produces a collision-rate vs time-out-rate Pareto plot — the paper's
central baseline comparison.

Two kinds of configs:
  1. Hand-tuned: fixed (α, φ, a, b, c) values, span conservative→aggressive.
  2. Trained teacher: actor loaded from an rsl_rl checkpoint.

Fair comparison: every config is evaluated on the same Isaac Lab env
instance with the same seed, so each config sees the same ensemble of
randomized obstacle positions / friction / masses.

Output:
  - <output_dir>/pareto.csv    — per-config metrics
  - <output_dir>/pareto.png    — Pareto plot

Usage (from IsaacLab dir):
    ./isaaclab.sh -p ../scripts/eval_pareto.py \\
        --checkpoint ../IsaacLab/logs/rsl_rl/cbf_go2_teacher/<run>/model_2999.pt \\
        --num_envs 64 --num_steps 2000 --headless

    # skip trained teacher (hand-tuned only):
    ./isaaclab.sh -p ../scripts/eval_pareto.py --num_envs 64 --headless
"""

import argparse

from isaaclab.app import AppLauncher

# --- CLI parse before sim app launches ---
parser = argparse.ArgumentParser(description="Hand-tuned vs trained teacher Pareto eval.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to rsl_rl model_*.pt. If omitted, only hand-tuned configs run.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_steps", type=int, default=2000,
                    help="Steps per config eval. 2000 × 64 envs ≈ 130+ episodes per config.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="logs/pareto_eval")
parser.add_argument("--condition", type=str, default="default",
                    choices=["default", "slip_calm", "slip_push", "grip_calm", "grip_push"],
                    help="Env condition preset. 'default' = full training DR. "
                         "'slip_*' pin friction to 0.25. 'grip_*' pin to 1.2. "
                         "'*_calm' zero disturbances. '*_push' apply full ±10N/±2Nm.")
parser.add_argument("--force_uniform_planner", action="store_true",
                    help="MODEL-3 Step 1 diagnostic: override planner weights to "
                         "uniform-only (1.0 / 0 / 0 / 0). Isolates dynamics "
                         "adaptation from any residual planner-coupling. By "
                         "default the OOD presets use uniform+goal 50/50; this "
                         "flag drops goal too.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- imports after sim app starts ---

import csv
import math
import os
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn as nn

import isaaclab_tasks  # noqa: F401  -- registers tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.manager_based.safety.cbf_go2.cbf_go2_env_cfg import (
    OBSTACLE_NAMES,
)


TASK = "Isaac-CBF-Go2-v0"   # training task: has DR, env-namespaced obstacles

# Param ranges MUST match cbf_go2_env._cbf_filter tanh-scale mapping.
# raw → tanh → scale: raw ∈ ℝ, squashed ∈ [-1, 1], then scaled to [lo, hi].
PARAM_RANGES = {
    "alpha": (0.1, 5.0),   # cbf_params[:, 0]
    "phi":   (0.0, 5.0),   # cbf_params[:, 1]
    "a":     (0.0, 1.0),   # cbf_params[:, 2]
    "c":     (0.0, 0.5),   # cbf_params[:, 4]
}
# cbf_params[:, 3] = b, ignored downstream


def inv_tanh_scale(target: float, lo: float, hi: float) -> float:
    """Inverse of the tanh+scale in _cbf_filter.

    _cbf_filter does:  out = lo + (tanh(raw) + 1) * 0.5 * (hi - lo)
    So:                raw = atanh( 2*(target - lo)/(hi - lo) - 1 )

    Clamp the argument away from ±1 to avoid atanh's singularity.
    """
    sq = 2.0 * (target - lo) / (hi - lo) - 1.0
    sq = max(min(sq, 0.9999), -0.9999)
    return float(math.atanh(sq))


def encode_config(alpha: float, phi: float, a_param: float, c_param: float, b_param: float = 0.0) -> list[float]:
    """Build the 5D raw action that, after _cbf_filter's tanh+scale,
    yields the requested physical (α, φ, a, b, c) values."""
    return [
        inv_tanh_scale(alpha, *PARAM_RANGES["alpha"]),
        inv_tanh_scale(phi,   *PARAM_RANGES["phi"]),
        inv_tanh_scale(a_param, *PARAM_RANGES["a"]),
        b_param,    # ignored by filter
        inv_tanh_scale(c_param, *PARAM_RANGES["c"]),
    ]


# --- Hand-tuned configs: sweep α (main knob) × a (margin). φ, c held ---
# Paper story needs a span from "paranoid" to "aggressive" so we can see
# the Pareto curve. α dominates CBF sensitivity; a is the robust margin.
HAND_TUNED = []
for alpha in [0.1, 0.5, 1.0, 2.0, 3.5, 5.0]:
    for a_param in [0.001, 0.1, 0.5]:
        HAND_TUNED.append({
            "name": f"ht_a={alpha}_m={a_param}",
            "kind": "hand-tuned",
            "physical": dict(alpha=alpha, phi=0.1, a=a_param, c=0.1),
            "raw": encode_config(alpha=alpha, phi=0.1, a_param=a_param, c_param=0.1),
        })


# --- Rebuild trained teacher from checkpoint state_dict (no rsl_rl needed) ---

def _build_linear_chain(state: dict, prefix: str, with_last_activation: bool) -> nn.Sequential:
    """Rebuild a Linear/ELU chain from state_dict keys under `prefix`."""
    keys = sorted(k for k in state if k.startswith(prefix) and k.endswith(".weight"))
    layers: list[nn.Module] = []
    for i, wk in enumerate(keys):
        bk = wk[: -len(".weight")] + ".bias"
        W = state[wk]
        out_dim, in_dim = W.shape
        lin = nn.Linear(in_dim, out_dim)
        lin.weight.data = W.clone()
        if bk in state:
            lin.bias.data = state[bk].clone()
        layers.append(lin)
        # env_encoder uses last_activation=elu → ELU after every Linear.
        # pi_teacher: no last_activation → ELU only between Linears.
        if with_last_activation or i < len(keys) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def load_trained_teacher(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("actor_state_dict") or ckpt.get("model_state_dict") or ckpt

    env_encoder = _build_linear_chain(state, "mlp.0.", with_last_activation=True).to(device).eval()
    pi_teacher  = _build_linear_chain(state, "mlp.1.", with_last_activation=False).to(device).eval()

    print(f"[teacher] rebuilt env_encoder ({sum(p.numel() for p in env_encoder.parameters())} params)")
    print(f"[teacher] rebuilt pi_teacher  ({sum(p.numel() for p in pi_teacher.parameters())} params)")
    return env_encoder, pi_teacher


# --- Per-config eval loop ---

def eval_config(env, get_action_fn, num_steps: int, verbose: bool = True):
    """Run one config for num_steps and aggregate per-episode stats.

    get_action_fn(obs_dict, inner_env) -> (N, 5) raw action tensor
    """
    inner = env.unwrapped
    N = inner.num_envs
    device = inner.device

    obs, _ = env.reset()

    # Per-env running state for the in-progress episode
    ep_min_dist = torch.full((N,), float("inf"), device=device)
    ep_u_safe_dev_sum = torch.zeros(N, device=device)
    ep_step_count = torch.zeros(N, dtype=torch.long, device=device)

    # Cross-episode aggregates
    n_eps = 0
    n_coll = 0
    n_timeout = 0
    n_other = 0
    sum_min_dist = 0.0
    sum_u_safe_dev_per_step = 0.0
    sum_ep_len = 0.0

    for step in range(num_steps):
        action = get_action_fn(obs, inner)

        # Pre-step robot/obstacle positions — env.step will auto-reset
        # done envs to new positions, so sample distance BEFORE stepping.
        # SCENE-1: K obstacles per env; take min distance over all of them.
        # Off-stage obstacles (~100m) trivially lose the min.
        robot_pos = inner.scene["robot"].data.root_pos_w[:, :2]
        obs_pos_stack = torch.stack(
            [inner.scene[n].data.root_pos_w[:, :2] for n in OBSTACLE_NAMES],
            dim=1,
        )                                                              # (N, K, 2)
        dists_all = torch.linalg.norm(
            robot_pos.unsqueeze(1) - obs_pos_stack, dim=-1
        )                                                              # (N, K)
        dist_pre = dists_all.min(dim=-1).values                        # (N,)

        obs, _, terminated, truncated, _ = env.step(action)

        # Update running state using pre-step dist and the just-applied action.
        # last_u_safe / last_u_des were cached by env.step during this frame.
        ep_min_dist = torch.minimum(ep_min_dist, dist_pre)
        u_safe_dev = ((inner.last_u_safe - inner.last_u_des) ** 2).sum(dim=-1)
        ep_u_safe_dev_sum += u_safe_dev
        ep_step_count += 1

        done = terminated | truncated
        if done.any():
            done_idxs = done.nonzero(as_tuple=False).squeeze(-1).tolist()
            for e in done_idxs:
                n_eps += 1
                sum_min_dist += ep_min_dist[e].item()
                sum_ep_len += float(ep_step_count[e].item())
                steps = max(int(ep_step_count[e].item()), 1)
                sum_u_safe_dev_per_step += (ep_u_safe_dev_sum[e] / steps).item()

                # Classify outcome. Order matters — collision takes precedence
                # (it terminates the episode, often also before time_out).
                # Threshold 0.65m covers the largest obstacle's effective
                # radius (max R_combined = 0.49 + 0.15 = 0.64) — conservative
                # since per-obstacle thresholds vary post SCENE-1.5.
                if bool(terminated[e].item()) and ep_min_dist[e].item() < 0.65:
                    n_coll += 1
                elif bool(truncated[e].item()) and not bool(terminated[e].item()):
                    n_timeout += 1
                else:
                    # base_contact (robot tipped) or other non-collision termination
                    n_other += 1

                # Reset per-env running state
                ep_min_dist[e] = float("inf")
                ep_u_safe_dev_sum[e] = 0.0
                ep_step_count[e] = 0

    eps = max(n_eps, 1)
    stats = {
        "n_episodes": n_eps,
        "collision_rate": n_coll / eps,
        "timeout_rate": n_timeout / eps,
        "other_term_rate": n_other / eps,
        "avg_min_dist": sum_min_dist / eps,
        "avg_u_safe_dev_per_step": sum_u_safe_dev_per_step / eps,
        "avg_ep_length": sum_ep_len / eps,
    }
    if verbose:
        print(
            f"  eps={n_eps:4d}  coll={stats['collision_rate']:.3f}  "
            f"to={stats['timeout_rate']:.3f}  oth={stats['other_term_rate']:.3f}  "
            f"min_d={stats['avg_min_dist']:.2f}  "
            f"|du|^2/step={stats['avg_u_safe_dev_per_step']:.4f}  "
            f"ep_len={stats['avg_ep_length']:.0f}"
        )
    return stats


# --- Plotting (matplotlib optional) ---

def plot_pareto(results: list[dict], output_png: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib unavailable, skipping PNG. CSV still written.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for r in results:
        marker = "*" if r["kind"] == "trained" else "o"
        color = "red" if r["kind"] == "trained" else "steelblue"
        size = 200 if r["kind"] == "trained" else 60
        ax.scatter(
            r["stats"]["collision_rate"],
            r["stats"]["timeout_rate"],
            c=color, marker=marker, s=size,
            edgecolors="black", linewidths=0.6,
        )
        # Annotate hand-tuned points with their α
        if r["kind"] == "hand-tuned":
            ax.annotate(
                f"α={r['physical']['alpha']}",
                (r["stats"]["collision_rate"], r["stats"]["timeout_rate"]),
                fontsize=7, alpha=0.7,
                xytext=(3, 3), textcoords="offset points",
            )

    ax.set_xlabel("Collision rate")
    ax.set_ylabel("Time-out rate")
    ax.set_title("CBF Pareto: hand-tuned (blue) vs trained teacher (red ★)")
    ax.grid(alpha=0.3)
    ax.set_xlim(left=-0.02)
    ax.set_ylim(bottom=-0.02)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"[plot] wrote {output_png}")


def write_csv(results: list[dict], path: Path):
    fieldnames = [
        "name", "kind", "alpha", "phi", "a", "c",
        "n_episodes", "collision_rate", "timeout_rate", "other_term_rate",
        "avg_min_dist", "avg_u_safe_dev_per_step", "avg_ep_length",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            phys = r.get("physical", {})
            s = r["stats"]
            w.writerow({
                "name": r["name"],
                "kind": r["kind"],
                "alpha": phys.get("alpha", ""),
                "phi":   phys.get("phi", ""),
                "a":     phys.get("a", ""),
                "c":     phys.get("c", ""),
                "n_episodes": s["n_episodes"],
                "collision_rate": s["collision_rate"],
                "timeout_rate": s["timeout_rate"],
                "other_term_rate": s["other_term_rate"],
                "avg_min_dist": s["avg_min_dist"],
                "avg_u_safe_dev_per_step": s["avg_u_safe_dev_per_step"],
                "avg_ep_length": s["avg_ep_length"],
            })
    print(f"[csv] wrote {path}")


# --- Main ---

def apply_condition(env_cfg, condition: str) -> None:
    """Pin env_cfg to one of the Wk3.5 OOD eval presets.

    All presets zero the COM randomization + force a uniform/goal
    planner mix to isolate the friction × disturbance axes. The
    'default' preset leaves env_cfg as-is (full training DR).

    | Preset     | Friction (static / dynamic) | Disturbance        |
    |------------|-----------------------------|--------------------|
    | default    | (0.3, 1.2) / (0.2, 1.0)     | ±10N, ±2 Nm        |
    | slip_calm  | 0.25 / 0.20                 | 0                  |
    | slip_push  | 0.25 / 0.20                 | ±10N, ±2 Nm        |
    | grip_calm  | 1.20 / 1.00                 | 0                  |
    | grip_push  | 1.20 / 1.00                 | ±10N, ±2 Nm        |
    """
    if condition == "default":
        return

    # Friction pin
    if condition.startswith("slip"):
        friction_s, friction_d = 0.25, 0.20
    elif condition.startswith("grip"):
        friction_s, friction_d = 1.20, 1.00
    else:
        raise ValueError(f"Unknown condition: {condition}")

    env_cfg.events.physics_material.params["static_friction_range"] = (friction_s, friction_s)
    env_cfg.events.physics_material.params["dynamic_friction_range"] = (friction_d, friction_d)

    # Disturbance pin
    if condition.endswith("calm"):
        force_range = (0.0, 0.0)
        torque_range = (0.0, 0.0)
    else:  # push
        force_range = (-10.0, 10.0)
        torque_range = (-2.0, 2.0)

    env_cfg.events.base_external_force_torque.params["force_range"] = force_range
    env_cfg.events.base_external_force_torque.params["torque_range"] = torque_range

    # Zero COM offset for controlled eval — isolate friction × disturbance
    env_cfg.events.randomize_com.params["com_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
    }

    # Lock planner to uniform + goal only (skip random-walk + adversarial)
    # so the eval focuses on physics effects, not planner noise.
    env_cfg.commands.base_velocity.uniform_weight = 0.5
    env_cfg.commands.base_velocity.goal_weight = 0.5
    env_cfg.commands.base_velocity.walk_weight = 0.0
    env_cfg.commands.base_velocity.adversarial_weight = 0.0


def main():
    torch.manual_seed(args_cli.seed)
    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # One env, reused across all configs — much faster than re-instantiating.
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args_cli.num_envs)
    apply_condition(env_cfg, args_cli.condition)

    # MODEL-3 Step 1 diagnostic: drop the residual goal-planner from the OOD
    # presets and force pure uniform planning. If teacher's per-condition
    # ranks change a lot vs the existing eval, planner-coupling is real.
    if args_cli.force_uniform_planner:
        env_cfg.commands.base_velocity.uniform_weight = 1.0
        env_cfg.commands.base_velocity.goal_weight = 0.0
        env_cfg.commands.base_velocity.walk_weight = 0.0
        env_cfg.commands.base_velocity.adversarial_weight = 0.0
        print("[planner] forced uniform-only (MODEL-3 diagnostic)")

    env = gym.make(TASK, cfg=env_cfg)
    inner = env.unwrapped
    device = inner.device
    N = inner.num_envs
    print(f"\n[env] {TASK}, num_envs={N}, device={device}\n")

    configs_to_eval = list(HAND_TUNED)
    if args_cli.checkpoint:
        env_encoder, pi_teacher = load_trained_teacher(Path(args_cli.checkpoint), device)

        def teacher_action(obs, _inner):
            priv = obs["policy"] if isinstance(obs, dict) else obs
            with torch.no_grad():
                z = env_encoder(priv)
                return pi_teacher(z)

        configs_to_eval.append({
            "name": "trained_teacher",
            "kind": "trained",
            "physical": {},
            "action_fn": teacher_action,
        })

    # Hand-tuned action fns: constant (N, 5) broadcast of the raw params.
    for cfg in configs_to_eval:
        if cfg["kind"] == "hand-tuned":
            raw = torch.tensor(cfg["raw"], device=device).unsqueeze(0).expand(N, -1).contiguous()
            def make_fn(r):
                return lambda _obs, _inner: r
            cfg["action_fn"] = make_fn(raw)

    results = []
    print(f"[eval] {len(configs_to_eval)} configs × {args_cli.num_steps} steps × {N} envs\n")
    for i, cfg in enumerate(configs_to_eval, 1):
        label = cfg["name"]
        if cfg["kind"] == "hand-tuned":
            label += f"  (physical {cfg['physical']})"
        print(f"[{i}/{len(configs_to_eval)}] {label}")
        stats = eval_config(env, cfg["action_fn"], args_cli.num_steps)
        results.append({**cfg, "stats": stats})

    suffix = f"_{args_cli.condition}" if args_cli.condition != "default" else ""
    write_csv(results, output_dir / f"pareto{suffix}.csv")
    plot_pareto(results, output_dir / f"pareto{suffix}.png")

    # Terminal summary
    print("\n" + "=" * 70)
    print("Pareto summary (sorted by collision rate):")
    print("=" * 70)
    print(f"{'name':<32}{'coll':>7}{'to':>7}{'min_d':>8}{'|du|²/st':>10}")
    for r in sorted(results, key=lambda x: x["stats"]["collision_rate"]):
        s = r["stats"]
        tag = "★" if r["kind"] == "trained" else " "
        print(f"{tag} {r['name']:<30}{s['collision_rate']:>7.3f}{s['timeout_rate']:>7.3f}"
              f"{s['avg_min_dist']:>8.2f}{s['avg_u_safe_dev_per_step']:>10.4f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
