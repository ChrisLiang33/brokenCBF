"""PAPER-1 scenario 1: tight-gap (set-shrink param c) sweep.

Compares the trained teacher against fixed-c baselines as the gap width
varies. Each config is evaluated at multiple gap widths; the win
condition is that the teacher's curve dominates every fixed-c curve
across the eval-width range — proving "no fixed c wins all gaps."

Train window: gap widths ∈ [0.6, 1.2] m (set in CbfGo2EnvCfg_TIGHT_GAP).
Eval window: extended to [0.5, 1.5] m to test extrapolation.

Usage (from IsaacLab dir):
    ./isaaclab.sh -p ../scripts/eval_tight_gap.py \\
        --checkpoint ../IsaacLab/logs/rsl_rl/cbf_go2_teacher/<run>/model_2999.pt \\
        --num_envs 64 --steps_per_width 600 --headless

Output:
  - <output_dir>/tight_gap.csv  — per-(config, width) metrics
  - <output_dir>/tight_gap.png  — collision-rate-vs-width curves per config
"""

import argparse

from isaaclab.app import AppLauncher

# --- CLI parse before sim app launches ---
parser = argparse.ArgumentParser(description="Tight-gap c-sweep eval.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="rsl_rl model_*.pt. If omitted, only fixed-c baselines run.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps_per_width", type=int, default=600,
                    help="Sim steps per (config, gap_width). 600 ≈ 30s × 64 envs.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="logs/tight_gap_eval")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- imports after sim app starts ---
import csv
import math
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn as nn

import isaaclab_tasks  # noqa: F401  -- registers tasks
from isaaclab_tasks.utils import parse_env_cfg


TASK = "Isaac-CBF-Go2-TightGap-v0"

# Param ranges MUST match cbf_go2_env._cbf_filter tanh-scale mapping.
PARAM_RANGES = {
    "alpha": (0.1, 5.0),
    "phi":   (0.0, 5.0),
    "a":     (0.0, 1.0),
    "c":     (0.0, 0.5),
}

# 5 fixed-c baselines spanning the action range. α/φ/a held at the
# teacher's typical "average" outputs (eyeballed from training logs).
C_VALUES = [0.05, 0.15, 0.25, 0.35, 0.45]
ALPHA_DEFAULT = 1.5
PHI_DEFAULT = 0.5
A_DEFAULT = 0.1

# Eval gap widths — wider than training [0.6, 1.2] to test extrapolation.
GAP_WIDTHS = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

# Success threshold: robot has crossed past the gap (gap_x = 3.0, plus
# margin so the robot's body is fully through).
SUCCESS_X = 4.0


def inv_tanh_scale(target: float, lo: float, hi: float) -> float:
    sq = 2.0 * (target - lo) / (hi - lo) - 1.0
    sq = max(min(sq, 0.9999), -0.9999)
    return float(math.atanh(sq))


def encode_config(alpha: float, phi: float, a_param: float, c_param: float) -> list[float]:
    return [
        inv_tanh_scale(alpha, *PARAM_RANGES["alpha"]),
        inv_tanh_scale(phi,   *PARAM_RANGES["phi"]),
        inv_tanh_scale(a_param, *PARAM_RANGES["a"]),
        0.0,  # b unused
        inv_tanh_scale(c_param, *PARAM_RANGES["c"]),
    ]


def _build_linear_chain(state: dict, prefix: str, with_last_activation: bool) -> nn.Sequential:
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
        if with_last_activation or i < len(keys) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def load_trained_teacher(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("actor_state_dict") or ckpt.get("model_state_dict") or ckpt
    env_encoder = _build_linear_chain(state, "mlp.0.", with_last_activation=True).to(device).eval()
    pi_teacher  = _build_linear_chain(state, "mlp.1.", with_last_activation=False).to(device).eval()
    return env_encoder, pi_teacher


def set_gap_width(inner, width: float):
    """Mutate the running event term's params so the next reset uses
    `width` as the deterministic gap width across all envs."""
    term_cfg = inner.event_manager.get_term_cfg("randomize_obstacles_position")
    term_cfg.params["gap_width_override"] = float(width)


def force_reset_all(env, inner):
    """Reset all envs so the placement event re-runs with the new override."""
    inner.episode_length_buf[:] = inner.max_episode_length
    obs, _ = env.reset()
    return obs


def eval_one(env, action_fn, num_steps: int, device, N: int):
    """Run a config for num_steps; aggregate collision / success / params."""
    obs, _ = env.reset()
    inner = env.unwrapped

    n_eps = 0
    n_coll = 0
    n_success = 0
    n_other = 0
    sum_c_output = 0.0
    sum_c_count = 0
    ep_max_x = torch.full((N,), -1e9, device=device)

    for _ in range(num_steps):
        action = action_fn(obs, inner)

        # Track teacher's c-output (if action is the teacher's tensor, the
        # 5th column is the raw c — gets tanh+scaled inside the env).
        sum_c_output += action[:, 4].mean().item()
        sum_c_count += 1

        robot_x = inner.scene["robot"].data.root_pos_w[:, 0] - inner.scene.env_origins[:, 0]
        ep_max_x = torch.maximum(ep_max_x, robot_x)

        obs, _, terminated, truncated, _ = env.step(action)

        done = terminated | truncated
        if done.any():
            done_idxs = done.nonzero(as_tuple=False).squeeze(-1).tolist()
            for e in done_idxs:
                n_eps += 1
                if bool(terminated[e].item()):
                    # base_contact OR obstacle_contact; tell them apart by max_x
                    if ep_max_x[e].item() < SUCCESS_X:
                        n_coll += 1
                    else:
                        n_other += 1
                elif bool(truncated[e].item()):
                    if ep_max_x[e].item() >= SUCCESS_X:
                        n_success += 1
                    else:
                        n_other += 1
                ep_max_x[e] = -1e9

    eps = max(n_eps, 1)
    return {
        "n_episodes": n_eps,
        "collision_rate": n_coll / eps,
        "success_rate": n_success / eps,
        "other_rate": n_other / eps,
        "avg_c_raw": sum_c_output / max(sum_c_count, 1),
    }


def main():
    torch.manual_seed(args_cli.seed)
    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args_cli.num_envs)
    env = gym.make(TASK, cfg=env_cfg)
    inner = env.unwrapped
    device = inner.device
    N = inner.num_envs
    print(f"\n[env] {TASK}, num_envs={N}, device={device}\n")

    # Build configs.
    configs = []
    for c in C_VALUES:
        raw = torch.tensor(
            encode_config(ALPHA_DEFAULT, PHI_DEFAULT, A_DEFAULT, c),
            device=device,
        ).unsqueeze(0).expand(N, -1).contiguous()
        configs.append({
            "name": f"fixed_c={c:.2f}",
            "kind": "fixed",
            "c_target": c,
            "action_fn": (lambda r: lambda _o, _i: r)(raw),
        })

    if args_cli.checkpoint:
        env_encoder, pi_teacher = load_trained_teacher(Path(args_cli.checkpoint), device)

        def teacher_action(obs, _inner):
            priv = obs["policy"] if isinstance(obs, dict) else obs
            with torch.no_grad():
                z = env_encoder(priv)
                return pi_teacher(z)

        configs.append({
            "name": "teacher",
            "kind": "trained",
            "c_target": None,
            "action_fn": teacher_action,
        })

    # Sweep over (gap_width, config).
    results = []
    print(f"[sweep] {len(GAP_WIDTHS)} widths × {len(configs)} configs × {args_cli.steps_per_width} steps\n")
    for w in GAP_WIDTHS:
        set_gap_width(inner, w)
        force_reset_all(env, inner)
        print(f"--- gap_width = {w:.2f} m ---")
        for cfg in configs:
            stats = eval_one(env, cfg["action_fn"], args_cli.steps_per_width, device, N)
            print(f"  {cfg['name']:<18s}  eps={stats['n_episodes']:3d}  "
                  f"coll={stats['collision_rate']:.3f}  "
                  f"succ={stats['success_rate']:.3f}  "
                  f"c_raw={stats['avg_c_raw']:+.2f}")
            results.append({
                "gap_width": w,
                "name": cfg["name"],
                "kind": cfg["kind"],
                "c_target": cfg["c_target"],
                **stats,
            })

    # CSV.
    csv_path = output_dir / "tight_gap.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["gap_width", "name", "kind", "c_target",
                      "n_episodes", "collision_rate", "success_rate",
                      "other_rate", "avg_c_raw"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\n[csv] wrote {csv_path}")

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        names = sorted({r["name"] for r in results})
        for name in names:
            rs = sorted([r for r in results if r["name"] == name], key=lambda r: r["gap_width"])
            xs = [r["gap_width"] for r in rs]
            ys_coll = [r["collision_rate"] for r in rs]
            ys_succ = [r["success_rate"] for r in rs]
            style = {"linewidth": 2.5, "color": "red", "marker": "*"} \
                    if name == "teacher" else {"linewidth": 1.0, "marker": "o"}
            ax1.plot(xs, ys_coll, label=name, **style)
            ax2.plot(xs, ys_succ, label=name, **style)
        for ax, ylab in [(ax1, "Collision rate"), (ax2, "Success rate")]:
            ax.set_xlabel("Gap width (m)")
            ax.set_ylabel(ylab)
            ax.axvspan(0.6, 1.2, alpha=0.1, color="gray", label="train window")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "tight_gap.png", dpi=150)
        print(f"[plot] wrote {output_dir / 'tight_gap.png'}")
    except ImportError:
        print("[plot] matplotlib unavailable, skipping PNG.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
