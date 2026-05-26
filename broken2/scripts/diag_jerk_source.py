"""DIAG-1: per-step jerk-source logger.

Records u_des (planner output), u_safe (CBF QP output), robot_vel
(actual achieved), and h(x) at every step of one episode in a single
env. Run at B0 with fixed α (default 0.5, the smoothest config —
gives the cleanest signal of intrinsic jerkiness vs CBF-induced).

Output:
    <output_dir>/diag_jerk.csv  - per-step trace
    <output_dir>/diag_jerk.png  - 4-panel plot

Reading the plot:
    Panel 1: u_des and u_safe overlaid. If they track tightly,
             CBF is barely deflecting (expected at B0 α=0.5).
    Panel 2: robot_vel + tracking error ||u_safe - robot_vel||.
             Spikes here = locomotion failed to track.
    Panel 3: per-step delta magnitudes. Identifies who's jerky:
             - u_des delta high → planner is jerky
             - u_safe delta high but u_des delta low → CBF amplified
             - robot_vel delta high → locomotion's reaction is jerky
    Panel 4: h(x) over time. Spikes in u_safe delta correlated with
             h getting small = CBF deflecting near obstacles (expected).
             Spikes when h is large = unexplained jerkiness.

Usage (from the IsaacLab dir):

    ./isaaclab.sh -p ../scripts/diag_jerk_source.py \\
        --steps 1000 \\
        --alpha 0.5 \\
        --headless
"""

import argparse

from isaaclab.app import AppLauncher

# --- CLI parse before sim app launches -----------------------------------
parser = argparse.ArgumentParser(description="Per-step jerk-source diagnostic.")
parser.add_argument("--task", type=str, default="Isaac-CBF-Go2-v0")
parser.add_argument("--steps", type=int, default=1000,
                    help="Max steps. Loop exits on episode end.")
parser.add_argument("--alpha", type=float, default=0.5,
                    help="Fixed CBF α to use (B0 mode). 0.5 = smoothest.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="logs/diag_jerk")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- imports after sim app starts ----------------------------------------
import csv
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  -- registers tasks
from isaaclab_tasks.utils import parse_env_cfg


# Planner-id → name lookup (matches cbf_go2_commands.py constants).
PLANNER_NAMES = {
    0: "uniform",
    1: "goal",
    2: "walk",
    3: "adversarial",
    4: "smooth_goal",
    5: "waypoint_path",
    6: "mpc",
}


# Action ranges MUST match cbf_go2_env._cbf_filter tanh-scale mapping.
PARAM_RANGES = {
    "alpha": (0.1, 5.0),
    "phi":   (0.0, 5.0),
    "a":     (0.0, 1.0),
    "c":     (0.0, 0.5),
}
# Near-zero φ for B0 (range lo is 0; atanh(-1) = -∞, so use 0.005).
PHI_FLOOR_B0 = 0.005


def _encode_dim(target: float, lo: float, hi: float, device, N: int) -> torch.Tensor:
    """Inverse of cbf_go2_env._cbf_filter's tanh+linear-scale.

    Returns a raw RL action component such that the env's tanh+scale
    produces the requested physical value.
    """
    x = torch.full((N,), float(target), device=device).clamp(lo, hi)
    sq = (2.0 * (x - lo) / (hi - lo)) - 1.0
    return torch.atanh(sq.clamp(-0.9999, 0.9999))


def encode_b0_action(alpha: float, device, N: int) -> torch.Tensor:
    """Build a (N, 5) raw action for B0: fixed α, near-zero φ, a=c=0."""
    return torch.stack([
        _encode_dim(alpha,         *PARAM_RANGES["alpha"], device, N),
        _encode_dim(PHI_FLOOR_B0,  *PARAM_RANGES["phi"],   device, N),
        _encode_dim(0.0,           *PARAM_RANGES["a"],     device, N),
        torch.zeros(N, device=device),                                  # b unused
        _encode_dim(0.0,           *PARAM_RANGES["c"],     device, N),
    ], dim=-1)


def main():
    torch.manual_seed(args_cli.seed)
    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single env for clean per-step trace.
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    device = inner.device
    N = inner.num_envs
    print(f"[env] {args_cli.task}, num_envs={N}, alpha={args_cli.alpha}")

    # B0 action — held constant for whole episode.
    action = encode_b0_action(args_cli.alpha, device, N)

    obs, _ = env.reset()

    # Read which planner this episode rolled. MultiPlannerCommand stores
    # planner_id per env on the command term; sample for env 0.
    try:
        cmd_term = inner.command_manager.get_term("base_velocity")
        planner_id = int(cmd_term.planner_id[0].item())
        planner_name = PLANNER_NAMES.get(planner_id, "?")
    except (AttributeError, KeyError):
        planner_id, planner_name = -1, "unknown"
    print(f"[planner] env 0 rolled planner_id={planner_id} ({planner_name})")

    rows = []
    end_reason = "step_limit"

    for t in range(args_cli.steps):
        obs, _, terminated, truncated, _ = env.step(action)

        u_des = inner.last_u_des[0].detach().cpu().numpy()           # (3,)
        u_safe = inner.last_u_safe[0].detach().cpu().numpy()         # (3,)
        robot_vel = inner.scene["robot"].data.root_lin_vel_b[0].detach().cpu().numpy()
        with torch.no_grad():
            h_vals, _ = inner._compute_h()
        h = float(h_vals[0].item())

        rows.append({
            "step":       t,
            "u_des_x":    float(u_des[0]),
            "u_des_y":    float(u_des[1]),
            "u_des_yaw":  float(u_des[2]),
            "u_safe_x":   float(u_safe[0]),
            "u_safe_y":   float(u_safe[1]),
            "u_safe_yaw": float(u_safe[2]),
            "robot_vx":   float(robot_vel[0]),
            "robot_vy":   float(robot_vel[1]),
            "robot_vz":   float(robot_vel[2]),
            "h":          h,
        })

        if terminated[0].item():
            end_reason = "fall_or_collision"
            break
        if truncated[0].item():
            end_reason = "timeout"
            break

    print(f"[diag] {len(rows)} steps logged, ended via {end_reason}")

    # Tag output files with seed + planner so multi-seed runs don't overwrite.
    suffix = f"seed{args_cli.seed}_{planner_name}"

    # --- CSV ------------------------------------------------------------
    csv_path = output_dir / f"diag_jerk_{suffix}.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[csv] {csv_path}")

    # --- Plot -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        steps      = np.array([r["step"] for r in rows])
        u_des_x    = np.array([r["u_des_x"] for r in rows])
        u_des_y    = np.array([r["u_des_y"] for r in rows])
        u_safe_x   = np.array([r["u_safe_x"] for r in rows])
        u_safe_y   = np.array([r["u_safe_y"] for r in rows])
        robot_vx   = np.array([r["robot_vx"] for r in rows])
        robot_vy   = np.array([r["robot_vy"] for r in rows])
        h_vals     = np.array([r["h"] for r in rows])

        # Per-step delta magnitudes (xy plane only — what locomotion sees).
        def _step_diff(arr_x, arr_y):
            dx = np.diff(arr_x, prepend=arr_x[0])
            dy = np.diff(arr_y, prepend=arr_y[0])
            return np.sqrt(dx ** 2 + dy ** 2)

        u_des_diff  = _step_diff(u_des_x, u_des_y)
        u_safe_diff = _step_diff(u_safe_x, u_safe_y)
        robot_diff  = _step_diff(robot_vx, robot_vy)

        # Tracking error per step.
        track_err = np.sqrt((u_safe_x - robot_vx) ** 2 + (u_safe_y - robot_vy) ** 2)

        fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

        # Panel 1: u_des vs u_safe overlay.
        ax = axes[0]
        ax.plot(steps, u_des_x,  label="u_des x",  linewidth=1.0, color="C0")
        ax.plot(steps, u_des_y,  label="u_des y",  linewidth=1.0, color="C1")
        ax.plot(steps, u_safe_x, label="u_safe x", linewidth=1.0, color="C0", linestyle="--")
        ax.plot(steps, u_safe_y, label="u_safe y", linewidth=1.0, color="C1", linestyle="--")
        ax.set_ylabel("velocity (m/s)")
        ax.set_title(
            f"u_des (solid) vs u_safe (dashed) — "
            f"B0 α={args_cli.alpha}, seed={args_cli.seed}, planner={planner_name}"
        )
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

        # Panel 2: robot velocity vs u_safe + tracking error.
        ax = axes[1]
        ax.plot(steps, u_safe_x,  label="u_safe x", linewidth=0.8, color="C0", alpha=0.5)
        ax.plot(steps, u_safe_y,  label="u_safe y", linewidth=0.8, color="C1", alpha=0.5)
        ax.plot(steps, robot_vx,  label="robot vx", linewidth=1.0, color="C2")
        ax.plot(steps, robot_vy,  label="robot vy", linewidth=1.0, color="C3")
        ax.plot(steps, track_err, label="track err ||u_safe - robot_vel||",
                linewidth=1.5, color="red")
        ax.set_ylabel("velocity / err (m/s)")
        ax.set_title("Locomotion tracking — does robot follow u_safe?")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

        # Panel 3: per-step delta magnitudes — who's jerky?
        ax = axes[2]
        ax.plot(steps, u_des_diff,  label="||Δ u_des||",   color="C0")
        ax.plot(steps, u_safe_diff, label="||Δ u_safe||",  color="C1")
        ax.plot(steps, robot_diff,  label="||Δ robot_vel||", color="C2")
        ax.set_ylabel("step delta magnitude")
        ax.set_title("Per-step change magnitudes — sources of jerkiness")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

        # Panel 4: h(x) over time.
        ax = axes[3]
        ax.plot(steps, h_vals, color="purple", linewidth=1.2)
        ax.axhline(0.0, color="red", linestyle=":", alpha=0.7, label="collision boundary")
        ax.set_xlabel("step (50 Hz; 1000 steps = 20s)")
        ax.set_ylabel("h(x) (margin from obstacles)")
        ax.set_title("Distance from obstacles — context for u_safe deflections")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / f"diag_jerk_{suffix}.png"
        plt.savefig(plot_path, dpi=150)
        print(f"[plot] {plot_path}")
    except ImportError:
        print("[plot] matplotlib unavailable, skipping PNG.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
