"""Render a matplotlib MP4 animation from dump_rollout.npz.

Three-panel layout:
  - Left:    top-down trajectory with obstacles, robot pose, CBF deflection markers
  - Top-right: time-series of α, φ, h
  - Bot-right: live LiDAR occupancy grid (2-channel: current + previous)

Usage:
  python3 scripts/animate_rollout.py \\
    data_from_lab/dump_v8_trainmatch.npz \\
    --env_idx 0 --fps 30 --output v8_trainmatch.mp4
"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz_path", type=str)
    p.add_argument("--env_idx", type=int, default=0,
                   help="Which parallel env to visualize.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--max_frames", type=int, default=800)
    p.add_argument("--step_skip", type=int, default=2,
                   help="Render every Nth simulation step.")
    args = p.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)
    pos = data["pos_history"]        # (S, N, 3)
    yaw = data["yaw_history"]        # (S, N)
    grid = data["grid_history"]      # (S, N, 2, 64, 64)
    alpha = data["alpha_history"]    # (S, N)
    phi = data["phi_history"]        # (S, N)
    h = data["h_history"]            # (S, N)
    defl = data["deflection_history"]  # (S, N)
    qp_active = data["qp_active_history"]  # (S, N) bool
    cmd = data["cmd_vel_history"]    # (S, N, 3) — vx, vy, w_yaw cmd
    # Obstacle data: prefer per-step `obstacle_history` (S, K, N, 3) if
    # present (post-2026-05-22 dump). Fall back to static `obstacle_positions`.
    if "obstacle_history" in data.files:
        obstacles_per_step = data["obstacle_history"][:, :, args.env_idx, :]  # (S, K, 3)
        obstacles_static = obstacles_per_step[0]
        has_moving_obstacles = True
    else:
        obstacles_raw = data["obstacle_positions"]
        if obstacles_raw.ndim == 3:
            obstacles_static = obstacles_raw[:, args.env_idx, :]  # (K, 3)
        else:
            obstacles_static = obstacles_raw  # legacy (K, 3)
        obstacles_per_step = None
        has_moving_obstacles = False

    # Goal position per step (world frame, xy). NaN where not exposed.
    if "goal_history" in data.files:
        goal_per_step = data["goal_history"][:, args.env_idx, :]  # (S, 2)
        has_goal = bool(np.isfinite(goal_per_step).any())
    else:
        goal_per_step = None
        has_goal = False

    S = pos.shape[0]
    e = args.env_idx
    frame_idxs = list(range(0, S, args.step_skip))[: args.max_frames]
    print(f"[animate] {len(frame_idxs)} frames at fps={args.fps}, env={e}")

    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1.0])
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_ts = fig.add_subplot(gs[0, 1])
    ax_grid = fig.add_subplot(gs[1, 1])

    # ── Trajectory panel ──
    # Limits: bounding box of trajectory + (if present) goal positions, ± buffer.
    px, py = pos[:, e, 0], pos[:, e, 1]
    xs_for_bounds = [px.min(), px.max()]
    ys_for_bounds = [py.min(), py.max()]
    if has_goal:
        finite_g = np.isfinite(goal_per_step).all(axis=1)
        if finite_g.any():
            xs_for_bounds.append(float(goal_per_step[finite_g, 0].min()))
            xs_for_bounds.append(float(goal_per_step[finite_g, 0].max()))
            ys_for_bounds.append(float(goal_per_step[finite_g, 1].min()))
            ys_for_bounds.append(float(goal_per_step[finite_g, 1].max()))
    margin = 1.5
    ax_traj.set_xlim(min(xs_for_bounds) - margin, max(xs_for_bounds) + margin)
    ax_traj.set_ylim(min(ys_for_bounds) - margin, max(ys_for_bounds) + margin)
    ax_traj.set_aspect("equal")
    # Tag panel title with corridor vs open scene type, if available.
    if "is_corridor" in data.files:
        ic = bool(data["is_corridor"][args.env_idx])
        scene_tag = "CORRIDOR" if ic else "open"
    else:
        scene_tag = "scene_type=?"
    ax_traj.set_title(f"Trajectory (top-down)  env={args.env_idx}  [{scene_tag}]",
                       fontsize=11)
    ax_traj.set_xlabel("x [m]"); ax_traj.set_ylabel("y [m]")
    ax_traj.grid(True, alpha=0.3)

    # Obstacles (fixed radius 0.3 m matching SHIELD).
    # Static path: draw once. Moving path: patches updated each frame.
    obstacle_patches = []
    if has_moving_obstacles:
        K = obstacles_per_step.shape[1]
        for k in range(K):
            ob = obstacles_static[k]
            patch = patches.Circle(
                (ob[0], ob[1]), 0.30, fc="#b91c1c", ec="#7f1d1d",
                alpha=0.7, zorder=3)
            if np.linalg.norm(ob[:2]) > 50:
                patch.set_visible(False)  # off-stage; hide rather than skip
                                          # so index alignment is preserved
            ax_traj.add_patch(patch)
            obstacle_patches.append(patch)
    else:
        for ob in obstacles_static:
            if np.linalg.norm(ob[:2]) > 50:
                continue
            ax_traj.add_patch(patches.Circle(
                (ob[0], ob[1]), 0.30, fc="#b91c1c", ec="#7f1d1d",
                alpha=0.7, zorder=3))

    # Active goal marker (green star). Updated per frame if `goal_history`
    # is present; static at first finite position otherwise.
    goal_marker, = ax_traj.plot([], [], marker="*", color="#10b981",
                                  ms=18, mew=1.5, mec="#065f46",
                                  ls="", zorder=7)

    # Past trajectory (trail) — drawn cumulatively
    trail_line, = ax_traj.plot([], [], color="#1f77b4", lw=1.5, alpha=0.7,
                                zorder=4)
    # Robot dot
    robot_dot, = ax_traj.plot([], [], "o", color="#1f77b4", ms=8, zorder=5)
    # Robot heading arrow
    heading_arrow = ax_traj.annotate("", xy=(0, 0), xytext=(0, 0),
                                      arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.8),
                                      zorder=6)
    # CBF intervention markers
    intervene_pts, = ax_traj.plot([], [], "*", color="#ff7f0e", ms=10, zorder=7,
                                    label="CBF active")
    ax_traj.legend(loc="upper right", fontsize=9)

    # ── Time series panel ──
    ax_ts.set_title("CBF parameters + h(x) over time", fontsize=11)
    ax_ts.set_xlim(0, S)
    ax_ts.grid(True, alpha=0.3)
    line_alpha, = ax_ts.plot([], [], color="#1f77b4", lw=1.5, label="α")
    line_phi, = ax_ts.plot([], [], color="#d62728", lw=1.5, label="φ")
    line_h, = ax_ts.plot([], [], color="#2ca02c", lw=1.5, label="h(x)")
    ax_ts.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
    # Compute y limits over full episode for this env
    all_min = float(min(alpha[:, e].min(), phi[:, e].min(), h[:, e].min(), 0))
    all_max = float(max(alpha[:, e].max(), phi[:, e].max(), h[:, e].max(), 0))
    ax_ts.set_ylim(all_min - 0.5, all_max + 0.5)
    ax_ts.set_xlabel("step"); ax_ts.legend(loc="upper right", fontsize=9)
    cursor_ts = ax_ts.axvline(0, color="black", lw=0.8, alpha=0.7)

    # ── Grid panel ──
    ax_grid.set_title("LiDAR occupancy grid (current frame)", fontsize=11)
    grid_img = ax_grid.imshow(grid[0, e, 0], cmap="binary", origin="lower",
                               extent=(-3.2, 3.2, -3.2, 3.2), vmin=0, vmax=1)
    # Robot at grid center
    ax_grid.plot(0, 0, "o", color="#1f77b4", ms=8)
    ax_grid.set_xlabel("body-x [m]"); ax_grid.set_ylabel("body-y [m]")

    # Title with metadata
    fig.suptitle(
        f"{data['task']}  |  env {e}/{pos.shape[1]}",
        fontsize=10,
    )

    # Update function
    def update(frame_idx):
        s = frame_idxs[frame_idx]
        # Trail (cumulative up to s)
        trail_line.set_data(pos[:s + 1, e, 0], pos[:s + 1, e, 1])
        # Robot dot
        robot_dot.set_data([pos[s, e, 0]], [pos[s, e, 1]])
        # Heading arrow (length 0.5 m)
        rx, ry = pos[s, e, 0], pos[s, e, 1]
        yz = yaw[s, e]
        ex, ey = rx + 0.5 * np.cos(yz), ry + 0.5 * np.sin(yz)
        heading_arrow.set_position((rx, ry))
        heading_arrow.xy = (ex, ey)
        # CBF intervention markers (cumulative)
        active_mask = qp_active[:s + 1, e]
        if active_mask.any():
            intervene_pts.set_data(pos[:s + 1, e, 0][active_mask],
                                    pos[:s + 1, e, 1][active_mask])

        # Moving obstacles — reposition patches every frame.
        if has_moving_obstacles:
            for k, patch in enumerate(obstacle_patches):
                ob = obstacles_per_step[s, k]
                if np.linalg.norm(ob[:2]) > 50:
                    patch.set_visible(False)
                else:
                    patch.set_visible(True)
                    patch.center = (ob[0], ob[1])

        # Active goal — update star position. Hidden if NaN at this step.
        if has_goal:
            gx, gy = goal_per_step[s, 0], goal_per_step[s, 1]
            if np.isfinite(gx) and np.isfinite(gy):
                goal_marker.set_data([gx], [gy])
            else:
                goal_marker.set_data([], [])

        # Time series (cumulative up to s)
        line_alpha.set_data(np.arange(s + 1), alpha[:s + 1, e])
        line_phi.set_data(np.arange(s + 1), phi[:s + 1, e])
        line_h.set_data(np.arange(s + 1), h[:s + 1, e])
        cursor_ts.set_xdata([s, s])

        # Grid
        grid_img.set_data(grid[s, e, 0])
        ax_grid.set_title(
            f"LiDAR grid t={s}  (α={alpha[s,e]:.2f}, φ={phi[s,e]:.2f}, "
            f"h={h[s,e]:+.2f})",
            fontsize=10,
        )

        return (trail_line, robot_dot, heading_arrow, intervene_pts,
                line_alpha, line_phi, line_h, cursor_ts, grid_img)

    print(f"[animate] rendering {len(frame_idxs)} frames...")
    anim = animation.FuncAnimation(fig, update, frames=len(frame_idxs),
                                    interval=1000.0 / args.fps, blit=False)

    # Pick writer based on what's available + requested format
    out = args.output or str(Path(args.npz_path).with_suffix(".mp4").name)
    import shutil
    has_ffmpeg = shutil.which("ffmpeg") is not None
    if out.endswith(".mp4"):
        if has_ffmpeg:
            anim.save(out, writer="ffmpeg", fps=args.fps,
                      extra_args=["-pix_fmt", "yuv420p"])
        else:
            print("[animate] ffmpeg not found, falling back to .gif")
            out = out.replace(".mp4", ".gif")
            anim.save(out, writer="pillow", fps=args.fps)
    elif out.endswith(".gif"):
        anim.save(out, writer="pillow", fps=args.fps)
    else:
        # default MP4
        if has_ffmpeg:
            anim.save(out + ".mp4", writer="ffmpeg", fps=args.fps,
                      extra_args=["-pix_fmt", "yuv420p"])
        else:
            anim.save(out + ".gif", writer="pillow", fps=args.fps)
    print(f"[animate] saved {out}")


if __name__ == "__main__":
    main()
