"""Plot a single hardware trial rosbag — trajectory + α/φ/h traces.

Reads a ros2 bag recorded by deploy/record_trial.sh and produces:
  - Trajectory plot (top-down) with start/end markers + obstacle position
    estimated from the closest LiDAR cluster
  - α(t), φ(t), h(t), deflection(t) time series
  - Summary stats: closest distance, completion-ish, mean v, deflection

Usage (laptop, after pulling the bag from Go2):
  python3 scripts/plot_trial_bag.py \\
    data_from_lab/trials/ours_v13_1/trial_01 \\
    --output docs/viz/trial_ours_v13_1_01.png \\
    --title "V13.1 student — trial 1"

Dependencies (laptop): rosbags (pip install rosbags), numpy, matplotlib.
The `rosbags` Python lib reads ros2 bags WITHOUT needing a full ROS install.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ros msg types we care about
TOPIC_TO_TYPE = {
    "/odom": "nav_msgs/msg/Odometry",
    "/cbf/params": "std_msgs/msg/Float32MultiArray",
    "/cbf/filter_h": "std_msgs/msg/Float32MultiArray",
    "/u_teleop": "geometry_msgs/msg/Twist",
    "/u_des": "geometry_msgs/msg/Twist",
}


def load_bag(bag_path: Path) -> dict:
    """Return dict {topic: list of (timestamp_s, msg_dict)}."""
    from rosbags.rosbag2 import Reader
    from rosbags.serde import deserialize_cdr

    out: dict[str, list] = {t: [] for t in TOPIC_TO_TYPE}

    with Reader(str(bag_path)) as reader:
        connections = [c for c in reader.connections if c.topic in TOPIC_TO_TYPE]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            t = timestamp * 1e-9  # ns → s
            msg = deserialize_cdr(rawdata, connection.msgtype)
            out[connection.topic].append((t, msg))
    return out


def extract_traj(odom_msgs: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, xy, v_xy) arrays from /odom."""
    if not odom_msgs:
        return np.empty((0,)), np.empty((0, 2)), np.empty((0, 2))
    t0 = odom_msgs[0][0]
    ts = np.array([m[0] - t0 for m in odom_msgs])
    pos = np.array([
        [m[1].pose.pose.position.x, m[1].pose.pose.position.y]
        for m in odom_msgs
    ])
    vel = np.array([
        [m[1].twist.twist.linear.x, m[1].twist.twist.linear.y]
        for m in odom_msgs
    ])
    return ts, pos, vel


def extract_params(param_msgs: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, [α, φ, a, b, c]) arrays from /cbf/params."""
    if not param_msgs:
        return np.empty((0,)), np.empty((0, 5))
    t0 = param_msgs[0][0]
    ts = np.array([m[0] - t0 for m in param_msgs])
    vals = np.array([list(m[1].data) for m in param_msgs])
    return ts, vals


def extract_filter_h(filter_msgs: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, [h, Lgh_x, Lgh_y, slack]) arrays from /cbf/filter_h."""
    if not filter_msgs:
        return np.empty((0,)), np.empty((0, 4))
    t0 = filter_msgs[0][0]
    ts = np.array([m[0] - t0 for m in filter_msgs])
    vals = np.array([list(m[1].data) for m in filter_msgs])
    return ts, vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bag_path", type=str, help="Path to ros2 bag directory")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--max_seconds", type=float, default=20.0)
    args = p.parse_args()

    bag = Path(args.bag_path)
    if not bag.exists():
        raise SystemExit(f"Bag not found: {bag}")

    print(f"[plot] loading {bag}...")
    data = load_bag(bag)
    for topic, msgs in data.items():
        print(f"  {topic:30s}  {len(msgs)} msgs")

    t_traj, pos, vel = extract_traj(data["/odom"])
    t_par, params = extract_params(data["/cbf/params"])
    t_h, filterh = extract_filter_h(data["/cbf/filter_h"])

    # Clip to max_seconds
    if t_traj.size > 0:
        mask = t_traj <= args.max_seconds
        t_traj, pos, vel = t_traj[mask], pos[mask], vel[mask]
    if t_par.size > 0:
        t_par = t_par[t_par <= args.max_seconds]
        params = params[: t_par.size]
    if t_h.size > 0:
        t_h = t_h[t_h <= args.max_seconds]
        filterh = filterh[: t_h.size]

    # ── Layout ──
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1.7])
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_params = fig.add_subplot(gs[0, 1])
    ax_h = fig.add_subplot(gs[1, 1], sharex=ax_params)
    ax_defl = fig.add_subplot(gs[2, 1], sharex=ax_params)

    # Trajectory
    if pos.size > 0:
        ax_traj.plot(pos[:, 0], pos[:, 1], lw=2, color="#1f77b4")
        ax_traj.scatter([pos[0, 0]], [pos[0, 1]], color="green", s=80,
                        zorder=5, label="start")
        ax_traj.scatter([pos[-1, 0]], [pos[-1, 1]], color="red", s=80,
                        zorder=5, label="end")
        ax_traj.set_aspect("equal")
        ax_traj.grid(True, alpha=0.3)
        ax_traj.set_xlabel("x [m]"); ax_traj.set_ylabel("y [m]")
        ax_traj.set_title("Robot trajectory (odom frame)", fontsize=11)
        ax_traj.legend(loc="upper right", fontsize=9)

    # Parameter time series (α, φ)
    if t_par.size > 0:
        ax_params.plot(t_par, params[:, 0], color="#1f77b4", lw=1.5, label="α")
        ax_params.plot(t_par, params[:, 1], color="#d62728", lw=1.5, label="φ")
        ax_params.set_ylabel("CBF param value", fontsize=10)
        ax_params.set_title("Adaptive parameters (α, φ)", fontsize=11)
        ax_params.legend(loc="upper right", fontsize=9)
        ax_params.grid(True, alpha=0.3)

    # h(x)
    if t_h.size > 0:
        ax_h.plot(t_h, filterh[:, 0], color="#2ca02c", lw=1.5, label="h(x)")
        ax_h.axhline(0, color="black", lw=0.5, ls="--", alpha=0.5)
        ax_h.set_ylabel("h(x) [m]", fontsize=10)
        ax_h.set_title("Safety margin h(x)", fontsize=11)
        ax_h.grid(True, alpha=0.3)
    else:
        ax_h.text(0.5, 0.5, "No /cbf/filter_h messages\n(no finite-h moments — robot may have been too far)",
                  transform=ax_h.transAxes, ha="center", va="center", fontsize=10, alpha=0.5)

    # Deflection ||u_des - u_safe||
    teleop_xy = []; udes_xy = []; t_tele = []; t_udes = []
    if data["/u_teleop"]:
        t0 = data["/u_teleop"][0][0]
        for ts, m in data["/u_teleop"]:
            t_tele.append(ts - t0)
            teleop_xy.append([m.linear.x, m.linear.y])
    if data["/u_des"]:
        t0u = data["/u_des"][0][0]
        for ts, m in data["/u_des"]:
            t_udes.append(ts - t0u)
            udes_xy.append([m.linear.x, m.linear.y])
    if t_tele and t_udes:
        # Interp u_des onto teleop timeline for a clean diff
        teleop_xy = np.array(teleop_xy)
        udes_xy = np.array(udes_xy)
        t_tele = np.array(t_tele)
        t_udes = np.array(t_udes)
        # Resample
        ude_x = np.interp(t_tele, t_udes, udes_xy[:, 0])
        ude_y = np.interp(t_tele, t_udes, udes_xy[:, 1])
        defl = np.sqrt((teleop_xy[:, 0] - ude_x) ** 2 + (teleop_xy[:, 1] - ude_y) ** 2)
        ax_defl.plot(t_tele, defl, color="#9467bd", lw=1.5)
        ax_defl.set_ylabel("|u_teleop − u_des| [m/s]", fontsize=10)
        ax_defl.set_title("CBF deflection magnitude", fontsize=11)
        ax_defl.set_xlabel("time [s]")
        ax_defl.grid(True, alpha=0.3)
    else:
        ax_defl.text(0.5, 0.5, "No teleop or u_des messages",
                      transform=ax_defl.transAxes, ha="center", va="center",
                      fontsize=10, alpha=0.5)

    title = args.title or f"{bag.parent.name} / {bag.name}"
    fig.suptitle(title, fontsize=12)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[plot] saved {out}")

    # ── Summary stats ──
    summary = {
        "title": title,
        "n_odom_msgs": len(data["/odom"]),
        "n_param_msgs": len(data["/cbf/params"]),
        "n_filter_h_msgs": len(data["/cbf/filter_h"]),
        "duration_s": float(t_traj[-1] if t_traj.size > 0 else 0.0),
    }
    if t_par.size > 0:
        summary["alpha"] = {"mean": float(params[:, 0].mean()),
                             "std": float(params[:, 0].std()),
                             "min": float(params[:, 0].min()),
                             "max": float(params[:, 0].max())}
        summary["phi"] = {"mean": float(params[:, 1].mean()),
                           "std": float(params[:, 1].std()),
                           "min": float(params[:, 1].min()),
                           "max": float(params[:, 1].max())}
    if t_h.size > 0:
        summary["h_min"] = float(filterh[:, 0].min())
        summary["frac_h_neg"] = float((filterh[:, 0] < 0).mean())
    if vel.size > 0:
        summary["mean_v_xy"] = float(np.linalg.norm(vel, axis=1).mean())
        summary["mean_v_x"] = float(vel[:, 0].mean())

    print("\n=== Trial summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    json_out = out.with_suffix(".json")
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[plot] summary → {json_out}")


if __name__ == "__main__":
    main()
