#!/usr/bin/env python3
"""ROS 2 inference node for CBF params on the Go2.

Loads V13.1 teacher + student adapter, subscribes to the Go2's sensor
topics, builds the proprio vector, runs the deploy model, and publishes
(α, φ, a, b, c) on /cbf/params.

Fail-safes (all standard practice for safety-critical deploy):
  1. Input watchdog — if any required input is stale (>200ms) → publish
     SAFE_DEFAULTS and reset history.
  2. NaN/Inf guard on inputs and outputs.
  3. Tilt detection — if projected gravity z < 0.5 (robot tilted >~60°)
     → publish DEFENSIVE_DEFAULTS (high α, high φ).
  4. Try/except around inference — any exception → SAFE_DEFAULTS.
  5. Heartbeat — /cbf/inference_status published at 10Hz with state info.
  6. Output bounds clamping — values forced into declared ranges
     regardless of model output (tanh+scale should already keep them in
     range, but defensive).
  7. History reset on long input gaps (>500ms) — prevents stale-history
     contamination at the start of a new run.

The Go2's walking_bridge.cpp adds its own heartbeat watchdog at the
robot-command layer (500ms timeout → Damp), so the full stack has two
independent safety layers.

Run on the Go2's Jetson Orin (ros2 humble):
  ros2 run go2_walking_lidar cbf_inference_node \\
    --ros-args -p teacher_ckpt:=<path> -p student_ckpt:=<path>
"""
from __future__ import annotations

import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, String

# Local import — cbf_deploy_model lives next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cbf_deploy_model import CbfDeployModel


# ──────────────────────────────────────────────────────────────────────
# Safety defaults — published when fail-safes trigger.
# ──────────────────────────────────────────────────────────────────────

# Conservative-but-functional CBF params. Robot can still move; CBF acts
# normally with hand-tuned values. Used on input timeout / inference fail.
SAFE_DEFAULTS = {
    "alpha": 2.0, "phi": 0.5, "a": 0.05, "b": 0.5, "c": -0.05,
}

# Maximally defensive — robot expected to almost stop. Used on tilt
# detection (robot falling / fallen).
DEFENSIVE_DEFAULTS = {
    "alpha": 3.0, "phi": 5.0, "a": 0.5, "b": 1.0, "c": -0.10,
}

# Param-clamp ranges (must match cbf_deploy_model.py).
PARAM_RANGES = {
    "alpha": (0.5, 3.0),
    "phi":   (0.0, 5.0),
    "a":     (0.0, 0.5),
    "b":     (0.0, 1.0),
    "c":     (-0.10, 0.0),
}

# Timing thresholds.
INPUT_STALE_MS = 200        # input considered stale → SAFE_DEFAULTS
HISTORY_RESET_MS = 500      # gap > this → reset 50-step history
INFERENCE_RATE_HZ = 50.0    # match sim dt = 0.02s
HEARTBEAT_RATE_HZ = 10.0
TRACKING_HISTORY_LEN = 5    # 5-step tracking_err history per V13 priv obs


class CbfInferenceNode(Node):
    def __init__(self):
        super().__init__("cbf_inference_node")

        # Parameters.
        self.declare_parameter("teacher_ckpt", "")
        self.declare_parameter("student_ckpt", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("base_height_nominal", 0.30)  # Go2 standing
        self.declare_parameter("input_stale_ms", float(INPUT_STALE_MS))
        self.declare_parameter("history_reset_ms", float(HISTORY_RESET_MS))
        self.declare_parameter("inference_rate_hz", INFERENCE_RATE_HZ)
        self.declare_parameter("tilt_gravity_z_min", 0.5)  # tilt fail-safe
        teacher_ckpt = self.get_parameter("teacher_ckpt").value
        student_ckpt = self.get_parameter("student_ckpt").value
        device = self.get_parameter("device").value
        self.h_nominal = float(self.get_parameter("base_height_nominal").value)
        self.input_stale_s = float(self.get_parameter("input_stale_ms").value) / 1000.0
        self.history_reset_s = float(self.get_parameter("history_reset_ms").value) / 1000.0
        self.tilt_gravity_z_min = float(self.get_parameter("tilt_gravity_z_min").value)
        rate_hz = float(self.get_parameter("inference_rate_hz").value)

        if not teacher_ckpt or not student_ckpt:
            self.get_logger().fatal(
                "teacher_ckpt and student_ckpt parameters required.")
            raise SystemExit(1)

        # Load deploy model — heavy, happens once at startup.
        self.get_logger().info(f"Loading teacher: {teacher_ckpt}")
        self.get_logger().info(f"Loading student: {student_ckpt}")
        self.model = CbfDeployModel(teacher_ckpt, student_ckpt, device=device)
        self.get_logger().info("Deploy model loaded.")

        # Sensor state — most-recent values + timestamps.
        # V14 (2026-05-21): /odom now carries both linear and angular velocity
        # (extended by odom_publisher.cpp from the SportModeState IMU
        # gyroscope). Tilt detection reads pose.orientation from the same msg.
        self.last_odom_time = None
        self.odom_lin_vel = np.zeros(3, dtype=np.float32)    # base lin vel
        self.odom_ang_vel = np.zeros(3, dtype=np.float32)    # body-frame ω
        self.odom_base_height = self.h_nominal
        self.odom_gravity_z = 1.0                             # cos(tilt)
        self.last_teleop_time = None
        self.teleop_cmd = np.zeros(3, dtype=np.float32)      # vx, vy, ωyaw
        self.last_grid_time = None
        self.latest_grid = np.zeros((2, 64, 64), dtype=np.float32)

        # Tracking-err history — 5 most-recent (cmd - measured) snapshots.
        self.tracking_history = deque(maxlen=TRACKING_HISTORY_LEN)
        for _ in range(TRACKING_HISTORY_LEN):
            self.tracking_history.append(np.zeros(3, dtype=np.float32))

        # State machine for fail-safes (informational, for heartbeat).
        self.last_state = "INIT"
        self.last_publish_time = None

        # QoS — best-effort for high-rate sensors, reliable for cmd.
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        cmd_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        # Subscriptions. V14: consolidated to /odom (lin+ang vel+pose) +
        # /u_teleop (raw user cmd) + /cbf/grid. Drops the separate /imu/data
        # subscription.
        self.create_subscription(Odometry, "/odom", self.odom_cb, sensor_qos)
        self.create_subscription(Twist, "/u_teleop", self.teleop_cb, cmd_qos)
        self.create_subscription(
            Float32MultiArray, "/cbf/grid", self.grid_cb, sensor_qos,
        )

        # Publishers.
        self.params_pub = self.create_publisher(
            Float32MultiArray, "/cbf/params", cmd_qos,
        )
        self.status_pub = self.create_publisher(String, "/cbf/inference_status", 1)

        # Timers.
        self.create_timer(1.0 / rate_hz, self.tick)
        self.create_timer(1.0 / HEARTBEAT_RATE_HZ, self.heartbeat)

        self.get_logger().info(
            f"CbfInferenceNode ready. inference_rate={rate_hz}Hz, "
            f"input_stale={self.input_stale_s*1000:.0f}ms, "
            f"history_reset={self.history_reset_s*1000:.0f}ms")

    # ── Callbacks ──

    def odom_cb(self, msg: Odometry):
        self.last_odom_time = self.get_clock().now()
        # Linear velocity (body frame, set by odom_publisher).
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        self.odom_lin_vel = np.array([vx, vy, vz], dtype=np.float32)
        # V14: angular velocity (body-frame ω, from SportModeState gyro).
        wx = msg.twist.twist.angular.x
        wy = msg.twist.twist.angular.y
        wz = msg.twist.twist.angular.z
        self.odom_ang_vel = np.array([wx, wy, wz], dtype=np.float32)
        # base_height from odom z.
        self.odom_base_height = float(msg.pose.pose.position.z)
        # Projected gravity z (body frame) — for tilt fail-safe. From the
        # body→odom quaternion, gravity_z_body = qw² − qx² − qy² + qz².
        qw, qx, qy, qz = (msg.pose.pose.orientation.w,
                          msg.pose.pose.orientation.x,
                          msg.pose.pose.orientation.y,
                          msg.pose.pose.orientation.z)
        self.odom_gravity_z = float(qw*qw - qx*qx - qy*qy + qz*qz)

    def teleop_cb(self, msg: Twist):
        self.last_teleop_time = self.get_clock().now()
        self.teleop_cmd = np.array(
            [msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32,
        )

    def grid_cb(self, msg: Float32MultiArray):
        self.last_grid_time = self.get_clock().now()
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size != 2 * 64 * 64:
            self.get_logger().warn(
                f"/cbf/grid wrong size: got {data.size}, expected 8192. "
                "Keeping previous grid."
            )
            return
        self.latest_grid = data.reshape(2, 64, 64)

    # ── Inference tick ──

    def tick(self):
        now = self.get_clock().now()
        self.last_state, params = self._compute_params(now)

        # Final output clamping (defense in depth).
        for k, (lo, hi) in PARAM_RANGES.items():
            params[k] = float(np.clip(params[k], lo, hi))
            if not math.isfinite(params[k]):
                self.get_logger().error(f"{k}=NaN after clamp; using SAFE_DEFAULTS")
                params = dict(SAFE_DEFAULTS)
                self.last_state = "NAN_OUTPUT"
                break

        msg = Float32MultiArray()
        msg.data = [params["alpha"], params["phi"],
                    params["a"], params["b"], params["c"]]
        self.params_pub.publish(msg)
        self.last_publish_time = now

    def _compute_params(self, now) -> tuple[str, dict]:
        """Returns (state_label, params_dict). Pure — no publish side-effect."""

        # 1. Input freshness check.
        def stale(stamp):
            if stamp is None:
                return True
            elapsed = (now - stamp).nanoseconds / 1e9
            return elapsed > self.input_stale_s

        if stale(self.last_odom_time) or stale(self.last_teleop_time):
            return "INPUT_STALE", dict(SAFE_DEFAULTS)

        # 2. Tilt fail-safe — robot is falling / fallen.
        if self.odom_gravity_z < self.tilt_gravity_z_min:
            return "TILT_FAIL", dict(DEFENSIVE_DEFAULTS)

        # 3. NaN/Inf guard on inputs.
        for arr, name in [(self.odom_ang_vel, "odom_ang_vel"),
                          (self.odom_lin_vel, "odom_lin_vel"),
                          (self.teleop_cmd, "teleop_cmd"),
                          (self.latest_grid, "grid")]:
            if not np.all(np.isfinite(arr)):
                self.get_logger().warn(f"non-finite values in {name}; SAFE_DEFAULTS")
                return f"NAN_INPUT_{name}", dict(SAFE_DEFAULTS)

        # 4. History reset on long gaps (e.g. recovery after stale).
        if self.last_publish_time is not None:
            gap = (now - self.last_publish_time).nanoseconds / 1e9
            if gap > self.history_reset_s:
                proprio_seed = self._build_proprio()
                self.model.reset_history(proprio_seed)
                return "HISTORY_RESET", self._infer_safely()

        # 5. Normal path.
        return "OK", self._infer_safely()

    def _build_proprio(self) -> np.ndarray:
        """Construct the 19-D observable proprio vector per V13 ordering:
        [base_height(1), tracking_err_history(15), base_ang_vel(3)]."""
        # Update tracking-err history (cmd - measured) using current values.
        # Order: vx, vy, ωyaw — matches MultiPlannerCommand convention.
        cur_err = self.teleop_cmd - np.array(
            [self.odom_lin_vel[0], self.odom_lin_vel[1], self.odom_ang_vel[2]],
            dtype=np.float32,
        )
        self.tracking_history.append(cur_err)
        # Flatten oldest-to-newest into a 15-D vector (matches sim's
        # history_length=5, flatten_history_dim=True).
        tracking_15 = np.concatenate(list(self.tracking_history), dtype=np.float32)

        # Use odom z if it looks reasonable; else nominal.
        bh = self.odom_base_height
        if not (0.15 < bh < 0.45):
            bh = self.h_nominal
        base_height_1 = np.array([bh], dtype=np.float32)
        base_ang_vel_3 = self.odom_ang_vel.astype(np.float32)

        return np.concatenate([base_height_1, tracking_15, base_ang_vel_3])

    def _infer_safely(self) -> dict:
        """Run model.infer; on any exception, return SAFE_DEFAULTS."""
        try:
            proprio = self._build_proprio()
            grid = self.latest_grid  # most-recent grid
            out = self.model.infer(proprio, grid)
            return {k: out[k] for k in ("alpha", "phi", "a", "b", "c")}
        except Exception as exc:
            self.get_logger().error(f"inference exception: {exc!r}; SAFE_DEFAULTS")
            return dict(SAFE_DEFAULTS)

    # ── Heartbeat / diagnostics ──

    def heartbeat(self):
        msg = String()
        msg.data = (
            f"state={self.last_state}  "
            f"gz={self.odom_gravity_z:.2f}  "
            f"|cmd|={np.linalg.norm(self.teleop_cmd):.2f}"
        )
        self.status_pub.publish(msg)


def main():
    rclpy.init()
    node = CbfInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
