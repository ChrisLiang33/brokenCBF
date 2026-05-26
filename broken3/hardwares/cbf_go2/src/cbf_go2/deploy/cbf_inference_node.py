#!/usr/bin/env python3
"""ROS 2 inference node — runs the trained CBF parameter policy and
publishes (alpha, phi) on /cbf/params at 50 Hz.

Subscribes:
    /odom              (nav_msgs/Odometry)        — body lin+ang vel, pose
    /u_teleop          (geometry_msgs/Twist)      — user command (for tracking error)
    /cbf/grid          (std_msgs/Float32MultiArray) — 2x64x64 ego-centric LiDAR grid

Publishes:
    /cbf/params              (std_msgs/Float32MultiArray)  [alpha, phi]
    /cbf/inference_status    (std_msgs/String)             heartbeat (10 Hz)

Adapted from hardwares/prototype/deploy/cbf_inference_node.py. Key
changes vs the prototype:
    - Action space trimmed to (alpha, phi); a/b/c dropped (MVP pins them 0).
    - Defaults to SAFE_DEFAULTS until cbf_deploy_model is wired with a
      real checkpoint (the stub returns SAFE values, so this node still
      publishes sensible params on day 1).
    - Proprio construction left as a clearly-marked TODO since the Go2
      observation layout is still being finalized in training.

Fail-safe layers (kept from prototype — they are policy-agnostic):
    1. Input freshness watchdog (>200 ms stale -> SAFE_DEFAULTS).
    2. Tilt fail-safe (gravity_z < 0.5 -> DEFENSIVE_DEFAULTS).
    3. NaN/Inf guards on inputs and outputs.
    4. try/except around inference -> SAFE_DEFAULTS.
    5. History reset after long input gaps (>500 ms).
    6. Output clamping to declared (alpha, phi) ranges.
    7. Heartbeat status (/cbf/inference_status @ 10 Hz).

The walking_bridge.cpp adds its own watchdog at the motor-command layer
(no Twist for >5 s -> Damp). Two independent safety layers.

Run:
    ros2 run cbf_go2 cbf_inference_node \\
        --ros-args -p checkpoint:=/home/unitree/cbf_go2/run/checkpoints/policy.pt
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

# cbf_deploy_model lives next to this file; install via FILES.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cbf_deploy_model import (
    CbfDeployModel,
    ALPHA_MIN, ALPHA_MAX,
    PHI_MIN, PHI_MAX,
    SAFE_ALPHA, SAFE_PHI,
)


# Conservative hand-tuned values. Used on input timeout / inference fail.
SAFE_DEFAULTS = {"alpha": SAFE_ALPHA, "phi": SAFE_PHI}

# Maximally defensive — high alpha, high phi. Used on tilt detection.
DEFENSIVE_DEFAULTS = {"alpha": 3.0, "phi": 5.0}

PARAM_RANGES = {
    "alpha": (ALPHA_MIN, ALPHA_MAX),
    "phi":   (PHI_MIN,   PHI_MAX),
}

# Timing.
INPUT_STALE_MS    = 200
HISTORY_RESET_MS  = 500
INFERENCE_RATE_HZ = 50.0    # match sim dt = 0.02s
HEARTBEAT_RATE_HZ = 10.0
TRACKING_HISTORY_LEN = 5


class CbfInferenceNode(Node):
    def __init__(self):
        super().__init__("cbf_inference_node")

        self.declare_parameter("checkpoint", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("base_height_nominal", 0.30)  # Go2 standing
        self.declare_parameter("input_stale_ms", float(INPUT_STALE_MS))
        self.declare_parameter("history_reset_ms", float(HISTORY_RESET_MS))
        self.declare_parameter("inference_rate_hz", INFERENCE_RATE_HZ)
        self.declare_parameter("tilt_gravity_z_min", 0.5)

        checkpoint = self.get_parameter("checkpoint").value or None
        device     = self.get_parameter("device").value
        self.h_nominal       = float(self.get_parameter("base_height_nominal").value)
        self.input_stale_s   = float(self.get_parameter("input_stale_ms").value) / 1000.0
        self.history_reset_s = float(self.get_parameter("history_reset_ms").value) / 1000.0
        self.tilt_gravity_z_min = float(self.get_parameter("tilt_gravity_z_min").value)
        rate_hz = float(self.get_parameter("inference_rate_hz").value)

        # Load policy (stub mode if no checkpoint).
        self.get_logger().info(f"Loading deploy model (checkpoint={checkpoint!r}, device={device})")
        self.model = CbfDeployModel(checkpoint=checkpoint, device=device)
        if self.model.is_stub:
            self.get_logger().warn(
                "Deploy model is in STUB mode — publishing SAFE_DEFAULTS for "
                "(alpha, phi). Wire CbfDeployModel.infer() before claiming "
                "this is a learned policy.")

        # Sensor state.
        self.last_odom_time = None
        self.odom_lin_vel = np.zeros(3, dtype=np.float32)
        self.odom_ang_vel = np.zeros(3, dtype=np.float32)
        self.odom_base_height = self.h_nominal
        self.odom_gravity_z = 1.0
        self.last_teleop_time = None
        self.teleop_cmd = np.zeros(3, dtype=np.float32)
        self.last_grid_time = None
        self.latest_grid = np.zeros((2, 64, 64), dtype=np.float32)

        # Tracking-err history (5-step (cmd - measured) snapshots).
        self.tracking_history = deque(maxlen=TRACKING_HISTORY_LEN)
        for _ in range(TRACKING_HISTORY_LEN):
            self.tracking_history.append(np.zeros(3, dtype=np.float32))

        self.last_state = "INIT"
        self.last_publish_time = None

        sensor_qos = QoSProfile(depth=1,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        cmd_qos = QoSProfile(depth=1,
                             reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(Odometry,            "/odom",      self.odom_cb,   sensor_qos)
        self.create_subscription(Twist,               "/u_teleop",  self.teleop_cb, cmd_qos)
        self.create_subscription(Float32MultiArray,   "/cbf/grid",  self.grid_cb,   sensor_qos)

        self.params_pub = self.create_publisher(Float32MultiArray, "/cbf/params", cmd_qos)
        self.status_pub = self.create_publisher(String, "/cbf/inference_status", 1)

        self.create_timer(1.0 / rate_hz, self.tick)
        self.create_timer(1.0 / HEARTBEAT_RATE_HZ, self.heartbeat)

        self.get_logger().info(
            f"CbfInferenceNode ready. rate={rate_hz}Hz, "
            f"input_stale={self.input_stale_s*1000:.0f}ms, "
            f"history_reset={self.history_reset_s*1000:.0f}ms")

    # ---- Callbacks ----

    def odom_cb(self, msg: Odometry):
        self.last_odom_time = self.get_clock().now()
        t = msg.twist.twist
        self.odom_lin_vel = np.array([t.linear.x,  t.linear.y,  t.linear.z],  dtype=np.float32)
        self.odom_ang_vel = np.array([t.angular.x, t.angular.y, t.angular.z], dtype=np.float32)
        self.odom_base_height = float(msg.pose.pose.position.z)
        q = msg.pose.pose.orientation
        # Projected gravity z (body frame) = qw^2 - qx^2 - qy^2 + qz^2.
        self.odom_gravity_z = float(q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z)

    def teleop_cb(self, msg: Twist):
        self.last_teleop_time = self.get_clock().now()
        self.teleop_cmd = np.array(
            [msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32)

    def grid_cb(self, msg: Float32MultiArray):
        self.last_grid_time = self.get_clock().now()
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size != 2 * 64 * 64:
            self.get_logger().warn(
                f"/cbf/grid wrong size: got {data.size}, expected 8192. Keeping previous grid.")
            return
        self.latest_grid = data.reshape(2, 64, 64)

    # ---- Inference tick ----

    def tick(self):
        now = self.get_clock().now()
        self.last_state, params = self._compute_params(now)

        for k, (lo, hi) in PARAM_RANGES.items():
            params[k] = float(np.clip(params[k], lo, hi))
            if not math.isfinite(params[k]):
                self.get_logger().error(f"{k}=NaN after clamp; using SAFE_DEFAULTS")
                params = dict(SAFE_DEFAULTS)
                self.last_state = "NAN_OUTPUT"
                break

        msg = Float32MultiArray()
        msg.data = [params["alpha"], params["phi"]]
        self.params_pub.publish(msg)
        self.last_publish_time = now

    def _compute_params(self, now) -> tuple[str, dict]:
        def stale(stamp):
            if stamp is None:
                return True
            return (now - stamp).nanoseconds / 1e9 > self.input_stale_s

        if stale(self.last_odom_time) or stale(self.last_teleop_time):
            return "INPUT_STALE", dict(SAFE_DEFAULTS)

        if self.odom_gravity_z < self.tilt_gravity_z_min:
            return "TILT_FAIL", dict(DEFENSIVE_DEFAULTS)

        for arr, name in [(self.odom_ang_vel, "odom_ang_vel"),
                          (self.odom_lin_vel, "odom_lin_vel"),
                          (self.teleop_cmd,   "teleop_cmd"),
                          (self.latest_grid,  "grid")]:
            if not np.all(np.isfinite(arr)):
                self.get_logger().warn(f"non-finite values in {name}; SAFE_DEFAULTS")
                return f"NAN_INPUT_{name}", dict(SAFE_DEFAULTS)

        if self.last_publish_time is not None:
            gap = (now - self.last_publish_time).nanoseconds / 1e9
            if gap > self.history_reset_s:
                proprio_seed = self._build_proprio()
                self.model.reset_history(proprio_seed)
                return "HISTORY_RESET", self._infer_safely()

        return "OK", self._infer_safely()

    def _build_proprio(self) -> np.ndarray:
        """Assemble the proprio observation vector for the policy.

        TODO(deploy): match the exact ordering used by the trained Go2
        policy (see go2/<latest_phase>/env.py). The current layout
        mirrors the prototype's 19-D V13 vector:
            [base_height(1), tracking_err_history(15), base_ang_vel(3)]
        which may not match the MVP/Go2 policy 1:1.
        """
        cur_err = self.teleop_cmd - np.array(
            [self.odom_lin_vel[0], self.odom_lin_vel[1], self.odom_ang_vel[2]],
            dtype=np.float32)
        self.tracking_history.append(cur_err)
        tracking_15 = np.concatenate(list(self.tracking_history), dtype=np.float32)

        bh = self.odom_base_height
        if not (0.15 < bh < 0.45):
            bh = self.h_nominal
        base_height_1 = np.array([bh], dtype=np.float32)
        base_ang_vel_3 = self.odom_ang_vel.astype(np.float32)

        return np.concatenate([base_height_1, tracking_15, base_ang_vel_3])

    def _infer_safely(self) -> dict:
        try:
            proprio = self._build_proprio()
            grid    = self.latest_grid
            out     = self.model.infer(proprio, grid)
            return {"alpha": out["alpha"], "phi": out["phi"]}
        except Exception as exc:
            self.get_logger().error(f"inference exception: {exc!r}; SAFE_DEFAULTS")
            return dict(SAFE_DEFAULTS)

    def heartbeat(self):
        msg = String()
        msg.data = (
            f"state={self.last_state}  gz={self.odom_gravity_z:.2f}  "
            f"|cmd|={np.linalg.norm(self.teleop_cmd):.2f}")
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
