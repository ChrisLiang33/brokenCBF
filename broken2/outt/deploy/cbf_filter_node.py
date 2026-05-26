#!/usr/bin/env python3
"""ROS 2 CBF QP filter node for the Go2.

Sits between teleop and walking_bridge:
  /u_teleop  (user cmd)
  /cbf/grid   (2-channel grid, optional — used for diagnostics only)
  /cbf/params (α, φ, a, b, c from cbf_inference_node)
  + LiDAR points (for h(x), L_g h computation)
  →
  /u_des  (filtered safe command)  → walking_bridge

CBF constraint (matches cbf_go2_env._cbf_filter):
  L_g h · u ≥ −α·(h − c) + φ·‖L_g h‖² + a

Closed-form half-space projection in 2D (no QP solver needed when the
constraint reduces to a single linear inequality — sim uses the same
projection).

h(x) = ‖robot − nearest_obs‖ − r_safety
L_g h ≈ unit vector from nearest obs to robot (gradient of h w.r.t. robot xy)

Fail-safes:
  - On stale params (>200ms): pass u_teleop through unchanged (no filter).
  - On no LiDAR data: pass u_teleop through.
  - On NaN/Inf: zero out output (walking_bridge's watchdog will dampen).
  - On infeasible QP (L_g h ≈ 0 but rhs > 0): emergency stop.
"""
from __future__ import annotations

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, String

import tf2_ros
from tf2_ros import TransformException


# Safety geometry (matches sim).
R_ROBOT = 0.40             # robot footprint radius (m)
R_OBS_NOMINAL = 0.30       # nominal obstacle radius assumed
SAFETY_MARGIN_DEFAULT = R_ROBOT + R_OBS_NOMINAL  # 0.70 m

# LiDAR/grid-derived h(x) filter.
Z_MIN_OBSTACLE = 0.05
Z_MAX_OBSTACLE = 0.80
BODY_MASK_X = 0.40
BODY_MASK_Y = 0.20

# Output rate. Match sim physics step.
PUBLISH_RATE_HZ = 50.0

# Timing.
PARAMS_STALE_S = 0.2
LIDAR_STALE_S = 0.3


class CbfFilterNode(Node):
    def __init__(self):
        super().__init__("cbf_filter_node")

        self.declare_parameter("cloud_topic", "/poisson_cloud")
        self.declare_parameter("body_frame", "body_link")
        self.declare_parameter("safety_margin", SAFETY_MARGIN_DEFAULT)
        self.declare_parameter("publish_rate_hz", PUBLISH_RATE_HZ)
        cloud_topic = self.get_parameter("cloud_topic").value
        self.body_frame = self.get_parameter("body_frame").value
        self.safety_margin = float(self.get_parameter("safety_margin").value)
        rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State.
        self.teleop_cmd = np.zeros(3, dtype=np.float32)  # vx, vy, ωyaw
        self.last_teleop_time = None
        self.params = {"alpha": 2.0, "phi": 0.5, "a": 0.05, "b": 0.5, "c": -0.05}
        self.last_params_time = None
        self.latest_cloud_msg: PointCloud2 | None = None
        self.last_cloud_time = None
        self.last_state = "INIT"

        # QoS.
        sensor_qos = QoSProfile(depth=1,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        cmd_qos = QoSProfile(depth=1,
                             reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)

        # Subs.
        self.create_subscription(Twist, "/u_teleop", self.teleop_cb, cmd_qos)
        self.create_subscription(
            Float32MultiArray, "/cbf/params", self.params_cb, cmd_qos,
        )
        self.create_subscription(
            PointCloud2, cloud_topic, self.cloud_cb, sensor_qos,
        )

        # Pubs.
        self.u_des_pub = self.create_publisher(Twist, "/u_des", cmd_qos)
        self.status_pub = self.create_publisher(String, "/cbf/filter_status", 1)
        self.h_pub = self.create_publisher(Float32MultiArray, "/cbf/filter_h", sensor_qos)

        # Timer.
        self.create_timer(1.0 / rate_hz, self.tick)
        self.create_timer(0.1, self.heartbeat)  # 10 Hz status

        self.get_logger().info(
            f"CbfFilterNode: cloud_topic={cloud_topic}, "
            f"safety_margin={self.safety_margin:.2f}m, rate={rate_hz}Hz"
        )

    # ── Callbacks ──

    def teleop_cb(self, msg: Twist):
        self.last_teleop_time = self.get_clock().now()
        self.teleop_cmd = np.array(
            [msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float32,
        )

    def params_cb(self, msg: Float32MultiArray):
        if len(msg.data) < 5:
            self.get_logger().warn(f"/cbf/params has {len(msg.data)} dims, expected 5")
            return
        self.last_params_time = self.get_clock().now()
        self.params = {
            "alpha": float(msg.data[0]),
            "phi":   float(msg.data[1]),
            "a":     float(msg.data[2]),
            "b":     float(msg.data[3]),
            "c":     float(msg.data[4]),
        }

    def cloud_cb(self, msg: PointCloud2):
        self.last_cloud_time = self.get_clock().now()
        self.latest_cloud_msg = msg

    # ── Geometry ──

    def _transform_points_to_body(self, msg: PointCloud2) -> np.ndarray | None:
        if msg.header.frame_id == self.body_frame:
            pts = np.asarray(list(pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True,
            )))
            if pts.size == 0: return None
            return np.array(pts.tolist(), dtype=np.float32).reshape(-1, 3)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.body_frame, msg.header.frame_id, msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except TransformException:
            return None
        q = tf.transform.rotation
        t = tf.transform.translation
        xx, yy, zz = q.x*q.x, q.y*q.y, q.z*q.z
        xy, xz, yz = q.x*q.y, q.x*q.z, q.y*q.z
        wx, wy, wz = q.w*q.x, q.w*q.y, q.w*q.z
        R = np.array([
            [1 - 2*(yy+zz), 2*(xy-wz),    2*(xz+wy)],
            [2*(xy+wz),     1 - 2*(xx+zz), 2*(yz-wx)],
            [2*(xz-wy),     2*(yz+wx),    1 - 2*(xx+yy)],
        ], dtype=np.float32)
        T_xyz = np.array([t.x, t.y, t.z], dtype=np.float32)
        pts = np.asarray(list(pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True,
        )))
        if pts.size == 0: return None
        pts = np.array(pts.tolist(), dtype=np.float32).reshape(-1, 3)
        return (pts @ R.T) + T_xyz

    def _compute_h_and_grad(self) -> tuple[float, np.ndarray] | None:
        """Find nearest obstacle point in body frame, return (h, L_g h).

        h     = ‖p_nearest‖ − safety_margin   (positive = safe)
        L_g h = direction in which moving the robot INCREASES h, i.e. away
                from the nearest point. In body frame, L_g h = -p̂_nearest.

        Returns None if no LiDAR data."""
        if self.latest_cloud_msg is None:
            return None
        body_pts = self._transform_points_to_body(self.latest_cloud_msg)
        if body_pts is None or body_pts.size == 0:
            return None

        # Filter: in-plane points only, not on robot body.
        z = body_pts[:, 2]
        mask_z = (z >= Z_MIN_OBSTACLE) & (z <= Z_MAX_OBSTACLE)
        x, y = body_pts[:, 0], body_pts[:, 1]
        mask_body = ~((np.abs(x) < BODY_MASK_X) & (np.abs(y) < BODY_MASK_Y))
        mask = mask_z & mask_body
        if not mask.any():
            # No obstacles in range — h is huge.
            return float("inf"), np.array([0.0, 0.0], dtype=np.float32)

        xy = body_pts[mask][:, :2]
        dists = np.linalg.norm(xy, axis=1)
        nearest_idx = int(np.argmin(dists))
        d = float(dists[nearest_idx])
        p_nearest = xy[nearest_idx]                     # (2,)
        h = d - self.safety_margin
        # Gradient of h w.r.t. robot xy. In body frame, robot is at origin,
        # obstacle is at p_nearest. h = ‖robot − obs‖ − R = ‖−p_nearest‖ − R.
        # ∂h/∂robot = (robot − obs)/‖robot − obs‖ = −p̂_nearest.
        L_g_h = -p_nearest / max(d, 1e-6)               # (2,)
        return h, L_g_h.astype(np.float32)

    # ── Filter tick ──

    def tick(self):
        now = self.get_clock().now()

        def stale(stamp, max_age_s):
            if stamp is None: return True
            return (now - stamp).nanoseconds / 1e9 > max_age_s

        # 1. No teleop → publish zero.
        if stale(self.last_teleop_time, 0.5):
            self._publish_safe(np.zeros(3, dtype=np.float32), state="NO_TELEOP")
            return

        # 2. Stale params → passthrough (no filter, no safety!).
        # walking_bridge has its own 500ms watchdog that catches genuine
        # failure modes; passthrough lets manual teleop work even if
        # inference node is restarting.
        if stale(self.last_params_time, PARAMS_STALE_S):
            self._publish_safe(self.teleop_cmd.copy(), state="PASSTHROUGH_PARAMS_STALE")
            return

        # 3. NaN/Inf guard on inputs.
        if not (np.all(np.isfinite(self.teleop_cmd))
                and all(math.isfinite(v) for v in self.params.values())):
            self._publish_safe(np.zeros(3, dtype=np.float32), state="NAN_INPUT")
            return

        # 4. No LiDAR → passthrough (better to let user move than block).
        if stale(self.last_cloud_time, LIDAR_STALE_S):
            self._publish_safe(self.teleop_cmd.copy(), state="PASSTHROUGH_LIDAR_STALE")
            return

        h_grad = self._compute_h_and_grad()
        if h_grad is None:
            self._publish_safe(self.teleop_cmd.copy(), state="NO_OBSTACLE_DATA")
            return
        h, L_g_h = h_grad

        # 5. No obstacle in range — passthrough.
        if math.isinf(h):
            self._publish_safe(self.teleop_cmd.copy(), state="OK_NO_OBSTACLE")
            return

        # 6. CBF QP — closed-form projection.
        u_des_xy = self.teleop_cmd[:2].copy()
        u_des_yaw = self.teleop_cmd[2]

        alpha = self.params["alpha"]
        phi   = self.params["phi"]
        a     = self.params["a"]
        c     = self.params["c"]

        Lgh_norm_sq = float(L_g_h @ L_g_h)
        rhs = -alpha * (h - c) + phi * Lgh_norm_sq + a

        # Slack: L_g h · u_des − rhs.  ≥0 means already safe → no deflection.
        slack = float(L_g_h @ u_des_xy) - rhs
        if slack < 0:
            # Project u_des onto the constraint half-space.
            if Lgh_norm_sq < 1e-8:
                # Infeasible (constraint binding but no direction to move).
                self._publish_safe(np.zeros(3, dtype=np.float32), state="INFEASIBLE")
                # Publish h for diagnostics.
                self._publish_h(h, L_g_h, slack)
                return
            projection_scale = slack / Lgh_norm_sq
            u_safe_xy = u_des_xy - L_g_h * projection_scale
        else:
            u_safe_xy = u_des_xy

        # Keep yaw unfiltered (CBF only constrains xy).
        u_safe = np.array([u_safe_xy[0], u_safe_xy[1], u_des_yaw], dtype=np.float32)
        self._publish_safe(u_safe, state="OK" if slack >= 0 else "OK_DEFLECTED")
        self._publish_h(h, L_g_h, slack)

    def _publish_safe(self, u_safe: np.ndarray, state: str):
        if not np.all(np.isfinite(u_safe)):
            u_safe = np.zeros(3, dtype=np.float32)
            state = "NAN_OUTPUT"
        msg = Twist()
        msg.linear.x = float(u_safe[0])
        msg.linear.y = float(u_safe[1])
        msg.angular.z = float(u_safe[2])
        self.u_des_pub.publish(msg)
        self.last_state = state

    def _publish_h(self, h: float, L_g_h: np.ndarray, slack: float):
        msg = Float32MultiArray()
        msg.data = [float(h), float(L_g_h[0]), float(L_g_h[1]), float(slack)]
        self.h_pub.publish(msg)

    def heartbeat(self):
        msg = String()
        msg.data = (
            f"state={self.last_state}  "
            f"α={self.params['alpha']:.2f}  φ={self.params['phi']:.2f}  "
            f"|cmd|={np.linalg.norm(self.teleop_cmd):.2f}"
        )
        self.status_pub.publish(msg)


def main():
    rclpy.init()
    node = CbfFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
