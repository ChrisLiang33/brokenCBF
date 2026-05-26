#!/usr/bin/env python3
"""ROS 2 grid construction node for the Go2 CBF stack.

Consumes the merged LiDAR cloud from `cloud_merger`, transforms points to
body frame, rasterizes a 64×64 ego-centric occupancy grid (10 cm / cell,
6.4 m × 6.4 m), and publishes the 2-channel grid (current + previous
frame, motion-encoding) on /cbf/grid.

Grid layout (matches sim exactly — see cbf_go2_observations.occupancy_grid_b):
  - Channels: 2 (current frame, previous frame)
  - Shape: (2, 64, 64)
  - Cell size: 0.10 m
  - Coverage: 6.4 m × 6.4 m
  - Origin: robot at grid center (cell [32, 32])
  - +x (body forward)  → increasing column index
  - +y (body left)     → increasing row index
  - Z filter: 0.05 m ≤ z ≤ 0.80 m (matches cloud_merger's body filter)
  - Output: Float32MultiArray, data = grid.flatten(), length 2 × 64 × 64 = 8192

Run:
  ros2 run go2_walking_lidar cbf_grid_node \\
    --ros-args -p cloud_topic:=/poisson_cloud
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

import tf2_ros
from tf2_ros import TransformException


# Sim conventions — must match cbf_go2_observations.occupancy_grid_b.
GRID_H = 64
GRID_W = 64
CELL_SIZE = 0.10       # 0.10 m / cell → 6.4 m × 6.4 m total
N_CHANNELS = 2         # current + previous frame

# Body filter (matches cloud_merger's static obstacle filter).
Z_MIN = 0.05
Z_MAX = 0.80
# Excludes the robot body itself (footprint).
BODY_MASK_X = 0.40
BODY_MASK_Y = 0.20

# Publication rate. Sim physics step is 0.02s = 50 Hz; the policy reads
# the grid at the same cadence.
PUBLISH_RATE_HZ = 50.0


class CbfGridNode(Node):
    def __init__(self):
        super().__init__("cbf_grid_node")

        self.declare_parameter("cloud_topic", "/poisson_cloud")
        self.declare_parameter("body_frame", "body_link")
        self.declare_parameter("publish_rate_hz", PUBLISH_RATE_HZ)
        cloud_topic = self.get_parameter("cloud_topic").value
        self.body_frame = self.get_parameter("body_frame").value
        rate_hz = float(self.get_parameter("publish_rate_hz").value)

        # TF buffer for cloud_frame → body_frame transform.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State.
        self.latest_cloud_msg: PointCloud2 | None = None
        # Prev-frame buffer initialized to empty.
        self.prev_grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)

        # QoS.
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.create_subscription(
            PointCloud2, cloud_topic, self.cloud_cb, sensor_qos,
        )
        self.grid_pub = self.create_publisher(
            Float32MultiArray, "/cbf/grid", sensor_qos,
        )
        self.create_timer(1.0 / rate_hz, self.tick)

        self.get_logger().info(
            f"CbfGridNode: cloud_topic={cloud_topic}, body_frame={self.body_frame}, "
            f"rate={rate_hz}Hz, grid={GRID_H}×{GRID_W}@{CELL_SIZE*100:.0f}cm"
        )

    def cloud_cb(self, msg: PointCloud2):
        self.latest_cloud_msg = msg

    def _transform_to_body(self, msg: PointCloud2) -> np.ndarray | None:
        """Return (N, 3) numpy array of points in body frame, or None on failure."""
        # If the cloud is already in body frame, no transform needed.
        if msg.header.frame_id == self.body_frame:
            pts = np.asarray(list(pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True,
            )))
            if pts.size == 0:
                return None
            return pts.view(np.float32).reshape(-1, 3).astype(np.float32)

        # Otherwise TF lookup at the cloud's timestamp.
        try:
            tf = self.tf_buffer.lookup_transform(
                self.body_frame, msg.header.frame_id, msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF lookup {msg.header.frame_id}→{self.body_frame} failed: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

        # Build 4x4 transformation matrix.
        q = tf.transform.rotation
        t = tf.transform.translation
        # Quaternion → 3x3 rotation.
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
        if pts.size == 0:
            return None
        pts = np.array(pts.tolist(), dtype=np.float32).reshape(-1, 3)
        # Apply: p_body = R · p_world + t
        return (pts @ R.T) + T_xyz

    def _rasterize(self, body_pts: np.ndarray) -> np.ndarray:
        """Map (N, 3) body-frame points → (H, W) binary occupancy grid."""
        grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        if body_pts.size == 0:
            return grid

        # Z filter.
        z = body_pts[:, 2]
        mask_z = (z >= Z_MIN) & (z <= Z_MAX)
        # Body mask — exclude points inside the robot footprint.
        x, y = body_pts[:, 0], body_pts[:, 1]
        mask_body = ~((np.abs(x) < BODY_MASK_X) & (np.abs(y) < BODY_MASK_Y))
        mask = mask_z & mask_body
        if not mask.any():
            return grid

        x = x[mask]; y = y[mask]
        # Grid coordinates: robot at center cell.
        # +x → +col, +y → +row.
        col = np.round(x / CELL_SIZE).astype(np.int32) + GRID_W // 2
        row = np.round(y / CELL_SIZE).astype(np.int32) + GRID_H // 2

        valid = (col >= 0) & (col < GRID_W) & (row >= 0) & (row < GRID_H)
        col = col[valid]; row = row[valid]
        grid[row, col] = 1.0
        return grid

    def tick(self):
        if self.latest_cloud_msg is None:
            return
        body_pts = self._transform_to_body(self.latest_cloud_msg)
        if body_pts is None:
            return
        current = self._rasterize(body_pts)

        # Stack with previous frame, publish, roll the buffer.
        grid_2c = np.stack([current, self.prev_grid], axis=0)  # (2, H, W)
        msg = Float32MultiArray()
        # Optional layout metadata — useful for downstream debugging.
        msg.layout.dim = [
            MultiArrayDimension(label="ch", size=N_CHANNELS, stride=N_CHANNELS * GRID_H * GRID_W),
            MultiArrayDimension(label="row", size=GRID_H, stride=GRID_H * GRID_W),
            MultiArrayDimension(label="col", size=GRID_W, stride=GRID_W),
        ]
        msg.data = grid_2c.flatten().tolist()
        self.grid_pub.publish(msg)
        self.prev_grid = current


def main():
    rclpy.init()
    node = CbfGridNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
