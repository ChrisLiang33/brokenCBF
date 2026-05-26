"""MVP: a blue dot tries to reach a red goal with a naive (obstacle-blind) planner.

Pipeline:
    planner.plan(start, goal)         -> straight-line waypoints to the goal
    PurePursuit(waypoints).command()  -> world-frame velocity command (vx, vy)
    set agent free-joint qvel         -> agent slides toward the next waypoint

Episode terminates on either:
    - goal reached (success)
    - agent contacts a cylinder obstacle (failure)

Run modes:
    mjpython mvp.py                 # interactive viewer (macOS)
    python3 mvp.py --record run.mp4 # offline render to mp4
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

import planner
import safety_filter

HERE = Path(__file__).parent
MODEL_PATH = HERE / "scene_mvp.xml"

START_XY = (0.0, 0.0)
GOAL_XY = (3.0, 0.0)
GOAL_TOLERANCE = 0.2
WAYPOINT_TOLERANCE = 0.15
CRUISE_SPEED = 0.6


class Sim:
    def __init__(self, verbose: bool = True) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.verbose = verbose

        self.goal_mocap = self._mocap_id("goal")
        self.next_wp_mocap = self._mocap_id("next_wp")

        self.agent_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "agent_geom")
        self.obstacle_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("obs1_geom", "obs2_geom")
        }

        self.filter = safety_filter.CBFSafetyFilter(planner.OBSTACLES)

        # Per-step velocity disturbance added to the filter's u_safe before
        # the integrator. Models actuator noise / wind / slip. Defaults to
        # zero; env writes to it each step for the φ experiments.
        self.disturbance = np.zeros(2)

        # Control loop delay (sim steps). Filter operates on position from
        # this many steps ago — models sensor-to-actuator latency.
        self.state_delay = 0
        self._pos_history: list[np.ndarray] = []

        # Filter recomputes every `control_period` sim steps. Between
        # recomputes the agent runs on the previous u_safe.
        self.control_period = 1
        self._sim_step_counter = 0
        self._last_u_safe = np.zeros(2)

        # Max norm of u_safe before it's applied — actuator saturation.
        # When disturbance exceeds this limit, the filter literally cannot
        # cancel it, so φ's reserved margin becomes the only safety budget.
        self.u_max = float("inf")

        # Unobservable position kick. External code can write a vector here;
        # we apply it once before the next mj_step, then clear it. Models
        # a sudden disturbance the filter has no time to pre-compensate.
        self.pending_kick = np.zeros(2)

        self.reset()

    def reset(self, goal_xy: tuple[float, float] | None = None) -> None:
        """Reset to initial state. Keeps the loaded model; cheap enough for RL."""
        mujoco.mj_resetData(self.model, self.data)
        self.goal_xy = tuple(goal_xy) if goal_xy is not None else GOAL_XY
        self.data.mocap_pos[self.goal_mocap] = [self.goal_xy[0], self.goal_xy[1], 0.1]
        self.data.qpos[:3] = [START_XY[0], START_XY[1], 0.1]

        waypoints = planner.plan(START_XY, self.goal_xy)
        self.pursuit = planner.PurePursuit(
            waypoints, reach_tolerance=WAYPOINT_TOLERANCE, speed=CRUISE_SPEED)
        if self.verbose:
            print(f"Planned path: {len(waypoints)} waypoints, "
                  f"length {self._path_length(waypoints):.2f} m  (naive — ignores obstacles)")

        self.terminated = False
        self.termination_reason: str | None = None
        self._filter_was_active = False
        self._pos_history = [self.data.qpos[:2].copy()]
        self._sim_step_counter = 0
        self._last_u_safe = np.zeros(2)
        self.last_info = None

    def _mocap_id(self, body_name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return int(self.model.body_mocapid[body_id])

    @staticmethod
    def _path_length(waypoints: list[tuple[float, float]]) -> float:
        pts = np.asarray(waypoints)
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    def _agent_obstacle_contact(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {int(c.geom1), int(c.geom2)}
            if self.agent_geom_id in pair and pair & self.obstacle_geom_ids:
                return True
        return False

    def step(self, alpha_override: float | None = None,
             phi_override: float | None = None) -> None:
        if self.terminated:
            # Hold the agent in place during the post-termination linger.
            self.data.qvel[0] = 0.0
            self.data.qvel[1] = 0.0
            mujoco.mj_step(self.model, self.data)
            return

        pos_xy_true = self.data.qpos[:2].copy()
        self._pos_history.append(pos_xy_true)

        if self._sim_step_counter % self.control_period == 0:
            # Filter sees position from `state_delay` sim steps ago.
            idx = max(0, len(self._pos_history) - 1 - self.state_delay)
            pos_xy = self._pos_history[idx]
            u_des = self.pursuit.command(pos_xy)
            u_safe, info = self.filter.filter(pos_xy, u_des,
                                              alpha_override=alpha_override,
                                              phi_override=phi_override)
            self._last_u_safe = u_safe
            self.last_info = info
        u_safe = self._last_u_safe
        # Saturate the controller's output, then add the disturbance.
        norm = float(np.linalg.norm(u_safe))
        if norm > self.u_max:
            u_safe = u_safe * (self.u_max / norm)
        u_actual = u_safe + self.disturbance
        self.data.qvel[0] = u_actual[0]
        self.data.qvel[1] = u_actual[1]
        # Apply an unobservable position kick, if requested.
        if float(np.linalg.norm(self.pending_kick)) > 1e-9:
            self.data.qpos[0] += self.pending_kick[0]
            self.data.qpos[1] += self.pending_kick[1]
            self.pending_kick = np.zeros(2)
        self._sim_step_counter += 1

        if self.verbose and self.last_info is not None:
            info = self.last_info
            pos_xy = pos_xy_true  # for log readout only
            if info["active"] and not self._filter_was_active:
                print(f"[CBF active]  pos=({pos_xy[0]:.2f}, {pos_xy[1]:.2f})  "
                      f"sdf={info['sdf_min']:+.3f}  h={info['h']:+.3f}  "
                      f"α={info['alpha']:.2f}  |Δu|={np.linalg.norm(u_safe - self.pursuit.command(pos_xy)):.3f}")
            elif self._filter_was_active and not info["active"]:
                print(f"[CBF released] pos=({pos_xy[0]:.2f}, {pos_xy[1]:.2f})  "
                      f"sdf={info['sdf_min']:+.3f}  α={info['alpha']:.2f}")
        if self.last_info is not None:
            self._filter_was_active = self.last_info["active"]

        tgt = self.pursuit.current_target()
        self.data.mocap_pos[self.next_wp_mocap] = [tgt[0], tgt[1], 0.05]

        mujoco.mj_step(self.model, self.data)

        if self._agent_obstacle_contact():
            self._terminate("collision",
                            f"hit an obstacle at ({self.data.qpos[0]:.2f}, "
                            f"{self.data.qpos[1]:.2f})")
            return

        dist = float(np.linalg.norm(np.asarray(self.goal_xy) - self.data.qpos[:2]))
        if dist <= GOAL_TOLERANCE:
            self._terminate("goal_reached",
                            f"reached goal at ({self.data.qpos[0]:.2f}, "
                            f"{self.data.qpos[1]:.2f})")

    def _terminate(self, reason: str, detail: str) -> None:
        self.terminated = True
        self.termination_reason = reason
        if self.verbose:
            tag = "SUCCESS" if reason == "goal_reached" else "FAILURE"
            print(f"[{tag}] {detail} at sim t={self.data.time:.2f}s")


def run_viewer(sim: Sim) -> None:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            sim.step()
            viewer.sync()
            sleep_for = sim.model.opt.timestep - (time.time() - step_start)
            if sleep_for > 0:
                time.sleep(sleep_for)


def run_record(sim: Sim, out_path: Path, fps: int, width: int, height: int,
               max_seconds: float, azimuth: float, elevation: float,
               distance: float) -> None:
    import imageio.v2 as imageio

    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [1.5, 0.0, 0.0]
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation

    sim_per_frame = max(1, int(round(1.0 / (fps * sim.model.opt.timestep))))
    max_steps = int(max_seconds / sim.model.opt.timestep)
    linger_steps = int(1.0 / sim.model.opt.timestep)

    print(f"Recording {out_path.name} at {width}x{height}@{fps}fps "
          f"(max {max_seconds:.1f}s)")

    frames: list[np.ndarray] = []
    terminated_step: int | None = None
    for step in range(max_steps):
        sim.step()
        if step % sim_per_frame == 0:
            renderer.update_scene(sim.data, camera=cam)
            frames.append(renderer.render())
        if sim.terminated and terminated_step is None:
            terminated_step = step
        if terminated_step is not None and step - terminated_step >= linger_steps:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out_path), frames, fps=fps, codec="libx264", quality=8)
    print(f"Wrote {len(frames)} frames -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=str, default=None,
                        help="Render offline to this mp4 path instead of opening the viewer.")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Max simulated seconds in record mode.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--azimuth", type=float, default=90.0,
                        help="Camera azimuth in degrees (record mode).")
    parser.add_argument("--elevation", type=float, default=-75.0,
                        help="Camera elevation in degrees, negative looks down.")
    parser.add_argument("--distance", type=float, default=4.5,
                        help="Camera distance from lookat (record mode).")
    args = parser.parse_args()

    sim = Sim()
    if args.record:
        run_record(sim, Path(args.record),
                   fps=args.fps, width=args.width, height=args.height,
                   max_seconds=args.duration,
                   azimuth=args.azimuth, elevation=args.elevation,
                   distance=args.distance)
    else:
        run_viewer(sim)


if __name__ == "__main__":
    main()
