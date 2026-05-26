"""Central configuration for the CBF-parameter-learning MVP.

Everything is a 2D single integrator in command space (x_dot = u + d).
The CBF lives in command space, so this toy is faithful to a
controller-agnostic design: nothing here knows the robot has legs.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ---- workspace geometry --------------------------------------------
    start: tuple = (1.0, 1.0)
    goal: tuple = (9.0, 9.0)
    # obstacle sits between start and goal but OFF the y=x line: colinear
    # start/obstacle/goal causes a CBF deadlock (purely radial push, robot
    # stalls). The offset gives the QP a clear side to slide around.
    obstacle: tuple = (5.0, 4.0)
    obstacle_radius: float = 1.5      # safe disk still crosses the straight
                                      # path, so the constraint always binds
    robot_radius: float = 0.3
    workspace: tuple = (10.0, 10.0)

    # ---- dynamics ------------------------------------------------------
    dt: float = 0.1
    u_max: float = 1.0                # 2-norm velocity command limit
    max_steps: int = 300
    goal_tol: float = 0.4

    # ---- nominal go-to-goal P controller -------------------------------
    kp: float = 1.0

    # ---- disturbance (the OOD dial) ------------------------------------
    disturbance: float = 0.0          # fixed magnitude of d (eval / Phase 2)
    disturbance_range: tuple = None   # if set, sample d ~ U(range) each
                                      # episode -> policy learns to condition
                                      # on the disturbance estimate (Phase 3)
    disturbance_resample: int = 20    # steps between re-sampling d's direction

    # ---- QP safety filter ----------------------------------------------
    slack_penalty: float = 1e4        # M in min ||u-u_nom||^2 + M*delta^2

    # ---- learned parameter bounds (action -> (phi, alpha)) -------------
    # h = ||x - p_obs|| - r_safe, so ||L_g h||^2 = 1 and the robustified
    # constraint reduces to:   grad_h . u  >=  phi - alpha * h
    #   phi   : constant rate margin -> directly the disturbance hedge
    #   alpha : how fast h is allowed to decrease (standard CBF gain)
    phi_bounds: tuple = (0.0, 1.0)
    alpha_bounds: tuple = (0.2, 4.0)

    # ---- fixed params for Phase 0 / grid-sweep baselines ---------------
    fixed_phi: float = 0.25
    fixed_alpha: float = 1.0

    # ---- reward weights ------------------------------------------------
    w_progress: float = 1.0           # + reduction in distance to goal
    w_intervention: float = 1.0       # - ||u_safe - u_nom||  (filter cost)
    w_collision: float = 100.0        # - on h_true < 0  (terminates episode)
    w_goal: float = 50.0              # + on reaching the goal

    # ---- RL ------------------------------------------------------------
    state_conditioned: bool = True    # False -> obs is zeroed (Phase 1)

    @property
    def r_safe(self) -> float:
        return self.obstacle_radius + self.robot_radius
