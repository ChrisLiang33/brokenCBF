"""Central configuration for the CBF-parameter-learning MVP.

The robot is a 2D single integrator (x_dot = u + d) in command space.
Phase 2 uses a CORRIDOR layout: two obstacles straddling the path
midpoint form one tight gap, with open space before and after. The
optimal conservatism therefore genuinely varies with position -- which
is exactly what lets an adaptive policy beat any fixed parameter.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ---- workspace geometry --------------------------------------------
    start: tuple = (1.0, 1.0)
    goal: tuple = (9.0, 9.0)

    # scatter: 4 obstacles along the start->goal path. A and D are
    # single near-miss obstacles -- the path grazes them, but with open
    # space to one side, so low phi clears them with a small nudge and
    # high phi just wastes intervention. B and C form a tight pinch
    # (~0.8-wide gap) that genuinely needs high phi. A fixed parameter
    # must therefore compromise (wasteful at near-misses OR unsafe at the
    # pinch); an adaptive policy can be cheap at A/D and conservative at
    # the pinch. Each obstacle is (center_x, center_y, radius).
    obstacles: tuple = (
        (3.2, 3.9, 0.9),   # A - early near-miss
        (4.6, 7.0, 1.0),   # B - pinch, upper
        (7.0, 4.6, 1.0),   # C - pinch, lower
        (7.3, 8.2, 0.9),   # D - late near-miss
    )
    robot_radius: float = 0.3
    workspace: tuple = (10.0, 10.0)

    # one obstacle oscillates -> creates a dynamic regime where the CBF
    # gain alpha has a reason to adapt. With a static field, committing
    # to the nominal and correcting late (high alpha) always works, so
    # alpha parks at its ceiling. With a closing obstacle, a late sharp
    # correction can fail -- engaging early and gently (lower alpha) can
    # be better. moving_obstacle = (index, axis_x, axis_y, amplitude,
    # period_steps); the obstacle oscillates about its base position:
    #   p(n) = base + amplitude * sin(2*pi*n/period) * axis_hat
    # None -> all obstacles static (the pre-experiment behaviour).
    moving_obstacle: tuple = (1, 1.0, 0.0, 1.0, 55)   # obstacle B in x

    # ---- dynamics ------------------------------------------------------
    dt: float = 0.1
    u_max: float = 1.0                # 2-norm velocity command limit
    max_steps: int = 300
    goal_tol: float = 0.4

    # ---- nominal go-to-goal P controller -------------------------------
    kp: float = 1.0

    # ---- disturbance (the OOD dial) ------------------------------------
    disturbance: float = 0.0          # fixed magnitude of d (eval / Phase 2)
    disturbance_range: tuple = None   # if set, sample d ~ U(range) per
                                      # episode -> policy learns to condition
                                      # on the disturbance estimate (Phase 3)
    disturbance_resample: int = 20    # steps between re-sampling d's direction

    # ---- QP safety filter ----------------------------------------------
    slack_penalty: float = 1e4        # M in min ||u-u_nom||^2 + M*sum(delta^2)

    # ---- learned parameter bounds (action -> (phi, alpha)) -------------
    # h_i = ||x - p_i|| - r_safe_i, so ||L_g h_i||^2 = 1 and each
    # robustified constraint reduces to:  grad_h_i . u >= phi - alpha*h_i
    phi_bounds: tuple = (0.0, 1.0)
    alpha_bounds: tuple = (0.2, 4.0)

    # ---- fixed params for Phase 0 / grid-sweep baselines ---------------
    fixed_phi: float = 0.25
    fixed_alpha: float = 1.0

    # ---- reward weights ------------------------------------------------
    # collision:intervention ratio set to 15:1. Too high (e.g. 100:1)
    # and the policy stays safety-greedy -- it runs phi high everywhere
    # and never learns it can be cheap at a near-miss. Too low and it
    # starts accepting collisions. 15:1 keeps safety dominant while
    # making invasiveness a real term the policy must compete on.
    w_progress: float = 1.0           # + reduction in distance to goal
    w_intervention: float = 1.0       # - ||u_safe - u_nom||  (filter cost)
    w_collision: float = 15.0         # - on any h_true < 0  (terminates)
    w_goal: float = 50.0              # + on reaching the goal

    # ---- RL ------------------------------------------------------------
    state_conditioned: bool = True    # False -> obs is zeroed (Phase 1)

    def obstacle_centers(self):
        import numpy as np
        return [np.array([o[0], o[1]], dtype=float) for o in self.obstacles]

    def safe_radii(self):
        return [o[2] + self.robot_radius for o in self.obstacles]

    def obstacle_state(self, step):
        """Obstacle centers and velocities at a given step.

        Returns (centers, velocities): centers is a list of np arrays
        (base position for static obstacles, oscillated for the moving
        one); velocities is a list of np arrays in units/sec (zero for
        static obstacles). The velocity feeds the time-varying CBF term
        d h/d t = -grad_h . v_obs.
        """
        import numpy as np
        centers = self.obstacle_centers()
        vels = [np.zeros(2) for _ in self.obstacles]
        if self.moving_obstacle is not None:
            idx, ax, ay, amp, period = self.moving_obstacle
            axis = np.array([ax, ay], dtype=float)
            axis /= (np.linalg.norm(axis) + 1e-9)
            w = 2.0 * np.pi / period
            centers[idx] = centers[idx] + amp * np.sin(w * step) * axis
            # d(center)/d(step) * (1/dt)  ->  units per second
            vels[idx] = (amp * w * np.cos(w * step) * axis) / self.dt
        return centers, vels
