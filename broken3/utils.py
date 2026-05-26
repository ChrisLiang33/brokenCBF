"""Rollout / evaluation / training / plotting helpers shared by phases."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from env import CBFParamEnv


# ----------------------------------------------------------------------
# action <-> parameter conversion
# ----------------------------------------------------------------------
def to_action(cfg, phi, alpha):
    """Inverse of env._map_action: (phi, alpha) -> normalized action."""
    lo_p, hi_p = cfg.phi_bounds
    lo_a, hi_a = cfg.alpha_bounds
    ap = 2.0 * (phi - lo_p) / (hi_p - lo_p) - 1.0
    aa = 2.0 * (alpha - lo_a) / (hi_a - lo_a) - 1.0
    return np.array([ap, aa], dtype=np.float32)


# ----------------------------------------------------------------------
# rollouts
# ----------------------------------------------------------------------
def rollout(env, policy, disturbance=None, seed=0):
    """Run one episode. `policy` is a callable obs -> action, or a fixed
    normalized action applied every step."""
    opts = None if disturbance is None else {"disturbance": disturbance}
    obs, _ = env.reset(seed=seed, options=opts)

    traj = [env.x.copy()]
    phi_series, alpha_series = [], []
    min_h = np.inf
    total_intervention = 0.0
    collided = reached = False

    done = False
    while not done:
        action = policy(obs) if callable(policy) else policy
        obs, _, term, trunc, info = env.step(action)
        traj.append(info["x"])
        phi_series.append(info["phi"])
        alpha_series.append(info["alpha"])
        min_h = min(min_h, info["h"])
        total_intervention += info["intervention"]
        collided = collided or info["collided"]
        reached = reached or info["reached"]
        done = term or trunc

    return {
        "traj": np.asarray(traj),
        "phi_series": np.asarray(phi_series),
        "alpha_series": np.asarray(alpha_series),
        "min_h": float(min_h),
        "intervention": float(total_intervention),
        "collided": collided,
        "reached": reached,
        "phi": float(np.mean(phi_series)),
        "alpha": float(np.mean(alpha_series)),
        "steps": len(phi_series),
    }


def evaluate(cfg, policy, disturbance, n_episodes=30, seed0=1000):
    """Average rollout metrics over many seeds at a fixed disturbance."""
    env = CBFParamEnv(cfg)
    results = [
        rollout(env, policy, disturbance=disturbance, seed=seed0 + i)
        for i in range(n_episodes)
    ]
    return {
        "collision_rate": float(np.mean([r["collided"] for r in results])),
        "reach_rate": float(np.mean([r["reached"] for r in results])),
        "min_h": float(np.mean([r["min_h"] for r in results])),
        "min_h_worst": float(np.min([r["min_h"] for r in results])),
        "intervention": float(np.mean([r["intervention"] for r in results])),
    }


# ----------------------------------------------------------------------
# PPO training
# ----------------------------------------------------------------------
def train_ppo(cfg, timesteps, n_envs=4, seed=0, verbose=0, init_from=None):
    """Train a PPO policy on CBFParamEnv. Returns the SB3 model.

    init_from: optional path to a saved model to continue training from
    (lets long runs be split into chunks across time limits)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv([(lambda: CBFParamEnv(cfg)) for _ in range(n_envs)])
    if init_from is not None:
        model = PPO.load(init_from)
        model.set_env(venv)
    else:
        model = PPO(
            "MlpPolicy", venv, seed=seed, verbose=verbose,
            n_steps=512, batch_size=256, gamma=0.99, gae_lambda=0.95,
            ent_coef=0.01, learning_rate=3e-4,   # small entropy: explore
                                                 # low-phi at near-misses
                                                 # instead of collapsing to
                                                 # safe-everywhere
            policy_kwargs=dict(net_arch=[64, 64]),
        )
    model.learn(total_timesteps=timesteps,
                reset_num_timesteps=(init_from is None))
    return model


def policy_from_model(model):
    """Wrap an SB3 model as a deterministic obs -> action callable."""
    return lambda obs: model.predict(obs, deterministic=True)[0]


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------
def _draw_obstacles(cfg, ax):
    moving_idx = cfg.moving_obstacle[0] if cfg.moving_obstacle else -1
    base = cfg.obstacle_centers()
    radii = [o[2] for o in cfg.obstacles]
    safe = cfg.safe_radii()
    for i, (c, r, rs) in enumerate(zip(base, radii, safe)):
        if i == moving_idx:
            _, axx, axy, amp, _ = cfg.moving_obstacle
            axis = np.array([axx, axy], dtype=float)
            axis /= (np.linalg.norm(axis) + 1e-9)
            for s in (-1.0, 1.0):                 # ghost the swept extremes
                ax.add_patch(Circle(c + s * amp * axis, r,
                                    color="#cc4444", alpha=0.16))
            ax.add_patch(Circle(c, r, color="#cc4444", alpha=0.4))
            ax.add_patch(Circle(c, rs, color="#cc4444",
                                fill=False, ls="--", lw=1))
        else:
            ax.add_patch(Circle(c, r, color="0.55", alpha=0.5))
            ax.add_patch(Circle(c, rs, color="0.55",
                                fill=False, ls="--", lw=1))


def _h_series(cfg, traj):
    """Min barrier value per step, using obstacle positions at that step
    (the moving obstacle is at obstacle_state(step))."""
    radii = cfg.safe_radii()
    out = []
    for step in range(1, len(traj)):
        centers, _ = cfg.obstacle_state(step)
        x = traj[step]
        out.append(min(float(np.linalg.norm(x - c)) - r
                       for c, r in zip(centers, radii)))
    return out


def plot_trajectory(cfg, labelled_rollouts, path):
    """labelled_rollouts: list of (label, rollout_dict).
    Four panels: trajectory, barrier value, learned phi, learned alpha."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    ax1, ax2, ax3, ax4 = axes

    _draw_obstacles(cfg, ax1)
    ax1.plot(*cfg.start, "ko", ms=9)
    ax1.plot(*cfg.goal, "k*", ms=16)
    for label, r in labelled_rollouts:
        ax1.plot(r["traj"][:, 0], r["traj"][:, 1], lw=2, label=label)
    ax1.set_xlim(0, cfg.workspace[0])
    ax1.set_ylim(0, cfg.workspace[1])
    ax1.set_aspect("equal")
    ax1.set_title("Trajectory through the obstacle field")
    ax1.legend(loc="lower right", fontsize=9)

    for label, r in labelled_rollouts:
        series = _h_series(cfg, r["traj"])
        ax2.plot(np.arange(len(series)), series, lw=2, label=label)
    ax2.axhline(0.0, color="r", ls="--", lw=1, label="h = 0 (contact)")
    ax2.set_xlabel("step")
    ax2.set_ylabel("min barrier value h")
    ax2.set_title("Safety margin over time")
    ax2.legend(fontsize=9)

    for label, r in labelled_rollouts:
        ax3.plot(np.arange(len(r["phi_series"])), r["phi_series"],
                 lw=2, label=label)
    ax3.set_ylim(cfg.phi_bounds[0] - 0.05, cfg.phi_bounds[1] + 0.05)
    ax3.set_xlabel("step")
    ax3.set_ylabel("learned phi")
    ax3.set_title("phi over time\n(robustness margin, ~1/epsilon)")
    ax3.legend(fontsize=9)

    for label, r in labelled_rollouts:
        ax4.plot(np.arange(len(r["alpha_series"])), r["alpha_series"],
                 lw=2, label=label)
    ax4.set_ylim(cfg.alpha_bounds[0] - 0.2, cfg.alpha_bounds[1] + 0.2)
    ax4.set_xlabel("step")
    ax4.set_ylabel("learned alpha")
    ax4.set_title("alpha over time\n(class-K gain; flat line = not adapting)")
    ax4.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _pareto_front(xs, ys):
    """Indices on the up-left frontier (min intervention, max min_h)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    front, best_y = [], -np.inf
    for i in order:
        if ys[i] > best_y:
            front.append(i)
            best_y = ys[i]
    return front


def plot_pareto(fixed_points, learned_point, path, disturbance):
    """fixed_points / learned_point: dicts with intervention, min_h,
    collision_rate. The view zooms into the informative low-intervention
    region -- extreme-phi points that over-deflect by 100x are real but
    nobody would pick them, and at full scale they bury everything."""
    fig, ax = plt.subplots(figsize=(7.5, 6))

    xs = [p["intervention"] for p in fixed_points]
    ys = [p["min_h"] for p in fixed_points]
    cs = [p["collision_rate"] for p in fixed_points]
    lx, ly = learned_point["intervention"], learned_point["min_h"]

    sc = ax.scatter(xs, ys, c=cs, cmap="RdYlGn_r", s=70,
                    edgecolor="k", lw=0.5, vmin=0, vmax=1,
                    label="fixed (phi, alpha) grid")
    fig.colorbar(sc, ax=ax, label="collision rate")

    front = _pareto_front(xs, ys)
    ax.plot([xs[i] for i in front], [ys[i] for i in front],
            "k--", lw=1, alpha=0.7, label="fixed-param frontier")

    ax.scatter([lx], [ly], marker="*", s=420, color="#534AB7",
               edgecolor="k", lw=1, zorder=5, label="learned adaptive policy")
    ax.annotate("learned", (lx, ly), textcoords="offset points",
                xytext=(10, 10), fontsize=9, color="#534AB7", weight="bold")

    # zoom: keep the learned point and the cheaper fixed points in view.
    # Extreme-phi points are real but over-deflect 100x and over-shoot
    # safety -- they bury the informative cluster on both axes.
    focus_x = max(lx * 3.0, float(np.percentile(xs, 55)) * 1.4,
                  min(xs) + 1e-6)
    in_view = [i for i in range(len(xs)) if xs[i] <= focus_x]
    ys_in = [ys[i] for i in in_view] + [ly]
    y_lo, y_hi = min(ys_in), max(ys_in)
    y_pad = max((y_hi - y_lo) * 0.18, 0.02)
    n_off = len(xs) - len(in_view)

    ax.set_xlim(-0.04 * focus_x, focus_x)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    if n_off:
        ax.text(0.98, 0.04,
                f"{n_off} extreme-phi fixed points off-screen "
                f"(intervention up to {max(xs):.0f}, min_h up to {max(ys):.1f})",
                transform=ax.transAxes, ha="right", fontsize=8,
                style="italic", color="0.4")

    ax.set_xlabel("mean intervention cost   (lower = closer to nominal)")
    ax.set_ylabel("mean min barrier value   (higher = safer)")
    ax.set_title(f"Invasiveness vs safety   (disturbance = {disturbance})\n"
                 "learned policy should sit up and/or left of the frontier")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_ood(levels, curves, path, train_level):
    """curves: dict name -> list of collision rates aligned with `levels`."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, rates in curves.items():
        ax.plot(levels, rates, "o-", lw=2, label=name)
    ax.axvline(train_level, color="0.4", ls=":", lw=1.5,
               label=f"train disturbance ceiling = {train_level}")
    ax.set_xlabel("test disturbance magnitude")
    ax.set_ylabel("collision rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Out-of-distribution robustness\n"
                 "adaptive policy should degrade more gracefully")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
