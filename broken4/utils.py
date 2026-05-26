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
    """Run one episode.

    `policy` is either a callable obs -> action, or a fixed normalized
    action (np.ndarray) applied every step.
    """
    opts = None if disturbance is None else {"disturbance": disturbance}
    obs, _ = env.reset(seed=seed, options=opts)

    traj = [env.x.copy()]
    min_h = np.inf
    total_intervention = 0.0
    phis, alphas = [], []
    collided = reached = False

    done = False
    while not done:
        action = policy(obs) if callable(policy) else policy
        obs, _, term, trunc, info = env.step(action)
        traj.append(info["x"])
        min_h = min(min_h, info["h"])
        total_intervention += info["intervention"]
        phis.append(info["phi"])
        alphas.append(info["alpha"])
        collided = collided or info["collided"]
        reached = reached or info["reached"]
        done = term or trunc

    return {
        "traj": np.asarray(traj),
        "min_h": float(min_h),
        "intervention": float(total_intervention),
        "collided": collided,
        "reached": reached,
        "phi": float(np.mean(phis)),
        "alpha": float(np.mean(alphas)),
        "steps": len(phis),
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
def train_ppo(cfg, timesteps, n_envs=4, seed=0, verbose=0):
    """Train a PPO policy on CBFParamEnv. Returns the SB3 model."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv([(lambda: CBFParamEnv(cfg)) for _ in range(n_envs)])
    model = PPO(
        "MlpPolicy",
        venv,
        seed=seed,
        verbose=verbose,
        n_steps=512,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    model.learn(total_timesteps=timesteps)
    return model


def policy_from_model(model):
    """Wrap an SB3 model as a deterministic obs -> action callable."""
    return lambda obs: model.predict(obs, deterministic=True)[0]


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------
def plot_trajectory(cfg, labelled_rollouts, path):
    """labelled_rollouts: list of (label, rollout_dict)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # workspace + obstacle
    ax1.add_patch(Circle(cfg.obstacle, cfg.obstacle_radius,
                         color="0.55", alpha=0.5, label="obstacle"))
    ax1.add_patch(Circle(cfg.obstacle, cfg.r_safe, color="0.55",
                         fill=False, ls="--", lw=1))
    ax1.plot(*cfg.start, "ko", ms=9)
    ax1.plot(*cfg.goal, "k*", ms=16)
    for label, r in labelled_rollouts:
        ax1.plot(r["traj"][:, 0], r["traj"][:, 1], lw=2, label=label)
    ax1.set_xlim(0, cfg.workspace[0])
    ax1.set_ylim(0, cfg.workspace[1])
    ax1.set_aspect("equal")
    ax1.set_title("Trajectory")
    ax1.legend(loc="lower right", fontsize=9)

    for label, r in labelled_rollouts:
        ax2.plot(r["traj"][:, 0] * 0 + np.arange(len(r["traj"])),
                 [np.nan] + _h_series(cfg, r["traj"]), lw=2, label=label)
    ax2.axhline(0.0, color="r", ls="--", lw=1, label="h = 0 (contact)")
    ax2.set_xlabel("step")
    ax2.set_ylabel("barrier value h")
    ax2.set_title("Safety margin over time")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _h_series(cfg, traj):
    p = np.asarray(cfg.obstacle)
    return [float(np.linalg.norm(x - p)) - cfg.r_safe for x in traj[1:]]


def plot_pareto(fixed_points, learned_point, path, disturbance):
    """fixed_points: list of dicts with intervention/min_h/collision_rate.
    learned_point: dict with intervention/min_h/collision_rate."""
    fig, ax = plt.subplots(figsize=(7.5, 6))

    xs = [p["intervention"] for p in fixed_points]
    ys = [p["min_h"] for p in fixed_points]
    cs = [p["collision_rate"] for p in fixed_points]
    sc = ax.scatter(xs, ys, c=cs, cmap="RdYlGn_r", s=70,
                    edgecolor="k", lw=0.5, vmin=0, vmax=1,
                    label="fixed (phi, alpha) grid")
    fig.colorbar(sc, ax=ax, label="collision rate")

    front = _pareto_front(xs, ys)
    ax.plot([xs[i] for i in front], [ys[i] for i in front],
            "k--", lw=1, alpha=0.7, label="fixed-param frontier")

    ax.scatter([learned_point["intervention"]], [learned_point["min_h"]],
               marker="*", s=420, color="#534AB7", edgecolor="k", lw=1,
               zorder=5, label="learned adaptive policy")
    ax.annotate("learned",
                (learned_point["intervention"], learned_point["min_h"]),
                textcoords="offset points", xytext=(12, 8),
                fontsize=9, color="#534AB7", weight="bold")

    ax.set_xlabel("mean intervention cost   (lower = closer to nominal)")
    ax.set_ylabel("mean min barrier value   (higher = safer)")
    ax.set_title(f"Invasiveness vs safety   (disturbance = {disturbance})\n"
                 "learned policy should sit up and/or left of the frontier")
    ax.legend(loc="lower right", fontsize=9)
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


def plot_ood(levels, curves, path, train_level):
    """curves: dict name -> list of collision rates aligned with `levels`."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, rates in curves.items():
        ax.plot(levels, rates, "o-", lw=2, label=name)
    ax.axvline(train_level, color="0.4", ls=":", lw=1.5,
               label=f"train disturbance = {train_level}")
    ax.set_xlabel("test disturbance magnitude")
    ax.set_ylabel("collision rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Out-of-distribution robustness\n"
                 "adaptive policy should degrade more gracefully")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
