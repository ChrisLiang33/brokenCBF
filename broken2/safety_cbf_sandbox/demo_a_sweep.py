"""Demo: a sweep with adversarial push events at varying magnitudes.

`a` is the Dean-style additive slack: it tightens the CBF constraint by
`a` units, pre-allocating margin against bounded additive disturbance d.

In the sim formulation:  L_g h . u + alpha(h - c) >= phi ||L_g h||^2 + a

For a worst-case disturbance d aligned with -grad_h, magnitude |d|:
  h_dot_actual = grad_h . (u + d) >= -alpha(h - c) - phi ||L_g h||^2 - a + 0 - |d|
               (where the -|d| comes from the worst-case d projection)
  Wait — for the constraint  L_g h . u + alpha(h-c) >= a  to absorb -|d|:
  Need:  alpha(h - c) >= a >= |d|  (intuitively, a = |d| pre-pays the worst-case push)

This experiment uses *impulsive push events*: at random intervals, a one-step
velocity injection of magnitude |Delta_v| aligned with -grad_h (worst-case
direction). Each push effectively reduces h by ~ |Delta_v| * dt over that step.

We expect a* ~ |Delta_v| at the boundary, roughly.

Sweep:
  a in linspace(0, 2.0, 21)
  push magnitudes |Delta_v| in {0.0, 0.5, 1.0, 1.5, 2.0}
  push interval: every 1.0 second
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from sandbox import CBFParams, Obstacle, simulate, barrier


def make_periodic_adversarial_push(interval_s, magnitude, dt, x_state_ref, obs):
    """Returns a push_fn(i, dt) that fires every interval_s with magnitude
    aligned with -grad_h at x[i]. x_state_ref is a list whose last element
    is updated each step by the simulator.
    """
    interval_steps = max(1, int(interval_s / dt))

    def push_fn(i, dt_inner):
        if i == 0 or (i % interval_steps) != 0:
            return np.zeros(2)
        # x_state_ref[-1] is the current state (updated by the caller).
        x_now = x_state_ref[-1]
        _, grad_h = barrier(x_now, obs, 0.0)
        gn = np.linalg.norm(grad_h)
        if gn < 1e-9:
            return np.zeros(2)
        # Push is a *velocity adjustment* applied for one step.
        # We want delta-x = -magnitude*(grad_h/gn) * dt, so velocity per
        # step = -magnitude*(grad_h/gn). push_fn is added to u in simulate(),
        # which is then multiplied by dt for delta-x.
        return -magnitude * (grad_h / gn)

    return push_fn


def run_grid(magnitudes, a_values, n_trials=1, alpha=2.0, push_interval_s=1.0,
             T=10.0, dt=0.05):
    """Returns dict (mag, a) -> {col_rate, min_h, t_to_goal}.

    Because each sim is deterministic given push_fn closure on state,
    we wrap state in a list and update it manually here.
    """
    obs = Obstacle(pos=np.array([0.0, 0.0]), radius=0.5)
    x0 = np.array([-3.0, 0.05])
    goal = np.array([3.0, 0.0])

    results = {}
    for mag in magnitudes:
        for a_val in a_values:
            params = CBFParams(alpha=alpha, phi=0.0, a=a_val, c=0.0)
            collided = 0
            ttg_list = []
            min_h_list = []
            for seed in range(n_trials):
                # State container so push_fn can read current x.
                x_ref = [x0.copy()]
                push_fn = make_periodic_adversarial_push(
                    push_interval_s, mag, dt, x_ref, obs
                )

                # Monkey-patch: simulate() doesn't pass x into push_fn, so
                # we manually drive the rollout here, mirroring simulate().
                hist = manual_rollout(
                    x0=x0, goal=goal, obs=obs, params=params,
                    T=T, dt=dt, u_max=2.0,
                    push_fn=push_fn, x_ref=x_ref, seed=seed,
                )
                if hist["collided"]:
                    collided += 1
                if hist["reached_goal"]:
                    ttg_list.append(hist["time_to_goal"])
                min_h_list.append(hist["min_h_true"])
            results[(mag, a_val)] = {
                "col_rate": collided / n_trials,
                "t_to_goal": float(np.median(ttg_list)) if ttg_list else float("inf"),
                "mean_min_h": float(np.mean(min_h_list)),
                "frac_reached": len(ttg_list) / n_trials,
            }
    return results


def manual_rollout(x0, goal, obs, params, T, dt, u_max, push_fn, x_ref, seed=0):
    """Stripped-down rollout where push_fn can read the current state via x_ref."""
    from sandbox import cbf_qp
    N = int(T / dt)
    x = x0.copy()
    h_list = []
    reached = False
    for i in range(N):
        x_ref[-1] = x  # update state container BEFORE push_fn fires
        dir_to_goal = goal - x
        norm = np.linalg.norm(dir_to_goal)
        if norm < 1e-9:
            u_des = np.zeros(2)
        else:
            u_des = (dir_to_goal / norm) * min(u_max, norm)
        u_safe, _, _ = cbf_qp(u_des, x, obs, params, 0.0)
        un = np.linalg.norm(u_safe)
        if un > u_max:
            u_safe = u_safe * (u_max / un)
        u_applied = u_safe + push_fn(i, dt)
        x = x + u_applied * dt
        h_true, _ = barrier(x, obs, 0.0)
        h_list.append(h_true)
        if np.linalg.norm(x - goal) < 0.1:
            reached = True
            break

    min_h = float(min(h_list)) if h_list else 0.0
    return {
        "collided": bool(min_h < 0.0),
        "reached_goal": reached,
        "min_h_true": min_h,
        "time_to_goal": (len(h_list) * dt) if reached else float("inf"),
    }


def find_a_star(results, mag, a_values, col_target=0.05):
    for a_val in a_values:
        if results[(mag, a_val)]["col_rate"] <= col_target:
            return a_val
    return None


def main():
    magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0]
    a_values = np.linspace(0.0, 2.0, 21)

    print(f"Running {len(magnitudes)} x {len(a_values)} = "
          f"{len(magnitudes)*len(a_values)} sims (periodic adversarial push)...")
    results = run_grid(magnitudes, a_values, n_trials=1, alpha=2.0)
    print("Done.\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(magnitudes)))

    ax = axes[0]
    for color, mag in zip(cmap, magnitudes):
        ys = [results[(mag, a_val)]["col_rate"] for a_val in a_values]
        ax.plot(a_values, ys, "-o", color=color, label=f"|dv|={mag:.1f}",
                markersize=4)
    ax.axhline(0.05, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("a")
    ax.set_ylabel("collision rate")
    ax.set_title("Collision rate vs a, periodic adversarial push (every 1s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for color, mag in zip(cmap, magnitudes):
        ys = [results[(mag, a_val)]["mean_min_h"] for a_val in a_values]
        ax.plot(a_values, ys, "-o", color=color, label=f"|dv|={mag:.1f}",
                markersize=4)
    ax.axhline(0.0, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("a")
    ax.set_ylabel("min h_true")
    ax.set_title("Safety margin vs a")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    a_stars = []
    mags_with_star = []
    for mag in magnitudes:
        a_star = find_a_star(results, mag, a_values, col_target=0.05)
        if a_star is not None:
            mags_with_star.append(mag)
            a_stars.append(a_star)
    ax.plot(mags_with_star, a_stars, "ko-", label="empirical a*", markersize=9)
    mag_fine = np.linspace(0, max(magnitudes) + 0.5, 100)
    ax.plot(mag_fine, mag_fine, "b--", alpha=0.6, label="reference a = |dv|")
    ax.plot(mag_fine, mag_fine * 0.05, "g--", alpha=0.6,
            label="a = |dv| * dt  (per-step impulse)")
    ax.set_xlabel("push magnitude |dv|")
    ax.set_ylabel("a*")
    ax.set_title("Empirical a* vs push magnitude")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("a sweep with periodic adversarial push — alpha=2.0")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "out_a_sweep.png")
    plt.savefig(out, dpi=110)
    print(f"Saved {out}\n")

    print(f"{'|dv|':>5} {'a*':>5} {'col@a*':>8} {'t_goal@a*':>11} {'min_h@a*':>10}")
    print("-" * 50)
    for mag in magnitudes:
        a_star = find_a_star(results, mag, a_values, col_target=0.05)
        if a_star is None:
            print(f"{mag:>5.1f} {'-':>5} {'no a clears 5% threshold':>30}")
            continue
        r = results[(mag, a_star)]
        print(f"{mag:>5.1f} {a_star:>5.2f} {r['col_rate']:>8.3f} "
              f"{r['t_to_goal']:>11.2f} {r['mean_min_h']:>10.3f}")

    print()
    print("min_h_true at a=0 across magnitudes (failure mode):")
    print(f"{'|dv|':>5} {'min_h(a=0)':>11} {'col(a=0)':>9}")
    for mag in magnitudes:
        r = results[(mag, 0.0)]
        print(f"{mag:>5.1f} {r['mean_min_h']:>11.3f} {r['col_rate']:>9.3f}")


if __name__ == "__main__":
    main()
