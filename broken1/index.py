"""
MVP: Learn CBF parameters (alpha, phi) via CEM for a 2D single integrator
navigating cylindrical obstacles using a SMOOTH SDF-based CBF.

System:    x_dot = u + w,  w ~ N(0, sigma^2 I)
SDF:       sdf(x) = min_i (||x - rho_i|| - R_i)        (closest obstacle)
CBF:       h(x) = lambda * (1 - exp(-gamma * sdf))    (single constraint for all)
           grad h = lambda*gamma * exp(-gamma*sdf) * grad_sdf
           ||L_g h||^2 = (lambda*gamma)^2 * exp(-2 gamma sdf)
                       peaks at the boundary, decays away.
Filter:    min ||u - u_nom||^2
           s.t.  L_g h . u - phi * ||L_g h||^2 + alpha * h >= 0
           Closed-form single-constraint projection.
Learner:   CEM over theta = (log alpha, log phi).
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------- Environment ----------
class SingleIntegratorEnv:
    def __init__(self, obstacles, goal=(5.0, 0.0),
                 dt=0.05, max_steps=400, noise_std=0.1, u_max=2.0,
                 h_smooth_lambda=1.0, h_smooth_gamma=2.0,
                 obs_drift_std=0.0, g_eps=0.0):
        # Store initial obstacle layout; current self.obstacles drifts per rollout
        self._initial = [(np.array(c, dtype=float), float(r)) for c, r in obstacles]
        self.obstacles = [(c.copy(), r) for c, r in self._initial]
        self.goal = np.array(goal, dtype=float)
        self.dt = dt
        self.max_steps = max_steps
        self.noise_std = noise_std
        self.u_max = u_max
        self.h_lambda = h_smooth_lambda
        self.h_gamma = h_smooth_gamma
        self.obs_drift_std = obs_drift_std
        self.g_eps = g_eps                      # actuation gain perturbation std
        self.obs_velocities = [np.zeros(2) for _ in self._initial]
        self.B = np.eye(2)

    def reset(self, x0, rng):
        self.x = np.array(x0, dtype=float)
        self.t = 0
        self.obstacles = [(c.copy(), r) for c, r in self._initial]
        if self.obs_drift_std > 0:
            self.obs_velocities = [
                self.obs_drift_std * rng.standard_normal(2) for _ in self._initial
            ]
        else:
            self.obs_velocities = [np.zeros(2) for _ in self._initial]
        # Per-rollout actuation gain matrix B = I + delta. The filter still assumes B=I.
        if self.g_eps > 0:
            self.B = np.eye(2) + self.g_eps * rng.standard_normal((2, 2))
        else:
            self.B = np.eye(2)
        return self.x.copy()

    def step(self, u, rng):
        n = np.linalg.norm(u)
        if n > self.u_max:
            u = u * (self.u_max / n)
        w = self.noise_std * rng.standard_normal(2)
        self.x = self.x + (self.B @ u + w) * self.dt
        if self.obs_drift_std > 0:
            self.obstacles = [(c + v * self.dt, r)
                              for (c, r), v in zip(self.obstacles, self.obs_velocities)]
        self.t += 1
        collided = any(np.linalg.norm(self.x - c) < r for c, r in self.obstacles)
        reached = np.linalg.norm(self.x - self.goal) < 0.2
        done = collided or reached or self.t >= self.max_steps
        return self.x.copy(), done, {'collided': collided, 'reached': reached}


def p_controller(x, goal, k=2.0):
    return k * (goal - x)


# ---------- Simulated LiDAR + clustering + cylinder fit ----------
# Stand-in for Livox Mid-360 + Euclidean clustering + circle fit pipeline.
class LidarPerception:
    def __init__(self, n_rays=180, max_range=6.0, range_noise_std=0.02,
                 cluster_threshold=0.35, min_cluster_size=3, max_fit_radius=2.5,
                 match_threshold=0.5):
        self.n_rays = n_rays
        self.max_range = max_range
        self.range_noise_std = range_noise_std
        self.cluster_threshold = cluster_threshold
        self.min_cluster_size = min_cluster_size
        self.max_fit_radius = max_fit_radius
        self.match_threshold = match_threshold
        self.angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        self._cos = np.cos(self.angles)
        self._sin = np.sin(self.angles)
        # Frame-to-frame tracking state (for velocity estimation)
        self._last_obstacles = []  # list of (center, radius)

    def reset_tracking(self):
        self._last_obstacles = []

    def _raycast(self, x, true_obstacles):
        """Vectorized ray-circle intersection. Returns clean range per ray."""
        ranges = np.full(self.n_rays, self.max_range)
        for c, r in true_obstacles:
            dx = x[0] - c[0]
            dy = x[1] - c[1]
            b = 2 * (dx * self._cos + dy * self._sin)
            c_quad = dx * dx + dy * dy - r * r
            disc = b * b - 4 * c_quad
            valid = disc >= 0
            sqrt_disc = np.sqrt(np.where(valid, disc, 0.0))
            t1 = np.where(valid, (-b - sqrt_disc) / 2, np.inf)
            t2 = np.where(valid, (-b + sqrt_disc) / 2, np.inf)
            t1 = np.where(t1 > 1e-6, t1, np.inf)
            t2 = np.where(t2 > 1e-6, t2, np.inf)
            t = np.minimum(t1, t2)
            ranges = np.minimum(ranges, t)
        return ranges

    def scan(self, x, true_obstacles, rng):
        """Clean raycast + range noise."""
        ranges = self._raycast(x, true_obstacles)
        hit = ranges < self.max_range
        noise = self.range_noise_std * rng.standard_normal(self.n_rays)
        return np.where(hit, ranges + noise, self.max_range), hit

    def _cluster_rays(self, ranges, hit):
        """Group contiguous hit rays whose ranges differ by less than threshold."""
        clusters, current = [], []
        for i in range(self.n_rays):
            if not hit[i]:
                if current:
                    clusters.append(current)
                    current = []
                continue
            if not current or abs(ranges[i] - ranges[current[-1]]) < self.cluster_threshold:
                current.append(i)
            else:
                clusters.append(current)
                current = [i]
        if current:
            clusters.append(current)
        # Merge wrap-around (last + first) if they look like the same surface
        if (len(clusters) >= 2 and clusters[0][0] == 0
                and clusters[-1][-1] == self.n_rays - 1
                and abs(ranges[0] - ranges[self.n_rays - 1]) < self.cluster_threshold):
            clusters[0] = clusters[-1] + clusters[0]
            clusters.pop()
        return clusters

    def _fit_circle(self, pts):
        """Algebraic LS circle fit. Returns (center, radius) or None."""
        if len(pts) < 3:
            return None
        A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
        b = (pts ** 2).sum(axis=1)
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        cx, cy = sol[0], sol[1]
        rsq = sol[2] + cx * cx + cy * cy
        if rsq <= 1e-6:
            return None
        r = float(np.sqrt(rsq))
        if r > self.max_fit_radius:
            return None
        return (np.array([cx, cy]), r)

    def perceive(self, x, true_obstacles, rng):
        """Full pipeline: scan -> cluster -> fit. Returns list of (center, radius)."""
        ranges, hit = self.scan(x, true_obstacles, rng)
        hits = np.stack([x[0] + ranges * self._cos,
                         x[1] + ranges * self._sin], axis=1)
        clusters = self._cluster_rays(ranges, hit)
        est = []
        for idxs in clusters:
            if len(idxs) < self.min_cluster_size:
                continue
            fit = self._fit_circle(hits[idxs])
            if fit is not None:
                est.append(fit)
        return est

    def perceive_with_velocity(self, x, true_obstacles, rng, dt):
        """Perceive + estimate per-obstacle velocity by matching to previous frame."""
        current = self.perceive(x, true_obstacles, rng)
        velocities = []
        for c_cur, _ in current:
            best_v = np.zeros(2)
            best_d = self.match_threshold
            for c_prev, _ in self._last_obstacles:
                d = float(np.linalg.norm(c_cur - c_prev))
                if d < best_d:
                    best_d = d
                    best_v = (c_cur - c_prev) / dt
            velocities.append(best_v)
        self._last_obstacles = current
        return current, velocities


# ---------- SDF and smoothed h ----------
def compute_sdf(x, obstacles):
    """Returns (sdf, grad_sdf) for the closest obstacle. ||grad_sdf|| = 1."""
    best_sdf = np.inf
    best_grad = np.zeros(2)
    for c_obs, r_obs in obstacles:
        diff = x - c_obs
        d = float(np.linalg.norm(diff)) + 1e-9
        sdf_i = d - r_obs
        if sdf_i < best_sdf:
            best_sdf = sdf_i
            best_grad = diff / d
    return best_sdf, best_grad


def compute_h_smooth(x, obstacles, lam, gamma):
    """Returns (h_smooth, grad_h_smooth)."""
    sdf, grad_sdf = compute_sdf(x, obstacles)
    exp_term = np.exp(-gamma * sdf)
    h = lam * (1.0 - exp_term)
    grad_h = lam * gamma * exp_term * grad_sdf
    return h, grad_h


# ---------- CBF safety filter (single smoothed-SDF constraint) ----------
# alpha, phi may each be a scalar OR a callable of h (and possibly more context).
# If obs_velocities is provided, the filter accounts for obstacle motion via
# the L_f h term: ḣ = A·u − A·ċ_obs (the obstacle drift adds to ḣ).
def _eval(term, h):
    return float(term(h)) if callable(term) else float(term)


def safety_filter(x, u_nom, obstacles, alpha, phi, lam, gamma,
                  obs_velocities=None, a=0.0, b=0.0, c=0.0, n_picard=5):
    """Full TISSf-style filter:
        slack(u) = L_f h + A·u + α·(h − c) − φ·‖A‖² − a − b·‖u‖   ≥ 0
    The b·‖u‖ term makes the constraint a second-order cone, not a half-space.
    We handle this with a Picard iteration: project assuming the current ‖u‖,
    recompute ‖u‖ from the projected point, repeat. n_picard=5 converges
    in a few passes for any reasonable b."""
    h, A = compute_h_smooth(x, obstacles, lam, gamma)
    alpha_v = _eval(alpha, h)
    phi_v   = _eval(phi,   h)
    a_v     = _eval(a,     h)
    b_v     = _eval(b,     h)
    c_v     = _eval(c,     h)
    A_sq = float(A @ A)

    # L_f h from obstacle motion
    L_f_h = 0.0
    if obs_velocities is not None and len(obs_velocities) == len(obstacles):
        sdfs = [float(np.linalg.norm(x - c_obs)) - r for c_obs, r in obstacles]
        idx = int(np.argmin(sdfs))
        L_f_h = -float(A @ obs_velocities[idx])

    # Constant part of slack (independent of u):
    #   slack(u) = const + A·u − b·‖u‖
    const = L_f_h + alpha_v * (h - c_v) - phi_v * A_sq - a_v
    A_dot_u_nom = float(A @ u_nom)

    # Fast path: b ~ 0 → single closed-form half-space projection
    if b_v < 1e-6:
        slack = const + A_dot_u_nom
        if slack >= 0:
            return u_nom
        return u_nom - (slack / (A_sq + 1e-9)) * A

    # Picard iteration: u_{k+1} = project(u_nom | A·u ≥ −const + b·‖u_k‖)
    u = u_nom.copy()
    for _ in range(n_picard):
        u_norm = float(np.linalg.norm(u))
        rhs = -const + b_v * u_norm       # required A·u
        gap = rhs - A_dot_u_nom            # how much A·u_nom misses by
        if gap <= 0:
            # u_nom already satisfies even with current ‖u‖ — done
            return u_nom
        u = u_nom + (gap / (A_sq + 1e-9)) * A
    return u


# ---------- Rollout ----------
# Reward constants — single source of truth, easy to tune in one place.
# Designed so: success >> 0,  collision very negative,  timeout moderately negative.
# Progress shaping densifies the signal so freezing is never a stable attractor.
PROGRESS_K   = 10.0     # per-meter-of-progress dense reward
TIME_PEN     = 0.05     # per-step
COLLISION_P  = 100.0
GOAL_BONUS   = 20.0
TIMEOUT_P    = 30.0


def rollout(env, ctrl, starts, rng, perception=None, adv_prob=0.0):
    """ctrl is a dict {'alpha':..., 'phi':..., 'a':..., 'b':..., 'c':...} where each
    value is a scalar or a callable(h). Missing keys default to 0 (or 1 for alpha).
    adv_prob > 0 injects adversarial actions into the nominal."""
    alpha = ctrl.get('alpha', 1.0)
    phi   = ctrl.get('phi',   0.0)
    a     = ctrl.get('a',     0.0)
    b     = ctrl.get('b',     0.0)
    c     = ctrl.get('c',     0.0)
    total_r, trajs = 0.0, []
    for x0 in starts:
        x = env.reset(x0, rng)
        if perception is not None:
            perception.reset_tracking()
        traj = [x.copy()]
        prev_dist = float(np.linalg.norm(x - env.goal))
        r = 0.0
        terminated = False
        for _ in range(env.max_steps):
            u_nom = adversarial_u_nom(x, env, rng, adv_prob)
            if perception is not None:
                obs_for_filter, vels = perception.perceive_with_velocity(
                    x, env.obstacles, rng, env.dt)
                if not obs_for_filter:
                    u = u_nom
                else:
                    u = safety_filter(x, u_nom, obs_for_filter, alpha, phi,
                                      env.h_lambda, env.h_gamma,
                                      obs_velocities=vels, a=a, b=b, c=c)
            else:
                u = safety_filter(x, u_nom, env.obstacles, alpha, phi,
                                  env.h_lambda, env.h_gamma, a=a, b=b, c=c)
            x, _, info = env.step(u, rng)
            traj.append(x.copy())
            cur_dist = float(np.linalg.norm(x - env.goal))
            r += PROGRESS_K * (prev_dist - cur_dist) - TIME_PEN
            prev_dist = cur_dist
            if info['collided']:
                r -= COLLISION_P
                terminated = True
                break
            if info['reached']:
                r += GOAL_BONUS
                terminated = True
                break
        if not terminated:
            r -= TIMEOUT_P        # explicit terminal penalty: dithering is now expensive
        total_r += r
        trajs.append(np.array(traj))
    return total_r / len(starts), trajs


# ---------- 1D CEM: learn phi only, alpha fixed ----------
def cem_phi_only(env, alpha_fixed, n_iters=25, pop=30, elite_frac=0.25, seed=0,
                 n_seeds_per_eval=15, verbose=False):
    rng = np.random.default_rng(seed)
    mu = np.log(0.05)
    sigma = 1.0
    n_elite = max(2, int(pop * elite_frac))
    starts = [np.array([0.0, 0.4]), np.array([0.0, -0.4]), np.array([0.0, 0.0])]
    history = []
    for it in range(n_iters):
        log_phis = rng.standard_normal(pop) * sigma + mu
        rewards = np.array([
            np.mean([
                rollout(env, alpha_fixed, float(np.exp(lp)), starts,
                        np.random.default_rng(it * 1000 + k))[0]
                for k in range(n_seeds_per_eval)
            ])
            for lp in log_phis
        ])
        elite = log_phis[np.argsort(rewards)[-n_elite:]]
        mu = float(elite.mean())
        sigma = float(elite.std()) + 1e-2
        history.append((mu, rewards.mean(), rewards.max()))
        if verbose:
            print(f"  iter {it:2d}  meanR={rewards.mean():6.2f}  "
                  f"phi={np.exp(mu):.4f}")
    return np.exp(mu), history


# ---------- CEM (2D) ----------
def cem(env, n_iters=25, pop=40, elite_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.array([np.log(0.5), np.log(0.05)])  # start: alpha=0.5, phi=0.05
    sigma = np.array([1.0, 1.0])
    n_elite = max(2, int(pop * elite_frac))
    starts = [np.array([0.0, 0.4]), np.array([0.0, -0.4]), np.array([0.0, 0.0])]
    history = []
    for it in range(n_iters):
        samples = rng.standard_normal((pop, 2)) * sigma + mu
        # Average reward over a few noise seeds for robustness
        rewards = np.array([
            np.mean([rollout(env, float(np.exp(s[0])), float(np.exp(s[1])),
                             starts, np.random.default_rng(it * 1000 + k))[0]
                     for k in range(5)])
            for s in samples
        ])
        elite = samples[np.argsort(rewards)[-n_elite:]]
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 1e-2
        history.append((mu.copy(), rewards.mean(), rewards.max()))
        print(f"iter {it:2d}  meanR={rewards.mean():7.2f}  maxR={rewards.max():7.2f}  "
              f"alpha={np.exp(mu[0]):.3f}  phi={np.exp(mu[1]):.3f}")
    return mu, history


# ---------- Plot ----------
def _plot_world(ax, env, theta, trajs, title):
    th = np.linspace(0, 2 * np.pi, 100)
    alpha_l, phi_l = np.exp(theta[0]), np.exp(theta[1])
    lam, gamma = env.h_lambda, env.h_gamma
    # Filter at u=0 binds where alpha*h = phi*||L_g h||^2
    #   alpha*lam*(1-e^{-gamma s}) = phi*(lam*gamma)^2 e^{-2 gamma s}
    # Solve for s by bisection; if it falls outside [-r, 10], skip.
    def buffer_sdf():
        s_lo, s_hi = 0.0, 5.0
        f = lambda s: alpha_l * lam * (1 - np.exp(-gamma * s)) \
                      - phi_l * (lam * gamma) ** 2 * np.exp(-2 * gamma * s)
        if f(s_lo) >= 0 or f(s_hi) <= 0:
            return None
        for _ in range(40):
            m = 0.5 * (s_lo + s_hi)
            (s_lo, s_hi) = (m, s_hi) if f(m) < 0 else (s_lo, m)
        return 0.5 * (s_lo + s_hi)
    s_buf = buffer_sdf()
    for i, (c, r) in enumerate(env.obstacles):
        ax.fill(c[0] + r * np.cos(th), c[1] + r * np.sin(th),
                color='red', alpha=0.3, label='obstacle' if i == 0 else None)
        if s_buf is not None:
            eff_r = r + s_buf
            ax.plot(c[0] + eff_r * np.cos(th), c[1] + eff_r * np.sin(th),
                    'r--', alpha=0.5,
                    label=f'buffer (+{s_buf:.2f})' if i == 0 else None)
    ax.plot(*env.goal, 'g*', markersize=18, label='goal')
    for tr in trajs:
        ax.plot(tr[:, 0], tr[:, 1], '-', lw=1.3, alpha=0.8)
        ax.plot(tr[0, 0], tr[0, 1], 'bo', markersize=4)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title(title)
    ax.grid(alpha=0.3)


def plot_comparison(env_lo, theta_lo, hist_lo, env_hi, theta_hi, hist_hi,
                    path='/Users/chrisliang8/Desktop/lastChance/result.png'):
    _, axes = plt.subplots(2, 2, figsize=(13, 9))
    starts = [np.array([0.0, y]) for y in (-0.4, -0.2, 0.0, 0.2, 0.4)]

    # Use many noise seeds per start to visualize the trajectory distribution
    def many_trajs(env, theta, seeds=8):
        all_tr = []
        for k in range(seeds):
            _, tr = rollout(env, float(np.exp(theta[0])), float(np.exp(theta[1])),
                            starts, np.random.default_rng(1000 + k))
            all_tr.extend(tr)
        return all_tr

    _plot_world(axes[0, 0], env_lo, theta_lo, many_trajs(env_lo, theta_lo),
                f"LOW noise={env_lo.noise_std}  →  "
                f"alpha={np.exp(theta_lo[0]):.2f}, phi={np.exp(theta_lo[1]):.3f}")
    _plot_world(axes[0, 1], env_hi, theta_hi, many_trajs(env_hi, theta_hi),
                f"HIGH noise={env_hi.noise_std}  →  "
                f"alpha={np.exp(theta_hi[0]):.2f}, phi={np.exp(theta_hi[1]):.3f}")

    for ax, hist, label in [(axes[1, 0], hist_lo, 'low'),
                            (axes[1, 1], hist_hi, 'high')]:
        ax.plot([h[1] for h in hist], label='mean (pop)')
        ax.plot([h[2] for h in hist], label='max (pop)')
        ax.set_xlabel('CEM iteration')
        ax.set_ylabel('reward')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title(f'CEM training — {label}')

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"saved {path}")


def plot_sweep(noise_levels, phis, alpha_fixed, env_template, obstacles,
               path='/Users/chrisliang8/Desktop/lastChance/result.png'):
    _, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: phi vs noise curve
    ax = axes[0]
    ax.plot(noise_levels, phis, 'o-', lw=2, color='steelblue')
    ax.set_xlabel('noise std (sigma)')
    ax.set_ylabel('learned phi')
    ax.set_title(f'phi adapts to noise (alpha fixed at {alpha_fixed})')
    ax.grid(alpha=0.3)

    # Right: sample trajectories at the highest noise level, with learned phi
    ax = axes[1]
    env = SingleIntegratorEnv(obstacles=obstacles, noise_std=noise_levels[-1],
                              h_smooth_lambda=env_template.h_lambda,
                              h_smooth_gamma=env_template.h_gamma)
    theta = (np.log(alpha_fixed), np.log(phis[-1]))
    starts = [np.array([0.0, y]) for y in (-0.4, -0.2, 0.0, 0.2, 0.4)]
    all_tr = []
    for k in range(8):
        _, tr = rollout(env, alpha_fixed, phis[-1], starts,
                        np.random.default_rng(1000 + k))
        all_tr.extend(tr)
    _plot_world(ax, env, theta, all_tr,
                f"sigma={noise_levels[-1]}, alpha={alpha_fixed}, learned phi={phis[-1]:.3f}")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"saved {path}")


# ---------- Baselines from arXiv 2103.08041 ----------
def phi_issf(eps_0):
    """Example 2: constant phi = 1/eps_0."""
    return 1.0 / eps_0


def phi_tissf(eps_0, lam_e):
    """Example 3: state-dependent phi(h) = 1/(eps_0 * exp(lam_e * h))."""
    return lambda h: 1.0 / (eps_0 * np.exp(lam_e * h))


# ---------- OUR state-dependent learned policy ----------
# Features per term: [intercept, h, sigma, drift_std, g_eps, adv_prob] = 6 features.
# Each of {alpha, phi, a, b, c} has 6 params → 30-vector theta.
TERMS_5 = ['alpha', 'phi', 'a', 'b', 'c']
N_FEATURES = 6
THETA_DIM = N_FEATURES * len(TERMS_5)    # 30


def _term_fn(theta, idx_block, sigma, drift_std, g_eps, adv_prob):
    """Closure: term(h) = exp(b0 + b1·h + b2·σ + b3·drift + b4·g_eps + b5·adv_prob)."""
    c = theta[idx_block:idx_block + N_FEATURES]
    base = float(c[0] + c[2] * sigma + c[3] * drift_std
                 + c[4] * g_eps + c[5] * adv_prob)
    w_h = float(c[1])
    return lambda h, base=base, w_h=w_h: float(np.exp(base + w_h * h))


def make_ctrl(theta, sigma, drift_std=0.0, g_eps=0.0, adv_prob=0.0):
    """Build a 5-term controller dict from theta + current uncertainty levels."""
    return {term: _term_fn(theta, N_FEATURES * i, sigma, drift_std, g_eps, adv_prob)
            for i, term in enumerate(TERMS_5)}


# ---------- Adversarial nominal controller ----------
def adversarial_u_nom(x, env, rng, adv_prob, k=2.0):
    """Mostly goal-seeking, but with probability adv_prob aims at the nearest
    obstacle's center. Simulates a sloppy / occasionally hostile teleoperator."""
    if adv_prob > 0 and rng.uniform() < adv_prob and env.obstacles:
        dists = [float(np.linalg.norm(x - c)) for c, r in env.obstacles]
        idx = int(np.argmin(dists))
        c_obs, _ = env.obstacles[idx]
        return k * (c_obs - x)
    return p_controller(x, env.goal, k=k)


# ---------- Scenes ----------
SCENES = {
    'open':     {'obstacles': [((2.5, 0.0), 0.6)],
                 'goal': (5.0, 0.0)},
    'spath':    {'obstacles': [((1.8, 0.5), 0.6), ((3.5, -0.5), 0.6)],
                 'goal': (5.0, 0.0)},
    'corridor': {'obstacles': [((2.5, 0.9), 0.775), ((2.5, -0.9), 0.775)],  # gap ~0.25m
                 'goal': (5.0, 0.0)},
    'slalom':   {'obstacles': [((1.5, 0.45), 0.5),
                               ((3.0, -0.45), 0.5),
                               ((4.0, 0.45), 0.5)],
                 'goal': (5.5, 0.0)},
    # Held-out, designed to break uniform-φ baselines:
    'narrow':   {'obstacles': [((2.5, 0.9), 0.825), ((2.5, -0.9), 0.825)],  # gap ~0.15m
                 'goal': (5.0, 0.0)},
    'gauntlet': {'obstacles': [((1.5,  0.45), 0.4),
                               ((2.5, -0.45), 0.4),
                               ((3.5,  0.45), 0.4),
                               ((4.5, -0.45), 0.4)],
                 'goal': (6.0, 0.0)},
}


def make_env(scene_name, noise_std, h_lambda=1.0, h_gamma=2.0,
             obs_drift_std=0.0, g_eps=0.0):
    sc = SCENES[scene_name]
    return SingleIntegratorEnv(obstacles=sc['obstacles'], goal=sc['goal'],
                               noise_std=noise_std,
                               h_smooth_lambda=h_lambda, h_smooth_gamma=h_gamma,
                               obs_drift_std=obs_drift_std, g_eps=g_eps)


# ---------- Multi-scene, multi-noise CEM (6D) ----------
def cem_state_dep(scene_names, sigmas, n_iters=25, pop=60, elite_frac=0.25,
                  n_seeds_per_eval=3, seed=0, verbose=False, perception=None,
                  drift_range=(0.0, 0.0), g_eps_range=(0.0, 0.0),
                  adv_range=(0.0, 0.0)):
    """Learn 30-D theta (5 terms × 6 features each).
    Per evaluation we sample sigma, drift, g_eps, adv_prob from their ranges and
    pass them BOTH to the env (as actual uncertainty) and to the policy (as
    features). The controller knows its operating regime."""
    rng = np.random.default_rng(seed)
    mu = np.zeros(THETA_DIM)
    initials = [3.0, 0.1, 0.05, 0.05, 0.05]
    for i, v in enumerate(initials):
        mu[N_FEATURES * i] = np.log(v)
    sig = np.full(THETA_DIM, 0.5)
    sig[0::N_FEATURES] = 0.7
    n_elite = max(2, int(pop * elite_frac))
    starts = [np.array([0.0, y]) for y in (-0.4, -0.2, 0.0, 0.2, 0.4)]
    history = []
    for it in range(n_iters):
        samples = rng.standard_normal((pop, THETA_DIM)) * sig + mu
        rewards = []
        for theta in samples:
            r = 0.0
            n = 0
            for sname in scene_names:
                for sigma in sigmas:
                    for k in range(n_seeds_per_eval):
                        rngk = np.random.default_rng(it * 10000 + k)
                        # Each uncertainty type independently 50% ON / 50% OFF
                        # so the policy sees all 8 corners (incl. fully-clean world).
                        drift = (rngk.uniform(drift_range[0], drift_range[1])
                                 if rngk.uniform() < 0.5 else 0.0)
                        g_eps = (rngk.uniform(g_eps_range[0], g_eps_range[1])
                                 if rngk.uniform() < 0.5 else 0.0)
                        adv   = (rngk.uniform(adv_range[0],   adv_range[1])
                                 if rngk.uniform() < 0.5 else 0.0)
                        env = make_env(sname, sigma,
                                       obs_drift_std=drift, g_eps=g_eps)
                        ctrl = make_ctrl(theta, sigma,
                                         drift_std=drift, g_eps=g_eps,
                                         adv_prob=adv)
                        rv, _ = rollout(env, ctrl, starts, rngk,
                                        perception=perception, adv_prob=adv)
                        r += rv
                        n += 1
            rewards.append(r / n)
        rewards = np.array(rewards)
        elite = samples[np.argsort(rewards)[-n_elite:]]
        mu = elite.mean(axis=0)
        sig = elite.std(axis=0) + 0.05
        history.append((mu.copy(), float(rewards.mean()), float(rewards.max())))
        if verbose:
            # Show intercepts (term values at h=0, σ=0, drift=0, g_eps=0)
            ints = [np.exp(mu[N_FEATURES * i]) for i in range(5)]
            print(f"iter {it:2d} R={rewards.mean():6.2f}  "
                  f"α0={ints[0]:.2f} φ0={ints[1]:.3f} "
                  f"a0={ints[2]:.3f} b0={ints[3]:.3f} c0={ints[4]:.3f}")
    return mu, history


# ---------- Grid comparison plot ----------
def plot_grid(results, scene_names, controller_names, sigma, alpha_fixed,
              path='/Users/chrisliang8/Desktop/lastChance/result.png'):
    n_rows = len(scene_names)
    n_cols = len(controller_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    th = np.linspace(0, 2 * np.pi, 100)
    for i, sname in enumerate(scene_names):
        sc = SCENES[sname]
        for j, cname in enumerate(controller_names):
            ax = axes[i, j]
            res = results[sname][cname]
            for c, r in sc['obstacles']:
                ax.fill(c[0] + r * np.cos(th), c[1] + r * np.sin(th),
                        color='red', alpha=0.3)
            ax.plot(*sc['goal'], 'g*', markersize=14)
            for tr in res['trajs']:
                color = 'C3' if res['collision_rate'] > 0.5 else 'C0'
                ax.plot(tr[:, 0], tr[:, 1], '-', lw=0.8, alpha=0.5, color=color)
            ax.set_aspect('equal')
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(cname, fontsize=9)
            if j == 0:
                ax.set_ylabel(sname, fontsize=11)
            ax.text(0.02, 0.98,
                    f"SAFETY\n"
                    f" coll {res['collision_rate']*100:.0f}%\n"
                    f" unsafe% {res['time_unsafe_frac']*100:.1f}\n"
                    f" min h {res['min_h']:+.2f}\n"
                    f"PERF\n"
                    f" reach {res['success_rate']*100:.0f}%\n"
                    f" t {res['mean_time']:.2f}s\n"
                    f" detour {res['mean_detour_ratio']:.2f}",
                    transform=ax.transAxes, fontsize=7,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    fig.suptitle(f"controllers vs scenes  (sigma={sigma}, alpha_fixed={alpha_fixed})",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    print(f"saved {path}")


# ---------- Controller evaluator ----------
def evaluate(env, ctrl, n_seeds=30, perception=None, adv_prob=0.0):
    """ctrl is a dict {'alpha':..., 'phi':..., 'a':..., 'b':..., 'c':...}.
    adv_prob > 0: nominal sometimes aims at nearest obstacle (sloppy teleop)."""
    alpha = ctrl.get('alpha', 1.0)
    phi   = ctrl.get('phi',   0.0)
    a     = ctrl.get('a',     0.0)
    b     = ctrl.get('b',     0.0)
    c     = ctrl.get('c',     0.0)
    starts = [np.array([0.0, y]) for y in (-0.4, -0.2, 0.0, 0.2, 0.4)]
    collisions = 0
    successes_t = []
    min_hs = []
    unsafe_fracs = []
    path_lens = []
    detour_ratios = []
    sample_trajs = []
    total = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        for x0 in starts:
            total += 1
            x = env.reset(x0, rng)
            if perception is not None:
                perception.reset_tracking()
            traj = [x.copy()]
            min_h = np.inf
            steps_unsafe = 0
            steps_taken = 0
            path_len = 0.0
            x_prev = x.copy()
            initial_dist = float(np.linalg.norm(x - env.goal))
            for t in range(env.max_steps):
                u_nom = adversarial_u_nom(x, env, rng, adv_prob)
                if perception is not None:
                    obs_for_filter, vels = perception.perceive_with_velocity(
                        x, env.obstacles, rng, env.dt)
                    if not obs_for_filter:
                        u = u_nom
                    else:
                        u = safety_filter(x, u_nom, obs_for_filter, alpha, phi,
                                          env.h_lambda, env.h_gamma,
                                          obs_velocities=vels, a=a, b=b, c=c)
                else:
                    u = safety_filter(x, u_nom, env.obstacles, alpha, phi,
                                      env.h_lambda, env.h_gamma, a=a, b=b, c=c)
                x, _, info = env.step(u, rng)
                traj.append(x.copy())
                # Track path length
                path_len += float(np.linalg.norm(x - x_prev))
                x_prev = x.copy()
                steps_taken = t + 1
                # Safety: always against the true geometry
                h_val, _ = compute_h_smooth(x, env.obstacles, env.h_lambda, env.h_gamma)
                if h_val < min_h:
                    min_h = h_val
                if h_val < 0:
                    steps_unsafe += 1
                if info['collided']:
                    collisions += 1
                    break
                if info['reached']:
                    successes_t.append((t + 1) * env.dt)
                    break
            min_hs.append(min_h)
            unsafe_fracs.append(steps_unsafe / max(steps_taken, 1))
            path_lens.append(path_len)
            detour_ratios.append(path_len / max(initial_dist, 1e-6))
            if seed < 6:
                sample_trajs.append(np.array(traj))
    return {
        # ----- performance -----
        'success_rate':       len(successes_t) / total,
        'mean_time':          float(np.mean(successes_t)) if successes_t else float('nan'),
        'mean_path_length':   float(np.mean(path_lens)),
        'mean_detour_ratio':  float(np.mean(detour_ratios)),
        # ----- safety -----
        'collision_rate':     collisions / total,
        'min_h':              float(np.min(min_hs)),
        'mean_min_h':         float(np.mean(min_hs)),
        'time_unsafe_frac':   float(np.mean(unsafe_fracs)),
        'trajs':              sample_trajs,
    }


def plot_baselines(env, results, alpha, path='/Users/chrisliang8/Desktop/lastChance/result.png'):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        # Visualize obstacles
        th = np.linspace(0, 2 * np.pi, 100)
        for i, (c, r) in enumerate(env.obstacles):
            ax.fill(c[0] + r * np.cos(th), c[1] + r * np.sin(th),
                    color='red', alpha=0.3, label='obstacle' if i == 0 else None)
        ax.plot(*env.goal, 'g*', markersize=18, label='goal')
        for tr in res['trajs']:
            color = 'C3' if res['collision_rate'] > 0.5 else 'C0'
            ax.plot(tr[:, 0], tr[:, 1], '-', lw=1.0, alpha=0.5, color=color)
            ax.plot(tr[0, 0], tr[0, 1], 'bo', markersize=3)
        ax.set_aspect('equal')
        ax.legend(loc='upper left', fontsize=7)
        ax.set_title(
            f"{name}\n"
            f"collide={res['collision_rate']*100:.0f}%  "
            f"reach={res['success_rate']*100:.0f}%\n"
            f"time={res['mean_time']:.2f}s  min h={res['min_h']:+.3f}",
            fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle(f"sigma={env.noise_std}, alpha={alpha}", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"saved {path}")


def plot_perception_demo(scene_names, perception,
                         path='/Users/chrisliang8/Desktop/lastChance/perception.png'):
    """Sanity-check: for each scene, draw true obstacles, lidar rays, and fitted circles."""
    rng = np.random.default_rng(0)
    _, axes = plt.subplots(1, len(scene_names),
                           figsize=(4.5 * len(scene_names), 4.5), squeeze=False)
    th = np.linspace(0, 2 * np.pi, 100)
    x = np.array([0.0, 0.0])      # robot at origin
    for ax, sname in zip(axes[0], scene_names):
        sc = SCENES[sname]
        ranges, hit = perception.scan(x, sc['obstacles'], rng)
        # True obstacles
        for c, r in sc['obstacles']:
            ax.fill(c[0] + r * np.cos(th), c[1] + r * np.sin(th),
                    color='red', alpha=0.3, label='true')
        # Lidar rays
        for i in range(perception.n_rays):
            if hit[i]:
                ax.plot([x[0], x[0] + ranges[i] * perception._cos[i]],
                        [x[1], x[1] + ranges[i] * perception._sin[i]],
                        'g-', lw=0.3, alpha=0.4)
        # Hit points
        hpx = x[0] + ranges * perception._cos
        hpy = x[1] + ranges * perception._sin
        ax.scatter(hpx[hit], hpy[hit], s=4, c='green', label='lidar pts')
        # Fitted obstacles
        est = perception.perceive(x, sc['obstacles'], rng)
        for k, (c, r) in enumerate(est):
            ax.plot(c[0] + r * np.cos(th), c[1] + r * np.sin(th),
                    'b--', lw=1.5,
                    label='fit' if k == 0 else None)
            ax.plot(*c, 'b+', markersize=8)
        ax.plot(*x, 'k^', markersize=10, label='robot')
        ax.set_aspect('equal')
        ax.set_xlim(-1, 7)
        ax.set_ylim(-3, 3)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f"{sname}: {len(est)} obstacle(s) fitted")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    print(f"saved {path}")


if __name__ == "__main__":
    # ----- Step 0: sanity-check the perception pipeline -----
    perception = LidarPerception(n_rays=180, max_range=6.0,
                                 range_noise_std=0.02, cluster_threshold=0.35)
    print("Visualizing perception on each scene...")
    plot_perception_demo(['open', 'spath', 'corridor', 'slalom'], perception)

    # ----- Step 1: train 6-feature 5-term policy under ALL uncertainty types -----
    train_scenes = ['open', 'spath', 'corridor']
    train_sigmas = [0.3, 0.6]
    drift_range = (0.0, 0.2)
    g_eps_range = (0.0, 0.15)
    adv_range   = (0.0, 0.25)
    print(f"\nTraining 6-feat 5-term policy: scenes={train_scenes}, "
          f"sigmas={train_sigmas}, drift~U{drift_range}, "
          f"g_eps~U{g_eps_range}, adv~U{adv_range} ...")
    theta_star, _ = cem_state_dep(train_scenes, train_sigmas,
                                  n_iters=25, pop=50, n_seeds_per_eval=2, seed=7,
                                  verbose=True, perception=perception,
                                  drift_range=drift_range, g_eps_range=g_eps_range,
                                  adv_range=adv_range)

    print("\nLearned 5-term policy "
          "(each = exp(b0 + b1·h + b2·σ + b3·drift + b4·g_eps + b5·adv_prob)):")
    for i, t in enumerate(TERMS_5):
        c = theta_star[N_FEATURES * i:N_FEATURES * (i + 1)]
        v_clean    = np.exp(c[0] + 0.6 * c[2])
        v_advdrift = np.exp(c[0] + 0.6 * c[2] + 0.15 * c[3]
                            + 0.10 * c[4] + 0.20 * c[5])
        print(f"  {t:5s}: int={c[0]:+.2f}  h:{c[1]:+.2f}  σ:{c[2]:+.2f}  "
              f"drift:{c[3]:+.2f}  g_eps:{c[4]:+.2f}  adv:{c[5]:+.2f}  "
              f"→ clean(σ=.6)={v_clean:.3f}  advdrift={v_advdrift:.3f}")

    # ----- Step 2: evaluate -----
    alpha_for_baselines = 3.0
    # All 6 scenes; narrow + gauntlet are designed to break uniform-φ baselines
    test_scenes = ['open', 'spath', 'corridor', 'slalom', 'narrow', 'gauntlet']

    def controllers_at(sigma, drift, g_eps, adv):
        return {
            "ISSf eps0=10\n(phi=0.10)":  {'alpha': alpha_for_baselines,
                                          'phi':   phi_issf(10.0)},
            "ISSf eps0=1\n(phi=1.00)":   {'alpha': alpha_for_baselines,
                                          'phi':   phi_issf(1.0)},
            "TISSf eps0=1, lam=3":       {'alpha': alpha_for_baselines,
                                          'phi':   phi_tissf(1.0, 3.0)},
            "OURS 6-feat\n(adaptive)":   make_ctrl(theta_star, sigma,
                                                   drift_std=drift, g_eps=g_eps,
                                                   adv_prob=adv),
        }

    # (label, sigma, drift, g_eps, adv_prob)
    test_configs = [
        ('static',      0.6, 0.0,  0.0,  0.0),
        ('drift',       0.6, 0.15, 0.10, 0.0),
        ('adversarial', 0.6, 0.0,  0.0,  0.20),
        ('worst_case',  0.6, 0.20, 0.15, 0.20),
    ]

    for label, ts, td, tg, ta in test_configs:
        print(f"\n=== {label}  σ={ts}  drift={td}  g_eps={tg}  adv={ta} ===")
        controllers = controllers_at(ts, td, tg, ta)
        results = {sname: {} for sname in test_scenes}
        for sname in test_scenes:
            env = make_env(sname, ts, obs_drift_std=td, g_eps=tg)
            held_out = '*HELD-OUT*' if sname not in train_scenes else ''
            for cname, ctrl in controllers.items():
                res = evaluate(env, ctrl, n_seeds=20,
                               perception=perception, adv_prob=ta)
                results[sname][cname] = res
                print(f"  {sname:10s}{held_out:11s} | "
                      f"{cname.split(chr(10))[0]:25s} "
                      f"coll={res['collision_rate']*100:4.0f}%  "
                      f"unsafe%={res['time_unsafe_frac']*100:4.1f}  "
                      f"reach={res['success_rate']*100:4.0f}%  "
                      f"t={res['mean_time']:.2f}s  "
                      f"detour={res['mean_detour_ratio']:.2f}  "
                      f"minh={res['min_h']:+.2f}")
        out_path = f'/Users/chrisliang8/Desktop/lastChance/grid_{label}.png'
        plot_grid(results, test_scenes, list(controllers.keys()),
                  ts, alpha_for_baselines, path=out_path)
