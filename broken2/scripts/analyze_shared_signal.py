"""Caveat-#4 analysis — multivariate regression of α and φ on priv features.

Reads the .npz dumped by diagnose_shared_signal.py and prints partial
correlations (standardized regression coefficients with t-stats).

The question: do α and φ heads have INDEPENDENT priv-feature sensitivities,
or is the marginal Pearson(head, tracking_err) hiding everything else?

Standardized β coefficient interpretation:
  β = standardized partial regression coefficient
  |β| > 0.20 with |t| > 2 → meaningful independent effect of that feature
  β ≈ 0 (after controlling for others) → no independent effect

Usage (local):
  python3 scripts/analyze_shared_signal.py data_from_lab/wk3tight8/diagnose_shared_signal_wk3tight8.npz
"""
from __future__ import annotations

import sys
import numpy as np
from pathlib import Path


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)


def multivar_regression(y: np.ndarray, X: np.ndarray, names: list[str]) -> list[dict]:
    """OLS regression y = β · X + ε with all-standardized inputs.

    Returns per-feature dict with standardized β, t-stat, p-value (rough),
    and marginal Pearson with y (for comparison).
    """
    y_std = standardize(y)
    X_std = np.stack([standardize(X[:, j]) for j in range(X.shape[1])], axis=1)

    # OLS via least squares
    beta, residuals, rank, _sv = np.linalg.lstsq(X_std, y_std, rcond=None)

    # Residual variance
    y_hat = X_std @ beta
    resid = y_std - y_hat
    n, p = X_std.shape
    rss = (resid ** 2).sum()
    sigma2 = rss / max(n - p, 1)
    # Covariance of β: σ² (X'X)⁻¹
    XtX_inv = np.linalg.inv(X_std.T @ X_std)
    se_beta = np.sqrt(np.maximum(0.0, np.diag(XtX_inv) * sigma2))

    out = []
    for j, name in enumerate(names):
        # Marginal Pearson with y (unstandardized X uses same correlation)
        x_j = X[:, j]
        mx, my = x_j.mean(), y.mean()
        sx, sy = x_j.std() + 1e-12, y.std() + 1e-12
        marginal_pearson = float(((x_j - mx) * (y - my)).mean() / (sx * sy))
        t_stat = beta[j] / (se_beta[j] + 1e-12)
        out.append({
            "feature": name,
            "marginal_pearson": marginal_pearson,
            "partial_beta_std": float(beta[j]),
            "t_stat": float(t_stat),
            "se_beta": float(se_beta[j]),
        })
    return out


def render_table(rows: list[dict], head_label: str):
    print("")
    print(f"  ── {head_label} ── (standardized partial regression)")
    print(f"  {'feature':<26} {'marginal r':>10}   {'partial β':>10}   {'t-stat':>8}   "
          f"{'meaning':<24}")
    print(f"  {'─'*26} {'─'*10}   {'─'*10}   {'─'*8}   {'─'*24}")
    rows_sorted = sorted(rows, key=lambda r: -abs(r["partial_beta_std"]))
    for r in rows_sorted:
        # ASCII bar for partial β
        b = r["partial_beta_std"]
        bar_len = int(min(20, abs(b) * 40))
        bar_pos = " " * 20
        bar_str = "█" * bar_len
        sign = "+" if b >= 0 else "−"
        meaning = ""
        if abs(b) > 0.20 and abs(r["t_stat"]) > 2:
            meaning = "INDEPENDENT effect"
        elif abs(b) > 0.10 and abs(r["t_stat"]) > 2:
            meaning = "weak independent"
        elif abs(b) < 0.05:
            meaning = "no effect"
        else:
            meaning = "noise"
        print(f"  {r['feature']:<26} {r['marginal_pearson']:>+10.3f}   "
              f"{sign}{abs(b):>9.3f}   {r['t_stat']:>+8.2f}   {meaning:<24}")


def compare_heads(alpha_rows, phi_rows):
    by_feat = {r["feature"]: r for r in alpha_rows}
    print("")
    print("  ── COMPARISON: do α and φ have DIFFERENT independent sensitivities? ──")
    print(f"  {'feature':<26}   {'α partial β':>12}   {'φ partial β':>12}   "
          f"{'difference':>11}   {'interpretation':<28}")
    print(f"  {'─'*26}   {'─'*12}   {'─'*12}   {'─'*11}   {'─'*28}")
    for pr in phi_rows:
        feat = pr["feature"]
        ar = by_feat[feat]
        a_b, p_b = ar["partial_beta_std"], pr["partial_beta_std"]
        diff = p_b - a_b
        # Heuristic interpretation
        if abs(a_b) > 0.15 and abs(p_b) > 0.15 and np.sign(a_b) == np.sign(p_b) and abs(diff) < 0.1:
            interp = "BOTH use it — coupled"
        elif abs(a_b) > 0.15 and abs(p_b) < 0.05:
            interp = "α only"
        elif abs(p_b) > 0.15 and abs(a_b) < 0.05:
            interp = "φ only — specialized"
        elif abs(diff) > 0.2:
            interp = "DIFFERENT magnitudes"
        else:
            interp = "neither"
        print(f"  {feat:<26}   {a_b:>+12.3f}   {p_b:>+12.3f}   "
              f"{diff:>+11.3f}   {interp:<28}")


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_shared_signal.py <path/to/diagnose_shared_signal.npz>")
        sys.exit(1)

    npz_path = Path(sys.argv[1])
    if not npz_path.exists():
        print(f"file not found: {npz_path}")
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)
    task = str(data["task"])
    ckpt = str(data["checkpoint"])
    N = int(data["n_envs"])
    S = int(data["n_steps"])
    print("")
    print("=" * 78)
    print(f"  Caveat-#4 analysis (partial correlations)")
    print(f"  task: {task}")
    print(f"  ckpt: {ckpt}")
    print(f"  N={N} envs × S={S} steps")
    print("=" * 78)

    alpha = data["alpha_per_env"]
    phi = data["phi_per_env"]
    feats = {
        "tracking_err_norm": data["tracking_err_norm_per_env"],
        "actuation_noise_σ": data["actuation_noise_sigma_per_env"],
        "base_height":       data["base_height_per_env"],
        "friction":          data["friction_per_env"],
        "base_mass":         data["base_mass_per_env"],
        "com_norm":          data["com_norm_per_env"],
        "base_ang_vel_norm": data["base_ang_vel_norm_per_env"],
    }
    feat_names = list(feats.keys())
    X = np.stack([feats[k] for k in feat_names], axis=1)

    # Feature-feature correlation matrix (between-env)
    print("")
    print("  ── feature ↔ feature correlations (between-env, to show "
          "tracking_err is the lump-sum symptom) ──")
    F = X
    F_std = np.stack([standardize(F[:, j]) for j in range(F.shape[1])], axis=1)
    R = (F_std.T @ F_std) / F.shape[0]
    print(f"  {'':>20}  " + "  ".join(f"{n[:10]:>10}" for n in feat_names))
    for i, ni in enumerate(feat_names):
        cells = "  ".join(f"{R[i, j]:>+10.3f}" for j in range(len(feat_names)))
        print(f"  {ni:>20}  {cells}")

    # Multivariate regression for α and φ
    alpha_rows = multivar_regression(alpha, X, feat_names)
    phi_rows = multivar_regression(phi, X, feat_names)
    render_table(alpha_rows, "α head — what predicts α_per_env")
    render_table(phi_rows, "φ head — what predicts φ_per_env")
    compare_heads(alpha_rows, phi_rows)

    # Within-episode partial corr (per env): regress α_t on [tracking_err_t, h_t]
    # and φ_t on [tracking_err_t, h_t], per env, then average β across envs.
    alpha_hist = data["alpha_history"]    # (S, N)
    phi_hist = data["phi_history"]
    h_hist = data["h_history"]
    te_hist = data["tracking_err_norm_history"]

    print("")
    print("  ── WITHIN-EPISODE partial regression (per env, averaged) ──")
    print(f"  Regress α_t on [tracking_err_t, h_t] within each env, "
          f"then mean β across envs.")

    def within_partial(y_hist, regressors_hist, names):
        n_envs = y_hist.shape[1]
        betas_all = np.zeros((n_envs, len(regressors_hist)), dtype=np.float64)
        valid = np.zeros(n_envs, dtype=bool)
        for e in range(n_envs):
            y = y_hist[:, e].astype(np.float64)
            if y.std() < 1e-6:
                continue
            X_e = np.stack([standardize(r[:, e].astype(np.float64))
                            for r in regressors_hist], axis=1)
            y_std = standardize(y)
            try:
                b, _, _, _ = np.linalg.lstsq(X_e, y_std, rcond=None)
                betas_all[e] = b
                valid[e] = True
            except Exception:
                continue
        return betas_all[valid], int(valid.sum())

    a_betas, n_valid_a = within_partial(alpha_hist, [te_hist, h_hist],
                                        ["tracking_err_t", "h_t"])
    p_betas, n_valid_p = within_partial(phi_hist, [te_hist, h_hist],
                                        ["tracking_err_t", "h_t"])
    print(f"  α within-ep partial β:  tracking_err_t = {a_betas[:,0].mean():+.3f}"
          f"  h_t = {a_betas[:,1].mean():+.3f}   (n_valid={n_valid_a})")
    print(f"  φ within-ep partial β:  tracking_err_t = {p_betas[:,0].mean():+.3f}"
          f"  h_t = {p_betas[:,1].mean():+.3f}   (n_valid={n_valid_p})")
    print("")
    print("  Read: if α has independent β on tracking_err_t but φ doesn't,")
    print("  the within-episode dynamics are different even though both")
    print("  show similar marginal Pearson on tracking_err.")
    print("=" * 78)


if __name__ == "__main__":
    main()
