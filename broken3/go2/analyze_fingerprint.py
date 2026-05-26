"""Phase 1.5 analyzer.

Loads per-disturbance .npz files produced by `phase1_5_fingerprint_sweep.py`,
constructs (X, y) where X is windowed deployable observations and y is the
disturbance magnitude, and trains a small regressor to see if y is
*perceivable* from X.

Decision rule (the Phase 1.5 GATE):
- R²_test > 0.7 -> Option 1: proprio + action history carry the fingerprint
                  directly. Phase 2 can train with these obs and no teacher.
- R²_test < 0.3 -> Option 2: locomotion launders the disturbance into
                  proprio. Build a teacher-student (RMA-style): teacher
                  trains on priv_obs incl. disturbance; student distills
                  the deployable encoder to match teacher's z.
- 0.3 -- 0.7    -> borderline; lengthen the observation window, then
                  re-evaluate. If still in this band, lean Option 2.

Usage:
    python analyze_fingerprint.py phase1_5_d*.npz [--window 20] [--test_frac 0.2]
"""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np
import torch
import torch.nn as nn


def load_runs(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (obs (T*, N, D), labels (T*,)) concatenated across files.
    Different files may have different T; we stack along the time axis
    after replicating the per-file label."""
    xs, ys = [], []
    for p in sorted(paths):
        z = np.load(p)
        obs = z["obs"]                                # (T, N, D)
        d = float(z["disturbance"])
        xs.append(obs)
        ys.append(np.full((obs.shape[0],), d, dtype=np.float32))
        print(f"  {p}  T={obs.shape[0]}  N={obs.shape[1]}  D={obs.shape[2]}  d={d}")
    X = np.concatenate(xs, axis=0)   # (T_total, N, D)
    y = np.concatenate(ys, axis=0)   # (T_total,)
    return X, y


def build_windows(obs: np.ndarray, labels: np.ndarray, window: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """obs: (T, N, D), labels: (T,). Returns
       X: ((T-window+1)*N, window*D), y: ((T-window+1)*N,).
    Windows are per-env, sliding stride 1."""
    T, N, D = obs.shape
    if window < 1 or window > T:
        raise ValueError(f"window={window} out of [1, T={T}]")
    nw = T - window + 1
    # build sliding windows along T, per-env
    # reshape: per t, per n, the obs[t:t+window, n, :] window
    X = np.zeros((nw, N, window * D), dtype=np.float32)
    for i in range(nw):
        X[i] = obs[i:i + window].transpose(1, 0, 2).reshape(N, window * D)
    y = np.broadcast_to(labels[window - 1:nw + window - 1, None], (nw, N))
    X = X.reshape(-1, window * D)
    y = y.reshape(-1).astype(np.float32)
    return X, y


def fit_linear(Xtr, ytr, Xte, yte, device) -> dict:
    """Closed-form ridge regression as a baseline."""
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    yte_t = torch.tensor(yte, dtype=torch.float32, device=device)

    # solve (X^T X + λI) w = X^T y
    lam = 1e-3
    n_feat = Xtr_t.shape[1]
    A = Xtr_t.T @ Xtr_t + lam * torch.eye(n_feat, device=device)
    b = Xtr_t.T @ ytr_t
    w = torch.linalg.solve(A, b)

    def r2(X, y):
        pred = X @ w
        ss_res = torch.sum((y - pred) ** 2)
        ss_tot = torch.sum((y - y.mean()) ** 2)
        return float(1.0 - (ss_res / (ss_tot + 1e-9)).item())

    return {"r2_train": r2(Xtr_t, ytr_t), "r2_test": r2(Xte_t, yte_t)}


def fit_mlp(Xtr, ytr, Xte, yte, device, hidden=64,
            epochs=60, batch_size=4096) -> dict:
    """Small MLP regressor."""
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    yte_t = torch.tensor(yte, dtype=torch.float32, device=device)

    n_feat = Xtr.shape[1]
    net = nn.Sequential(
        nn.Linear(n_feat, hidden), nn.ELU(),
        nn.Linear(hidden, hidden), nn.ELU(),
        nn.Linear(hidden, 1),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    n = Xtr_t.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        losses = []
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = net(Xtr_t[idx]).squeeze(-1)
            loss = ((pred - ytr_t[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if (ep + 1) % 10 == 0:
            mean_loss = sum(losses) / max(len(losses), 1)
            print(f"    mlp ep {ep+1:3d}/{epochs}  train_mse {mean_loss:.3f}")

    def r2(X, y):
        with torch.no_grad():
            pred = net(X).squeeze(-1)
            ss_res = torch.sum((y - pred) ** 2)
            ss_tot = torch.sum((y - y.mean()) ** 2)
        return float(1.0 - (ss_res / (ss_tot + 1e-9)).item())

    return {"r2_train": r2(Xtr_t, ytr_t), "r2_test": r2(Xte_t, yte_t)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", nargs="+", help=".npz files from the sweep (glob ok)")
    ap.add_argument("--window", type=int, default=20,
                    help="Number of past steps to stack into the observation. "
                         "Larger = more temporal context for the regressor.")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths: list[str] = []
    for p in args.npz:
        m = sorted(glob.glob(p))
        paths.extend(m if m else [p])
    if not paths:
        print("No npz files found.", file=sys.stderr)
        sys.exit(2)

    print(f"[analyze] loading {len(paths)} files:")
    obs, labels = load_runs(paths)
    print(f"[analyze] obs={obs.shape}, unique_d={sorted(set(labels.tolist()))}")

    X, y = build_windows(obs, labels, args.window)
    print(f"[analyze] windowed: X={X.shape}, y={y.shape}, window={args.window}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(y))
    n_test = int(args.test_frac * len(y))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]
    print(f"[analyze] train {Xtr.shape[0]:,}   test {Xte.shape[0]:,}")

    # baseline: predict mean (R²=0 by construction)
    y_train_mean = float(ytr.mean())
    print(f"[analyze] y mean={y_train_mean:.2f}   y std={float(ytr.std()):.2f}")

    print("[analyze] linear (closed-form ridge) ...")
    lin = fit_linear(Xtr, ytr, Xte, yte, device)
    print(f"    R²_train={lin['r2_train']:+.3f}   R²_test={lin['r2_test']:+.3f}")

    print("[analyze] MLP (3-layer, hidden 64) ...")
    mlp = fit_mlp(Xtr, ytr, Xte, yte, device)
    print(f"    R²_train={mlp['r2_train']:+.3f}   R²_test={mlp['r2_test']:+.3f}")

    best_r2 = max(lin["r2_test"], mlp["r2_test"])
    print()
    print("=" * 70)
    print("  Phase 1.5 FINGERPRINT GATE")
    print("=" * 70)
    print(f"  best R²_test                     : {best_r2:+.3f}")
    if best_r2 > 0.7:
        verdict = "STRONG  -> Option 1 (direct training, no teacher needed)"
    elif best_r2 < 0.3:
        verdict = "WEAK    -> Option 2 (RMA-style teacher-student required)"
    else:
        verdict = ("BORDERLINE -> rerun with larger --window; if still in "
                   "[0.3, 0.7], lean Option 2")
    print(f"  verdict                          : {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
