"""Offline student distillation training for V13 two-stream teacher.

Loads a dumped rollout .npz (from dump_teacher_rollout_for_student.py),
slides a temporal window over (proprio, action) history per env, and
trains the StudentAdaptationModule to regress to z_env_target.

This is pure PyTorch — no IsaacLab. Can run on laptop CPU/GPU.

Usage:
  python3 scripts/train_student_v13.py \\
    --dump data_from_lab/dump_v13_for_student.npz \\
    --history_len 50 --batch_size 1024 --epochs 50 \\
    --output checkpoints/student_v13.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Add IsaacLab module so we can import the student class without launching
# IsaacLab itself (the student module has no IsaacLab dependencies).
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(
    0,
    str(THIS_DIR.parent / "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/"
                          "manager_based/safety/cbf_go2"),
)
from cbf_go2_student import StudentAdaptationModule


def build_windows(
    proprio: np.ndarray,      # (S, N, F_p)
    action: np.ndarray,       # (S, N, F_a)
    z_env: np.ndarray,        # (S, N, F_z)
    resets: np.ndarray,       # (S, N) bool
    history_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) windows. X: (M, T, F_p + F_a); y: (M, F_z).

    A window at (env e, step s) is valid iff no reset occurred during the
    history range [s - history_len + 1, s]. Otherwise the history spans
    multiple episodes.
    """
    S, N, _ = proprio.shape
    F_p = proprio.shape[-1]
    F_a = action.shape[-1]
    F_z = z_env.shape[-1]
    T = history_len

    feat = np.concatenate([proprio, action], axis=-1)  # (S, N, F_p+F_a)

    X_list, y_list = [], []
    # Use a sliding window per env. Filter out windows that cross resets.
    for s in range(T - 1, S):
        # window is [s - T + 1, s] inclusive — T steps. Last step is the
        # target's input frame.
        # Reset mask within window — if ANY reset in [s-T+1, s-1] (not
        # the last step itself, since reset_history[s] is whether THIS
        # step caused a reset), drop the window.
        # Use the convention: keep windows where no reset in last T-1 steps.
        reset_window = resets[s - T + 1: s].any(axis=0)  # (N,)
        keep = ~reset_window                              # (N,)
        if not keep.any():
            continue
        win = feat[s - T + 1: s + 1, keep]  # (T, K, F)
        win = win.transpose(1, 0, 2)         # (K, T, F)
        targ = z_env[s, keep]                # (K, F_z)
        X_list.append(win.reshape(-1, T, F_p + F_a))
        y_list.append(targ)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X.astype(np.float32), y.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True, type=str,
                   help="Path to dump_v13_for_student.npz")
    p.add_argument("--history_len", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train_frac", type=float, default=0.85)
    p.add_argument("--device", type=str, default="auto",
                   help="cuda | cpu | auto")
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    print(f"[student] device={device}", flush=True)

    print(f"[student] loading {args.dump}...", flush=True)
    data = np.load(args.dump, allow_pickle=True)
    proprio = data["proprio_history"]
    action = data["action_history"]
    z_env = data["z_env_history"]
    resets = data["reset_history"]
    print(f"  proprio: {proprio.shape}  action: {action.shape}  "
          f"z_env: {z_env.shape}  resets: {resets.shape}", flush=True)

    print(f"[student] building windows (history_len={args.history_len})...", flush=True)
    t0 = time.time()
    X, y = build_windows(proprio, action, z_env, resets, args.history_len)
    print(f"  X={X.shape}  y={y.shape}  built in {time.time()-t0:.1f}s", flush=True)

    # Split.
    M = X.shape[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(M)
    split = int(M * args.train_frac)
    train_idx, test_idx = perm[:split], perm[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"  train={len(X_train)}  test={len(X_test)}", flush=True)

    # Model.
    F_p = proprio.shape[-1]
    F_a = action.shape[-1]
    F_z = z_env.shape[-1]
    model = StudentAdaptationModule(
        proprio_dim=F_p, action_dim=F_a, z_env_dim=F_z,
        history_len=args.history_len,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[student] model: {model.__class__.__name__}  n_params={n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Compute teacher y_test variance as a baseline (R²-style).
    y_test_var = float(((y_test - y_test.mean(axis=0)) ** 2).mean())
    print(f"[student] y_test variance baseline: {y_test_var:.4f}", flush=True)

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    best_test = float("inf")
    history = []
    for epoch in range(args.epochs):
        # Shuffle and mini-batch.
        perm = torch.randperm(len(X_train_t), device=device)
        train_losses = []
        model.train()
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i: i + args.batch_size]
            x_b = X_train_t[idx]
            y_b = y_train_t[idx]
            pred = model(x_b)
            loss = loss_fn(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Eval.
        model.eval()
        with torch.no_grad():
            pred_test = model(X_test_t)
            test_mse = loss_fn(pred_test, y_test_t).item()
        train_mse = float(np.mean(train_losses))
        r2_test = 1.0 - test_mse / max(y_test_var, 1e-12)
        history.append({"epoch": epoch, "train_mse": train_mse,
                         "test_mse": test_mse, "test_r2": r2_test})
        print(f"  epoch {epoch:>3}  train_mse={train_mse:.5f}  "
              f"test_mse={test_mse:.5f}  test_R²={r2_test:.4f}", flush=True)

        if test_mse < best_test:
            best_test = test_mse
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "proprio_dim": F_p, "action_dim": F_a,
                    "z_env_dim": F_z, "history_len": args.history_len,
                },
                "best_test_mse": best_test,
                "best_test_r2": r2_test,
                "history": history,
            }, args.output)

    # Final summary.
    print(f"\n[student] best test MSE: {best_test:.5f}", flush=True)
    print(f"[student] best test R²: {1.0 - best_test / max(y_test_var, 1e-12):.4f}",
          flush=True)
    print(f"[student] saved to {args.output}", flush=True)

    # Also write a JSON summary for plotting.
    summary_path = Path(args.output).with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump({
            "best_test_mse": best_test,
            "best_test_r2": 1.0 - best_test / max(y_test_var, 1e-12),
            "y_test_variance": y_test_var,
            "n_train": len(X_train), "n_test": len(X_test),
            "history": history,
        }, f, indent=2)
    print(f"[student] summary → {summary_path}", flush=True)


if __name__ == "__main__":
    main()
