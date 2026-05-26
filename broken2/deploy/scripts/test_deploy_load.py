"""Sanity test for cbf_deploy_model.

Loads teacher + student checkpoints, runs forward on fake data, verifies
the output looks reasonable. Use this BEFORE wiring up ROS 2 nodes to
confirm the deploy model loads without IsaacLab.

Usage on lab box (where the checkpoints live):
  cd ~/Desktop/safety-go2
  python3 deploy/test_deploy_load.py \\
    --teacher IsaacLab/logs/rsl_rl/cbf_go2_teacher_rma/2026-05-20_18-25-01/model_2499.pt \\
    --student checkpoints/student_v13_1.pt
"""
from __future__ import annotations

import argparse
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cbf_deploy_model import CbfDeployModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num_steps", type=int, default=100)
    args = p.parse_args()

    model = CbfDeployModel(args.teacher, args.student, device=args.device)
    print()

    # Fake "rollout" — random proprio + sparse occupancy grid.
    rng = np.random.default_rng(42)
    proprio_dim = 19
    print(f"[test] running {args.num_steps} dummy inference steps...")
    print(f"{'step':>4s}  {'alpha':>6s}  {'phi':>6s}  {'a':>6s}  {'b':>6s}  {'c':>6s}")
    for s in range(args.num_steps):
        # Simulate a (mostly stationary) proprio: small noise around 0.
        proprio = rng.normal(0.0, 0.1, proprio_dim).astype(np.float32)
        proprio[0] += 0.30  # base_height nominal Go2 standing height

        # Sparse grid: 2% occupancy, like sim.
        grid = (rng.random((2, 64, 64)) < 0.02).astype(np.float32)

        out = model.infer(proprio, grid)
        if s % 10 == 0 or s == args.num_steps - 1:
            print(f"{s:>4d}  {out['alpha']:6.3f}  {out['phi']:6.3f}  "
                  f"{out['a']:6.3f}  {out['b']:6.3f}  {out['c']:6.3f}")

    print()
    print("Sanity checks:")
    print(f"  α in [0.5, 3.0]: {0.5 <= out['alpha'] <= 3.0}")
    print(f"  φ in [0.0, 5.0]: {0.0 <= out['phi'] <= 5.0}")
    print(f"  a in [0.0, 0.5]: {0.0 <= out['a'] <= 0.5}")
    print(f"  b in [0.0, 1.0]: {0.0 <= out['b'] <= 1.0}")
    print(f"  c in [-0.1, 0.0]: {-0.1 <= out['c'] <= 0.0}")
    print()
    print("OK if values look reasonable. If α/φ are pinned at min or max,")
    print("the policy is being run on out-of-distribution input — expected")
    print("for random data. Real test is with real rollouts.")


if __name__ == "__main__":
    main()
