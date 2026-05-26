"""Student distillation via DAgger (Dataset Aggregation, Ross & Bagnell 2011).

Trains a student policy to match a frozen v2.15 teacher's CBF-parameter
output (alpha, phi, a, b, c). The teacher reads the clean privileged
occupancy grid; the student reads a noised version derived from the
synthetic-LiDAR + clustering pipeline.

Usage (from the IsaacLab dir, after sim env is set up):

    ./isaaclab.sh -p ../scripts/train_distillation.py \\
        --task Isaac-CBF-Go2-Distill-v0 \\
        --teacher_checkpoint logs/rsl_rl/cbf_go2_teacher/<v215_run>/model_4999.pt \\
        --num_envs 1024 \\
        --max_iter 3000 \\
        --headless

Output:
  <output_dir>/student.pt          - distilled student weights
  <output_dir>/distill.csv         - per-iter loss + diagnostic stats
  <output_dir>/distill_loss.png    - loss curve

Status: SKELETON. The noised-occupancy-grid obs term is not yet wired
in; until it lands (post-v2.15 implementation), the student trains on
the same clean grid as the teacher. That degenerates the experiment to
"can DAgger trivially recover the teacher's policy from itself" — useful
as an end-to-end pipeline smoke test, not a real distillation result.

See docs/student_distillation_spec.md for the full design and the
implementation checklist for what's still missing.
"""

import argparse

from isaaclab.app import AppLauncher

# --- CLI parse before sim app launches -----------------------------------
parser = argparse.ArgumentParser(description="DAgger student distillation.")
parser.add_argument("--task", type=str, default="Isaac-CBF-Go2-Distill-v0",
                    help="Distillation env id. Default uses the placeholder "
                         "alias; will get the noised-grid obs term post-v2.15.")
parser.add_argument("--teacher_checkpoint", type=str, required=True,
                    help="Frozen teacher checkpoint (rsl_rl model_*.pt). "
                         "Typically the final model from a v2.15-style run.")
parser.add_argument("--num_envs", type=int, default=1024,
                    help="Lower than v2.15 training (4096) — DAgger is "
                         "matching, not exploring. 1024 is the spec default.")
parser.add_argument("--max_iter", type=int, default=3000,
                    help="DAgger iterations. Distillation typically converges "
                         "faster than RL from scratch (~3K vs v2.15's 6K).")
parser.add_argument("--rollout_steps", type=int, default=24,
                    help="Steps the student rolls out per iter before the "
                         "loss gradient step. Matches PPO's nsteps default.")
parser.add_argument("--lr", type=float, default=3e-4,
                    help="Adam learning rate for the student.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="logs/distillation")
parser.add_argument("--save_every", type=int, default=200,
                    help="Iterations between student.pt checkpoints.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- imports after sim app starts ----------------------------------------
import csv
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  -- registers tasks
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry


# ──────────────────────────────────────────────────────────────────────
# Teacher and student loading
# ──────────────────────────────────────────────────────────────────────

def load_teacher(checkpoint_path: Path, env, agent_cfg, device):
    """Load the v2.15 teacher as a frozen inference policy.

    Same loading path as eval_baseline's make_br_provider — the teacher's
    weights stay fixed throughout student training.
    """
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(checkpoint_path))
    teacher_policy = runner.get_inference_policy(device=device)

    return teacher_policy


def init_student(env, agent_cfg, device):
    """Initialize a student with the same architecture as the teacher.

    Currently bootstraps from agent_cfg's default network shape (no
    weight init from teacher — DAgger learns from scratch). Future
    revision: warm-start from teacher weights for faster convergence.
    """
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    student_policy = runner.alg.actor_critic                          # the nn.Module

    return runner, student_policy


# ──────────────────────────────────────────────────────────────────────
# Observation noising — DEFERRED IMPLEMENTATION
# ──────────────────────────────────────────────────────────────────────

def noise_obs_for_student(obs):
    """Return the obs the student should see (noised obstacle channel).

    PLACEHOLDER. The noised occupancy grid obs term is not yet wired
    in — when this is called pre-implementation, the student sees the
    same clean grid as the teacher.

    Post-v2.15 implementation:
        - Replace the obstacle channel of obs["policy"] (last 8192 dims)
          with the output of `noised_occupancy_grid_b`, which runs the
          synthetic-LiDAR + clustering pipeline + DR layers.
        - The 15 dynamics scalars at the front stay clean (first-pass
          deploy-realism caveat — see spec doc).
    """
    # TODO (post-v2.15): wire in noised_occupancy_grid_b. For now the
    # student sees the same priv obs as the teacher.
    return obs


# ──────────────────────────────────────────────────────────────────────
# DAgger training loop
# ──────────────────────────────────────────────────────────────────────

def train_dagger(env, teacher_policy, student_runner, student_policy,
                 device, args, output_dir):
    """Outer DAgger loop. One iteration is:

        1. Roll out `student` for `rollout_steps` steps in the env, using
           the noised obs at each step.
        2. At every visited state, query teacher with the CLEAN obs and
           record its action as the label.
        3. Backprop MSE(student_action, teacher_action) on the collected
           batch and step the student's optimizer.
    """
    optimizer = torch.optim.Adam(student_policy.parameters(), lr=args.lr)
    log_rows: list[dict] = []

    for it in range(args.max_iter):
        student_policy.train()
        obs, _ = env.reset() if it == 0 else (None, None)
        if obs is None:
            obs, _ = env.get_observations() if hasattr(env, "get_observations") else env.reset()

        student_actions: list[torch.Tensor] = []
        teacher_actions: list[torch.Tensor] = []

        for step in range(args.rollout_steps):
            # Student sees noised obs.
            obs_student = noise_obs_for_student(obs)
            action_student = student_policy.act(obs_student["policy"])

            # Teacher sees clean obs at the SAME state.
            with torch.no_grad():
                action_teacher = teacher_policy(obs)

            student_actions.append(action_student)
            teacher_actions.append(action_teacher)

            # Step env using the student's action — DAgger learns from
            # the student's actual visited trajectory, not the teacher's.
            obs, _, _, _, _ = env.step(action_student)

        s_batch = torch.cat(student_actions, dim=0)                   # (N*T, 5)
        t_batch = torch.cat(teacher_actions, dim=0)                   # (N*T, 5)

        loss = torch.nn.functional.mse_loss(s_batch, t_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_rows.append({"iter": it, "loss": float(loss.item())})

        if it % 10 == 0:
            print(f"[distill] iter {it:>5d}  loss={loss.item():.5f}")

        if it > 0 and it % args.save_every == 0:
            ckpt_path = output_dir / f"student_{it}.pt"
            torch.save(student_policy.state_dict(), ckpt_path)

    final_path = output_dir / "student.pt"
    torch.save(student_policy.state_dict(), final_path)
    print(f"[distill] wrote final student to {final_path}")

    return log_rows


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(args_cli.seed)
    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    device = inner.device
    print(f"[env] {args_cli.task}, num_envs={inner.num_envs}, device={device}")

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    print(f"[teacher] loading from {args_cli.teacher_checkpoint}")
    teacher_policy = load_teacher(
        Path(args_cli.teacher_checkpoint), env, agent_cfg, device,
    )
    print("[student] initializing fresh actor-critic with teacher's architecture")
    student_runner, student_policy = init_student(env, agent_cfg, device)

    log_rows = train_dagger(
        env, teacher_policy, student_runner, student_policy, device,
        args_cli, output_dir,
    )

    # CSV.
    csv_path = output_dir / "distill.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "loss"])
        w.writeheader()
        for r in log_rows:
            w.writerow(r)
    print(f"[csv] wrote {csv_path}")

    # Loss curve.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([r["iter"] for r in log_rows], [r["loss"] for r in log_rows])
        ax.set_xlabel("DAgger iteration")
        ax.set_ylabel("MSE(student, teacher)")
        ax.set_yscale("log")
        ax.set_title("Student distillation loss")
        ax.grid(alpha=0.3)
        plot_path = output_dir / "distill_loss.png"
        plt.savefig(plot_path, dpi=150)
        print(f"[plot] wrote {plot_path}")
    except ImportError:
        print("[plot] matplotlib unavailable, skipping PNG.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
