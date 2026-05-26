"""B0 / B1 / B2 / BR baseline comparison for the CBF teacher.

Implements the professor's plan from the TISSf paper (Cosner et al., 2103.08041):

    B0  plain CBF        ε = ∞   (φ = 0)               Fig 1(a)/(b)
    B1  ISSf-CBF         φ = const                     Fig 1(c), Example 2
    B2  TISSf-CBF        φ(h) = (1/ε₀) · exp(-λ·h)     Fig 1(d), Example 3
    BR  RL-tuned         φ from trained teacher        ours

Our env's CBF inequality (see _cbf_filter in cbf_go2_env.py):

    L_g h · u  ≥  -α·(h - c)  +  φ·‖L_g h‖²  +  a

The 5-D action (α, φ, a, b, c) goes through tanh + linear scaling:
    α  ∈ [0.1, 5.0]
    φ  ∈ [0.0, 5.0]      ← this is 1/ε from the paper
    a  ∈ [0.0, 1.0]      ← extra additive slack (NOT in TISSf paper)
    b  unused (reserved for SOCP)
    c  ∈ [0.0, 0.5]      ← h-shift (NOT in TISSf paper)

For B0/B1/B2 we lock a = c = 0 (TISSf-as-written). α is also held at a
constant per config — sweep multiple α values to find each baseline's
strongest version (Option 3 from the discussion: pick the BEST hand-tuned
baseline as opponent, not a strawman).

Usage (from the IsaacLab dir, after sim env is set up):

    ./isaaclab.sh -p ../scripts/eval_baseline.py \\
        --num_envs 64 \\
        --steps_per_config 600 \\
        --modes B0,B1,B2 \\
        --headless

Add `--checkpoint <path/to/model_*.pt>` to also evaluate the trained
teacher (BR). Without the flag, only the hand-tuned baselines run.

Output:
  <output_dir>/baseline.csv  — one row per (mode, config) sweep point
  <output_dir>/baseline.png  — collision-rate vs intervention-cost scatter
"""

import argparse

from isaaclab.app import AppLauncher

# --- CLI parse before sim app launches -----------------------------------
parser = argparse.ArgumentParser(description="B0/B1/B2/BR baseline eval.")
parser.add_argument("--task", type=str, default="Isaac-CBF-Go2-v0",
                    help="Task name. Default is the v2.2 training distribution.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps_per_config", type=int, default=600,
                    help="Sim steps per config. 600 ≈ 12s × 64 envs of episodes.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="logs/baseline_eval")

parser.add_argument("--modes", type=str, default="B0,B1,B2",
                    help="Comma-separated subset of B0,B1,B2,BR,BS.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="rsl_rl model_*.pt for BR. Required if BR in --modes.")
parser.add_argument("--student_adapter_checkpoint", type=str, default=None,
                    help="V13 two-stream student-adapter .pt (from train_student_v13.py). "
                         "When set + 'BS-A' in --modes: loads teacher from --checkpoint, "
                         "replaces teacher's priv_encoder with the student adaptation "
                         "module. Closed-loop test of student-vs-teacher.")
parser.add_argument("--student_checkpoint", type=str, default=None,
                    help="rsl_rl model_*.pt for the distilled student (BS mode). "
                         "Distinct from --checkpoint so BS and BR can be evaluated "
                         "together in one run for direct teacher-vs-student "
                         "comparison.")

# Sweep grids. Lists of comma-separated floats.
parser.add_argument("--alpha_grid", type=str, default="0.5,1.5,3.0",
                    help="α values to sweep across all of B0/B1/B2.")
parser.add_argument("--phi_grid",   type=str, default="0.5,1.5,3.0",
                    help="Constant φ values for B1.")
parser.add_argument("--epsilon0_grid", type=str, default="0.1,0.5",
                    help="ε₀ values for B2 (recall: φ(h) = (1/ε₀)·exp(-λh)).")
parser.add_argument("--lambda_grid",   type=str, default="1.0,3.0",
                    help="λ values for B2.")

parser.add_argument("--no_obstacles", action="store_true",
                    help="Force K_actual=0 every reset. CBF can't constrain "
                         "(no obstacles to avoid) — measures planner+locomotion "
                         "stability in isolation. Use with --modes B0 to "
                         "isolate whether falls come from CBF deflection or "
                         "from the planner+loco pair on its own.")

parser.add_argument("--planner_resample_s", type=float, default=None,
                    help="Override env_cfg.commands.base_velocity."
                         "resampling_time_range to (X, X). E.g., 10.0 for "
                         "v2.6-style mid-episode planner switching, 100.0 "
                         "for v2.8-style locked. Default (None) = use whatever "
                         "the env_cfg specifies.")

# v2.11 (2026-05-07): paper-Table-2 ablations — B-fixed-X clamps one CBF
# slot to a fixed physical value while letting BR policy drive the rest.
# Modes: Bf-alpha, Bf-phi, Bf-a, Bf-c. Requires --checkpoint (loads BR).
# Targets default to mid-range of each param's physical range.
parser.add_argument("--bf_alpha_target", type=float, default=2.5,
                    help="Fixed physical α for Bf-alpha mode (default 2.5; "
                         "valid range [0.1, 5.0]).")
parser.add_argument("--bf_phi_target",   type=float, default=2.5,
                    help="Fixed physical φ for Bf-phi mode (default 2.5; "
                         "valid range [0.0, 5.0]).")
parser.add_argument("--bf_a_target",     type=float, default=1.5,
                    help="Fixed physical a for Bf-a mode (default 1.5; "
                         "valid range [0.0, 3.0] under WIDE_PARAM_RANGES, "
                         "[0.0, 1.0] otherwise).")
parser.add_argument("--bf_c_target",     type=float, default=0.5,
                    help="Fixed physical c for Bf-c mode (default 0.5; "
                         "valid range [0.0, 1.0] under WIDE_PARAM_RANGES, "
                         "[0.0, 0.5] otherwise).")

# v2.16-diagnostic (2026-05-10): Bf-all + Bhc-alpha modes.
# Bf-all: all 4 slots fixed simultaneously (uses --bf_alpha/phi/a/c_target).
#   Tests "is BR's per-step variation better than its own averages?"
# Bhc-alpha: state-conditional α(h) = floor + (ceil - floor) * exp(-λ·h),
#   other slots constant. Tests "can ANY hand-crafted state-conditional α
#   beat fixed α at BR's mean?"
parser.add_argument("--hca_floor", type=float, default=0.1,
                    help="Bhc-alpha: α target when h is large (far from obstacle). "
                         "Eval-space value; env applies tanh+ALPHA_MIN linear map. "
                         "0.1 → env α=1.0 (the env's ALPHA_MIN floor).")
parser.add_argument("--hca_ceil",  type=float, default=5.0,
                    help="Bhc-alpha: α target when h is near 0. 5.0 → env α=5.0.")
parser.add_argument("--hca_lambda", type=float, default=2.0,
                    help="Bhc-alpha: decay rate. α(h) = floor + (ceil-floor)·exp(-λh).")
parser.add_argument("--hca_phi", type=float, default=2.26,
                    help="Bhc-alpha: held-constant φ (default = v2.15 BR mean).")
parser.add_argument("--hca_a",   type=float, default=0.015,
                    help="Bhc-alpha: held-constant a (default = v2.15 BR mean).")
parser.add_argument("--hca_c",   type=float, default=0.118,
                    help="Bhc-alpha: held-constant c (default = v2.15 BR mean).")

# v2.16-diagnostic Option 1 (2026-05-10): post-hoc shrinkage test.
# Wraps BR's policy output and shrinks the chosen slots toward the
# cross-env mean (computed at each step from the policy's actual output).
# shrink=0 → all envs get the cross-env mean (≈ Bf-all behavior, but with
# the policy's instantaneous mean instead of a hardcoded target).
# shrink=1 → no shrinkage (BR unchanged).
# shrink=0.5 → halves the cross-env variation around the mean each step.
# Tests whether narrowing BR's actor output band gives indist gains while
# preserving OOD wins (pure eval-only test of the "tighter actor mean
# would help" hypothesis).
parser.add_argument("--brs_shrink", type=float, default=0.5,
                    help="BR-shrunk: shrinkage factor in [0, 1]. 0 = collapse "
                         "to cross-env mean (Bf-all-like), 1 = full BR variation.")
parser.add_argument("--brs_slots", type=str, default="alpha",
                    help="BR-shrunk: comma-separated slots to shrink. "
                         "Choices: alpha, phi, a, c. Default 'alpha'.")

parser.add_argument("--bimodal_resample", action="store_true", default=None,
                    help="If set, leave bimodal_resample on at eval time. "
                         "By default, eval forces bimodal_resample=False so "
                         "the eval is deploy-realistic locked-planner. Pass "
                         "this flag only for debug or training-distribution "
                         "verification runs.")

# DIAG-4b (2026-05-08): isolate planner-mix vs DR vs 50Hz contributions to
# the 0.5% → 8.3% fall-rate jump (native loco env vs our env at K=0). Swaps
# the whole command term out for a vanilla UniformVelocityCommand at the
# locomotion's training distribution (10 s held). Holds DR + 50 Hz inner
# loop + obstacle setup constant. If fall stays ~8% → DR is the cause; if
# fall drops to ~1% → our planner mix is the cause.
parser.add_argument("--vanilla_uniform_command", action="store_true",
                    help="DIAG-4b: replace MultiPlannerCommand with a vanilla "
                         "UniformVelocityCommand (10s held, native loco "
                         "training distribution) at eval time. Use with "
                         "--no_obstacles to factor planner mix out of the "
                         "no-obstacle locomotion floor. Pair with the same "
                         "flag on the v2.6 ckpt for cross-version comparison.")

# DIAG-4b sub-bisection (2026-05-08): force the MultiPlannerCommand to use
# ONE planner at 100% weight, others at 0%. Use with --no_obstacles to
# attribute fall rate per-planner (compare each to the --vanilla_uniform_command
# baseline). Tells us which planner(s) in the mix actually stress locomotion,
# so we can drop only the bad apples instead of stripping the whole regularizer.
parser.add_argument("--solo_planner", type=str, default=None,
                    choices=["walk", "adversarial", "smooth_goal", "waypoint",
                             "mpc", "legacy_goal", "uniform"],
                    help="DIAG-4b sub-bisection: force planner mix to 100%% one "
                         "planner. All other weights set to 0. Use with "
                         "--no_obstacles to attribute K=0 fall rate per-planner.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- imports after sim app starts ----------------------------------------
import csv
import math
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  -- registers tasks
from isaaclab_tasks.utils import parse_env_cfg


# Action ranges MUST match cbf_go2_env._cbf_filter tanh-scale mapping.
# v2.11: read WIDE_PARAM_RANGES from env so a/c ranges stay in sync.
try:
    from isaaclab_tasks.manager_based.safety.cbf_go2.cbf_go2_env import (
        WIDE_PARAM_RANGES as _WIDE_PARAM_RANGES,
    )
except ImportError:
    _WIDE_PARAM_RANGES = False  # fall back to v2.10 ranges if env isn't reachable

PARAM_RANGES = {
    "alpha": (0.1, 5.0),
    "phi":   (0.0, 5.0),
    "a":     (0.0, 3.0) if _WIDE_PARAM_RANGES else (0.0, 1.0),
    "c":     (0.0, 1.0) if _WIDE_PARAM_RANGES else (0.0, 0.5),
}

# A near-zero φ for B0 (we can't represent exactly 0 because the lo
# of the range is 0 and atanh(-1) = -∞). 0.005 is "effectively no buffer."
PHI_FLOOR_B0 = 0.005

# "Stuck" detection — locomotion failure mode where the robot is alive
# (no fall, no collision) but barely moving despite being commanded to
# walk. Tracked via a sliding window of consecutive low-speed steps:
# if at episode end the streak is >= STUCK_WINDOW, the episode is
# counted as stuck (in addition to being a timeout). DIAG-1 found this
# pattern after big planner direction reversals — locomotion enters a
# zero-velocity attractor it doesn't recover from.
STUCK_LOW_SPEED_THRESHOLD = 0.10  # m/s — below this is "barely moving"
STUCK_WINDOW = 100                 # consecutive steps = 2 s at 50 Hz


# ──────────────────────────────────────────────────────────────────────
# Encoding helpers
# ──────────────────────────────────────────────────────────────────────

def _encode_dim(target, lo: float, hi: float, device, N: int) -> torch.Tensor:
    """Inverse of the tanh+linear-scale used in cbf_go2_env._cbf_filter.

    target may be a Python scalar OR a (N,) tensor; returns a (N,) tensor
    such that env's tanh+scale produces the requested physical value.

    Degenerate range handling: when hi == lo (e.g. c_param_range =
    (-0.05, -0.05) on FROZEN_AC and TIGHTCOR envs), the env's tanh+scale
    produces the single clamp value regardless of the raw action input.
    Returning zeros here avoids a 0/0 → NaN that would otherwise propagate
    through the env's CBF math and freeze the robot at spawn.
    """
    if hi == lo:
        return torch.zeros(N, device=device)
    if isinstance(target, torch.Tensor):
        x = target.to(device).clamp(lo, hi)
    else:
        x = torch.full((N,), float(target), device=device).clamp(lo, hi)
    sq = (2.0 * (x - lo) / (hi - lo)) - 1.0
    return torch.atanh(sq.clamp(-0.9999, 0.9999))


def encode_action(alpha, phi, a, c, device, N) -> torch.Tensor:
    """Build a (N, 5) raw action tensor from physical (α, φ, a, c).

    Each arg may be scalar (broadcast) or a (N,) tensor (per-env values,
    used by B2 which varies φ with h(x)).
    """
    return torch.stack([
        _encode_dim(alpha, *PARAM_RANGES["alpha"], device, N),
        _encode_dim(phi,   *PARAM_RANGES["phi"],   device, N),
        _encode_dim(a,     *PARAM_RANGES["a"],     device, N),
        torch.zeros(N, device=device),                       # b unused
        _encode_dim(c,     *PARAM_RANGES["c"],     device, N),
    ], dim=-1)


# ──────────────────────────────────────────────────────────────────────
# Action providers (one per mode)
# ──────────────────────────────────────────────────────────────────────

def make_b0_provider(alpha: float, device, N):
    """Plain CBF: φ ≈ 0, no robustness buffer. a = c = 0."""
    raw = encode_action(alpha, PHI_FLOOR_B0, 0.0, 0.0, device, N)
    return (lambda r: lambda _obs, _inner: r)(raw)


def make_b1_provider(alpha: float, phi: float, device, N):
    """ISSf-CBF: φ = constant. a = c = 0."""
    raw = encode_action(alpha, phi, 0.0, 0.0, device, N)
    return (lambda r: lambda _obs, _inner: r)(raw)


def make_b2_provider(alpha: float, epsilon0: float, lambda_: float, device, N):
    """TISSf-CBF: φ(h) = (1/ε₀) · exp(-λ · h). Queries inner._compute_h()
    each step to get h, computes per-env φ, encodes."""
    inv_eps0 = 1.0 / float(epsilon0)
    lam = float(lambda_)

    def fn(_obs, inner):
        with torch.no_grad():
            # v2.11: _compute_h now returns 3-tuple (h_vals, L_g_h, closest_idx).
            h_vals = inner._compute_h()[0]                      # (N,)
        phi_per_env = inv_eps0 * torch.exp(-lam * h_vals)  # (N,)
        return encode_action(alpha, phi_per_env, 0.0, 0.0, device, N)
    return fn


_BFIXED_SLOT_INDEX = {"alpha": 0, "phi": 1, "a": 2, "c": 4}  # cbf_params 5D layout


def make_b_fixed_provider(slot_name: str, target: float, br_provider_fn, device, N):
    """v2.11 paper-Table-2 ablation: run BR policy normally but override
    one CBF param slot with a fixed physical value.

    Tests whether the policy's online adaptation of the named slot is
    load-bearing for performance. If BR (full adaptive) outperforms
    B-fixed-X for any reasonable choice of `target`, the policy IS
    using slot X meaningfully. If BR ties, slot X isn't doing useful work.

    Implementation: pre-compute the inverse-tanh raw value for the target
    physical value (via _encode_dim — same logic as B0/B1/B2 use), then
    overwrite that slot of the BR policy's output. The env's tanh+scale
    in _cbf_filter then maps it back to the requested physical value.
    """
    slot_idx = _BFIXED_SLOT_INDEX[slot_name]
    lo, hi = PARAM_RANGES[slot_name]
    if not (lo <= target <= hi):
        raise ValueError(
            f"B-fixed-{slot_name} target {target} outside range [{lo}, {hi}]"
        )
    raw_target = _encode_dim(target, lo, hi, device, N)  # (N,)

    def fn(obs, inner):
        action = br_provider_fn(obs, inner)              # (N, 5) raw
        # Clone so we don't mutate the policy output (rsl_rl reuses it)
        action = action.clone()
        action[:, slot_idx] = raw_target
        return action
    return fn


def make_br_provider(checkpoint_path: Path, env, agent_cfg, device):
    """BR: load v2.2 trained teacher via rsl_rl OnPolicyRunner."""
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(checkpoint_path))
    policy = runner.get_inference_policy(device=device)

    def fn(obs, _inner):
        # rsl_rl's policy expects the dict (keyed by obs_group, e.g. "policy"),
        # not the unwrapped tensor — its MLPModel.get_latent does obs[group].
        with torch.no_grad():
            return policy(obs)
    return fn


def make_bf_all_provider(alpha: float, phi: float, a: float, c: float, device, N):
    """Bf-all: all 4 CBF slots fixed at provided values. No policy involvement.

    Diagnostic: tests whether BR's per-step variation helps over its own
    converged averages. If Bf-all (with targets = BR's means) ≈ BR, the
    policy's adaptation isn't doing useful work; it's variation around the
    mean that doesn't beat the constant.
    """
    raw = encode_action(alpha, phi, a, c, device, N)
    return (lambda r: lambda _obs, _inner: r)(raw)


def make_bhc_alpha_provider(floor: float, ceil: float, lambda_: float,
                             phi: float, a: float, c: float, device, N):
    """Bhc-alpha (B-handcrafted-alpha): state-conditional α driven by h(x).

        α(h) = floor + (ceil - floor) · exp(-λ · h)

    Close to obstacle (h≈0): α ≈ ceil (max aggressive)
    Far from obstacle (large h): α ≈ floor (relaxed)

    Other slots (φ, a, c) held at provided constants — defaults to v2.15
    BR converged means. Tests whether ANY hand-crafted state-conditional α
    beats fixed α=BR-mean. Mirrors B2's TISSf-style exp-decay for φ but
    applies it to α instead.
    """
    af = float(floor)
    ac = float(ceil)
    lam = float(lambda_)

    def fn(_obs, inner):
        with torch.no_grad():
            h_vals = inner._compute_h()[0]                            # (N,)
        alpha_per_env = af + (ac - af) * torch.exp(-lam * h_vals)     # (N,)
        return encode_action(alpha_per_env, phi, a, c, device, N)
    return fn


_SHRINK_SLOT_INDEX = {"alpha": 0, "phi": 1, "a": 2, "c": 4}


def make_br_shrunk_provider(br_provider_fn, shrink: float, slots: list[str], device, N):
    """Wrap BR's action provider; shrink chosen slots toward cross-env mean.

    For each step:
      action[:, slot] = mean + shrink * (action[:, slot] - mean)

    where mean is computed across envs at THIS step. shrink=0 collapses
    every env's slot value to the same mean (pure constant); shrink=1
    leaves BR's output untouched.

    Operates on raw (pre-tanh) action — the env's tanh+scale will then
    map shrunk raw values to physical params. Shrinking raw is monotonic
    with respect to post-tanh shrinkage, so the qualitative effect
    (narrower actor band) is preserved.
    """
    s = float(shrink)
    slot_idxs = [_SHRINK_SLOT_INDEX[name.strip()] for name in slots]

    def fn(obs, inner):
        action = br_provider_fn(obs, inner)                           # (N, 5) raw
        action = action.clone()
        for idx in slot_idxs:
            slot_mean = action[:, idx].mean()
            action[:, idx] = slot_mean + s * (action[:, idx] - slot_mean)
        return action
    return fn


def make_bs_provider(student_checkpoint_path: Path, env, agent_cfg, device):
    """BS: load distilled student checkpoint via rsl_rl OnPolicyRunner.

    Identical loading path to BR — student inherits the teacher's MLP
    architecture, just with weights from the DAgger run. The agent_cfg
    is shared (same network shape). Distinct loader function for clarity:
    BR and BS may run in the same eval, each loaded from its own ckpt.
    """
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(student_checkpoint_path))
    policy = runner.get_inference_policy(device=device)

    def fn(obs, _inner):
        with torch.no_grad():
            return policy(obs)
    return fn


def make_bs_adapter_provider(
    teacher_checkpoint_path: Path,
    student_adapter_checkpoint_path: Path,
    env, agent_cfg, device, N: int,
):
    """V13 two-stream student bridge.

    The V13 student is just the ADAPTATION MODULE — it predicts ẑ_env
    from a temporal history of (proprio, prev_action). The rest of the
    policy (pi_teacher MLP, grid encoder, proprio normalizer) stays as
    the trained teacher.

    Pipeline per step:
      obs_t  →  priv_proprio_t (observable slice, 19-D)
              →  z_proprio_t = teacher.proprio_normalizer(priv_proprio_t)
              →  grid_t
              →  z_grid_t = teacher.grid_encoder(grid_t)
              →  push (priv_proprio_t, prev_action_{t-1}) onto history buffer
              →  ẑ_env_t = student(history_buffer[-50:])
              →  action_t = teacher.pi_teacher(ẑ_env_t ⊕ z_proprio_t ⊕ z_grid_t)
              →  store action_t for next step's history

    Implementation: monkey-patch the teacher's priv_encoder (mlp[0]) with
    a thin shim that returns ẑ_env from the student's prediction over the
    stateful history buffer. The teacher's policy(obs) call then works
    unchanged, but z_env comes from the student.

    Episode resets clear the history for the reset envs.
    """
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    # Student module — pure PyTorch, no IsaacLab dep.
    from isaaclab_tasks.manager_based.safety.cbf_go2.cbf_go2_student import (
        StudentAdaptationModule,
    )
    from isaaclab_tasks.manager_based.safety.cbf_go2 import (
        cbf_go2_teacher_rma as _tr_module,
    )  # for _PRIV_HIDDEN_DIM/_PRIV_DIM globals
    import torch.nn as nn

    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(teacher_checkpoint_path))

    # Locate teacher actor.mlp (a _SplitRMAMLP). rsl_rl's PPO can expose the
    # actor at different attribute paths depending on version — try a few.
    actor = None
    for path in [("alg", "actor_critic", "actor"),
                 ("alg", "actor_critic"),
                 ("alg", "actor"),
                 ("policy",), ("actor",)]:
        m = runner
        try:
            for p in path:
                m = getattr(m, p)
            if hasattr(m, "mlp"):
                actor = m
                break
        except AttributeError:
            continue
    if actor is None:
        raise RuntimeError("could not locate actor.mlp on runner")
    inner_mlp = actor.mlp

    # Load student adapter.
    sckpt = torch.load(str(student_adapter_checkpoint_path), map_location=device)
    cfg = sckpt["config"]
    student = StudentAdaptationModule(
        proprio_dim=cfg["proprio_dim"],
        action_dim=cfg["action_dim"],
        z_env_dim=cfg["z_env_dim"],
        history_len=cfg["history_len"],
    ).to(device)
    student.load_state_dict(sckpt["state_dict"])
    student.eval()
    print(f"[BS-adapter] student loaded. R²_test_best={sckpt.get('best_test_r2', '?')}",
          flush=True)

    H = cfg["history_len"]
    F_p = cfg["proprio_dim"]
    F_a = cfg["action_dim"]
    P_hidden = _tr_module._PRIV_HIDDEN_DIM
    P_total = _tr_module._PRIV_DIM

    # Stateful history buffer per env. Initialized lazily on first step.
    # _PRIV_HIDDEN_DIM / _PRIV_DIM exported into state so eval_one can read
    # them when handling resets (without re-importing the teacher_rma module).
    state = {
        "history": torch.zeros(N, H, F_p + F_a, device=device),
        "prev_action": torch.zeros(N, F_a, device=device),
        "initialized": False,
        "priv_hidden_dim": P_hidden,
        "priv_total_dim": P_total,
    }

    class _StudentPrivEncoderShim(nn.Module):
        """Drop-in replacement for the teacher's priv_encoder. Ignores the
        ground-truth priv_hidden input and returns the student's prediction
        from the stateful history buffer."""
        def __init__(self, student_module, state_ref, z_priv_dim):
            super().__init__()
            self.student = student_module
            self.state = state_ref
            self.z_priv_dim = z_priv_dim

        def forward(self, _priv_hidden):
            return self.student(self.state["history"])

    # Save original encoder weights (unused by shim, but kept for potential restore).
    original_priv_encoder = inner_mlp[0]
    z_priv_dim = getattr(original_priv_encoder, "z_priv_dim", cfg["z_env_dim"])
    inner_mlp[0] = _StudentPrivEncoderShim(student, state, z_priv_dim)
    print(f"[BS-adapter] priv_encoder replaced with student shim. "
          f"z_env_dim={cfg['z_env_dim']}, history_len={H}", flush=True)

    policy = runner.get_inference_policy(device=device)

    def fn(obs, _inner):
        obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
        priv_proprio = obs_tensor[:, P_hidden:P_total]                # (N, F_p)

        # Push (proprio_t, prev_action_{t-1}) onto the history buffer.
        new_step = torch.cat([priv_proprio, state["prev_action"]], dim=-1)  # (N, F)
        state["history"] = torch.cat(
            [state["history"][:, 1:], new_step.unsqueeze(1)], dim=1
        )
        state["initialized"] = True

        with torch.no_grad():
            action = policy(obs)

        state["prev_action"] = action.detach()
        return action

    return fn, state


# ──────────────────────────────────────────────────────────────────────
# Eval loop
# ──────────────────────────────────────────────────────────────────────

def eval_one(env, action_fn, num_steps: int, device, N: int,
             bs_adapter_state=None):
    """Run num_steps; aggregate metrics across episodes that finished
    inside the window. Counts collisions via min h(x) crossing zero."""
    obs, _ = env.reset()
    inner = env.unwrapped

    n_eps = 0
    n_collision = 0   # min_h < 0 inside the episode (boundary breached)
    n_fall = 0        # terminated, no boundary breach (base_contact)
    n_timeout = 0     # truncated
    n_stuck = 0       # subset of timeout: robot was barely moving at end
    sum_min_h = 0.0
    sum_min_h_count = 0
    sum_infeas_steps = 0.0
    total_steps = 0
    sum_phi_step = 0.0   # mean φ actually used (after tanh+scale)
    phi_step_count = 0

    # v2.16 diagnostic: track per-step cbf_log values from inner._cbf_log.
    # Each step the env's _cbf_filter populates this dict with mean/std
    # of α/φ/a/c across envs. We accumulate to compute per-task averages.
    # If BR varies α a lot (high cbf_alpha_std avg) on OOD but little on
    # in-dist, the policy IS regime-aware. If similar across tasks, it's
    # varying regardless of state.
    cbf_log_keys = [
        "cbf_alpha_mean", "cbf_alpha_std",
        "cbf_phi_mean", "cbf_phi_std",
        "cbf_a_mean", "cbf_a_std",
        "cbf_c_mean", "cbf_c_std",
        "h_min", "h_mean", "qp_active_rate", "u_safe_clamp_rate",
        # v3.0: ||u_safe - u_des|| magnitude (mean+std across envs per step).
        # avg_deflection_mean = how hard the CBF pushes when it fires;
        # combined with avg_qp_active_rate this captures both intervention
        # frequency and force, separating "filter idle" from "filter active".
        "deflection_mean", "deflection_std",
    ]
    cbf_log_sums = {k: 0.0 for k in cbf_log_keys}
    cbf_log_count = 0

    ep_min_h = torch.full((N,), float("inf"), device=device)
    # Sliding "consecutive low-speed steps" counter, per env. Reset on
    # each step where speed crosses back above threshold, so it always
    # reflects the most recent low-speed streak.
    low_speed_streak = torch.zeros(N, dtype=torch.long, device=device)

    # v2.15 (Path A): richer 2-axis metrics for safety/performance.
    # Safety: per-episode worst-case h, fraction of steps in close-call zone.
    # Performance: mean speed, start-and-go latency, per-episode distance.
    # All derived from existing state (robot_speed, h_vals); no planner
    # introspection (deferred to goal-reach metrics, see todo).
    H_CLOSE_THRESHOLD     = 0.05   # "close-call zone": h < this = nearly at boundary
    START_SPEED_THRESHOLD = 0.30   # m/s — robot considered "moving" past this
    # Path C: goal-reach proxy. Env has no fixed goal target (planner is a
    # velocity-command mix), so we use displacement from spawn as a "did the
    # robot get somewhere" signal — distinct from mean_dist_traveled (path
    # length) because circling-in-place inflates path length but not
    # displacement. 1.5m ≈ 3 robot body lengths, well past locomotion noise.
    GOAL_REACH_DISTANCE   = 1.5    # m of displacement from spawn

    sum_episode_min_h = 0.0
    sum_episode_min_h_count = 0
    sum_v_xy = 0.0                 # accumulator for mean speed
    sum_h_close = 0                # count of (env, step) pairs with h < threshold
    sum_episode_dist = 0.0
    sum_episode_dist_count = 0
    sum_start_latency = 0.0
    sum_start_latency_count = 0    # count only episodes where robot DID start moving

    # Path C goal-reach accumulators.
    n_goal_reached            = 0
    sum_time_to_goal          = 0.0
    sum_time_to_goal_count    = 0  # only episodes that actually reached goal
    sum_final_displacement    = 0.0
    sum_final_displacement_count = 0

    # v3.0 task-progress accumulators.
    # path_efficiency = displacement / distance_traveled per episode,
    # captures wandering vs straight-line motion (∈ [0,1] by triangle
    # inequality). Skip episodes that barely moved (dist < 0.1m) — ratio
    # is noise there. mean_v_along_cmd projects actual velocity onto
    # commanded direction, so a robot scuttling sideways doesn't get the
    # same credit as one tracking teleop.
    sum_path_efficiency       = 0.0
    sum_path_efficiency_count = 0
    sum_v_along_cmd           = 0.0
    sum_v_along_cmd_count     = 0  # steps where ‖u_des_xy‖ > 0 (cmd active)

    # Per-env, per-episode trackers (reset at done):
    ep_dist_traveled    = torch.zeros(N, device=device)              # Σ ‖v_xy‖·dt
    ep_first_motion_step = torch.full((N,), -1, dtype=torch.long, device=device)
    ep_step_counter     = torch.zeros(N, dtype=torch.long, device=device)
    # Path C: integrate world-frame velocity for displacement (avoids the
    # post-step root_pos_w ambiguity caused by IsaacLab's auto-reset).
    ep_disp_x          = torch.zeros(N, device=device)
    ep_disp_y          = torch.zeros(N, device=device)
    ep_first_goal_step  = torch.full((N,), -1, dtype=torch.long, device=device)
    step_dt = float(inner.step_dt)

    # Per-scene-type telemetry (2026-05-16). Tags each ENDED episode with
    # its scene type (corridor vs open). Reads inner.cbf_scene_is_corridor
    # which the env sets per-reset in randomize_obstacles_position.
    if hasattr(inner, "cbf_scene_is_corridor"):
        ep_scene_is_corridor = inner.cbf_scene_is_corridor.clone()
    else:
        ep_scene_is_corridor = torch.zeros(N, dtype=torch.bool, device=device)
    n_eps_corridor = 0
    n_collision_corridor = 0
    n_fall_corridor = 0
    n_timeout_corridor = 0
    n_stuck_corridor = 0
    n_goal_reached_corridor = 0
    n_eps_open = 0
    n_collision_open = 0
    n_fall_open = 0
    n_timeout_open = 0
    n_stuck_open = 0
    n_goal_reached_open = 0

    # Collision-source split (2026-05-16). Our existing "collision" counter
    # fires whenever perceived h dipped below 0 anywhere in the episode.
    # With shield_v0c's +0.20-0.50 m over-protection bias, that can happen
    # even when the robot never touched a real obstacle (perceived surface
    # closer than truth → robot enters perceived-collision zone while still
    # physically clear). Split into:
    #   actual = obstacle_contact termination fired this episode
    #   perceived_only = perceived h < 0 but no obstacle_contact
    # Stacked via per-env tag latched at episode-end below.
    has_obstacle_term = (
        hasattr(inner, "termination_manager")
        and hasattr(inner.termination_manager, "get_term")
        and "obstacle_contact" in getattr(inner.termination_manager, "active_terms", [])
    )
    # Fallback: try the API; if the term doesn't exist we'll skip the split.
    try:
        _probe = inner.termination_manager.get_term("obstacle_contact")
        has_obstacle_term = _probe is not None
    except Exception:
        has_obstacle_term = False
    n_collision_actual = 0
    n_collision_perceived_only = 0
    # 2026-05-22: terminations of type `goal_reached` (eval-corridor-strict).
    # Distinct from `n_goal_reached` (above), which uses the 1.5m displacement
    # threshold. This counts the actual termination event ("robot got within
    # 0.5m of the active goal") and matches the task's success criterion.
    # Returns 0 on tasks without a goal_reached termination wired up.
    n_goal_reached_term = 0
    n_goal_reached_term_corridor = 0
    n_goal_reached_term_open = 0

    for _ in range(num_steps):
        action = action_fn(obs, inner)

        # Track what the env will see for φ AFTER the tanh+scale.
        with torch.no_grad():
            phi_actual = (torch.tanh(action[:, 1]) + 1.0) * 0.5 * PARAM_RANGES["phi"][1]
        sum_phi_step += float(phi_actual.mean().item())
        phi_step_count += 1

        with torch.no_grad():
            # _compute_h returns (h_vals, L_g_h, closest_idx); we only need h_vals.
            # The closest_idx return was added in v2.11 for L_f h obstacle drift.
            h_vals = inner._compute_h()[0]
        ep_min_h = torch.minimum(ep_min_h, h_vals)
        sum_min_h += float(h_vals.mean().item())
        sum_min_h_count += 1

        obs, _, terminated, truncated, _ = env.step(action)
        total_steps += N

        # v2.16 diagnostic: accumulate cbf_log from this step. inner._cbf_log
        # was populated inside _cbf_filter (called inside env.step). Keys are
        # per-step scalars (already averaged across envs by the env code).
        if hasattr(inner, "_cbf_log") and isinstance(inner._cbf_log, dict):
            for k in cbf_log_keys:
                if k in inner._cbf_log:
                    cbf_log_sums[k] += float(inner._cbf_log[k])
            cbf_log_count += 1

        # Update per-env low-speed streak using xy plane robot velocity.
        with torch.no_grad():
            robot_speed = inner.scene["robot"].data.root_lin_vel_b[:, :2].norm(dim=-1)
        low_speed_streak = torch.where(
            robot_speed < STUCK_LOW_SPEED_THRESHOLD,
            low_speed_streak + 1,
            torch.zeros_like(low_speed_streak),
        )

        # v2.15 (Path A) per-step accumulators. All read from already-
        # computed h_vals and robot_speed; no extra GPU work.
        sum_v_xy   += float(robot_speed.sum().item())
        sum_h_close += int((h_vals < H_CLOSE_THRESHOLD).sum().item())
        ep_dist_traveled = ep_dist_traveled + robot_speed * step_dt

        # First-motion latency: per env, log step count when speed first
        # crosses START_SPEED_THRESHOLD. Stays -1 for envs that never
        # started moving in the current episode.
        moving_now   = robot_speed > START_SPEED_THRESHOLD
        not_yet_moved = ep_first_motion_step < 0
        just_started = moving_now & not_yet_moved
        ep_first_motion_step = torch.where(
            just_started, ep_step_counter, ep_first_motion_step,
        )

        # Path C: integrate world-frame xy velocity to get displacement
        # from spawn. Mark first step where displacement crosses
        # GOAL_REACH_DISTANCE; -1 means never reached this episode.
        with torch.no_grad():
            v_w = inner.scene["robot"].data.root_lin_vel_w[:, :2]
        ep_disp_x = ep_disp_x + v_w[:, 0] * step_dt
        ep_disp_y = ep_disp_y + v_w[:, 1] * step_dt
        displacement = (ep_disp_x.pow(2) + ep_disp_y.pow(2)).sqrt()

        # v3.0 task progress: project robot velocity onto commanded
        # direction. inner.last_u_des is set by env.step() right after the
        # CBF filter; (vx, vy, yaw) — we want xy components. Skip steps
        # where teleop is commanding ~0 (no direction defined).
        if hasattr(inner, "last_u_des"):
            with torch.no_grad():
                u_des_xy = inner.last_u_des[:, :2]
                u_des_norm = u_des_xy.norm(dim=-1)
                active = u_des_norm > 1e-3
                if active.any():
                    v_proj = (v_w * u_des_xy).sum(dim=-1) / u_des_norm.clamp(min=1e-6)
                    sum_v_along_cmd += float(v_proj[active].sum().item())
                    sum_v_along_cmd_count += int(active.sum().item())
        reached_now      = displacement > GOAL_REACH_DISTANCE
        not_yet_reached  = ep_first_goal_step < 0
        just_reached_now = reached_now & not_yet_reached
        ep_first_goal_step = torch.where(
            just_reached_now, ep_step_counter, ep_first_goal_step,
        )

        ep_step_counter = ep_step_counter + 1

        if hasattr(inner, "last_infeasibility"):
            sum_infeas_steps += float(inner.last_infeasibility.sum().item())

        # Snapshot the per-env obstacle_contact done flag for THIS step
        # BEFORE we handle the done-loop, since auto-reset may clear it.
        if has_obstacle_term:
            try:
                obstacle_contact_now = inner.termination_manager.get_term(
                    "obstacle_contact"
                )
            except Exception:
                obstacle_contact_now = torch.zeros(N, dtype=torch.bool, device=device)
        else:
            obstacle_contact_now = torch.zeros(N, dtype=torch.bool, device=device)

        # 2026-05-22: same snapshot for the goal_reached termination (used
        # by eval-corridor-strict). Returns all-False on tasks without it,
        # so the elif branch below is a no-op for legacy tasks.
        try:
            goal_reached_now = inner.termination_manager.get_term(
                "goal_reached"
            )
        except Exception:
            goal_reached_now = torch.zeros(N, dtype=torch.bool, device=device)

        done = terminated | truncated
        # V13 BS-A: clear the student's per-env history buffer for envs
        # that just reset. Otherwise the history would span the boundary
        # between episodes, corrupting the student's input distribution.
        # Pre-fill with the post-reset obs's proprio (replicated across the
        # full history window) so the student doesn't have to spend 50 steps
        # warming up from zeros at the start of every episode.
        if bs_adapter_state is not None and done.any():
            P_hidden = bs_adapter_state["priv_hidden_dim"]
            P_total = bs_adapter_state["priv_total_dim"]
            F_a = bs_adapter_state["prev_action"].shape[-1]
            obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
            new_proprio = obs_tensor[:, P_hidden:P_total]                 # (N, F_p)
            zero_action = torch.zeros(N, F_a, device=device)
            fill_step = torch.cat([new_proprio, zero_action], dim=-1)      # (N, F_p+F_a)
            H = bs_adapter_state["history"].shape[1]
            fill = fill_step.unsqueeze(1).expand(-1, H, -1).clone()        # (N, H, F)
            bs_adapter_state["history"][done] = fill[done]
            bs_adapter_state["prev_action"][done] = 0.0
        if done.any():
            done_idxs = done.nonzero(as_tuple=False).squeeze(-1).tolist()
            for e in done_idxs:
                n_eps += 1
                # Snapshot the COMPLETED episode's scene type (auto-reset
                # has not yet rewritten inner.cbf_scene_is_corridor for
                # this env at this point — but to be safe we read from
                # the cloned tag captured at the prior episode's start).
                is_corridor = bool(ep_scene_is_corridor[e].item())
                if is_corridor:
                    n_eps_corridor += 1
                else:
                    n_eps_open += 1
                perceived_collision = ep_min_h[e].item() < 0.0
                actual_obstacle_contact = bool(obstacle_contact_now[e].item())
                goal_term_now = bool(goal_reached_now[e].item())

                # 2026-05-22: navigation_success counted INDEPENDENTLY of
                # collision/fall/timeout buckets. An episode can both reach
                # the goal AND have h<0 transients during traversal — those
                # are different facts. Headline metric = navigation_success.
                # collision_rate is a secondary safety stat.
                if goal_term_now:
                    n_goal_reached_term += 1
                    if is_corridor: n_goal_reached_term_corridor += 1
                    else:           n_goal_reached_term_open += 1

                # Original failure-mode categorization (mutually exclusive
                # among themselves). Note: goal_reached episodes may also
                # appear in collision buckets if h<0 happened during
                # corridor traversal; this is expected.
                if perceived_collision:
                    n_collision += 1
                    if is_corridor: n_collision_corridor += 1
                    else:           n_collision_open += 1
                    if actual_obstacle_contact:
                        n_collision_actual += 1
                    else:
                        n_collision_perceived_only += 1
                elif goal_term_now:
                    # Episode succeeded without margin breach — count as
                    # neither fall nor timeout. (Already counted in
                    # n_goal_reached_term above.)
                    pass
                elif bool(terminated[e].item()):
                    n_fall += 1
                    if is_corridor: n_fall_corridor += 1
                    else:           n_fall_open += 1
                else:
                    n_timeout += 1
                    if is_corridor: n_timeout_corridor += 1
                    else:           n_timeout_open += 1
                    if low_speed_streak[e].item() >= STUCK_WINDOW:
                        n_stuck += 1
                        if is_corridor: n_stuck_corridor += 1
                        else:           n_stuck_open += 1

                # v2.15 (Path A) per-episode aggregations.
                ep_min_h_val = float(ep_min_h[e].item())
                if ep_min_h_val != float("inf"):
                    sum_episode_min_h += ep_min_h_val
                    sum_episode_min_h_count += 1
                sum_episode_dist += float(ep_dist_traveled[e].item())
                sum_episode_dist_count += 1
                if ep_first_motion_step[e].item() >= 0:
                    sum_start_latency += float(ep_first_motion_step[e].item())
                    sum_start_latency_count += 1

                # Path C goal-reach aggregations.
                final_disp = float(displacement[e].item())
                sum_final_displacement += final_disp
                sum_final_displacement_count += 1
                if ep_first_goal_step[e].item() >= 0:
                    n_goal_reached += 1
                    if is_corridor: n_goal_reached_corridor += 1
                    else:           n_goal_reached_open += 1
                    sum_time_to_goal += float(ep_first_goal_step[e].item())
                    sum_time_to_goal_count += 1

                # v3.0 path efficiency: ratio of net displacement to total
                # walked distance. Episodes that barely moved are noise —
                # threshold at 10cm so we only score real motion.
                ep_dist = float(ep_dist_traveled[e].item())
                if ep_dist > 0.1:
                    sum_path_efficiency += final_disp / ep_dist
                    sum_path_efficiency_count += 1

                # Reset per-env trackers for the NEXT episode on this env.
                ep_min_h[e] = float("inf")
                low_speed_streak[e] = 0
                ep_dist_traveled[e] = 0.0
                ep_first_motion_step[e] = -1
                ep_step_counter[e] = 0
                ep_disp_x[e] = 0.0
                ep_disp_y[e] = 0.0
                ep_first_goal_step[e] = -1
                # Snapshot the NEW episode's scene type (env auto-reset has
                # already run randomize_obstacles_position by this point,
                # which updated inner.cbf_scene_is_corridor for this env).
                if hasattr(inner, "cbf_scene_is_corridor"):
                    ep_scene_is_corridor[e] = inner.cbf_scene_is_corridor[e]

    eps_safe = max(n_eps, 1)
    eps_corridor_safe = max(n_eps_corridor, 1)
    eps_open_safe = max(n_eps_open, 1)
    return {
        "n_episodes":         n_eps,
        "collision_rate":     n_collision / eps_safe,
        "collision_rate_actual":         n_collision_actual / eps_safe,
        "collision_rate_perceived_only": n_collision_perceived_only / eps_safe,
        "fall_rate":          n_fall      / eps_safe,
        "timeout_rate":       n_timeout   / eps_safe,
        "stuck_rate":         n_stuck     / eps_safe,
        # Per-scene-type telemetry (2026-05-16). Tells us the corridor
        # failure mode separately from random-obstacle failures.
        "n_episodes_corridor":     n_eps_corridor,
        "collision_rate_corridor": n_collision_corridor / eps_corridor_safe,
        "fall_rate_corridor":      n_fall_corridor      / eps_corridor_safe,
        "timeout_rate_corridor":   n_timeout_corridor   / eps_corridor_safe,
        "stuck_rate_corridor":     n_stuck_corridor     / eps_corridor_safe,
        "goal_reach_rate_corridor": n_goal_reached_corridor / eps_corridor_safe,
        "n_episodes_open":         n_eps_open,
        "collision_rate_open":     n_collision_open / eps_open_safe,
        "fall_rate_open":          n_fall_open      / eps_open_safe,
        "timeout_rate_open":       n_timeout_open   / eps_open_safe,
        "stuck_rate_open":         n_stuck_open     / eps_open_safe,
        "goal_reach_rate_open":    n_goal_reached_open / eps_open_safe,
        "infeasibility_rate": sum_infeas_steps / max(total_steps, 1),
        "mean_h":             sum_min_h    / max(sum_min_h_count, 1),
        "mean_phi_used":      sum_phi_step / max(phi_step_count, 1),
        # v2.15 (Path A) richer 2-axis metrics:
        "mean_min_h":          sum_episode_min_h    / max(sum_episode_min_h_count, 1),
        "frac_steps_close":    sum_h_close          / max(total_steps, 1),
        "mean_v_xy":           sum_v_xy             / max(total_steps, 1),
        "mean_start_latency":  sum_start_latency    / max(sum_start_latency_count, 1),
        "mean_dist_traveled":  sum_episode_dist     / max(sum_episode_dist_count, 1),
        # Path C goal-reach metrics (displacement from spawn proxy):
        "goal_reach_rate":         n_goal_reached         / eps_safe,
        "mean_time_to_goal":       sum_time_to_goal       / max(sum_time_to_goal_count, 1),
        "mean_final_displacement": sum_final_displacement / max(sum_final_displacement_count, 1),
        # 2026-05-22: termination-based navigation success.
        # Counts episodes that terminated with `goal_reached` (env was within
        # the threshold of the active goal). 0 on tasks without that
        # termination wired up. Use this — not `goal_reach_rate` — for
        # navigation-task success metrics on eval-corridor-strict.
        "navigation_success_rate":          n_goal_reached_term          / eps_safe,
        "navigation_success_rate_corridor": n_goal_reached_term_corridor / eps_corridor_safe,
        "navigation_success_rate_open":     n_goal_reached_term_open     / eps_open_safe,
        # v3.0 task progress richer metrics:
        "path_efficiency":   sum_path_efficiency / max(sum_path_efficiency_count, 1),
        "mean_v_along_cmd":  sum_v_along_cmd     / max(sum_v_along_cmd_count, 1),
        # v2.16 diagnostic: per-task average of inner._cbf_log values.
        # cbf_*_std reflects across-env variance at one step — high means
        # the policy is producing different α/φ/a/c across envs at the
        # same step, i.e., responding to state differences.
        **{f"avg_{k}": cbf_log_sums[k] / max(cbf_log_count, 1) for k in cbf_log_keys},
    }


# ──────────────────────────────────────────────────────────────────────
# Sweep config builder
# ──────────────────────────────────────────────────────────────────────

def parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def build_configs(modes: list[str], device, N, args):
    """Produce a list of dicts with keys {name, mode, action_fn, params}."""
    alphas    = parse_floats(args.alpha_grid)
    phis      = parse_floats(args.phi_grid)
    eps0s     = parse_floats(args.epsilon0_grid)
    lambdas   = parse_floats(args.lambda_grid)
    configs: list[dict] = []

    if "B0" in modes:
        for alpha in alphas:
            configs.append({
                "name":   f"B0_α={alpha:.2f}",
                "mode":   "B0",
                "params": {"alpha": alpha},
                "action_fn": make_b0_provider(alpha, device, N),
            })

    if "B1" in modes:
        for alpha in alphas:
            for phi in phis:
                configs.append({
                    "name":   f"B1_α={alpha:.2f}_φ={phi:.2f}",
                    "mode":   "B1",
                    "params": {"alpha": alpha, "phi": phi},
                    "action_fn": make_b1_provider(alpha, phi, device, N),
                })

    if "B2" in modes:
        for alpha in alphas:
            for eps0 in eps0s:
                for lam in lambdas:
                    configs.append({
                        "name":   f"B2_α={alpha:.2f}_ε₀={eps0:.2f}_λ={lam:.2f}",
                        "mode":   "B2",
                        "params": {"alpha": alpha, "epsilon0": eps0, "lambda": lam},
                        "action_fn": make_b2_provider(alpha, eps0, lam, device, N),
                    })

    return configs


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(args_cli.seed)
    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = [m.strip() for m in args_cli.modes.split(",") if m.strip()]
    print(f"[modes] {modes}")

    needs_ckpt = ("BR" in modes) or ("BR-shrunk" in modes) or any(m.startswith("Bf-") for m in modes)
    if needs_ckpt and not args_cli.checkpoint:
        raise SystemExit("--checkpoint required when BR or Bf-X is in --modes")
    if "BS" in modes and not args_cli.student_checkpoint:
        raise SystemExit("--student_checkpoint required when BS is in --modes")
    if "BS-A" in modes and not args_cli.student_adapter_checkpoint:
        raise SystemExit("--student_adapter_checkpoint required when BS-A is in --modes")
    if "BS-A" in modes and not args_cli.checkpoint:
        raise SystemExit("--checkpoint (teacher) required when BS-A is in --modes")

    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    # Seed the env (DR sampling, episode init, obstacle layout) — without this
    # eval-to-eval variance is 5–15pp on BR composite. With seed pinned, the
    # only remaining variance is from intra-episode CBF noise (small).
    env_cfg.seed = args_cli.seed
    print(f"[eval] env_cfg.seed={env_cfg.seed} (DR + episode init pinned)")

    if args_cli.no_obstacles:
        # Force K_actual=0 — every obstacle slot stays parked off-stage.
        # CBF h(x) is huge everywhere → no constraint binds → u_safe = u_des
        # always. This isolates the planner+locomotion pair: any falls now
        # are caused by the planner output being too jerky for locomotion,
        # NOT by CBF deflection.
        env_cfg.events.randomize_obstacles_position.params["k_min"] = 0
        env_cfg.events.randomize_obstacles_position.params["k_max"] = 0
        print("[diag] --no_obstacles: K_actual forced to 0 (planner+loco isolation)")

    # v2.11: by default, force eval-time deploy-realistic locked planner.
    # Training uses bimodal_resample=True (mid-switch [5,15]s OR locked 100s)
    # for the stuck-recovery regularizer, but eval should match real Go2
    # deployment which has a single locked nav stack. The training/eval
    # mismatch is the central technique. --bimodal_resample on the CLI
    # forces it on at eval (debug / training-distribution verification only).
    if args_cli.bimodal_resample:
        print("[eval] --bimodal_resample: keeping bimodal_resample=True at eval")
    else:
        if hasattr(env_cfg.commands.base_velocity, "bimodal_resample"):
            env_cfg.commands.base_velocity.bimodal_resample = False
            env_cfg.commands.base_velocity.resampling_time_range = (100.0, 100.0)
            print("[eval] forced bimodal_resample=False, resample=(100,100) — deploy-realistic locked")

    if args_cli.planner_resample_s is not None:
        # Probe A (2026-05-06): override resampling_time_range to test whether
        # v2.8's policy is brittle outside its locked-planner training regime.
        # v2.11 dual-regime eval: pass --planner_resample_s 10 to match v2.6's
        # mid-switch eval regime (apples-to-apples comparison vs v2.6 paper).
        s = float(args_cli.planner_resample_s)
        env_cfg.commands.base_velocity.resampling_time_range = (s, s)
        # Also turn off bimodal so the override actually takes effect.
        if hasattr(env_cfg.commands.base_velocity, "bimodal_resample"):
            env_cfg.commands.base_velocity.bimodal_resample = False
        print(f"[diag] --planner_resample_s={s}: planner switches every {s}s")

    if args_cli.solo_planner is not None:
        # DIAG-4b sub-bisection: force one planner at 100% weight, zero out
        # all others. Same MultiPlannerCommand class — only the mix changes.
        # Combined with --no_obstacles + locked-planner eval (default), this
        # measures THIS planner's fall-rate contribution in isolation against
        # the DR-only floor measured by --vanilla_uniform_command.
        bv = env_cfg.commands.base_velocity
        weight_attrs = ("uniform_weight", "goal_weight", "walk_weight",
                        "adversarial_weight", "smooth_goal_weight",
                        "waypoint_weight", "mpc_weight")
        # Map "legacy_goal" CLI choice → goal_weight (legacy naming).
        cli_to_attr = {
            "walk":         "walk_weight",
            "adversarial":  "adversarial_weight",
            "smooth_goal":  "smooth_goal_weight",
            "waypoint":     "waypoint_weight",
            "mpc":          "mpc_weight",
            "legacy_goal":  "goal_weight",
            "uniform":      "uniform_weight",
        }
        target_attr = cli_to_attr[args_cli.solo_planner]
        for w in weight_attrs:
            if hasattr(bv, w):
                setattr(bv, w, 1.0 if w == target_attr else 0.0)
        print(f"[diag] --solo_planner={args_cli.solo_planner}: "
              f"{target_attr}=1.0, all other planner weights=0.0")

    if args_cli.vanilla_uniform_command:
        # DIAG-4b (2026-05-08): replace MultiPlannerCommand with the native
        # locomotion's training-time UniformVelocityCommand. Same ranges,
        # heading mode, and 10s resample as Isaac-Velocity-Flat-Unitree-Go2-v0.
        # Holds DR + obstacle setup constant — isolates whether our planner mix
        # (smooth_goal/wp/mpc/walk/adv at 50 Hz inner) is what stresses
        # locomotion (vs DR axes alone).
        import math
        from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

        _old_bv = env_cfg.commands.base_velocity
        env_cfg.commands.base_velocity = UniformVelocityCommandCfg(
            asset_name=_old_bv.asset_name,
            resampling_time_range=(10.0, 10.0),         # native loco training default
            debug_vis=False,
            rel_standing_envs=0.0,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=1.0,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
                heading=(-math.pi, math.pi),
            ),
        )
        print(f"[diag] --vanilla_uniform_command: replaced MultiPlannerCommand "
              f"→ UniformVelocityCommand (native loco training distribution; "
              f"10s resample, heading mode)")

    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    device = inner.device
    N = inner.num_envs
    print(f"[env] {args_cli.task}, num_envs={N}, device={device}")

    # Sync PARAM_RANGES["c"] with the env's actual _c_param_range BEFORE
    # building action providers.
    # Bug fix (2026-05-16): module-level PARAM_RANGES["c"] is hardcoded to
    # (0.0, 1.0) under WIDE_PARAM_RANGES, which doesn't match LAYER3_PUSH_A_C's
    # symmetric (-0.20, +0.20). When encode_action(c=0.0) ran with the wrong
    # (lo, hi), the inverse-tanh produced a raw action that the env's
    # _cbf_filter (using its actual range) decoded as c = c_lo (the lower
    # cap). On LAYER3_PUSH_A_C that's c=-0.20, NOT c=0 — so fixed baselines
    # ran with a phantom-outward shifted boundary, inflating their collision
    # rates and tainting the BR-vs-baseline comparison.
    if hasattr(inner, "_c_param_range"):
        env_c_lo, env_c_hi = inner._c_param_range
        if (float(env_c_lo), float(env_c_hi)) != PARAM_RANGES["c"]:
            print(
                f"[eval] aligning PARAM_RANGES['c'] {PARAM_RANGES['c']} "
                f"→ env._c_param_range ({env_c_lo:.4f}, {env_c_hi:.4f})"
            )
            PARAM_RANGES["c"] = (float(env_c_lo), float(env_c_hi))

    # Same alignment for α range. Bug fix (2026-05-16, post AClamp):
    # PARAM_RANGES["alpha"] is hardcoded (0.1, 5.0); LAYER3_PUSH_A_C_ACLAMP
    # clamps env range to (0.5, 3.0). Without alignment, baseline target
    # α=2.0 gets encoded against (0.1, 5.0) → raw → decoded by env against
    # (0.5, 3.0) → actual α ≈ 1.47. Off-target baselines.
    if hasattr(inner, "_alpha_param_range"):
        env_a_lo, env_a_hi = inner._alpha_param_range
        if (float(env_a_lo), float(env_a_hi)) != PARAM_RANGES["alpha"]:
            print(
                f"[eval] aligning PARAM_RANGES['alpha'] {PARAM_RANGES['alpha']} "
                f"→ env._alpha_param_range ({env_a_lo:.4f}, {env_a_hi:.4f})"
            )
            PARAM_RANGES["alpha"] = (float(env_a_lo), float(env_a_hi))

    # Build configs.
    configs = build_configs(modes, device, N, args_cli)

    # BR provider needed by both BR and any Bf-X mode
    bf_modes = [m for m in modes if m.startswith("Bf-")]
    needs_br_provider = ("BR" in modes) or ("BR-shrunk" in modes) or (len(bf_modes) > 0)

    br_provider = None
    if needs_br_provider:
        if not args_cli.checkpoint:
            raise SystemExit(
                "--checkpoint required when BR or any Bf-X is in --modes"
            )
        from isaaclab_tasks.utils import load_cfg_from_registry
        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        br_provider = make_br_provider(
            Path(args_cli.checkpoint), env, agent_cfg, device,
        )

    if "BR" in modes:
        configs.append({
            "name":      "BR_teacher",
            "mode":      "BR",
            "params":    {"checkpoint": str(args_cli.checkpoint)},
            "action_fn": br_provider,
        })

    # BS: distilled student. Loaded from --student_checkpoint via the same
    # rsl_rl OnPolicyRunner path as BR (student inherits teacher MLP shape).
    # Reuses agent_cfg if BR is also in modes; loads it standalone otherwise.
    if "BS" in modes:
        if br_provider is None:
            from isaaclab_tasks.utils import load_cfg_from_registry
            agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        bs_provider = make_bs_provider(
            Path(args_cli.student_checkpoint), env, agent_cfg, device,
        )
        configs.append({
            "name":      "BS_student",
            "mode":      "BS",
            "params":    {"student_checkpoint": str(args_cli.student_checkpoint)},
            "action_fn": bs_provider,
        })

    # BS-A: V13 two-stream student-adapter. Loads teacher from --checkpoint,
    # student adapter from --student_adapter_checkpoint, then monkey-patches
    # priv_encoder so the teacher's policy() call internally uses the
    # student's ẑ_env prediction from the temporal history.
    bs_adapter_state = None  # exposed so we can reset history on env reset
    if "BS-A" in modes:
        if br_provider is None:
            from isaaclab_tasks.utils import load_cfg_from_registry
            agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        bs_adapter_provider, bs_adapter_state = make_bs_adapter_provider(
            Path(args_cli.checkpoint),
            Path(args_cli.student_adapter_checkpoint),
            env, agent_cfg, device, args_cli.num_envs,
        )
        configs.append({
            "name":      "BS-A_student_adapter",
            "mode":      "BS-A",
            "params":    {
                "teacher": str(args_cli.checkpoint),
                "student_adapter": str(args_cli.student_adapter_checkpoint),
            },
            "action_fn": bs_adapter_provider,
            "_bs_adapter_state": bs_adapter_state,
        })

    # v2.11 Table-2 ablations: Bf-alpha / Bf-phi / Bf-a / Bf-c.
    # Each runs the BR policy but overrides one CBF slot with a fixed
    # physical value. Compare BR vs Bf-X to show that slot X is doing
    # adaptive work.
    bf_specs = [
        ("Bf-alpha", "alpha", args_cli.bf_alpha_target),
        ("Bf-phi",   "phi",   args_cli.bf_phi_target),
        ("Bf-a",     "a",     args_cli.bf_a_target),
        ("Bf-c",     "c",     args_cli.bf_c_target),
    ]
    for mode_name, slot, target in bf_specs:
        if mode_name in modes:
            configs.append({
                "name":      f"{mode_name}_target={target:.2f}",
                "mode":      mode_name,
                "params":    {f"{slot}_target": target,
                              "checkpoint": str(args_cli.checkpoint)},
                "action_fn": make_b_fixed_provider(slot, target, br_provider, device, N),
            })

    # v2.16-diagnostic: Bf-all (4 slots fixed simultaneously) + Bhc-alpha
    # (state-conditional α). Both ignore the policy entirely — they're
    # diagnostic baselines for whether adaptation has signal at all.
    if "Bf-all" in modes:
        configs.append({
            "name":      f"Bf-all_α={args_cli.bf_alpha_target:.2f}_"
                         f"φ={args_cli.bf_phi_target:.2f}_"
                         f"a={args_cli.bf_a_target:.3f}_"
                         f"c={args_cli.bf_c_target:.3f}",
            "mode":      "Bf-all",
            "params":    {"alpha_target": args_cli.bf_alpha_target,
                          "phi_target":   args_cli.bf_phi_target,
                          "a_target":     args_cli.bf_a_target,
                          "c_target":     args_cli.bf_c_target},
            "action_fn": make_bf_all_provider(
                args_cli.bf_alpha_target, args_cli.bf_phi_target,
                args_cli.bf_a_target, args_cli.bf_c_target,
                device, N,
            ),
        })

    if "BR-shrunk" in modes:
        # Reuses br_provider built above for the BR/Bf-X path.
        if br_provider is None:
            raise SystemExit("BR-shrunk requires --checkpoint (BR provider).")
        slot_list = [s.strip() for s in args_cli.brs_slots.split(",") if s.strip()]
        configs.append({
            "name":      f"BR-shrunk_s={args_cli.brs_shrink:.2f}_"
                         f"slots={'+'.join(slot_list)}",
            "mode":      "BR-shrunk",
            "params":    {"shrink": args_cli.brs_shrink,
                          "slots":  args_cli.brs_slots},
            "action_fn": make_br_shrunk_provider(
                br_provider, args_cli.brs_shrink, slot_list, device, N,
            ),
        })

    if "Bhc-alpha" in modes:
        configs.append({
            "name":      f"Bhc-alpha_floor={args_cli.hca_floor:.2f}_"
                         f"ceil={args_cli.hca_ceil:.2f}_λ={args_cli.hca_lambda:.2f}",
            "mode":      "Bhc-alpha",
            "params":    {"alpha_floor":  args_cli.hca_floor,
                          "alpha_ceil":   args_cli.hca_ceil,
                          "alpha_lambda": args_cli.hca_lambda,
                          "phi":          args_cli.hca_phi,
                          "a":            args_cli.hca_a,
                          "c":            args_cli.hca_c},
            "action_fn": make_bhc_alpha_provider(
                args_cli.hca_floor, args_cli.hca_ceil, args_cli.hca_lambda,
                args_cli.hca_phi, args_cli.hca_a, args_cli.hca_c,
                device, N,
            ),
        })

    print(f"[sweep] {len(configs)} configs × {args_cli.steps_per_config} steps each")

    # Run.
    results = []
    for i, cfg in enumerate(configs):
        stats = eval_one(env, cfg["action_fn"], args_cli.steps_per_config, device, N,
                         bs_adapter_state=cfg.get("_bs_adapter_state"))
        print(
            f"[{i+1:>2d}/{len(configs)}] {cfg['name']:<32s}  "
            f"eps={stats['n_episodes']:3d}  "
            f"coll={stats['collision_rate']:.3f}  "
            f"fall={stats['fall_rate']:.3f}  "
            f"stuck={stats['stuck_rate']:.3f}  "
            f"v̄_xy={stats['mean_v_xy']:.3f}  "
            f"min_h̄={stats['mean_min_h']:+.3f}  "
            f"close%={stats['frac_steps_close']:.3f}  "
            f"goal%={stats['goal_reach_rate']:.3f}"
        )
        results.append({
            "name":   cfg["name"],
            "mode":   cfg["mode"],
            **cfg["params"],
            **stats,
        })

    # CSV.
    csv_path = output_dir / "baseline.csv"
    fieldnames = sorted({k for r in results for k in r.keys()})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\n[csv] wrote {csv_path}")

    # Plot: 2-panel scatter. CBF prevents virtually all collisions in
    # this env — collision_rate is uniformly 0, so the meaningful failure
    # axis is fall_rate. Show fall_rate vs both φ-spend (X1: how big a
    # buffer was used) and mean h (X2: how far from obstacles the policy
    # actually stayed). BR's mean_h being high while fall_rate is high
    # is the diagnostic for "over-conservative" learned behavior.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = {"B0": "gray", "B1": "tab:blue", "B2": "tab:orange",
                  "BR": "red", "BS": "purple",
                  "Bf-all": "tab:green", "Bhc-alpha": "tab:cyan"}
        markers = {"B0": "s", "B1": "o", "B2": "^", "BR": "*", "BS": "P",
                   "Bf-all": "D", "Bhc-alpha": "X"}
        for mode in colors:
            rs = [r for r in results if r["mode"] == mode]
            if not rs:
                continue
            ys = [r["fall_rate"] for r in rs]
            size = 200 if mode == "BR" else 80
            ax1.scatter([r["mean_phi_used"] for r in rs], ys, label=mode,
                        c=colors[mode], marker=markers[mode], s=size, alpha=0.7)
            ax2.scatter([r["mean_h"] for r in rs], ys, label=mode,
                        c=colors[mode], marker=markers[mode], s=size, alpha=0.7)
        for ax, xlab in [(ax1, "Mean φ actually used"),
                         (ax2, "Mean h(x) (margin from obstacles)")]:
            ax.set_xlabel(xlab)
            ax.set_ylabel("Fall rate")
            ax.legend()
            ax.grid(alpha=0.3)
        ax1.set_title("Buffer size vs locomotion stability")
        ax2.set_title("Distance from obstacles vs locomotion stability")
        plt.tight_layout()
        plot_path = output_dir / "baseline.png"
        plt.savefig(plot_path, dpi=150)
        print(f"[plot] wrote {plot_path}")
    except ImportError:
        print("[plot] matplotlib unavailable, skipping PNG.")

    env.close()


if __name__ == "__main__":
    import os
    rc = 0
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    # Skip simulation_app.close() — v17 confirmed it hangs unreliably (10+ min
    # waits observed). OS reclaims handles when the process dies via os._exit.
    os._exit(rc)
