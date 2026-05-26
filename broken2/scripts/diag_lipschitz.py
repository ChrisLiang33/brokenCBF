"""DIAG-3: Lipschitz constant estimator for the CBF teacher policy.

============================================================
WHAT IS THE LIPSCHITZ CONSTANT?
============================================================

A function f: X → Y is "L-Lipschitz" if for any two inputs x1 and x2,
the output change is bounded by L times the input change:

    ||f(x1) - f(x2)||  ≤  L · ||x1 - x2||

The smallest L for which this holds is THE Lipschitz constant of f.
You can read it as "the maximum amplification factor between input
and output."

Examples:
- f(x) = x:           L = 1   (output changes 1:1 with input)
- f(x) = 5x:          L = 5   (output changes 5x faster than input)
- f(x) = sin(x):      L = 1   (slope is bounded by ±1)
- f(x) = x²:          L = ∞   (slope grows without bound)
- f(x) = constant:    L = 0   (output never changes — flat)

For a neural network f(x) = W_n · σ(W_{n-1} · ... · σ(W_1·x) ...),
where σ is an activation with Lipschitz constant ≤ 1 (ELU, ReLU,
tanh all qualify), the overall L is bounded above by the product
of spectral norms of the linear/conv layers:

    L  ≤  ∏ σ_max(W_i)

where σ_max(W) is the largest singular value of W (the "operator
norm" — the maximum stretch factor when applying W to any vector).

============================================================
WHY WE CARE FOR THIS PROJECT
============================================================

Our policy is f: priv_obs (8207-D) → CBF params (5-D, 4 effective:
α, φ, a, c). Each step, the obs changes a little (robot moved,
obstacles moved). If our f has high L, those small obs changes get
amplified into BIG CBF param changes step-to-step. That makes
u_safe (the CBF-filtered velocity command) jerky. The locomotion
policy was trained to track smooth commands and can't follow jerky
ones — so the robot falls.

This is the "Lipschitz continuity problem" your professor flagged
early in the project. v2.5 tries to reduce L via two mechanisms:
- Action-rate penalty:  bounds the *temporal* L empirically (in
                        the reward, not the network architecture)
- Weight decay:         bounds the *network* L by shrinking the
                        weight magnitudes (smaller σ_max)

This script measures whether those mechanisms actually worked —
i.e., is the v2.5 checkpoint's L lower than v2.4's?

============================================================
THREE ESTIMATION METHODS (cheapest to most expensive)
============================================================

1. SPECTRAL NORM PRODUCT
   - Concept: for each linear/conv layer, compute σ_max(W_i),
     then multiply them together. ELU activations contribute
     factor ≤ 1, so they don't add to the bound.
   - Caveat: this is an UPPER BOUND, often loose by 10-100x.
     It's a worst-case "what's the absolute max L could be."
   - Cost: ~1 second, just SVDs of weight matrices.
   - When useful: comparing two checkpoints — if one has 10x
     smaller spectral product, its L is provably ≤ 10x smaller.

2. LOCAL JACOBIAN
   - Concept: at a specific input x, compute J = ∂f/∂x (a matrix).
     σ_max(J) is the LOCAL Lipschitz constant — how much f
     amplifies in the immediate neighborhood of x.
   - Caveat: this is state-dependent. Different x's give different
     local L's. We average over many x's to get a typical value.
   - Cost: ~1 second per state via torch.autograd; we sample N
     states → N seconds. Much tighter than method 1.
   - When useful: actual sensitivity at typical operating points.

3. EMPIRICAL FINITE-DIFFERENCE
   - Concept: pick state x, perturb to x + δ, measure
     ||f(x+δ) - f(x)|| / ||δ||. This is a discrete-secant
     approximation to the Jacobian's σ_max along direction δ.
   - Caveat: only sees L *along the perturbation directions*
     you sample. For a true L estimate, average over many
     random δ directions and take the max.
   - Cost: ~2 forward passes per measurement. Cheap.
   - When useful: gives the most realistic answer for "if obs
     changes by THIS amount in THIS direction, how much does
     output change?" Match perturbations to real obs noise
     (e.g., one grid-cell flip = real LiDAR scan changing).

============================================================
USAGE
============================================================

Load weights from a saved checkpoint and report all three.
Pure PyTorch — no Isaac Sim needed (we re-implement the
forward pass directly from the state-dict so we don't have
to launch the simulator just to look at the weights):

    python scripts/diag_lipschitz.py \\
        --checkpoint IsaacLab/logs/rsl_rl/cbf_go2_teacher/<run>/model_2999.pt \\
        --num_samples 100

Compare v2.4 and v2.5:

    python scripts/diag_lipschitz.py \\
        --checkpoint IsaacLab/logs/rsl_rl/cbf_go2_teacher/2026-05-03_13-44-46/model_2999.pt \\
        --label v2.4

    python scripts/diag_lipschitz.py \\
        --checkpoint IsaacLab/logs/rsl_rl/cbf_go2_teacher/2026-05-03_18-54-48/model_2999.pt \\
        --label v2.5

Then eyeball the differences. v2.5 should have noticeably
smaller numbers if weight decay is doing its job.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


# ---- Architecture constants (must match cbf_go2_teacher_cnn.py) ----------
# We need these to manually unpack the obs vector and route the right
# slices to the right layers. Hardcoded here to keep the script self-
# contained (no Isaac Lab import → no simulator launch needed).
DYN_DIM = 15           # First 15 dims of obs are dynamics priv (mass, friction, etc.)
GRID_CHANNELS = 2      # Two stacked occupancy grid frames (current + previous)
GRID_H = 64            # Grid height in cells
GRID_W = 64            # Grid width in cells
OBS_DIM = DYN_DIM + GRID_CHANNELS * GRID_H * GRID_W   # 15 + 2*64*64 = 8207
ACTION_DIM = 5         # CBF params (α, φ, a, b, c) — b unused but present


# ---- State-dict key map (matches the saved checkpoint structure) --------
# The model wraps everything in `mlp.0.*` (encoder) and `mlp.1.*` (head MLP).
# These keys are inferred from the boot-time print of the model class —
# if you ever rename layers in cbf_go2_teacher_cnn.py, update these.
KEYS = {
    "dyn_w":      "mlp.0.dyn_path.0.weight",
    "dyn_b":      "mlp.0.dyn_path.0.bias",
    "conv1_w":    "mlp.0.conv.0.weight",
    "conv1_b":    "mlp.0.conv.0.bias",
    "conv2_w":    "mlp.0.conv.2.weight",
    "conv2_b":    "mlp.0.conv.2.bias",
    "grid_w":     "mlp.0.grid_proj.0.weight",
    "grid_b":     "mlp.0.grid_proj.0.bias",
    "head1_w":    "mlp.0.head.0.weight",
    "head1_b":    "mlp.0.head.0.bias",
    "head2_w":    "mlp.0.head.2.weight",
    "head2_b":    "mlp.0.head.2.bias",
    "out1_w":     "mlp.1.0.weight",
    "out1_b":     "mlp.1.0.bias",
    "out2_w":     "mlp.1.2.weight",
    "out2_b":     "mlp.1.2.bias",
}


def _flatten_state_dict(d, prefix=""):
    """Recursively flatten nested dicts into a single {dotted-key: tensor} dict.

    rsl_rl checkpoints can nest several layers deep:
        blob = {"model_state_dict": {"actor": OrderedDict(...), "critic": ...}, ...}
    We walk all sub-dicts until we hit tensors, then return one flat map.
    """
    out = {}
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_state_dict(v, full))
        elif isinstance(v, torch.Tensor):
            out[full] = v
        # ignore non-tensor leaves (ints, strings, etc.)
    return out


def load_weights(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Load the state_dict from rsl_rl's checkpoint format.

    rsl_rl checkpoints vary in structure across versions. We flatten the
    whole thing into one {dotted-key: tensor} map, then pattern-match
    our expected weight names by their *suffixes* — that way the script
    works regardless of how deeply the actor's state_dict is nested.
    """
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    flat = _flatten_state_dict(blob)

    # For each expected nice-name, find the tensor whose flattened key
    # *ends* with the expected suffix. We restrict to keys containing
    # "actor" or no critic-specific marker so we don't accidentally pick
    # up the critic head's weights (which have the same shape).
    weights = {}
    missing = []
    for nice_name, sd_suffix in KEYS.items():
        # Prefer keys that mention "actor"; fall back to any match.
        actor_matches = [
            k for k in flat
            if k.endswith(sd_suffix) and "actor" in k.lower()
            and "critic" not in k.lower()
        ]
        any_matches = [k for k in flat if k.endswith(sd_suffix)]
        match = (actor_matches or any_matches)
        if not match:
            missing.append(sd_suffix)
            continue
        # Use the first match (or warn if multiple — shouldn't happen).
        if len(match) > 1:
            print(f"[diag] note: {sd_suffix} has multiple matches, picking first: {match}")
        weights[nice_name] = flat[match[0]]

    if missing:
        # Diagnostic: dump every flat tensor key + shape so we can see
        # exactly what's in this checkpoint and update KEYS if needed.
        print(f"[diag] missing key suffixes: {missing}")
        print(f"[diag] all tensor keys in checkpoint (sorted, first 60):")
        for k in sorted(flat.keys())[:60]:
            print(f"  {k}  shape={tuple(flat[k].shape)}")
        raise KeyError(
            f"State dict missing expected key suffixes: {missing}. "
            f"See dump above to update KEYS map."
        )
    return weights


# =====================================================================
# Forward pass re-implementation
# =====================================================================
#
# We don't import the model class — we re-do the forward by hand.
# This makes the script standalone (no Isaac Lab dependency) and
# also makes the architecture explicit, which helps with the Jacobian
# math later.
#
# Input: priv_obs (B, 8207) — batch of B observation vectors
# Output: action (B, 5)     — batch of B CBF param vectors
#
# The path:
#   priv_obs (B, 8207)
#     ↓ split
#   dyn (B, 15)         ─→ Linear+ELU ─→ dyn_feat (B, 64) ─┐
#   grid (B, 2, 64, 64) ─→ Conv+ELU+Conv+ELU+flatten        ├─ concat (B, 128)
#                       ─→ Linear+ELU ─→ grid_feat (B, 64) ─┘    ↓
#                                                             head1+ELU ─→ head2+ELU ─→ Z (B, 12)
#                                                                                        ↓
#                                                                             out1+ELU ─→ out2 ─→ action (B, 5)
# =====================================================================
def forward(priv_obs: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
    """Manual forward pass through the CBF teacher policy.

    Mirrors `_GridDynamicsEncoder.forward` + `MLP.forward` from
    cbf_go2_teacher_cnn.py but uses torch.nn.functional + raw weight
    tensors instead of nn.Module instances. Easier to introspect.
    """
    B = priv_obs.shape[0]
    assert priv_obs.shape[1] == OBS_DIM, (
        f"Expected obs dim {OBS_DIM}, got {priv_obs.shape[1]}"
    )

    # --- Split obs into dynamics + grid ----------------------------------
    dyn = priv_obs[:, :DYN_DIM]                                # (B, 15)
    grid = priv_obs[:, DYN_DIM:].reshape(B, GRID_CHANNELS, GRID_H, GRID_W)

    # --- Dynamics path: 15 → 64 ------------------------------------------
    dyn_feat = F.linear(dyn, w["dyn_w"], w["dyn_b"])           # (B, 64)
    dyn_feat = F.elu(dyn_feat)

    # --- Grid path: (B, 2, 64, 64) → 64 ----------------------------------
    g = F.conv2d(grid, w["conv1_w"], w["conv1_b"], stride=2, padding=1)
    g = F.elu(g)                                               # (B, 16, 32, 32)
    g = F.conv2d(g, w["conv2_w"], w["conv2_b"], stride=2, padding=1)
    g = F.elu(g)                                               # (B, 32, 16, 16)
    g = g.reshape(B, -1)                                       # (B, 8192)
    grid_feat = F.linear(g, w["grid_w"], w["grid_b"])          # (B, 64)
    grid_feat = F.elu(grid_feat)

    # --- Fuse + head: (B, 128) → Z (B, 12) -------------------------------
    fused = torch.cat([dyn_feat, grid_feat], dim=-1)           # (B, 128)
    h = F.elu(F.linear(fused, w["head1_w"], w["head1_b"]))     # (B, 128)
    z = F.elu(F.linear(h, w["head2_w"], w["head2_b"]))         # (B, 12) — bottleneck

    # --- Output MLP: Z (12) → action (5) ---------------------------------
    o = F.elu(F.linear(z, w["out1_w"], w["out1_b"]))           # (B, 128)
    action = F.linear(o, w["out2_w"], w["out2_b"])             # (B, 5)
    return action


# =====================================================================
# Method 1 — Spectral norm product (worst-case upper bound on L)
# =====================================================================
#
# Idea: each layer is a linear map. The composition's L is bounded
# above by the product of per-layer L's. For each weight matrix W,
# its L (operator norm) equals σ_max(W) — its largest singular value.
#
# For convolutional layers, this is more nuanced (the operator norm
# differs from σ_max of the unfolded kernel matrix). We use the
# unfolded kernel matrix as a SIMPLE BUT NOT TIGHT upper bound. It's
# fine for relative comparison between checkpoints, just not the
# absolute number.
#
# ELU, ReLU, tanh activations all have L ≤ 1 — they don't amplify,
# so we ignore them in the product.
#
# Result is an UPPER BOUND on the network's L. Often very loose
# (10-100x larger than the true L), but a clean number to compare
# v2.4 vs v2.5: if v2.5's product is smaller, its L definitely is
# smaller.
# =====================================================================
def method1_spectral_product(w: dict[str, torch.Tensor], verbose: bool = True):
    # Layers contributing to the bound, in forward order along the
    # path obs → action. We don't include ELU since it's L=1.
    # Conv weights get unfolded (out, in*kH*kW) before SVD — that's
    # the simplest valid spectral-norm proxy for a Conv2D.
    layers = [
        ("dyn_path.linear",   w["dyn_w"],   "linear"),
        ("conv1",             w["conv1_w"], "conv"),
        ("conv2",             w["conv2_w"], "conv"),
        ("grid_proj.linear",  w["grid_w"],  "linear"),
        ("head.linear1",      w["head1_w"], "linear"),
        ("head.linear2",      w["head2_w"], "linear"),
        ("out_mlp.linear1",   w["out1_w"],  "linear"),
        ("out_mlp.linear2",   w["out2_w"],  "linear"),
    ]

    if verbose:
        print()
        print("─" * 70)
        print("METHOD 1: spectral-norm product (upper bound on L)")
        print("─" * 70)
        print(f"{'layer':<25} {'shape':<22} {'σ_max':>10}")

    spectral_product = 1.0
    for name, W, kind in layers:
        if kind == "conv":
            # Unfold (out_ch, in_ch, kH, kW) → (out_ch, in_ch*kH*kW)
            W_2d = W.reshape(W.shape[0], -1)
        else:
            W_2d = W
        # σ_max = first singular value (descending order).
        # svdvals returns the singular values in descending order, so [0]
        # is the largest. Cast to float64 to avoid fp32 sloppiness.
        sigma_max = torch.linalg.svdvals(W_2d.double())[0].item()
        spectral_product *= sigma_max
        if verbose:
            print(f"  {name:<23} {str(tuple(W.shape)):<22} {sigma_max:>10.3f}")

    if verbose:
        print(f"{'─' * 70}")
        print(f"  PRODUCT (upper bound on L_network):  {spectral_product:>14.3f}")
        print(
            "  Interpretation: a 1.0-norm change in obs cannot produce more\n"
            "  than this much change in action. Realistic L is usually 10-\n"
            "  100x smaller; this is a worst-case bound, not a tight one."
        )

    return spectral_product


# =====================================================================
# Method 2 — Local Jacobian (state-dependent local L)
# =====================================================================
#
# Idea: at a specific obs `x`, the local linearization of f is just
# its Jacobian J = ∂f/∂x. The local Lipschitz constant at x is
# σ_max(J). Average over many x's to get a "typical" local L.
#
# This is much TIGHTER than method 1 (it's the actual derivative,
# not a worst-case bound), but only valid in a small neighborhood
# of x — for large input changes you can't extrapolate.
#
# Cost per x: O(action_dim × forward_pass) via reverse-mode autograd
# = 5 backward passes. Run on B states in a batch with vmap if available.
# =====================================================================
def method2_local_jacobian(
    w: dict[str, torch.Tensor],
    obs_samples: torch.Tensor,
    verbose: bool = True,
):
    if verbose:
        print()
        print("─" * 70)
        print("METHOD 2: local Jacobian L (averaged over sampled states)")
        print("─" * 70)

    # Mark the input as a leaf with grad enabled. We then call forward,
    # ask autograd for the Jacobian (5×8207 per sample), and SVD it.
    obs_samples = obs_samples.clone().detach().requires_grad_(False)

    local_Ls = []
    for i, x in enumerate(obs_samples):
        x = x.unsqueeze(0).requires_grad_(True)        # (1, 8207)

        # functional.jacobian computes ∂f/∂x for f: x → action.
        # Returns shape (1, 5, 1, 8207); squeeze the batch dims.
        def _f(z):
            return forward(z, w)
        J = torch.autograd.functional.jacobian(
            _f, x, create_graph=False, vectorize=True
        )                                              # (1, 5, 1, 8207)
        J = J.reshape(ACTION_DIM, OBS_DIM)             # (5, 8207)

        # σ_max(J) = local L at this state.
        sigma = torch.linalg.svdvals(J.double())[0].item()
        local_Ls.append(sigma)

    L_tensor = torch.tensor(local_Ls)
    stats = {
        "mean":   L_tensor.mean().item(),
        "median": L_tensor.median().item(),
        "min":    L_tensor.min().item(),
        "max":    L_tensor.max().item(),
        "p95":    L_tensor.quantile(0.95).item(),
    }

    if verbose:
        print(f"  N states sampled:  {len(local_Ls)}")
        print(f"  Local L stats:")
        for k, v in stats.items():
            print(f"    {k:<8}  {v:>10.3f}")
        print(
            "  Interpretation: at a typical (mean) obs, a small obs change\n"
            "  is amplified by ~mean times in the action. The p95 tells you\n"
            "  the worst-case in the sampled distribution. Compare across\n"
            "  checkpoints to see if regularization shrunk the typical L."
        )

    return stats


# =====================================================================
# Method 3 — Empirical finite-difference (real-world sensitivity)
# =====================================================================
#
# Idea: for each obs x and each random direction δ (with chosen norm),
# measure ||f(x + δ) - f(x)|| / ||δ||. This is the secant slope along
# direction δ, which lower-bounds σ_max(J) and approaches it as δ→0.
#
# More realistic than method 2 because we choose perturbations that
# match what the policy actually sees (e.g., flipping a single grid
# cell models real LiDAR noise; small Gaussian on dyn models DR
# uncertainty).
#
# We provide three perturbation flavors:
#   - "gaussian_obs":      δ ~ N(0, σ²I) on the entire obs
#   - "single_grid_cell":  flip one occupancy cell (1.0 → 0.0 or vice versa)
#   - "gaussian_dyn":      δ ~ N(0, σ²I) on only the dynamics priv (15-D)
# =====================================================================
def method3_finite_diff(
    w: dict[str, torch.Tensor],
    obs_samples: torch.Tensor,
    perturb_kind: str,
    n_trials_per_state: int = 5,
    sigma: float = 0.01,
    verbose: bool = True,
):
    if verbose:
        print()
        print("─" * 70)
        print(f"METHOD 3: empirical finite-difference L  (perturbation: {perturb_kind})")
        print("─" * 70)

    ratios = []
    with torch.no_grad():
        for x in obs_samples:
            x = x.unsqueeze(0)                         # (1, 8207)
            f_x = forward(x, w)                        # (1, 5)
            for _ in range(n_trials_per_state):
                if perturb_kind == "gaussian_obs":
                    delta = torch.randn_like(x) * sigma
                elif perturb_kind == "gaussian_dyn":
                    delta = torch.zeros_like(x)
                    delta[:, :DYN_DIM] = torch.randn(DYN_DIM) * sigma
                elif perturb_kind == "single_grid_cell":
                    # Pick one grid cell uniformly at random and flip its value.
                    # Grid lives in obs[15:8207], reshape conceptually as (2, 64, 64).
                    flat_idx = torch.randint(DYN_DIM, OBS_DIM, (1,)).item()
                    delta = torch.zeros_like(x)
                    # Flip 0→1 or 1→0; if intermediate, just toggle by ±1.
                    delta[0, flat_idx] = 1.0 - 2.0 * x[0, flat_idx]
                else:
                    raise ValueError(f"Unknown perturb_kind: {perturb_kind}")

                f_xd = forward(x + delta, w)           # (1, 5)
                num = (f_xd - f_x).norm().item()
                den = delta.norm().item()
                if den > 0.0:
                    ratios.append(num / den)

    R = torch.tensor(ratios)
    stats = {
        "mean":   R.mean().item(),
        "median": R.median().item(),
        "max":    R.max().item(),
        "p95":    R.quantile(0.95).item(),
        "n":      R.numel(),
    }

    if verbose:
        print(f"  Trials sampled:  {stats['n']}  ({len(obs_samples)} states × {n_trials_per_state} perturbations)")
        print(f"  ||Δf|| / ||Δx|| stats:")
        for k in ("mean", "median", "p95", "max"):
            print(f"    {k:<8}  {stats[k]:>10.3f}")
        print(
            "  Interpretation: if you perturb obs in this way, the output\n"
            "  changes by ~mean * ||δ|| on average. The 'max' is the worst\n"
            "  amplification we observed in the sample — it's a LOWER bound\n"
            "  on the true L. With tiny δ, this approaches the local Jacobian\n"
            "  σ_max from below."
        )

    return stats


# =====================================================================
# Synthetic obs sampler
# =====================================================================
#
# We don't have access to a saved batch of real rollout obs without
# launching the simulator. As a stand-in, we generate synthetic obs
# that mimic the rough distribution:
#   - Dynamics priv: small random values (DR is bounded, ~unit-scale)
#   - Occupancy grid: mostly zero, a few clusters of 1's
#
# CAVEAT: synthetic obs aren't truly in-distribution. Method 2/3
# numbers from synthetic obs are NOT the same as numbers from real
# rollout obs. But for a v2.4 vs v2.5 RELATIVE comparison they're
# fine — both checkpoints are tested against the same synthetic batch.
# =====================================================================
def synthetic_obs_batch(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    # Dynamics: random noise centered at 0, std ~ 0.5 (matches our DR scale).
    dyn = torch.randn(n, DYN_DIM, generator=g) * 0.5
    # Grid: zero baseline + 1-3 random "obstacles" per env (clusters of 1's).
    grid = torch.zeros(n, GRID_CHANNELS, GRID_H, GRID_W)
    for i in range(n):
        n_obs = torch.randint(1, 4, (1,), generator=g).item()
        for _ in range(n_obs):
            cy = torch.randint(8, GRID_H - 8, (1,), generator=g).item()
            cx = torch.randint(8, GRID_W - 8, (1,), generator=g).item()
            r = torch.randint(2, 6, (1,), generator=g).item()
            grid[i, :, cy - r:cy + r, cx - r:cx + r] = 1.0
    grid_flat = grid.reshape(n, -1)
    return torch.cat([dyn, grid_flat], dim=-1)         # (n, 8207)


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Lipschitz constant estimator.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model_*.pt from rsl_rl run.")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of synthetic obs states for methods 2 and 3.")
    parser.add_argument("--label", type=str, default="",
                        help="Tag to print in headers (e.g. v2.4, v2.5).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    print(f"[diag-3] Lipschitz analysis: {args.label or '(unlabeled)'}")
    print(f"[diag-3] checkpoint: {ckpt}")

    # Load weights from disk (no model class needed).
    w = load_weights(ckpt)
    print(f"[diag-3] loaded {len(w)} weight tensors")

    # Method 1: spectral product (no obs needed — just weights).
    L1 = method1_spectral_product(w)

    # Methods 2 + 3: need obs samples.
    obs = synthetic_obs_batch(args.num_samples, seed=args.seed)
    print(f"\n[diag-3] generated {args.num_samples} synthetic obs (caveat: not in-distribution; "
          f"useful for relative checkpoint comparison)")

    L2_stats = method2_local_jacobian(w, obs)

    # Run method 3 with three perturbation flavors — each tells us
    # something different about the policy's sensitivity:
    #   - gaussian_obs:      generic input noise (e.g., obs encoder jitter)
    #   - gaussian_dyn:      DR-style uncertainty (mass/friction noise)
    #   - single_grid_cell:  realistic LiDAR noise (one cell flips)
    L3_obs   = method3_finite_diff(w, obs, "gaussian_obs",      sigma=0.01)
    L3_dyn   = method3_finite_diff(w, obs, "gaussian_dyn",      sigma=0.01)
    L3_grid  = method3_finite_diff(w, obs, "single_grid_cell")

    # ---- Concise summary line (greppable for diff'ing checkpoints) -----
    print()
    print("=" * 70)
    print(f"SUMMARY ({args.label or 'unlabeled'}):")
    print(f"  Method 1 (spectral upper bound):     L ≤ {L1:>12.2f}")
    print(f"  Method 2 (local Jacobian, mean):     L ≈ {L2_stats['mean']:>12.2f}")
    print(f"  Method 2 (local Jacobian, p95):      L ≈ {L2_stats['p95']:>12.2f}")
    print(f"  Method 3 (gaussian_obs δ, mean):     {L3_obs['mean']:>12.2f}")
    print(f"  Method 3 (gaussian_dyn δ, mean):     {L3_dyn['mean']:>12.2f}")
    print(f"  Method 3 (single_grid_cell, mean):   {L3_grid['mean']:>12.2f}")
    print("=" * 70)
    print(
        "Lower numbers = smoother policy = less twitchy CBF params = better\n"
        "tracking by the locomotion stage. v2.5 (with weight decay) should\n"
        "have noticeably smaller numbers than v2.4."
    )


if __name__ == "__main__":
    main()
