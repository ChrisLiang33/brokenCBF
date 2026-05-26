#!/bin/bash
# Wk3 Tracker Ablation (2026-05-15): LAYER3 (perception + persistence) +
# tracker noise on v_obs, with `a` and `c` STILL FROZEN at 0.
#
# Purpose: isolate whether the LAYER3_FULL collision regression (0.586 →
# 0.826) came from tracker noise alone or from the c-shaving direction.
# Single-variable change from Wk3 v1 pers: tracker noise on (everything
# else identical, including a/c frozen).
#
# Interpretation matrix when this lands:
#   - If tracker BR collision ≈ pers BR collision (≈0.586): tracker noise
#     alone is harmless, the c-shaving was the culprit in LAYER3_FULL.
#     Next move: re-launch LAYER3_FULL variant with c clamped to non-neg
#     or partial-shave range.
#   - If tracker BR collision ≈ LAYER3_FULL BR collision (≈0.826):
#     tracker noise is the killer (independent of c). Need to revisit
#     the noise spec or accept it as inherently harder.
#   - If tracker BR collision in between (~0.65-0.75): both contribute.
#     The c-shaving still hurts but tracker noise is also painful.
#
# Wk3 v1 pers (2026-05-15 fourth attempt): PERCEPTION SWITCH + per-cluster
# radius + grid_res 0.60m + PER-OBSTACLE POSITION PERSISTENCE.
#
# Wk3 v1 gr60 (the third attempt) still had collision_rate ~0.72 because
# the radius/grid fixes didn't address the REAL bug. A standalone
# perception diagnostic revealed: shield_v0c is single-frame with no
# temporal persistence. When a closer obstacle occludes a farther one,
# the LiDAR scan loses the far obstacle entirely, and the QP forgets it
# exists. The robot then drives forward through the corridor thinking
# the back row of cylinders isn't there.
#
# Static perception bias was actually OVER-protective in almost every
# scenario (perceived near-surface 0.15-0.22m closer to robot than true).
# The collisions weren't from miscalibrated radii — they were from
# obstacles disappearing entirely.
#
# Wk3 v1 pers fix: per-obstacle position tracker. Maintains a per-(env,
# obstacle-slot) cache of last-observed perceived (centroid, radius).
# Each frame:
#   - If any LiDAR ray hits obstacle k → update cache; age=0; valid=True
#   - If no rays hit obstacle k → hold cache for K_persist=10 frames
#     (~200ms at 50Hz CBF rate); valid=False after expiration
# Sim-time oracle: uses synthetic_lidar_raycast's per-ray obstacle index
# to bucket hits by stable simulator obstacle slot. On hardware, real
# Kalman+Hungarian matching produces the equivalent assignment.
#
# Both QP h (_compute_h) and policy occupancy grid (noised_occupancy_grid_b)
# read from the SAME persistent cache, so QP and policy see one
# consistent perceived world — same constraint as Wk3 v1.
#
# Single-variable change from gr60: obstacle_tracker_enabled = True
# in CbfGo2EnvCfg_LAYER3 (and the supporting state + tracker step).
#
# Wk3 v1 gr60 (2026-05-15 third attempt): PERCEPTION SWITCH +
# per-cluster radius + grid_res 0.40 → 0.60m.
#
# Wk3 v1 rfit fixed the fixed-R=0.3m bug (per-cluster radius from hit
# spread), but the eval revealed a SECOND perception bug: at grid_res
# 0.40m, obstacles with R>0.4m fragment across multiple cells, creating
# multiple sub-clusters with spatial gaps the robot navigates into.
# Collision_rate stayed broken (0.629 → 0.740).
#
# Wk3 v1 gr60: bump grid_res to 0.60m so obstacles up to R=0.50m fit
# in single cells. Side effect: adjacent obstacles closer than 0.6m
# may merge into one cluster (cluster-merge bias). Acceptable — milder
# failure mode than synthetic-gap-creates-collision.
#
# Single-variable change from rfit: SHIELD_GRID_RES_DEFAULT only.
#
# Wk3 v1 rfit (2026-05-15 re-run): PERCEPTION SWITCH + per-cluster radius fit.
#
# Wk3 v1 (first attempt) trained successfully but the eval revealed a hidden
# bug in shield_v0c: every cluster was modeled as a fixed-R=0.3m cylinder,
# while the obstacle pool has radii 0.10-0.50 m. Obstacles with R>0.3m were
# systematically under-protected → BR collision_rate jumped 0.0 → 0.629.
# Fall_rate dropped to 0.13 only because collisions replaced falls.
#
# Wk3 v1 rfit: per-cluster radius estimated from hit spread.
#   r_est = max(dist(hit, centroid)) + SHIELD_R_SAFETY_MARGIN (0.10),
#           clamped to [SHIELD_R_MIN=0.15, SHIELD_R_MAX=0.55].
# Computed inside cluster_points_grid_v alongside centroids using scatter
# ops; no extra Python loop. Single-variable change from the broken Wk3 v1.
#
# Wk3 v1 / Layer 3 (2026-05-15): PERCEPTION SWITCH. SINGLE-VARIABLE CHANGE
# from v22 (LAYER2). Two coordinated swaps that together flip the world the
# QP and policy see from ground-truth to LiDAR-derived:
#
#   1. perception_mode "priv_fov" → "shield_v0c"
#      QP `_compute_h` reads obstacle positions from synthetic-LiDAR
#      raycast (64 rays, 6m range) → grid clustering → fixed-R=0.3m cylinder
#      per cluster. h and L_g h now reflect cluster centroids, not truth.
#
#   2. obs.policy: TeacherPrivCfg → CbfLayer3ObservationsCfg.TeacherPrivLidarCfg
#      The 8192-D occupancy grid the policy ingests is built from the SAME
#      cluster output (noised_occupancy_grid_b). Both QP and policy see one
#      consistent perceived world; no privileged-info asymmetry.
#
# Inherited from LAYER2 (v22), UNCHANGED:
#   - 30% corridor scenes (Fat Robot fix)
#   - v22 reward stack
#   - a, c slots frozen at 0
#   - actuation_noise_sigma_max = 0.10, COM DR ±5cm xy / ±3cm z
#   - 45% adversarial planner mix
#
# Motivation: v22's φ_within_env_std = 0.29 vs φ_pop_std = 1.64 showed that
# under ground-truth perception, the policy uses φ as a one-shot scene
# classifier (corridor=low, open=high), not as a continuous adaptive control
# variable. Pearson(φ, σ_actuation) = 0.02 — flat. Under ground-truth h, the
# QP solution at any state is deterministic in σ, so the policy has no
# constraint-level reason to couple φ to σ. With LiDAR-derived h, the
# constraint LHS picks up stochastic structure that φ (and eventually a/c)
# can compensate for state-by-state.
#
# PASS for Wk3 v1:
#   - φ within-env std rises from 0.29 toward > 0.5 (within-episode
#     modulation, not scene classification)
#   - Pearson(φ, σ_actuation) climbs above 0.10 (was 0.02 in v22)
#   - Encoder R²(COM/σ) preserved (no regression from v22's 0.50-0.60 / 0.56)
#   - BR combined fall+stuck not worse than v22's 0.507 by more than 5pp
#     (perception noise is HARDER, some regression expected — gate is "not
#     catastrophic", not "must improve". Improvement comes after a/c release.)
#
# If φ within-env std stays ~0.3, perception alone wasn't enough and we
# proceed to Wk3 Step 5 (release `a` and `c`).
#
# Sync before launch (only env_cfg + __init__ changed for Wk3 v1):
#   rsync -av ~/Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/{cbf_go2_env_cfg.py,cbf_go2_env.py,__init__.py} \
#       chrisliang@130.64.84.163:Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/
#   rsync -av ~/Desktop/safety-go2/scripts/train_and_eval_layer3.sh \
#       chrisliang@130.64.84.163:Desktop/safety-go2/scripts/
#
# Usage on lab box (in tmux):
#   tmux new -s wk3tracker
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_layer3.sh 2>&1 | tee logs/train_and_eval_wk3tracker.log
#
# Expect ~12h total (shield_v0c clustering is per-env loop at 4096 envs;
# v22 priv_fov ran ~3.5h).
#
# ─────────────────── HISTORICAL CONTEXT (v22 LAYER2 / pre-Wk3) ───────────────────
#
# v22 (2026-05-14): CORRIDOR SCENES at 30% of envs. SINGLE-VARIABLE CHANGE
# from v19 (reward stack reverted to v19; v20/v21 reward changes dropped).
#
# Motivation: v20 (u_safe_rate -0.05) and v21 (split action_rate) both
# hit the "Fat Robot" exploit — policy locked φ at 3.83 / 3.99 because
# the open-spawn-area training never punished a large safety bubble. The
# policy just detoured around obstacles. fall_rate 0.526 / 0.443,
# combined fall+stuck 0.566 / 0.572 (worst on record).
#
# Three reward-shape attempts in a row failed because the issue isn't
# reward shape — it's training distribution. The simulation is a 5.5m
# × 5m parking lot where a 1m safety bubble has zero physical cost.
#
# v22 fix: replace 30% of episodes with a "corridor" scene — two
# parallel cylinder rows forming a ~0.6m gap perpendicular to the
# robot's path (x = 2.5-4.0, y = ±0.55). Robot footprint is 0.30m.
# φ pegged at 4.0 → safety bubble can't fit through the gap → robot
# stuck at corridor entrance → stuck_rate penalty fires → policy
# learns to drop φ for tight-space navigation.
#
# Implementation: cbf_go2_events.randomize_obstacles_position takes a
# new scene_corridor_prob param. LAYER2 env_cfg sets it to 0.30.
# Backward-compatible (default 0 = pre-v22 behavior). Random scene
# (existing rejection sampling) covers the other 70%.
#
# Reward stack reverted to v19:
#   - action_rate uniform -0.05 (v19 baseline)
#   - action_rate_split = 0 (v21 disabled)
#   - u_safe_rate = 0 (v20/v21 disabled)
#   - u_safe_deviation = 0 (v18 baseline, kept)
#   - tilt_penalty = -2.0 (v15 baseline, kept)
#
# PASS for v22:
#   - BR fall_rate drops below 0.35 (v19: 0.413, v16: 0.284)
#   - avg_cbf_phi_mean stays moderate (NOT pegged near 4.0)
#   - Encoder R²(COM/σ) preserved (no regression from v19)
#   - φ-grid coupling re-emerges (Pearson(φ, grid_change) < -0.2 maybe)
#
# Expected initial training behavior:
#   - First ~200 iters: fall_rate and stuck_rate spike as the v21-style
#     "lock φ high" policy fails miserably in corridor scenes
#   - PPO discovers dynamic-φ as it learns to squeeze through corridors
#   - Hold nerve through the spike
#
# v21 (2026-05-14, REJECTED): SPLIT action_rate across CBF param dims.
#
# Motivation: v20's u_safe_rate=-0.05 (penalizing ‖Δu_safe‖²) Goodharted —
# policy locked φ at 3.83 (2× v19) to satisfy zero-derivative cheaply.
# avg_deflection +60% to 0.46, fall_rate jumped 0.413 → 0.526 (worst on
# record). QP clamp rate 28× v19. Mechanism was sighted (φ-grid coupling
# revived to -0.45, near v10's -0.53 record) but expressed as permanent
# panic instead of dynamic ramping. Buddy conceded the derivative-tax
# was structurally flawed.
#
# v21 attacks aggression via a different lever: policy-output smoothness
# with per-dim weights. v16-v20 used uniform action_rate=-0.05 across all
# 5 dims, which killed φ-grid coupling in v16 (-0.53 → +0.03). The
# rationale: α must stay smooth (maps to physical braking; α-jerk →
# u_safe-jerk → locomotion can't track → falls), but φ adjusts a VIRTUAL
# spatial buffer and SHOULD ramp instantly when the CNN sees an obstacle
# approach.
#
# Implementation: new action_rate_weighted reward function in
# cbf_go2_rewards.py takes per-dim weight vector. Registered as
# action_rate_split RewTerm (weight=0 by default). LAYER2 overrides
# action_rate.weight=0 (disable uniform) + action_rate_split.weight=-1.0
# with dim_weights=[0.05, 0.0025, 0, 0, 0]. Also reverts u_safe_rate=0.
#
# Why this doesn't re-trigger Goodhart on φ:
#   - No u_safe magnitude penalty either way → no "lock φ high to satisfy
#     zero-derivative" trap (v20's failure mode)
#   - φ-changes are nearly free (-0.0025) but not zero → no incentive to
#     spam jerky φ either
#   - Dense terms (collision -100, base_contact -500, tilt -2) push toward
#     "ramp φ only when needed" instead of "lock high"
#   - α weight stays at -0.05, same as v19 where α was well-behaved
#     (3.03 ± 0.67) → no re-saturation risk
#
# PASS for v21:
#   - BR fall_rate drops below 0.35 (v19: 0.413, v16: 0.284)
#   - φ-grid coupling stays strong (v20: -0.45, v19: -0.19)
#   - Encoder R²(COM/σ) preserved (v19: 0.36-0.51 for COM, 0.42 for σ)
#   - HeavyCOM OOD combined fall+stuck stays at v19 level or improves
#
# v20 (2026-05-14, REJECTED): u_safe_rate penalty at -0.05.
#
# Motivation: v19 unblocked the encoder (R²(COM) 0.04 → 0.36–0.51,
# R²(σ) 0.08 → 0.42). α now adapts to σ (+0.12) and COM (+0.11).
# v19 OOD HeavyCOM improved meaningfully vs v16 (BR combined 0.536 →
# 0.456, gap to best baseline −17pp → −9pp). BUT in-dist fall_rate
# is still high (0.413 vs v16's 0.284) and HighActuationNoise stayed
# at v16 levels. The dominant unresolved problem is over-aggression:
# v18 removed the u_safe_deviation tax to unlock α adaptation, and
# that tax was also keeping deflection magnitude in check.
# avg_deflection_mean jumped 0.23 → 0.44 in v18 and stayed 0.29 in
# v19 — still way above v16. The locomotion controller can't track
# the violent safe-velocity changes, so the robot falls.
#
# v20 fix: register `u_safe_rate` (penalizes ‖u_safe_t − u_safe_{t-1}‖²)
# at weight −0.05. The implementation exists in cbf_go2_rewards.py
# (line 400) since 2026-05-06; it was deliberately left unregistered
# pending exactly this diagnostic state ("smooth-params-but-jerky-u_safe").
#
# Why this instead of bringing u_safe_deviation back at a small weight:
# u_safe_deviation taxed the EXISTENCE of intervention → policy avoided
# intervention → α saturated (v17 failure mode). u_safe_rate taxes the
# CHANGE in intervention → policy keeps intervening when needed, but
# does so smoothly. No re-saturation of α.
#
# Buddy's split-action_rate (α at −0.05, φ at −0.0025) held as v21
# candidate if v20 fixes fall but HighActuationNoise stays stuck.
#
# PASS for v20:
#   - BR in-dist fall_rate drops below 0.35 (v19: 0.413; v16 was 0.284)
#   - avg_deflection_mean drops to ~0.30 (v19: 0.29 already lower; expect ~0.20)
#   - Encoder R² values preserved (no regression on z_priv → COM/σ)
#   - HeavyCOM OOD combined fall+stuck stays at v19 level or improves
#
# FAIL paths → v21 candidates:
#   - Fall still > 0.35: penalty too weak, bump to −0.10 OR add back a
#     small u_safe_deviation tax (−0.025) for combined effect
#   - HighActuationNoise still loses by >10pp: v21 split action_rate
#   - Encoder R² regresses: probably means u_safe_rate is fighting
#     normalization somehow (unlikely but check)
#
# v19 (2026-05-14): Per-feature running mean/std normalization at the priv
# encoder input. SINGLE-VARIABLE CHANGE from v18.
#
# Motivation: v18 confirmed u_safe_deviation=0 unlocks α/φ adaptation
# mechanism (α de-saturated 4.26 → 3.67; Pearson(φ, grid_change) revived
# 0.03 → +0.30). But encoder STILL blind to COM/σ (R² < 0.10 — unchanged
# from v16 and v17). Two reward-side fixes have failed to wake the
# encoder, confirming the failure mode is not reward-related.
#
# Root cause (third opinion + buddy converged): scale mismatch in priv
# obs. From v18 feature distributions:
#     applied_force std ~5.5 (N)
#     base_mass     std  1.17 (kg)
#     COM offset    std  0.029 (m)
#     σ_actuation   std  0.029 (m/s)
# 190× range across features. PyTorch's Kaiming init assumes N(0,1)
# inputs. With raw mixed-unit priv, the encoder weights on COM/σ would
# need to be ~100× larger than force weights to give equal gradient
# signal — which init doesn't favor. The encoder drops the small-std
# features.
#
# v19 fix: add per-feature running mean/std normalization at the start
# of _PrivEncoder.forward(). Welford's online algorithm; updates only
# during rollout collection (no_grad), not during PPO gradient updates.
# Scoped to the 16 priv dims; grid stays binary (no normalization).
# All other v18 settings preserved: u_safe_deviation=0, action_rate=−0.05
# uniform, tilt_penalty=−2.0, DR at v16 levels, cbf_state out of obs.
#
# PASS for v19:
#   R²(z_priv → COM offset axes) climbs from ~0.04 → > 0.20
#   R²(z_priv → σ_actuation) similar
#   BR combined fall+stuck doesn't regress below v18's 0.463 (best case
#     comes back to v16's 0.359 if encoder adaptation reduces aggression)
#
# FAIL paths → v20 candidates (held in back pocket):
#   - Split action_rate (α at −0.05, φ at −0.0025) — boost φ-perception
#   - u_safe_derivative penalty — replace v16-style deviation tax with
#     jerk tax; targets aggression without re-saturating α
#   - Add proprio to teacher obs (third opinion's "trap" — only if
#     normalization fails to wake encoder)
#
# v18 (2026-05-14): u_safe_deviation weight −0.1 → 0.0. DR reverted to v16
# levels. SINGLE-VARIABLE CHANGE from v16.
#
# Motivation: v17 decisively rejected the widen-DR hypothesis. Encoder
# R²(z_priv → COM) stayed at 0.04, R²(σ) at 0.06 even with 2× DR. BR
# in-dist combined fall+stuck regressed 18pp (won −5pp in v16, lost +13pp
# in v17). BR fall_rate exploded 0.284 → 0.513.
#
# Diagnostic α-corr (new in v17): α saturated at 4.26 ± 0.33 (cap is 5.0).
# Pearson(α, dynamics features) all |r| < 0.10. α was being used as a
# "loosen QP" lever, not for adaptation. Buddy's diagnosis: PPO converged
# to "high α, low intervention, accept falls" because every step of QP
# intervention cost the u_safe_deviation penalty (−0.1 × ‖u_safe − u_des‖²).
# With encoder blind to COM/σ, intervention couldn't be targeted, so
# intervention's expected value was negative. Policy chose high-α
# saturation as the dominant strategy.
#
# v18 test: set u_safe_deviation = 0.0. If α de-saturates and BR's
# combined-win returns to v16 level (or better), buddy's diagnosis is
# right and v19 layers split-action_rate. If α still saturates, the
# diagnosis is incomplete and v19 moves to proprio or action-rate floor.
#
# Caveat: α saturation is an old failure mode (v8: 4.999 ± 0.005 with
# u_safe_dev = −0.5). v17 had u_safe_dev = −0.1 and α saturated at 4.26.
# So u_safe_dev isn't the only driver — but setting to 0 is the cheapest
# isolated test.
#
# v17 (2026-05-14, REJECTED): widen training DR to match v16's OOD distributions.
#
# Motivation: v16 won combined fall+stuck in-dist by 5pp but reversed OOD
# (−12pp on σ=0.20, −17pp on COM ±8cm). Linear probe showed z_priv blind
# to both axes (R²(COM) ~ 0.04, R²(σ) ~ 0.03). Diagnosis: encoder ignores
# variables that don't drive enough return variance at training DR. Widen
# training DR so the encoder is forced to retain σ and COM in z_priv.
#
# Changes (env_cfg.py, single-axis DR widening — single experimental
# variable from v16, action_rate=-0.05 and tilt_penalty=-2.0 kept):
#   - LAYER2 σ_max:  0.10  → 0.20  (matches what was v16 HighActuationNoise OOD)
#   - LAYER2 COM:    ±5cm  → ±8cm  (matches what was v16 HeavyCOM OOD)
#   - LAYER2_HIGH_ACTUATION_NOISE σ:  0.20  → 0.30  (1.5× new training edge)
#   - LAYER2_HEAVY_COM COM:           ±8cm  → ±12cm (1.5× new training edge)
#
# Eval changes:
#   - B2 (TISSf φ(h) = (1/ε₀)·exp(−λh)) added back. ε₀=[0.5], λ=[1.0,3.0].
#     Keeps eval ~50 min instead of ~30 (vs B0+B1+BR alone).
#
# Predictions for v17:
#   PASS: R²(z_priv → COM) climbs from 0.04 → >0.30 and R²(σ) similar.
#         BR combined fall+stuck win extends to the new (harder) OOD tasks.
#   FAIL: encoder still blind even at widened DR. Then v18 = split
#         action_rate penalty (α at -0.05, φ at -0.0025; buddy's idea)
#         OR add proprio to teacher.
#
# v16 (2026-05-14): action_rate weight bumped 20× from -0.0025 → -0.05.
#
# Motivation: v15 added a dense tilt_penalty (flat_orientation_l2, weight -2.0)
# to attack the dense-vs-sparse gradient asymmetry that caused v12's 35% fall
# rate. But v15 only moved fall by 3pp (0.358 → 0.348, within noise). The
# policy routed around the tilt penalty by setting smoother *baseline* CBF
# params (α 3.5→4.1, φ 1.7→2.3) and going faster (v̄ 0.44→0.49) instead of
# learning to ramp params when needed. v15 punished the *symptom* (tilt), not
# the *cause* (jerky CBF commands).
#
# v16 directly penalizes the cause via action_rate (per-step ‖Δaction‖²).
# 20× bump from -0.0025 → -0.05 puts smoothness pressure at ~3% of velocity
# reward (vs prior 0.17%). Tilt penalty kept at v15's -2.0 for combined effect.
# Single variable from v15 — only action_rate changes.
#
# v15 (2026-05-14): dense tilt penalty added to combat the speed-over-safety
# bias measured in v12 paper-grade evals.
#
# Motivation: v12's BR teacher loses on combined fall+stuck across all three
# distributions (in-dist, HighActuationNoise, HeavyCOM) because PPO updates
# over-weight the per-step velocity_tracking (+1.5) relative to the sparse
# terminal base_contact_penalty (−500). Result: BR is fastest + least stuck
# across the board but falls at 35-40% (vs best baseline's 19-27%).
#
# Fix: add a dense per-step penalty proportional to ||projected_gravity_xy||²
# (flat_orientation_l2), weight −2.0. Eats into the velocity profit margin
# when the body tilts off-vertical, providing dense gradient pressure toward
# upright behavior that PPO can actually train against. At 30° tilt the
# penalty is −0.5/step (33% of velocity reward); at full tipover, −2.0/step.
# In normal walking (~5° tilt) the penalty is negligible (~−0.02/step).
#
# Inherits everything else from v12: TTC off, u_des noise injection at
# σ_max=0.10 m/s, k=10 grid stride, no proprio in policy obs.
#
# v14 (2026-05-13): proprioception added to teacher obs. (ABORTED)
#
# Structural change — obs space grows from 8212 to 8245 dims. v12 checkpoint
# CANNOT be reused (different architecture). New training from scratch.
#
# Obs layout (v14):
#     dims  0–15 : priv          (env class, 16 dims)
#     dims 16–19 : cbf_state     (h, L_g h·u_des, ‖L_g h‖², slack)
#                                  Kept in obs for measurement; policy
#                                  slices it out via _SplitRMAMLP.forward.
#     dims 20–52 : proprio       (33 dims, RMA-style)
#                                  base_lin_vel (3) + base_ang_vel (3) +
#                                  projected_gravity (3) + joint_pos_rel (12) +
#                                  joint_vel_rel (12). Raw, concat to policy head.
#     dims 53–   : grid          (2 × 64 × 64 = 8192 dims)
#
# Why now: required prerequisite for the student's adaptation module
# (Goal B.5). The student maps history of proprio quantities to ẑ_priv via
# MSE — the teacher's policy head needs to be trained with proprio in its
# input so the student can swap in ẑ_priv with the same downstream pathway.
# Also gives the policy direct access to body state for reactive φ adaptation.
# Inherits everything else from v12: TTC OFF, σ_max=0.10 on u_des, k=10 stride.
#
# v12 (2026-05-13): u_des noise injection. Single-variable change from v11.
#
# Re-routes actuation noise from post-locomotion joint targets (v6–v11) to
# u_des (the planner's velocity command, before the CBF filter). Tests the
# locomotion-absorption hypothesis: v9–v11 had R²(σ) stuck at ~0.05–0.11
# because joint-level noise was being absorbed/diffused before reaching the
# CBF teacher as a return-variance signal. By injecting on u_des, the QP
# sees a noisy command and the policy has to choose CBF params that are
# robust against the σ being used.
#
# Everything else identical to v11: TTC OFF (v11 showed safety doesn't need
# it), σ_max=0.10 (semantically m/s on linear velocity now, not rad on
# joint angle), k=10 stride, 1500 iters.
#
# v11 (2026-05-13): TTC penalty OFF. Single-variable change from v10.
#
# Motivation: v10 result (diagnose_temporal_grid_v10.json) showed strong
# Pearson(φ, grid_change) = -0.53 and Pearson(α, grid_change) = +0.27 —
# the CNN now uses the temporal channel. But v10 still had TTC at -1.0,
# so we can't yet attribute the temporal coupling to pure perception. v11
# drops TTC and re-measures. Two possible outcomes:
#   (a) Pearson(φ, grid_change) survives → temporal pathway is real,
#       driven by collision penalty alone. TTC was redundant.
#   (b) Pearson(φ, grid_change) collapses to ~0 → TTC reward was the
#       curriculum that taught the CNN to read the grid.
#
# Also tests whether the spatial pathway from v9 (Pearson(φ, h) = -0.28)
# re-emerges without TTC pushing toward closure-rate behavior.
#
# Everything else identical to v10: σ_max=0.10, k=10 stride, u_safe_dev
# -0.1, 1500 iters.
#
# Layer 2 (2026-05-11): RMA-style architecture (from v3.1) +
# adversarial-heavy planner mix + AUX_COEF=0.
#
# The bet: v3.0/v3.1 suffered from sparse CBF-active states in training
# data — most cooperative planners avoided obstacles, so the CBF rarely
# fired. v3.0f's aux loss tried to extract per-step gradient signal from
# this sparse-event data, but instead pre-saturated the policy.
#
# Layer 2 fixes the data, not the loss:
#   - 45% adversarial planner episodes (was 5%) — robot is commanded
#     toward obstacles ~half the time, CBF stress-tests dense.
#   - AUX_COEF=0 — no monkey-patched PPO. The existing reward stack
#     (collision -100, base_contact -500, stuck -1, proximity -0.5,
#     cbf_lhs_margin -0.1) gets dense natural signal because the data
#     itself exercises the CBF in nearly every episode.
#   - Inherits v3.1 split encoders + cbf_state-excluded policy input.
#
# Sync before launch (from local Mac) — v10 touches observations.py and env.py
# in addition to the launch script:
#   rsync -av ~/Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/{cbf_go2_observations.py,cbf_go2_env.py,cbf_go2_env_cfg.py,__init__.py} \
#       chrisliang@130.64.84.163:Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/
#   rsync -av ~/Desktop/safety-go2/scripts/train_and_eval_layer2.sh \
#       chrisliang@130.64.84.163:Desktop/safety-go2/scripts/
#
# Usage on lab box (in tmux):
#   tmux new -s v22
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_layer2.sh 2>&1 | tee logs/train_and_eval_v22.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

# Layer 2: aux loss OFF. Adversarial planner provides the dense signal.
AUX_COEF=0.0

echo "================================================================"
echo "Layer 3 / Wk3 v1 (RMA + perception switch: shield_v0c + noised grid)"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# Pre-flight: confirm Layer 2 changes are in place
echo ""
echo "Pre-flight checks"
test -f source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_rma.py \
  && echo "  ✓ cbf_go2_teacher_rma.py present" \
  || { echo "  ✗ cbf_go2_teacher_rma.py missing — sync first"; exit 1; }
test -f source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/agents/rsl_rl_ppo_rma_cfg.py \
  && echo "  ✓ rsl_rl_ppo_rma_cfg.py present" \
  || { echo "  ✗ rsl_rl_ppo_rma_cfg.py missing — sync first"; exit 1; }
grep -q "CbfGo2EnvCfg_LAYER3_TRACKER" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ CbfGo2EnvCfg_LAYER3_TRACKER config present" \
  || { echo "  ✗ Layer 3 config missing — sync env_cfg.py"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-Tracker-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ Isaac-CBF-Go2-RMA-Layer3-Tracker-v0 task registered" \
  || { echo "  ✗ Layer 3 task not registered — sync __init__.py"; exit 1; }
grep -q "noised_occupancy_grid_b" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_observations.py \
  && echo "  ✓ noised_occupancy_grid_b observation present" \
  || { echo "  ✗ noised_occupancy_grid_b missing"; exit 1; }

# ---------- PHASE 1: PPO TRAINING ----------
echo ""
echo "================================================================"
echo "[1/2] PPO TRAINING: ${CBF_ITERATIONS:-1500} iters, 4096 envs (vectorized clustering: shield_v0c overhead now negligible, matches v22 sampling)"
echo "      RMA split encoders + 45% adversarial planner mix + AUX_COEF=0"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

# Aux loss OFF by setting CBF_AUX_COEF=0 — the monkey-patch in __init__.py
# only activates when CBF_AUX_COEF>0, so the patched PPO.update is NOT
# loaded. Vanilla rsl_rl PPO is used.
export CBF_AUX_COEF=$AUX_COEF

# Defensive: don't accidentally inherit v3.0e/f's pretrain-load env vars.
unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER
unset CBF_PRETRAINED_PRIV
unset CBF_FREEZE_PRIV

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Tracker-v0 \
  --num_envs 4096 --max_iterations ${CBF_ITERATIONS:-1500} \
  --headless

unset CBF_AUX_COEF

echo ""
echo "[1/2] PPO TRAINING done at $(date '+%H:%M:%S')"

# ---------- LOCATE CHECKPOINT ----------
LATEST_DIR=$(ls -1td logs/rsl_rl/cbf_go2_teacher_rma/*/ | head -1)
LATEST_DIR=${LATEST_DIR%/}
CKPT=$(ls -1t "${LATEST_DIR}"/model_*.pt 2>/dev/null | head -1)

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found in ${LATEST_DIR}"
    exit 1
fi
echo "Using checkpoint: $CKPT"

# ---------- PHASE 2: FAST DIAGNOSTICS (v9 minimal) ----------
# v9 (2026-05-13): fast-iter pipeline. Dropped eval phase + α-corr +
# temporal grid (α stats readable from training log; temporal grid is
# uninterpretable until v9 confirms α isn't saturated).
# Keep phi-corr + probe — the actual decision numbers.

echo ""
echo "================================================================"
echo "[2/2] φ-CORR + PROBE DIAGNOSTICS (fast mode)"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=0.0

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Tracker-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 16 \
  --use_locked \
  --output diagnose_phi_corr_wk3tracker.json \
  --headless

echo ""
echo "Running α-corr diagnostic (does α track the dynamics-uncertainty axes it's designed for?)..."
# v17: added so we measure α's adaptation, not just φ's. α defends against
# friction/mass/COM/applied-force uncertainty; we'd been measuring φ-correlations
# comprehensively but only Pearson(α, grid_change). This closes the gap.
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Tracker-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 16 \
  --output diagnose_alpha_corr_wk3tracker.json \
  --headless

echo ""
echo "Running linear probe Z_priv → priv features..."
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Tracker-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3tracker.json \
  --headless

echo ""
echo "Running temporal-grid diagnostic (does it survive proprio addition?)..."
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_temporal_grid.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Tracker-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --cbf_state_dim 0 \
  --output diagnose_temporal_grid_wk3tracker.json \
  --headless

# ---------- PHASE 3: HEADLINE EVAL (B0+B1+BR, in-dist only) ----------
# v14 (2026-05-13): added "is the method actually any good" check.
# Not a full paper-grade sweep — just enough to know if BR (our adaptive
# teacher) beats the best hand-tuned fixed-α baseline (B0) and best
# constant-φ ISSf baseline (B1) on combined safety + performance.
#
# Skipping B2 (TISSf with exp(-λh) form) to keep eval ~30 min instead of
# ~60 min. B0 + B1 are sufficient for a "are we even better than fixed?"
# read. Add B2 + multiple eval distributions later if v14 passes here.

echo ""
echo "================================================================"
echo "[3/3] HEADLINE EVAL: B0 + B1 + BR vs in-dist task"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_OUT="logs/baseline_eval_wk3tracker_indist"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Tracker-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,4.0" \
  --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" \
  --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "$EVAL_OUT" \
  --headless

echo ""
echo "[3/3] HEADLINE EVAL done at $(date '+%H:%M:%S')"

unset CBF_AUX_COEF

echo ""
echo "================================================================"
echo "PIPELINE DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint:        $CKPT"
echo "α-corr JSON:           diagnose_alpha_corr_wk3tracker.json"
echo "φ-corr JSON:           diagnose_phi_corr_wk3tracker.json"
echo "Linear probe JSON:     probe_z_linear_wk3tracker.json"
echo "Temporal-grid JSON:    diagnose_temporal_grid_wk3tracker.json"
echo "Eval CSV:              logs/baseline_eval_wk3tracker_indist/baseline.csv    ← Wk3 v1 HEADLINE"
echo ""
echo "v11 decision criterion (single-variable test: drop TTC):"
echo "  Compare v11 vs v10 numbers directly:"
echo "    v10 Pearson(φ, grid_change) per-env: -0.5276"
echo "    v10 Pearson(α, grid_change) per-env: +0.2723"
echo "    v10 Pearson(φ, h):                   +0.158"
echo "    v10 Pearson(φ, slack):               +0.040"
echo ""
echo "  (a) If φ-grid_change correlation survives (|r| > 0.3): temporal"
echo "      pathway is real, driven by collision penalty alone. TTC redundant."
echo "  (b) If it collapses (|r| < 0.10): TTC was the curriculum that"
echo "      taught the CNN to read the grid."
echo ""
echo "  Also check: does φ-h coupling RE-emerge without TTC pushing toward"
echo "  closure-rate behavior? v9 had -0.28, v10 dropped to +0.16."
echo "================================================================"
