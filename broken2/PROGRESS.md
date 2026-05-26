# Progress — CoRL 2026

Active tracker. Session notes in [LOG.md](LOG.md). Methodology in
[TODO_training.md](TODO_training.md).

---

## TL;DR

**Late-afternoon addendum (2026-05-09):** v2.15 launched ~14:01 lab
time, training in flight (~9-10h, alarm 12:30 AM Sun EDT). In-flight
work hardened the eval pipeline (Path C displacement-based goal-reach
metrics, `--student_checkpoint` + `BS` mode for direct teacher-vs-student
eval, 4 paper-grade plot functions in `plot_results.py`), archived
v2.12 + v2.14 checkpoints to Mac, and pre-staged the full Goal B.5
student distillation infrastructure: registered `Isaac-CBF-Go2-Distill-v0`,
new `noised_occupancy_grid_b` obs term that calls `shield_perceive_v0c`
(built earlier today as Goal B.4 closure) and rasterizes clusters into
the same 64×64 grid shape as the teacher's privileged view, per-episode
DR state on env (sensor dropout + range gating, mirrors v2.12/v2.15
patterns), and a DAgger training skeleton. Conceptual clarification:
the c-parameter is **state-aware not radius-adaptive** — the policy
can't observe true radius, so c learns a Bayes-optimal robust baseline
plus a robot-state modulation. SHIELD differentiation survives. All
files synced to lab, md5 verified, additive-only (zero impact on v2.15
training in flight). Critical path remaining: smoke-test the noised
grid on lab, calibrate DR ranges from empirical clustering output once
v2.15 lands. See [LOG.md](LOG.md) entry "2026-05-09 (late afternoon)".

---

**As of 2026-05-09 (afternoon) — v2.14 ran and FAILED; v2.15 spec'd,
implemented, parsing clean, awaiting launch (~9-10h on lab).**

v2.14 (per-episode φ lock + train under shield_v0a perception) lost
0W/0T/8L, avg margin **-7.62pp**. Two-axis breakdown shocked us:
**safety axis -17.4pp LOSS, performance axis +9.6pp WIN.** BR was
trading falls for un-stuck — the policy was being too aggressive.
Bf-φ on HighDist STILL inverted at -6.81pp despite the per-episode
lock. So we re-diagnosed.

**Three findings drove v2.15 (all caught today):**

1. **2-axis eval framing required.** Combined-metric was masking that
   v2.14's "loss" was really a fall-rate problem (BR fall 0.26 vs
   baseline 0.085) hidden by a stuck-rate win (BR stuck 0.02 vs
   baseline 0.17). Paper goal is reframed: **beat baselines on BOTH
   safety AND performance, by large margin.**

2. **Reward asymmetry was wrong.** Under -100 fall + -2.0/step stuck,
   one fall ≈ 50 stuck-steps — policy learned "never be stuck, falling
   is acceptable." REWARD-3 (v2.15): -500 fall + -1.0 stuck. Now one
   fall ≈ 500 stuck-steps. Forces safety-first equilibrium.

3. **The φ inversion is DR-mismatch, not slot redundancy.** User-led
   re-diagnosis: HighDisturbance pushes the robot's pose, not its
   u-space tracking — that's not actuation uncertainty (φ's domain).
   Per-episode lock was correct but couldn't help because the DR axis
   wasn't actually exercising φ. Solution: add a proper actuation
   noise DR (per-episode σ_act on u_safe BEFORE locomotion).

   Same logic applies to slot c: HeavyCOM is an indirect proxy. Real
   c-target DR is **per-episode obstacle-radius perception error**.
   This doubles as the **Goal B.2 closure** — radius uncertainty is
   exactly what real LiDAR-cluster-fit-cylinder pipelines deliver.

**v2.15 spec — packed, all 6 changes in one run (9-10h):**

- REWARD-3: fall -500, stuck -1.0 (safety axis)
- α floor 1.0 + c floor 0.10 (safety axis — policy can't disable CBF)
- Revert perception_mode → priv (drop shield_v0a, replaced by c-DR below)
- NEW: per-episode actuation noise DR (σ_act_max=0.10) — **φ slot signal**
- NEW: per-episode obstacle-radius perception error (δ_R_max=0.10) — **c slot signal + Goal B.2**
- Train 6K iters (was 5K — 50% harder reward landscape)
- Keep per-episode φ lock (theory-consistent, harmless)
- Trim: skip dual-regime mid-switch eval, steps_per_config 2000→1500
- 2 new eval envs registered: HighActuationNoise-v0, RadiusError-v0

**Predicted v2.15 (success criteria):**

- 8-eval safety margin: from -17.4pp to ≥ 0pp (cross threshold)
- 8-eval combined: ≥ +5pp WIN
- Bf-φ on HighActuationNoise: from -7.6pp (v2.12 HighDist) to +20 to +40pp (load-bearing)
- Bf-c on RadiusError: similar large positive degradation
- Bf-a on NoisyPerception: still ≥ +20pp (carries from v2.12)

**Decision philosophy locked:** Goal A (sim-only 4-param win) confirmed
by Bf-a +44pp on v2.12. Goal B decomposed into B.1-B.6:
B.1 centroid bias DONE; B.2 radius uncertainty in v2.15; B.3 FOV gating
post-train eval (cheap); B.4 cluster errors via shield_v0c (1-2 days);
B.5 student distillation Wk3; B.6 hardware deploy Wk4.

**Checkpoints preserved:**
v2.10 — `logs/rsl_rl/cbf_go2_teacher/_archive/v2.10_model.pt` (lab) +
local backup. v2.11 ckpt to be archived after v2.12 launches.
v2.6 — `logs/rsl_rl/cbf_go2_teacher/2026-05-04_01-04-47/model_2999.pt`.
v2.9b — `logs/rsl_rl/cbf_go2_teacher/2026-05-06_21-53-01/model_4999.pt`.

**v2.6 (paper baseline) — 5 of 6 single-axis WINs (avg +6.4pp), 1
compound TIE.** With locked-planner eval (PLANNER-2a applied at
eval only), in-dist margin grows from +6.9pp → +10.5pp.

| v2.6 eval | Type | Margin |
| --- | --- | --- |
| In-dist (eval-1) | mixed | **+6.9pp WIN** |
| In-dist + locked-eval | mixed | **+10.5pp WIN** |
| Slippery | priv-obs (continuous) | **+5.6pp WIN** |
| DensePack | scene-only (h(x) symmetric) | +0.6pp tie |
| HighDisturbance | priv-obs (episodic) | **+9.0pp WIN** |
| FastObstacles | priv-obs (grid history) | **+10.6pp WIN** |
| HeavyCOM | priv-obs (startup) | **+5.9pp WIN** |
| RealisticCompound | all 5 modest pushes simultaneously | −0.3pp TIE |

**v2.7 outcome (abandoned 05-06 day):** widened training DR + 3000
iters → under-converged. v2.7 ckpt 0.424 combined vs v2.6 0.316 on
identical setup. Diagnostic also showed PLANNER-2a (locked planner)
is a free +3.6pp at eval time — but only on the v2.6 ckpt.

**v2.8 outcome (abandoned 05-06 evening):** v2.6 hyperparams +
PLANNER-2a (locked training) + PLANNER-2b (drop walk/adv) + mild DR
widening + 5000 iters. **Worse than v2.6 on every front.** In-dist
combined 49.3% vs v2.6's 30.6% (+18.7pp regression). **Stuck rate
tripled** (7.5% → 22.8%) uniformly across evals — policy-level
brittleness signature. Diagnosis: removing mid-episode planner
switches removed a regularizer; policy never learned to recover from
CBF deflection. Compound CSV missing (run status not yet checked).

**Direction this cycle — Probe A → B-α' rejected → REWARD-2 v2.9 → v2.9b:**

**Probe A done.** v2.8 ckpt under mid-switch eval: BR combined 0.456
vs best B 0.457 (+0.1pp tie). v2.6 ckpt under same regime: +6.9pp.
v2.8 fundamentally worse policy regardless of eval regime.

**B-α' rejected.** Mid-switch training fixes stuck *artificially*
via external disturbance — policy never learns *intrinsic* recovery,
disappears at deploy. Same critique applies backward to v2.6.

**REWARD-2 v2.9 trained + 5/7 evals.** Three reward changes:
NEW `base_contact_penalty -50` (terminal on fall — closes structural
gap), NEW `stuck -2.0` per step when ‖v_xy‖<0.15, CHANGE `proximity
-1.0 → -0.5`. Result: stuck FIXED (22.8%→7.7%) but fall WORSENED
(26.5%→40.8%). Combined ~tied with v2.8 (-1.2pp LOSS in-dist vs
best B). Reward shaping was right direction; -50 was too weak.

**`--no_obstacles` BR diagnostic on v2.9 ckpt:** fall 8.3%
(locomotion-internal floor under v2.8/v2.9 DR). So 80% of v2.9's
40.8% in-dist fall = CBF-attributable (controllable lever). Reward
shaping has substantial room.

**v2.9b launched** (single-knob retune): `base_contact_penalty:
-50 → -100`. Empirical evidence refuted the original "−100 will
lock policy into caution → stuck" worry: v2.9 stuck is 7.7% with
-50 + stuck-term, way under v2.8's 22.8%. Headroom on stuck axis to
add fall pressure. -100 matches `collision -100` symmetrically.
Predicted: fall ~15-20%, stuck ~7-10%, combined ~25-30%.

A `u_safe_rate` term was considered + rejected for v2.9: action_rate
already covers controllable u_safe jerk; penalizing
geometric/constraint-switch jerk would conflict with proximity
reduction. Code in place, unregistered. Revisit as REWARD-3 if
v2.9b doesn't land.

CBF param `c` randomization claim from earlier (05-04) **retracted**:
c is a per-step action output, not an env DR knob (verified in
`cbf_go2_env.py:179-183`).

**CBF QP audit (05-07 very late, while v2.10 trains):** verified the
4-param mapping against literature — α (Molnar 2021, model/tracking
error), φ (Kolathaya 2018 ISSf, actuation uncertainty via tightening
the `‖L_g h‖²` term), a (Dean 2019, state-indep measurement uncertainty
via additive RHS slack), c (boundary correction via `h(x) − c` inward
shift). `b` slot reserved but unused (would need SOCP solver).
**Real DR gap:** zero measurement noise in training → `a` is dead
weight. Post-v2.10 plan reframed around 4 param-aligned ablations
(BR vs B-fixed-{α,φ,a,c}) as paper Table 2; v2.12 will add
NoisyPerception OOD env to exercise `a`. **Two distinct L_f h items
parked:** (1) obstacle-drift term in QP constraint (~50-line
extension, no solver change) and (2) HOCBF for robot model change
(only relevant if we switch to double-integrator/torque control —
not on roadmap). Rollback strategy: git tag v2.10 + archive ckpt;
v2.11+ changes additive only.

Wk3 student distillation gated on recovering ≥ v2.6 paper baseline.

---

## Milestones

**Chain:** teacher beats baseline → student matches teacher → robot
demo → paper.

- **M1 — Teacher beats baseline.** v2.x converges and wins TISSf-style
  eval vs best-tuned B1/B2. Paper *result*.
- **M2 — Student matches teacher.** `(LiDAR, base_vel, history) → Ẑ`
  reproduces Z(12); collision rate within ~2× teacher's. *Deploy-readiness*.
- **M3 — Hardware demo (stretch).** Student on Jetson, LiDAR-driven
  obstacle avoidance with CBF active. Sim-only paper still submittable.
- **M4 — Paper submitted.** Abstract **May 25**, full **May 28**,
  supp **June 4**.

---

## This week

**Status:** **v2.11 done + failed (2W/1T/4L; in-dist regressed to 0.472).
v2.12 implemented today, ready to launch tonight (~10h on lab).** v2.6
remains paper baseline; v2.10 + v2.11 archived as data points. Today's
session pivoted multiple times — see LOG.md 2026-05-08 entry for the
full arc. Net result:

**v2.11 outcome.** All v2.10 design pieces + bimodal + L_f h + WIDE
params + motion DR. Score 2W/1T/4L — worse than v2.10 (4W/2T/1L). The
new 12 CBF training stats showed `a` slot collapsed to mean 0.06 (range
[0, 3.0]) and `c` slot collapsed to mean 0.09 (range [0, 1.0]) — both
slots had no gradient signal because analytical h(x) was exact. WIDE
ranges were wasted; α and φ overloaded. Bf-X ablations still show
adaptation is load-bearing (4 of 5 confirmed) even on a regressed
checkpoint — confirms the architecture is sound, the issue is the
training distribution didn't exercise all 4 slots.

**v2.12 design** (built today, sanity-check passes 17 axes):
SHIELD-style commitment to **cylinder-only obstacle pool** (drop boxes,
walls, rect boxes) + **per-episode persistent perception bias** on
obstacle positions (σ ~ Uniform(0, 0.05); per-(env, obstacle) ε ~
N(0, σ²·I), persistent for the whole episode — see `_compute_h` in
`cbf_go2_env.py`). Per-episode persistence is critical: per-step IID
noise would average to zero under LLN over 1000-step episodes, leaving
`a`/`c` no signal (caught during sanity check, fixed). Matches the
biased error structure of a real LiDAR-cluster-fit-cylinder pipeline
(SHIELD reference). New `Isaac-CBF-Go2-NoisyPerception-v0` OOD env
(σ_max=0.10) for the Bf-a Table-2 row. PLANNER-2b reverted (restored
walk + adversarial → 6-planner mix matching v2.6).

**v2.13 SHIELD-path (deferred):** real synthetic-LiDAR raycast →
cluster-fit-cylinder → analytical SDF on fitted cylinders. ~150 lines,
1-2 days. Build only AFTER v2.12 ships Goal A. Distance-transform v2.13
plan (the 4-layer original) was over-spec'd given the cylinder
commitment — replaced by SHIELD-path framing.

**Decision philosophy locked.** Two orthogonal goals, sequence them:

- **Goal A (sim-only paper claim):** per-step adaptive 4-param > fixed
  baselines. v2.12 has every piece. If v2.12 wins, ship.
- **Goal B (deploy story):** train h(x) pipeline = deploy h(x) pipeline.
  v2.13 SHIELD-path. Future work / extension.

If v2.12 wins, paper baseline. If v2.12 fails on the same pattern as
v2.11 (both `a`/`c` collapsed despite real signal), the architecture
itself needs work — v2.13 wouldn't have helped, save the dev time.

**CBF QP audited (2026-05-07 very late):** 4-param mapping (α/φ/a/c)
verified against Kolathaya 2018 / Dean 2019 / Molnar 2021 / boundary-
correction literature. Methods outline drafted at
`docs/class_paper/methods_outline.md`.

### Immediate to-dos

1. **Sync v2.12 to lab + launch tonight (~10h).** Sync 3 modified
   Python files (`cbf_go2_env_cfg.py`, `cbf_go2_env.py`, `__init__.py`)
   + new `train_and_eval_v212.sh` + `watch_training_health.sh` +
   `extract_training_summary.py`. Launch via `tee` to
   `train_and_eval_v212.log`.
2. **Pane B watch script — actually run it this time.** v2.11 lesson:
   "I'll just check at the end" burned 9h on a doomed run. v2.12 has
   new `cbf_obs_noise_sigma_mean / std` stats that confirm noise
   injection is wired. Trip wires at iter ~1000-2000:
   - `cbf_obs_noise_sigma_mean ≈ 0.025` (= σ_max/2; confirms DR active)
   - `cbf_a_std > 0.20` (was 0.23 with collapsed mean in v2.11; want
     ≥ 0.30 with mean > 0.3 — slot live, doing work)
   - `cbf_c_std > 0.20`
   - `term_base_contact < 0.10`, `r_stuck > -0.20` (training health)
3. **Early-abort criteria.** If at iter 2000 `cbf_a_std < 0.10`, the
   noise σ isn't producing useful signal. Ctrl-C, bump σ_max from 0.05
   → 0.10 in `cbf_go2_env_cfg.py`, re-sync, relaunch. Saves ~7h vs.
   waiting for the full run.
4. **Run 6-block analysis script post-eval** and paste back.
5. **Compare v2.12 to v2.6 + v2.11:**
   - In-dist combined ≤ 0.31 + HeavyCOM margin ≥ +3pp + compound holds
     + Bf-{α,φ,a,c} all show BR > Bf-X by ≥ 3pp → v2.12 ships as paper
     baseline. Tag + archive + scp ckpt to Mac. Proceed to Wk3 student
     distillation, OR v2.13 SHIELD-path if time permits.
   - `a`/`c` slots STILL collapse despite persistent bias → architecture
     needs deeper rework. v2.13 won't help; revisit reward shaping or
     latent design.
   - Otherwise → diagnose, decide v2.13 vs v2.6 + locked-eval headline.
6. **(if v2.12 ships) Tag v2.12 + archive ckpt + mirror to local.**
7. **(during the wait, alternative to v2.13 dev)** paper progress:
   methods section draft (TOC at `docs/class_paper/methods_outline.md`),
   Table 2 mockup with placeholder numbers.
8. **v2.13 — SHIELD-path perception pipeline (deferred; build only if
   v2.12 wins Goal A).** Replaces v2.12's synthetic per-episode bias
   with a real synthetic-LiDAR raycast → cluster-fit-cylinder →
   analytical SDF pipeline. Steps:
   - 72-ray analytical raycast against cylinder pool (closed-form
     ray-cylinder intersection on GPU; no Isaac Sim raycaster needed).
   - Distance-based cluster on hit points (connected-component on
     ray-adjacency).
   - Per-cluster cylinder fit (centroid + max radius, or min-bounding
     circle for tighter fit).
   - Analytical SDF on FITTED cylinders (reuse `compute_shape_sdf_batch`).
   ~150 lines, 1-2 days. Closes structural sim2real gap; `a`/`c` then
   absorb natural cluster-fit error rather than synthetic Gaussian
   bias. Original 4-layer grid distance-transform plan REPLACED — the
   distance transform was load-bearing only for arbitrary-shape
   support, which we dropped with the cylinder commitment.
9. **4-axis fixed-param ablation table** on final ckpt: BR vs
   B-fixed-{α,φ,a,c}. **Paper Table 2.**
10. **(Stretch, paper-polish)** Paired evaluation (pre-generated seed
    list). Eval-side only, no retrain. Cuts std-err 5-10×.

---

## 4-week plan

Today: **2026-05-08**. Abstract May 25, paper May 28.

| Week | Focus | Status |
| --- | --- | --- |
| **Wk1** (Apr 28 – May 3) | Multi-obstacle + varying-radius retrain + OOD eval | ✓ done; v1 mean rank 1.5 |
| **Wk2** (May 4 – 10) | v2 architecture redo + headline win | v2.6 suite closed (5 wins / 1 compound tie); v2.7-v2.11 all abandoned (v2.11 failed: a/c slots collapsed under analytical SDF); v2.12 implemented (cylinder pool + per-episode persistent perception bias for a/c signal); ready to launch tonight |
| **Wk3** (May 11 – 17) | Student distillation + paper draft | not started |
| **Wk4** (May 18 – 25) | Sim-to-real prep, polish, submit | not started |

**Wk2 to-dos:**

- [x] Wk2 train+eval pipeline through v2.4 (REWARD-1 Variant C; custom
  loco built + reverted; DIAG-1 stuck_rate; PLANNER-1 realistic mix;
  B0 on new mix combined 42-55%; v2.4 base_contact 8.5%).
- [x] v2.5 retrain with smoothness terms — lost eval-1 (over-
  regularized; action_std collapsed to 0.06).
- [x] v2.6 retrain (gentler regularization + entropy bump) — trained
  clean (action_std 0.42, base_contact 5.8% — lowest ever).
- [x] **In-distribution eval headline: BR combined 0.306 vs best
  baseline 0.375; +6.9pp.**
- [x] OOD eval on `Isaac-CBF-Go2-TightGap-v0`. Lost on combined,
  root-caused to planner-contract mismatch. **Retired** from headline.
- [x] **5-axis near-OOD suite:** Slippery (+5.6), DensePack (+0.6 tie),
  HighDisturbance (+9.0), FastObstacles (+10.6), HeavyCOM (+5.9).
  4 priv-obs WINS, 1 scene-only TIE.
- [x] **RealisticCompound** (all 5 modest pushes): −0.3pp TIE.
  Diagnosed as compositional generalization gap → motivated v2.7.
- [x] v2.7 (REALISM-1) retrain — **failed**. Wider DR + 3000 iters
  → under-converged. In-dist −0.5pp tie, Slippery −11.4pp loss.
  Diagnostic confirmed v2.7 ckpt is genuinely worse than v2.6.
- [x] **PLANNER-2a (locked planner) validated at eval time.** v2.6
  ckpt + locked planner eval: BR margin grew +6.9pp → **+10.5pp**
  (free win — but only on v2.6 ckpt).
- [x] **v2.8 retrain — failed.** v2.6 hyperparams + locked-during-
  training (PLANNER-2a) + drop walk/adv (PLANNER-2b) + mild DR
  widening + 5000 iters. **Worse than v2.6 on every eval.**
  In-dist combined 49.3% (vs v2.6's 30.6%, +18.7pp regression).
  Stuck rate tripled (7.5% → 22.8%) uniformly across envs —
  policy-level brittleness. Compound CSV missing.
- [x] **CBF param `c` claim retracted.** Earlier "c needs DR
  randomization" note was wrong; c is a per-step action output, not
  an env DR knob. Verified in `cbf_go2_env.py:179-183`.
- [x] **Probe A done.** v2.8 ckpt mid-switch eval: BR 0.456, best B
  0.457 (+0.1pp tie). v2.8 fundamentally worse policy than v2.6
  regardless of eval regime. Stuck dropped (22.8→12.9%) but fall
  rose (26.5→32.7%) — partial-attractor evidence.
- [x] **B-α' rejected** on principle: mid-switch fixes stuck
  artificially via external disturbance, not intrinsic recovery.
  Doesn't generalize to deployment (one stable nav stack). User
  insight applies backward to v2.6 too.
- [x] **REWARD-2 v2.9 implemented + trained + 5/7 evals**. Result:
  stuck fixed (22.8% → 7.7%), fall worsened (26.5% → 40.8%); combined
  ~tied with v2.8 (-1.2pp LOSS in-dist vs best B).
- [x] **`--no_obstacles` BR diagnostic on v2.9 ckpt**: fall 8.3% =
  loco-internal floor; 80% of v2.9 falls are CBF-attributable.
- [x] **`extract_training_summary.py` written** — parses rsl_rl logs
  to per-iter CSV. Handles partial logs.
- [x] **v2.9b trained + 7-eval**: -100 retune. Result: combined 0.479
  in-dist (-0.5pp tie vs best B); 5 OOD WINS / 1 in-dist tie / 1
  DensePack loss. **Compound flipped to WIN** (+1.7pp) where v2.6
  tied (-0.3pp). But combined didn't beat v2.6 (0.306).
- [x] **v2.10 implemented**: revert training DR + OOD ranges to v2.6;
  keep PLANNER-2a/2b + REWARD-2 retune. Hypothesis: narrow DR was
  dominant regression driver.
- [x] **v2.10 sync + launch + 7-eval.** Training done; combined 0.343
  in-dist (+9.9pp), 4 wins / 2 ties / 1 LOSS (HeavyCOM -7.8pp).
  Partial recovery zone per decision criteria.
- [x] **v2.10 iter 2500 gate check** — trip wires fired (r_stuck
  past -0.20, term_base_contact ≈ 0.10) but eval results consistent.
- [x] **CBF QP audit (2026-05-07 very late).** Verified 4-param
  mapping (α/φ/a/c) against Kolathaya 2018 / Dean 2019 / Molnar 2021 /
  boundary-correction. **Real DR gap identified:** `a` is dead weight
  (zero measurement noise in training). Post-v2.10 plan reframed
  around 4 param-aligned ablations as paper Table 2.
- [x] **HeavyCOM mid-switch diagnostic (post-v2.10):** v2.10 BR
  combined dropped 0.485 → 0.397 (-8.8pp) just from eval-time
  planner regime change. **Smoking gun: PLANNER-2a is the dominant
  HeavyCOM regression cause.**
- [x] **Tag v2.10 + archive ckpt + mirror to local.** Tag exists;
  ckpt at `_archive/v2.10_model.pt` (lab) + `checkpoints/` (Mac).
- [x] **v2.11 — DR-implicit shaping + planner regime fix — FAILED.**
  Trained 5K iters, ran 7-eval + dual-regime + Bf-{α,φ,c} ablations.
  In-dist combined 0.472 (-1.9pp LOSS); 2W/1T/4L; HeavyCOM marginal
  recovery -7.8pp → -4.8pp but still LOSS. Diagnostic gold from the 12
  new CBF training stats: **`a` slot collapsed to mean 0.06 / std 0.23
  and `c` slot collapsed to mean 0.09 / std 0.24** despite WIDE_PARAM_RANGES.
  Both slots had no gradient signal under analytical h(x). α and φ used
  full ranges (std 2.10 / 5.0 each) but were overloaded carrying both
  their own + the dead slots' work. Bf-X ablations still confirm
  adaptation is load-bearing on 4 of 5 axes (Bf-α losing by +11.9pp on
  v0, +13.8pp on DensePack; Bf-c losing by +3.5pp on HeavyCOM, +5.2pp
  on FastObs); only Bf-φ on HighDist had policy mistune φ. Architecture
  is sound; training distribution missing perception-noise axis.
- [~] **(post-v2.11) v2.12 — Cylinder commitment + per-episode persistent
  perception bias — LAUNCHING 2026-05-08 overnight (~10h ETA):**
  - Cylinder-only obstacle pool (drop boxes, walls, rect boxes; SHIELD-style
    commitment to analytical SDF math).
  - Per-episode persistent bias on QP-side obstacle positions (σ ~
    Uniform(0, 0.05); ε_{e,k} ~ N(0, σ²·I) per env per obstacle, fixed
    for the whole episode). Per-step IID noise rejected — would average
    to zero under LLN over 1000-step episodes. Persistent bias matches
    LiDAR cluster-fit error structure.
  - Priv obs grid stays clean (policy "sees truth", QP "sees noisy").
  - Restored 6-planner mix (PLANNER-2b reverted — walk + adversarial back).
  - New `Isaac-CBF-Go2-NoisyPerception-v0` OOD env (σ_max=0.10, 2× train).
  - Bf-a now meaningful (paired with NoisyPerception OOD); 4-row paper
    Table 2 ablation matrix.
  - All v2.11 stack pieces RETAINED (bimodal resample DR, L_f h drift
    term, WIDE param ranges, REWARD-2 stack, 12 health stats — now 14
    with noise σ stats). 17-axis sanity check all pass.
- [ ] **(post-v2.12) v2.13 — SHIELD-path perception (deferred,
  build only if v2.12 wins Goal A):** synthetic-LiDAR raycast →
  cluster-fit-cylinder → analytical SDF on fitted cylinders.
  ~150 lines, 1-2 days. Closes structural sim2real gap; `a`/`c`
  absorb natural cluster-fit error rather than synthetic Gaussian.
  Original 4-layer grid distance-transform plan REPLACED — distance
  transform was load-bearing only for arbitrary-shape support, dropped
  with cylinder commitment.
- [ ] **(post-v2.12) 4-axis fixed-param ablation table** on final ckpt
  (paper Table 2). Now includes Bf-a row meaningfully (NoisyPerception
  exercises `a` slot).
- [ ] Per-planner breakdown (optional; headline + near-OOD covers the
  claim) and Fig 1 reproduction (optional).

**Wk3 — Student distillation + paper draft:**

- [ ] Re-enable LiDAR (multi-mesh workaround for K>1 obstacles).
- [ ] Adapter (LSTM or 1D conv over history) → Ẑ, dim 12.
- [ ] Stage-2 supervised on teacher rollouts (~1-2 days compute).
- [ ] Student OOD eval across slip/grip × calm/push.
- [ ] Paper draft (intro, method, Wk1-2 results) by Wed May 13.
- **Exit:** student matches teacher; OOD within ~2× collision rate.

**Wk4 — Sim-to-real + polish + submit:**

- [ ] TorchScript export of student adapter + π_teacher.
- [ ] `walking_bridge` Sport Mode → low-level Unitree motor API.
- [ ] Iterate paper with Cosner; final figures; anonymize; supplementary.
- [ ] **Stretch:** 1-2 hardware demos (chair, doorway).
- [ ] **Submit:** Abstract May 25, paper May 28, supp June 4.
- **Exit:** teacher Pareto-dominates hand-tuned, or clear paper-claim pivot.

---

## Backlog

### Active

#### PAPER-1 — Adaptation-mechanism baseline

Status: **v2.6 paper baseline holds** — 5 single-axis wins
(5.6–10.6pp), 1 compound TIE. Four attempts to widen the gap (v2.7,
v2.8, v2.9, v2.9b) all regressed. v2.9b best result so far: in-dist
combined 0.479, 5 OOD WINS / 1 in-dist tie / 1 loss. Compound flipped
to WIN. But in-dist still below v2.6.
Current direction: **v2.10 implemented** — revert training DR + OOD
ranges to v2.6; keep PLANNER-2a/2b + REWARD-2 retune. Hypothesis:
v2.6's narrow DR was the dominant regression driver across all the
v2.7-v2.9b experiments. Tests this in one ~7h run.

**Closest published related work: SHIELD** (Yang et al. 2025,
arXiv:2505.11494). Same exponential-smoothed SDF form (their Eqs.
19-20), real LiDAR + cluster-fit-cylinder perception, Unitree G1
humanoid hardware. Differentiation points:
(1) **adaptivity** — their α calibrated per-episode via Freedman's
inequality; we learn α/φ/a/c per-step;
(2) **multi-param** — they have α only; we have 4 robust slacks tied
to specific uncertainty classes (Molnar/Kolathaya/Dean/boundary) +
B-fixed-X ablations as paper Table 2;
(3) **arbitrary shapes** — their cluster-fit-cylinder restricts to
humans + simple objects; v2.12 L4 grid pipeline handles irregular
meshes;
(4) **deterministic vs stochastic** — distinct design point; not
strictly better/worse.
Co-author overlap: Cosner is on both; same lab as our advisor.
Different methodology, platform, claim — not a conflict, just
related work to position against. See LOG.md 2026-05-07 (very late)
entry for full comparison table. Citation in
`docs/class_paper/references.bib` as `yang2025shield`.

#### EVAL-1 — v2.6 + v2.7 + v2.8 + v2.9 + v2.9b eval matrices closed; v2.10 pending

v2.6 (paper baseline, see LOG.md for detail):

- In-dist: BR +6.9pp combined.
- Slippery (priv-obs): +5.6pp.
- DensePack (scene-only): +0.6pp tie.
- HighDisturbance (priv-obs): +9.0pp.
- FastObstacles (priv-obs via grid history): +10.6pp.
- HeavyCOM (priv-obs): +5.9pp.
- RealisticCompound (compositional): −0.3pp tie.

v2.6 + locked planner (PLANNER-2a, eval-time only) diagnostic:
in-dist margin grew to **+10.5pp** (BR 0.316 vs best baseline 0.421).
Free win on v2.6 ckpt; does NOT generalize to v2.8 ckpt.

v2.7 (REALISM-1, abandoned): in-dist −0.5pp, Slippery −11.4pp,
DensePack −0.8pp. v2.7 BR 0.424 vs v2.6 BR 0.316 on identical setup
— under-converged checkpoint, not bad OOD design.

v2.8 (PLANNER-2a/2b + mild DR widening, abandoned): in-dist combined
49.3% vs v2.6's 30.6% (+18.7pp regression). Stuck rate 7.5% → 22.8%
across all envs. v2.8 ckpt path:
`logs/rsl_rl/cbf_go2_teacher/2026-05-06_01-33-37/model_4999.pt`.

Probe A (done): v2.8 ckpt mid-switch eval BR 0.456 = best B 0.457
(+0.1pp tie); v2.8 has zero margin even in v2.6's training-matched
regime → fundamentally a worse policy.

v2.9 (REWARD-2, abandoned): 5 of 7 evals (FastObs + Compound killed
to launch v2.9b). In-dist BR 0.486 vs best B 0.474 (-1.2pp LOSS).
Stuck FIXED across the 5 evals (5.8-15.4%); fall WORSENED across the
5 evals (39.0-50.3%). Net combined ~tied with v2.8. v2.9 ckpt path:
`logs/rsl_rl/cbf_go2_teacher/2026-05-06_14-25-49/model_4999.pt`.

`--no_obstacles` diagnostic on v2.9 ckpt: BR fall 8.3% (loco floor),
stuck 16.5% (planner artifact — not actionable). 80% of v2.9's
in-dist 40.8% fall is CBF-attributable.

v2.9b (REWARD-2 retune, abandoned): single-knob `base_contact_penalty
-50 → -100`. 7 evals: in-dist BR 0.479 = best B 0.474 (-0.5pp tie).
5 OOD WINS (Slippery +8.2pp, FastObs +8.7pp, HeavyCOM +3.9pp,
HighDist +2.7pp, Compound +1.7pp), DensePack -1.7pp loss. Trade-off
(vs v2.9): fall 40.8 → 35.0% (-5.8pp), stuck 7.7 → 12.9% (+5.2pp).
v2.9b ckpt: `logs/rsl_rl/cbf_go2_teacher/2026-05-06_21-53-01/model_4999.pt`.

v2.10 (DR revert + REWARD-2 retune, training pending): training DR
plus OOD ranges reverted to v2.6 levels; keeps PLANNER-2a/2b plus
REWARD-2 retune (-100/-2.0/-0.5). Eval matrix pending; OOD ranges now
match v2.6 paper baseline so numbers compare directly.

(Optional) Per-planner breakdown — less load-bearing now.

#### INFRA-1 — Constants cleanup

Medium priority. `LOCOMOTION_CHECKPOINT` move blocks Jetson deploy.
Other constants are post-paper polish.

### Closed this cycle

One-liners only. Detail in [LOG.md](LOG.md).

- **04-25 → 04-30:** BUG-1/MODEL-1/MODEL-2/MODEL-3 Step 1; REWARD-1
  Variant B; SCENE-1/1.5 Phase-1; v1 OOD eval (rank 10.25→1.5);
  PRIV-2 + SCENE-3 (CNN + multi-shape SDF); SCENE-2 (moving obs);
  SCENE-4 (K_MAX=20 pool); v2 OOM, v2.1 skipped.
- **05-01:** PAPER-1 baseline plan; v2.2 trained (lost baselines,
  rank 23/25); BR provider bug fixed; eval re-ran clean; plot
  fixed; **REWARD-1 Variant C + v2.3** (falls 25→9%); locomotion
  task built.
- **05-02:** Custom locomotion failed deploy (28-32% B0); reverted,
  reframed as drop-in safety filter. **DIAG-1** revealed 10-13%
  hidden stuck_rate; walk/adversarial planners flagged as
  unrealistic.
- **05-03:** **PLANNER-1** (smooth_goal + waypoint + mpc, 90%
  realistic mix). New B0 baseline 42-55% combined (~2× old).
  v2.4 trained (3h 18m, base_contact 8.5%).
- **05-04 (early):** v2.4 eval ties baselines (37.3% vs 37.5%
  combined). **DIAG-2** attributes ~28pp of falls to CBF
  deflection. v2.5 launched (action_rate -0.01 + AdamW wd=1e-4)
  via OnPolicyRunner monkey-patch.
- **05-04 (afternoon):** v2.5 trained but lost — action_std
  collapsed to 0.06; combined 0.485 (worse than v2.4). **DIAG-3**
  Lipschitz estimator confirmed regularization mechanically worked
  (-54% spectral); diagnosis was over-regularization +
  exploration starvation. v2.6 planned (wd 1e-5, action_rate
  -0.005, **entropy_coef 0.001 → 0.005** = load-bearing fix).
- **05-04 (evening):** **v2.6 trained clean** (action_std 0.42,
  base_contact 5.8% lowest ever). **Eval-1 in-dist WIN: BR 0.306
  vs best baseline 0.375 (+6.9pp).** TightGap OOD lost (planner-
  contract mismatch, not teacher quality) → retired. 5-axis
  near-OOD suite + RealisticCompound committed.
- **05-05 (day):** **v2.6 5-axis suite closed** — Slippery +5.6,
  DensePack +0.6 tie, HighDisturbance +9.0, FastObstacles +10.6,
  HeavyCOM +5.9 (avg +6.4pp). Pattern: wins where teacher has
  asymmetric info; ties where symmetric.
- **05-05 (evening):** **RealisticCompound TIE (−0.3pp)** —
  adaptations don't compose; joint high-tail untrained. **REALISM-1
  v2.7 launched** (4-axis training widening; OOD pushed further).
  **PLANNER-2a applied** (planner locked per episode). REWARD-2
  (base_contact + Δu_safe) and PLANNER-2b (drop walk/adversarial)
  queued for v2.8.
- **05-05 (late evening):** v2.7 first launch had train/eval
  distribution mismatch; killed and restarted with locked planner;
  TightGap dead code removed (~150 lines).
- **05-06 (day):** v2.7 trained + eval'd. **v2.7 failed**: in-dist
  −0.5pp, DensePack −0.8pp, Slippery −11.4pp. Wider DR + 3000 iters
  under-converged. Diagnostic (v2.7 ckpt vs v2.6 ckpt on identical
  setup): v2.7 BR 0.424 vs v2.6 BR 0.316 — v2.7 is genuinely a
  worse checkpoint. Same diagnostic showed **PLANNER-2a (locked
  planner) grows v2.6 margin to +10.5pp** (free win). Decision:
  abandon v2.7. Build v2.8 = v2.6 hyperparams + locked planner +
  drop walk/adv (PLANNER-2b) + mild DR widening + 5000 iters.
  `scripts/train_and_eval_v28.sh` written for unattended ~7h run.
- **05-06 (evening):** **v2.8 finished training + 6/7 evals; failed
  on every front.** In-dist combined 49.3% vs v2.6's 30.6%
  (+18.7pp regression). Stuck rate tripled (7.5% → 22.8%) uniformly
  across envs — policy-level brittleness. RealisticCompound CSV
  missing (status pending). Diagnosis: v2.6's mid-episode planner
  switching is a regularizer; v2.8's PLANNER-2a (locked training)
  combined with PLANNER-2b (dropped walk/adv) removed two regularizers and
  policy never learned to recover from CBF deflection. **Earlier
  CBF param `c` randomization claim retracted** — c is the
  teacher's per-step action output, not env DR. v2.6 exact planner
  mix recovered from saved `env.yaml`: smooth_goal 0.40, waypoint
  0.25, mpc 0.20, goal 0.05, walk 0.05, adversarial 0.05.
  Designed Probe A (re-eval v2.8 ckpt with mid-switch eval) +
  Probe B (single-knob revert retrain: B-α'/B-β'/B-γ'). Added
  `--planner_resample_s` flag to `eval_baseline.py` for Probe A.
- **05-06 (late evening):** **Probe A done.** v2.8 ckpt mid-switch
  eval: BR 0.456, best B 0.457 (+0.1pp tie). v2.8 fundamentally
  worse policy than v2.6 regardless of eval regime; fall went
  26.5%→32.7% but stuck dropped 22.8%→12.9% (partial-attractor
  evidence). **B-α' rejected on principle**: mid-switch fixes
  stuck *artificially* via external disturbance, not intrinsic
  recovery — doesn't generalize at deploy. User insight applies
  backward to v2.6 too. **REWARD-2 v2.9 designed + implemented**
  (4 files): NEW `base_contact_penalty -50` (closes structural
  gap; collision -100 only fires on obstacle hits, falls had no
  terminal penalty), NEW `stuck -2.0` per-step when ‖v_xy‖<0.15,
  CHANGE `proximity -1.0 → -0.5` (halve dominance). u_safe_rate
  considered + rejected (would conflict with proximity reduction);
  code kept unregistered for REWARD-3. `train_and_eval_v29.sh`
  written; ready to sync + launch.
- **05-06 (overnight):** **v2.9 launched on lab.** SPS 27K (vs
  v2.8's 22K), GPU 26.8/32.6 GB, 48% util — all healthy. Wrote
  `scripts/extract_training_summary.py` to parse rsl_rl logs to
  per-iter CSV. Iter 1211 spot check: `mean_ep_len` 12 → 818
  (good), `term_base_contact` 0.227 → 0.216 (slowly dropping),
  but `r_stuck` -0.009 → -0.300 (growing in magnitude — early
  caution-lock-in signature). Gate check at iter 2500.
- **05-07:** **v2.9 abandoned + v2.9b launched.** v2.9 finished
  training (5K iters) and ran 5 of 7 evals; killed FastObs +
  Compound to launch v2.9b sooner. Result: stuck FIXED (22.8% →
  7.7%, back to v2.6 level — REWARD-2 stuck term worked) but fall
  WORSENED (26.5% → 40.8% in-dist). Net combined ~tied with v2.8
  (-1.2pp LOSS in-dist vs best B). Pattern consistent across all
  5 evals (fall 39-50% everywhere). `--no_obstacles` BR diagnostic:
  fall 8.3% = loco-internal floor under v2.8/v2.9 DR; **80% of
  v2.9 falls are CBF-attributable** — reward shaping has room.
  Original worry that -100 base_contact would lock policy into
  caution → stuck is empirically refuted (v2.9 stuck = 7.7% with
  -50 + stuck-term, way under v2.8's 22.8% — headroom on stuck).
  v2.9b single-knob retune: `base_contact_penalty -50 → -100`
  (matches collision -100 symmetrically). Predicted fall 15-20%,
  stuck 7-10%, combined 25-30%. Trip wire same: iter 2500 gate
  check on r_stuck. Also caught + corrected an awk bug — earlier
  `sort -k2` was sorting "best baseline" by name lexicographically,
  not by combined value; corrected to numeric sort by combined.
- **05-07 (late):** **v2.9b done + v2.10 implemented.** v2.9b
  finished 7 evals: in-dist combined 0.479 (-0.5pp tie vs best B,
  basically tied). 5 OOD WINS (Slippery +8.2pp, FastObs +8.7pp,
  HeavyCOM +3.9pp, HighDist +2.7pp, Compound +1.7pp), DensePack
  -1.7pp loss. Compound flipped to WIN where v2.6 had tied. -100
  retune did push fall down (40.8 → 35.0%) but at cost of caution
  lock-in (stuck 7.7 → 12.9%); net combined barely moved. Pattern
  across all 7 evals: BR absolute combined values uniformly higher
  than v2.6's even where margins are decent → wider DR is making
  the env uniformly harder. action_std rose 0.68 → 0.81 in second
  half of training (same instability as v2.9). **v2.10 implemented**:
  revert training DR plus OOD ranges to v2.6 levels; keep
  PLANNER-2a/2b plus REWARD-2 retune. Hypothesis: narrow DR was the dominant
  regression cause across v2.7-v2.9b. Predicted training base_contact
  6-8%, in-dist combined 25-30% (would beat v2.6's 30.6%).
  `train_and_eval_v210.sh` written; ready to sync + launch.
- **05-07 (very late, while v2.10 trains):** **CBF QP audit +
  post-v2.10 plan reframed.** Read `cbf_go2_env.py:168-230`; verified
  the 4-param mapping against literature — α (Molnar 2021,
  model/tracking error), φ (Kolathaya 2018 ISSf, actuation
  uncertainty via tightening the `‖L_g h‖²` term), a (Dean 2019,
  state-indep measurement uncertainty via additive RHS slack),
  c (boundary correction via shifting safe-set inward through
  `h(x) − c`). `b` slot reserved but unused (would need SOCP solver;
  closed-form half-space projection breaks). **Identified real DR
  gap:** training has zero measurement noise → `a` has no gradient
  signal → likely settles to constant. Reframed post-v2.10 plan
  around 4 param-aligned ablations as paper Table 2: each param gets
  dedicated OOD axis + B-fixed-X ablation baseline (clamp the param
  to a tuned constant at eval, ignore policy's output for that slot).
  v2.11 will widen `a` ([0,1]→[0,3]) + `c` ([0,0.5]→[0,1]) behind
  `WIDE_PARAM_RANGES` flag (rollback-safe), add **variable
  obstacle-motion DR** (per-episode v_obs magnitude sampled in
  [0, 0.4] m/s — sometimes static, sometimes fast — exercises `c` for
  kinematic margin), add B-fixed-X eval modes (~30 lines/param).
  **No new reward terms** — user preference (later 2026-05-07): DR
  shapes params implicitly (cleaner paper claim than hand-crafted
  reward signals). Original `reward_alpha_aggressive_when_safe` /
  `reward_c_no_excess_when_stuck` proposals dropped. v2.12 will be
  the realistic-perception rebuild in 4 layers: L1 replaces analytical
  `_compute_h()` with grid-based distance transform (train/deploy
  parity); L2 censored grid (raycast occlusion + range falloff,
  no more perfect ego-centric crop); L3 Gaussian noise on grid +
  dedicated `Isaac-CBF-Go2-NoisyPerception-v0` OOD env to exercise
  `a`; L4 arbitrary obstacle shapes (irregular meshes — free side
  benefit once L1 drops the analytical-SDF requirement, strengthens
  deploy realism). ~4-5 days total. **Rollback strategy committed:** `git tag v2.10`
  after 7-eval ships + archive ckpt to `_archive/v2.10_model.pt` +
  mirror to local; v2.11+ code changes additive only (new functions /
  new flags / new env IDs, never destructive). **Future work parked
  (clarified two distinct L_f h sources):**
  (1) **L_f h obstacle-drift term** — currently MISSING from constraint.
  h depends on `(p_robot, p_obstacle)`; true
  `ḣ = L_g h · u + ∂h/∂p_obs · v_obs`. Our QP omits the second piece,
  treating obstacles as static during ḣ computation. ~50-line
  extension (read `cbf_obstacle_velocities`, compute gradient w.r.t.
  obstacle pos, add to RHS). NOT HOCBF, no solver change. Matters
  most for FastObstacles. **Magnitude justifies explicit handling:**
  v_obs up to 0.4 m/s ⇒ ḣ deviation ~0.4 m/s ⇒ too big for α to absorb.
  (2) **HOCBF for robot drift** — only relevant if we change robot
  model to double-integrator (`v̇ = u`, `ṗ = v`). Then `L_g h = 0`
  and we'd need `ψ̇_1 + α_2(ψ_1) ≥ 0` with `ψ_1 = ḣ + α_1(h)`.
  Currently `ẋ_robot = u` (single-integrator), so `L_g h ≠ 0` and
  standard CBF works. Robot velocity tracking error (~0.05 m/s) is
  small enough for α to absorb cleanly — that's why we leave robot
  drift implicit and obstacle drift explicit (asymmetry by magnitude,
  not by physics). Not on roadmap; kept as honest paper limitation.
  Drop `b` from 5D action OR commit to SOCP — paper-cleanliness
  parking lot.
- **05-08 — v2.11 done + failed; v2.12 designed and built; v2.13 reframed:**
  v2.11 7-eval finished. In-dist combined 0.472 (-1.9pp LOSS); 2W/1T/4L
  vs v2.10's 4W/2T/1L. HeavyCOM marginal recovery (-7.8 → -4.8pp) but
  still LOSS. Compound -0.2pp tie. The 12 new CBF training stats showed
  the smoking gun: **`a` slot collapsed to mean 0.06 (range [0, 3.0]) and
  `c` slot to mean 0.09 (range [0, 1.0])** — both had no gradient signal
  under analytical h(x). WIDE_PARAM_RANGES wasted; α and φ overloaded
  (std 2.10/5.0 each). Bf-X ablations confirm adaptation IS load-bearing
  on 4 of 5 axes despite the regressed checkpoint (Bf-α losing +11.9pp
  on v0; Bf-c losing +3.5pp on HeavyCOM, +5.2pp on FastObs). Architecture
  is sound; training distribution missing perception-noise axis. v2.12
  designed and built today: cylinder-only obstacle pool (SHIELD-style
  commitment, drops boxes/walls/rect boxes; v2.13 grid distance transform
  reframed as overkill since arbitrary shapes were dropped) + per-episode
  persistent perception bias on QP-side obstacle positions (σ ~
  Uniform(0, 0.05); per-(env, obstacle) ε ~ N(0, σ²·I) **persistent**
  for the whole episode — caught + fixed during sanity check that
  per-step IID would average to zero under LLN over 1000-step episodes).
  Restored 6-planner mix (PLANNER-2b reverted). New
  `Isaac-CBF-Go2-NoisyPerception-v0` OOD env (σ_max=0.10) for paper
  Table 2 Bf-a row. All v2.11 stack pieces retained (bimodal, L_f h,
  WIDE ranges, REWARD-2). **17-axis sanity check all pass.** Decision
  philosophy locked: method-first (Goal A, sim-only adaptation claim) →
  v2.12 has every piece; deploy-second (Goal B, train/deploy h(x)
  parity) → v2.13 SHIELD-path (synthetic-LiDAR raycast + cluster-fit-
  cylinder + analytical SDF) ~150 lines / 1-2 days, deferred until
  v2.12 ships Goal A. Methods section TOC drafted at
  `docs/class_paper/methods_outline.md`. v2.12 ready to launch tonight
  on lab (~10h train + parallel 8-eval).
- **05-07 (overnight) — v2.11 prep + launch:** Implemented full v2.11
  scope (8 components: bimodal resample DR via
  `MultiPlannerCommand._resample_command` override, variable obstacle-
  motion DR with `max_speed_range=(0.0, 0.4)`, `USE_LFH_OBSTACLE_DRIFT`
  flag adding `L_g h · v_obs` to RHS, `WIDE_PARAM_RANGES` flag widening
  `a`→[0,3] / `c`→[0,1], B-fixed-{α,φ,a,c} via
  `make_b_fixed_provider` with inverse-tanh override, 12 new CBF
  training stats injected through `extras["log"]`,
  `watch_training_health.sh` with `tee`-based dual-write to
  `.health.log`, 2-up parallel eval with 30s stagger). Sanity-check
  caught + fixed `task_tag` rename bug (bare `v0` was producing `_v0`
  dir, now `_indist`). User pushed back on "directly subtracting
  absolute velocity" in L_f h discussion — confirmed math is angle-
  aware via dot product (`cos θ` projection); no separate angle
  term needed. Originally-planned reward terms
  (`r_alpha_aggressive_when_safe`, `r_c_no_excess_when_stuck`)
  dropped per user preference for DR-implicit shaping (saved as
  feedback memory). `time_left` write-order concern documented but
  not pre-verifiable — will surface as missing trip-wire data if
  wrong. Synced to lab + launched via `tee` to
  `train_and_eval_v211.log`. Single-pane workflow (no second tmux
  pane). Drafted 6-block post-run analysis script with header-aware
  Python parsers (`csv.DictReader` in `combined_for_row()` /
  `best_baseline_combined()`) to handle B-fixed-X column drift. ETA
  ~9h. Standing by until results land.
- **05-07 (very late, post-eval) — v2.10 done; partial recovery;
  HeavyCOM diagnostic SMOKING GUN:** v2.10 7-eval finished. In-dist
  BR combined 0.343 (+9.9pp WIN vs best B 0.442). 4 wins / 2 ties /
  1 LOSS (HeavyCOM -7.8pp). Per script's decision criteria
  (`0.31 < c < 0.40` → partial recovery), v2.6 stays canonical
  paper baseline. v2.10 contributed real improvements: compound
  flipped to win (+4.1pp where v2.6 tied), DensePack improved
  (+2.8pp). But HeavyCOM regression broke the "wins on every
  priv-obs OOD" narrative. Trip wires fired during training
  (r_stuck -0.34 past -0.20; term_base_contact 0.114 above 0.10) —
  consistent with eval underperformance vs v2.6.
  **HeavyCOM mid-switch diagnostic** (re-eval v2.10 ckpt with
  `--planner_resample_s 10` instead of locked): combined 0.485
  → 0.397 (-8.8pp) just from eval-time planner regime change.
  fall 0.420 → 0.348, stuck 0.065 → 0.050. Both dropped.
  **Smoking gun: PLANNER-2a (locked-planner training) is the
  dominant HeavyCOM regression cause.** Mechanism: when CBF deflects
  velocity to zero near a COM-tilted obstacle, locked planner
  doesn't kick the policy out of stall, robot loses balance during
  freeze. v2.10 ckpt archived at
  `_archive/v2.10_model.pt` (lab) +
  `~/Desktop/safety-go2/checkpoints/v2.10_model.pt` (Mac).
  **v2.11 plan adjusted:** add **variable resample DR**
  (`resampling_time_range = (5.0, 100.0)` per episode — sometimes
  mid-switch, sometimes locked) to restore stuck-recovery
  regularizer. **Eval stays always-locked (deploy-realistic);
  training/eval mismatch is the point** — training mid-switch
  teaches intrinsic recovery, locked-planner eval verifies it
  transfers. Combined with v2.11's planned variable obstacle-motion
  DR + L_f h obstacle-drift fix + a/c range widening + B-fixed-X
  eval modes. No new reward terms (DR-implicit shaping
  preference).

**Adding items:** prefix (`PAPER`/`INFRA`/`MODEL`/`SCENE`/`STUDENT`/`BUG`/`REWARD`),
`####` block under Active. On close, detail to LOG.md, one-line here.

---

## Future OOD experiment design

Planning reference for OOD evals beyond the 5-axis suite.

### Framing principle

Adaptive learned methods have an in-dist/OOD axis that fixed-filter
baselines don't. Adversarial OOD favors baselines by construction.
Fair OOD design needs:

1. Shift along an axis the teacher *can* react to (priv obs), OR a
   scene-level axis affecting both methods symmetrically.
2. Single-knob attribution. Compositional only after single-axis
   wins are clean.
3. Symmetric exposure — never harder for the teacher than baselines.

### Axis inventory

**5-axis suite — v2.6 (working ckpt) and v2.10 (on-disk env_cfg, ranges reverted to v2.6):**

| Axis | Knob | v2.6 train | v2.6 OOD | v2.10 train (on-disk) | v2.10 OOD (on-disk) |
| --- | --- | --- | --- | --- | --- |
| Obstacle density | separation_buffer | 0.4m | 0.2m | 0.4m | 0.2m |
| Friction | static / dynamic | (0.30,1.20) / (0.20,1.00) | (0.15,1.50) / (0.10,1.30) | (0.30,1.20) / (0.20,1.00) | (0.15,1.50) / (0.10,1.30) |
| Disturbance | force / torque | ±10N / ±2Nm | ±18N / ±3.5Nm | ±10N / ±2Nm | ±18N / ±3.5Nm |
| COM offset | xy / z | ±5cm / ±3cm | ±8cm / ±5cm | ±5cm / ±3cm | ±8cm / ±5cm |
| Obstacle motion | max_speed | 0.2 m/s | 0.4 m/s | 0.2 m/s | 0.4 m/s |

(v2.10 = v2.6 ranges + REWARD-2 retune + locked planner. v2.7-v2.9b
DR widening abandoned: aggressive widening regressed convergence
quality at fixed compute. v2.10 tests whether reverting DR alone
recovers v2.6's clean training while retaining REWARD-2's compound
WIN.)

**Untapped axes (single-knob reachable):** payload mass (currently
fixed), goal range (push x to [3, 12] for sustained-nav stress).
Two previously-listed axes now in concrete plans: novel obstacle
shapes (irregular meshes — folded into v2.12 L4 once distance-
transform-on-grid drops the analytical-SDF requirement) and sensor
noise on occupancy grid (folded into v2.12 L3 with dedicated
`Isaac-CBF-Go2-NoisyPerception-v0` OOD env).

**Avoid for OOD eval (changes contract, not distribution):** planner
mode (the TightGap pattern), reward shaping at eval.

### Stress-curve experiment (paper figure)

Pick one priv-obs axis (Slippery / HighDisturbance / HeavyCOM).
Sweep the knob from training-edge to ~2× past in 4–5 steps. Plot BR
vs best baseline combined as a function of magnitude. If BR's edge
grows with difficulty → strong "adaptivity matters more under harder
conditions" figure.

### Strategic priority order (post-v2.11 launch)

Done: v2.6 baseline locked. v2.7-v2.9b abandoned. v2.10 done (partial
recovery, HeavyCOM smoking gun: PLANNER-2a). CBF QP audited; 4-param
mapping verified against Kolathaya/Dean/Molnar/boundary-correction
literature. DR gap identified (`a` has no measurement-noise signal
in training). Post-v2.10 plan reframed around 4 param-aligned
ablations as paper Table 2. **v2.11 launched on lab (2026-05-07
overnight, ~9h ETA).**

1. **v2.11 train + 7-eval + dual-regime HeavyCOM diagnostic +
   Bf-{α,φ,c} ablations** (running, ~9h). Tests bimodal resample DR
   + variable obstacle-motion DR + L_f h drift term + WIDE param
   ranges as a single attribution test against v2.10's HeavyCOM
   regression.
2. **v2.11 iter 2500 gate check** (optional, via watch script) —
   trip wires same as v2.10 + confirm new CBF stats are populating
   (validates bimodal `time_left` write-order assumption).
3. **Run 6-block analysis script post-eval** + paste back. Header-
   aware Python parsers handle B-fixed-X CSV column drift.
4. **Compare v2.11 to v2.10 + v2.6.** If in-dist ≤ 0.31 + HeavyCOM
   margin ≥ +3pp + compound holds → v2.11 = paper baseline.
5. **(if v2.11 ships) Tag v2.11 + archive ckpt + mirror to local.**
   Same pattern as v2.10. v2.11+ code changes additive only.
6. **v2.12 — Realistic perception (4 layers).** L1: replace
   `_compute_h()` with grid-based distance-transform variant
   (train/deploy parity). L2: censored grid (raycast occlusion +
   range falloff). L3: Gaussian grid noise + new
   `Isaac-CBF-Go2-NoisyPerception-v0` OOD env. L4: arbitrary
   obstacle shapes (irregular meshes — once L1 drops analytical-SDF
   requirement, this is ~1 day extra). Train + eval.
7. **4-axis fixed-param ablation table** on final ckpt: BR vs
   B-fixed-α (DensePack vs open) / B-fixed-φ (HighDist) / B-fixed-a
   (NoisyPerception) / B-fixed-c (HeavyCOM, FastObs). **Paper Table 2.**
8. **Stress curve** on strongest single axis (paper-figure quality).
9. **(Parking lot) Drop `b` from 5D action OR commit to SOCP** —
   paper-cleanliness decision after Table 2 lands.
10. **(Landed in v2.11) L_f h obstacle-drift term in constraint.**
    h depends on `(p_robot, p_obstacle)`; pre-v2.11 QP constraint
    omitted the `∂h/∂p_obs · v_obs` term. v2.11 adds it behind
    `USE_LFH_OBSTACLE_DRIFT` flag in `cbf_go2_env.py` — by symmetry
    `∂h/∂p_obs = -L_g h`, so the addition is `+ (L_g_h * v_obs).sum()`
    on RHS (angle-aware via dot product). NOT HOCBF, no solver
    change. Matters most for FastObstacles. Empirical effect TBD
    in v2.11 results.
11. **(Honest limitation, only if robot model changes) HOCBF.**
    Currently we model `ẋ_robot = u` (single-integrator); CBF QP
    works fine because `L_g h ≠ 0`. HOCBF would only be needed if
    we switched to acceleration/torque-level control (`v̇ = u`,
    `ṗ = v`), which makes `L_g h = 0` and requires a higher-order
    formulation. Not on roadmap. List as future work in paper
    limitations.

### v2.7 + v2.8 + v2.9 + v2.9b retrospectives

**v2.7 (REALISM-1, abandoned).** Hypothesis was right; magnitude
was wrong. Wider training *does* grow the gap when the policy can
converge on the wider distribution — but at 3000 iters with v2.7's
aggressive widening (friction span +50%, force +80%, motion 5×),
the teacher under-converged. Training base_contact went 5.8%
(v2.6) → 19% (v2.7), reflecting unfinished optimization.

**v2.8 (PLANNER-2a + 2b + mild DR widening, abandoned).** Three
simultaneous changes: locked-during-training (PLANNER-2a), drop
walk + adversarial planners (PLANNER-2b), mild DR widening (~30-50%
per axis). Each was justified individually; together they regressed
the policy. **Stuck rate tripled** (7.5% → 22.8%) uniformly across
envs — policy-level brittleness. Diagnosis: v2.6's mid-episode
planner switching during training is a *regularizer* that taught
the policy to recover from CBF deflection. Removing it, plus
removing adversarial command-space coverage, left the policy
unable to recover wherever the CBF intervened.

**Lesson learned:** at fixed compute, "make training more realistic"
trades against regularization. v2.6's train/eval mismatch
(switches in train, fewer at eval) was a feature; PLANNER-2a's
free win at eval-time only does NOT translate to a free win at
training-time.

**Probe A confirmed v2.8 is fundamentally worse** (BR 0.456 = best
B 0.457 under v2.6's training-matched mid-switch eval, vs v2.6's
+6.9pp on the same regime). **Probe B series rejected** — B-α'
(revert PLANNER-2a) would fix stuck *artificially* via external
disturbance, not intrinsic recovery. Same critique applies backward
to v2.6 too: v2.6's locked-eval +10.5pp is partly regression-to-mean.

**Real fix is reward shaping (REWARD-2 v2.9)**: gradient pressure on
specific failure modes (stuck + fall) so the policy learns intrinsic
recovery without depending on planner switching.

**v2.9 (REWARD-2, abandoned).** Three reward changes: NEW
`base_contact_penalty -50` (terminal on fall — closes structural
gap; falls had no terminal penalty before), NEW `stuck -2.0/step`
when ‖v_xy‖<0.15 (direct gradient on stuck attractor), CHANGE
`proximity -1.0 → -0.5` (halve dominance of artificial "more far is
more better" pressure). 5K iters, 5/7 evals (FastObs + Compound
killed for v2.9b launch).

Outcome: stuck FIXED (22.8% → 7.7% in-dist, back to v2.6 level —
the stuck term worked) but fall WORSENED (26.5% → 40.8%). Combined
~tied with v2.8 (-1.2pp LOSS in-dist vs best B). Pattern consistent
across all 5 evals (fall 39-50% everywhere) — policy-level signature.

**Diagnosis:** stuck penalty pushed policy to keep moving (good);
base_contact_penalty -50 wasn't strong enough to keep the moving
policy safe. Training trajectory was unstable: action_std went UP
0.69 → 0.81 in second half, mean_reward regressed -8.56 → -10.59,
r_stuck oscillated. PPO couldn't find a stable optimum with
conflicting reward gradients.

**`--no_obstacles` BR diagnostic (v2.9 ckpt):** with K=0 obstacles
forced, BR fall = 8.3% (= loco-internal floor under our DR). So
**80% of v2.9's 40.8% in-dist fall is CBF-attributable** —
controllable lever; reward shaping has substantial room. The
original -100 → -50 worry (caution lock-in → stuck) is empirically
refuted: v2.9 stuck = 7.7% with -50, way under v2.8's 22.8%.
Headroom on the stuck axis to add fall pressure.

**v2.9b retune:** `base_contact_penalty -50 → -100` (single knob).
Matches `collision -100` symmetrically. Predicted fall 15-20%, stuck
7-10%, combined 25-30%.

**v2.9b actual outcome (abandoned).** 7-eval matrix: in-dist combined
0.479 = best B 0.474 (-0.5pp tie). 5 OOD WINS (Slippery +8.2pp,
FastObs +8.7pp, HeavyCOM +3.9pp, HighDist +2.7pp, Compound +1.7pp),
DensePack -1.7pp loss. The retune did push fall down (40.8 → 35.0%)
but caused caution lock-in (stuck 7.7 → 12.9%); net combined barely
moved. **Compound flipped to a WIN** where v2.6 had tied — a real
positive signal that REWARD-2 fixes something on compositional stress.
But in-dist couldn't recover v2.6's +6.9pp WIN.

**Diagnosis (v2.9b → v2.10):** the trade-off pattern wasn't a tuning
problem. It was structural: BR absolute combined values are uniformly
higher than v2.6's across all 7 evals, regardless of margin. Wider
training DR is making the env uniformly harder; the policy can't
converge as cleanly at 5K iters as v2.6 could on its narrow DR.
action_std rose 0.68 → 0.81 in v2.9b's second half (same instability
as v2.9). Reward shaping isn't the dominant lever; **DR width is.**

**v2.10 plan (implemented, ready to launch):** revert training DR
plus matching OOD ranges to v2.6 levels. Keep PLANNER-2a/2b plus
REWARD-2 retune. Single-attribution test of the "narrow DR was the
dominant regression cause" hypothesis. OOD ranges now match v2.6
paper baseline so headline numbers compare apples-to-apples.

Predicted: training base_contact 6-8% (v2.6 was 5.8%), in-dist
combined 25-30% (v2.6 was 30.6%). Plus retain the compound WIN +
the OOD strength v2.9b showed.

**Catch to remember:** OOD ranges define the eval task. When training
DR shifts, OOD ranges must shift too for fair comparison to prior
versions. v2.10 reverted both — comparison to v2.6 paper is direct.

### Anti-patterns to avoid

- **Adversarial OOD that changes the planner contract.** (TightGap
  pattern.) Resist reviewer pushes for "harder" evals that break
  contract symmetry.
- **Compositional before single-axis is clean.** Confounded.
- **Pushing so far every method fails.** If BR=0.80 and best=0.82,
  the win is noise. Stay where best baseline combined < 0.5.
- **Cherry-picking eval scenes.** Same `--num_envs`,
  `--steps_per_config`, seeded RNG across configs.

---

## Done so far

Detail in [LOG.md](LOG.md).

- **Wk1-Wk3 (Apr 13-21):** sim infra; Go2 locomotion; `Isaac-CBF-Go2-v0`
  end-to-end (scene, 5D action, CBF-QP, priv obs, reward, termination);
  min-bar teacher with env_encoder / π_teacher split.
- **Wk3.5 (Apr 22-28):** wider DR + 4-planner mix + 45D priv obs;
  multi-obstacle + SCENE-1.5.
- **04-29:** v1 paper claim landed (rank 1.5). v2 architecture +
  SCENE-2 same evening.
- **04-30 → 05-01:** SCENE-4 (K_MAX=20 random subset); v2.1 skipped;
  v2.2 trained but lost baselines; v2.3 retrained with REWARD-1
  Variant C (falls 25→9%); locomotion task built.
- **05-02:** custom locomotion failed in deploy (28-32% B0 vs ~20%
  off-the-shelf). Reverted; reframed as drop-in safety filter.
  DIAG-1 revealed 10-13% hidden stuck_rate.
- **05-03:** PLANNER-1 (Smooth GOAL + A\*-like + MPC-like). B0 on new
  mix combined 42-55% (~2× old mix). v2.4 trained.
- **05-04:** v2.4 ties baselines; DIAG-2 attributes ~28pp falls to
  CBF deflection; v2.5 over-regularizes (action_std 0.06, lost eval);
  **v2.6 wins eval-1 by +6.9pp combined.** TightGap retired; 5-axis
  near-OOD suite designed and committed.
- **05-05:** v2.6 5-axis suite closed: Slippery +5.6, DensePack +0.6
  tie, HighDisturbance +9.0, FastObstacles +10.6, HeavyCOM +5.9.
  RealisticCompound TIE (−0.3pp) → diagnosed as compositional
  generalization gap. REALISM-1 launched (v2.7 training DR widened).
- **05-06 (day):** v2.7 evals lost across the board (in-dist −0.5pp,
  Slippery −11.4pp, DensePack −0.8pp). Diagnostic isolated v2.7
  ckpt as the cause; **v2.6 + locked planner pushed in-dist margin
  to +10.5pp** (free win from PLANNER-2a alone). v2.7 abandoned.
  v2.8 retrain plan: v2.6 hyperparams + locked planner +
  PLANNER-2b (drop walk/adv) + mild DR widening + 5000 iters.
  `scripts/train_and_eval_v28.sh` runs unattended.
- **05-06 (evening):** **v2.8 abandoned.** Trained 5000 iters; eval
  combined 49.3% in-dist vs v2.6's 30.6% — regression on every
  axis. Stuck rate tripled (7.5% → 22.8%) uniformly across envs;
  policy-level brittleness from removed regularizers. Retracted
  earlier "CBF param `c` randomization" claim (c is action output,
  not env DR — verified in `cbf_go2_env.py:179-183`). Recovered
  v2.6 exact planner mix from saved env.yaml.
- **05-06 (late evening):** **Probe A done.** v2.8 ckpt mid-switch
  eval: BR 0.456 = best B 0.457 (+0.1pp tie); v2.8 fundamentally
  worse policy regardless of eval regime. **B-α' rejected on
  principle** (mid-switch is artificial fix; doesn't generalize at
  deploy; same critique applies backward to v2.6). **REWARD-2 v2.9
  designed + implemented**: NEW `base_contact_penalty -50` (closes
  structural gap — collision -100 only fires on obstacle hits, falls
  had no terminal penalty), NEW `stuck -2.0/step` when ‖v_xy‖<0.15,
  CHANGE `proximity -1.0 → -0.5`. u_safe_rate considered + rejected
  (would conflict with proximity reduction). Code in 4 files,
  ready to sync + launch (~7h).
- **05-06 (overnight):** **v2.9 launched on lab.** SPS 27K (vs
  v2.8's 22K), GPU 26.8/32.6 GB, 48% util — all healthy. Wrote
  `scripts/extract_training_summary.py` (parses rsl_rl logs to
  per-iter CSV; handles partial logs for mid-training spot checks).
  Iter 1211 spot check: `mean_ep_len` 12 → 818 (good), `term_base_contact`
  0.227 → 0.216 (slowly dropping, base_contact_penalty biting), but
  `r_stuck` -0.009 → -0.300 (growing in magnitude — early caution-
  lock-in signature).
- **05-07:** **v2.9 abandoned + v2.9b launched.** v2.9 trained 5K
  iters, ran 5/7 evals (FastObs + Compound killed). Stuck FIXED
  (22.8% → 7.7% in-dist) but fall WORSENED (26.5% → 40.8%); combined
  ~tied with v2.8 (-1.2pp LOSS in-dist). `--no_obstacles` BR
  diagnostic on v2.9 ckpt: fall 8.3% (loco-internal floor), stuck
  16.5% (planner artifact); 80% of v2.9 falls are CBF-attributable.
  v2.9b: single-knob retune `base_contact_penalty: -50 → -100`
  (matches collision symmetrically). Predicted combined 25-30%.
  Also caught + corrected awk bug in best-baseline lookup one-liner.
- **05-07 (late):** **v2.9b done + v2.10 implemented.** v2.9b 7-eval
  matrix: in-dist 0.479 (-0.5pp tie vs best B); 5 OOD WINS / 1
  in-dist tie / 1 DensePack loss. **Compound flipped to WIN** (+1.7pp)
  where v2.6 tied. -100 retune pushed fall down (40.8 → 35.0%) but
  stuck rose (7.7 → 12.9%); net combined barely moved. Pattern
  across all 7 evals: BR absolute combined uniformly higher than
  v2.6's → wider DR is making env uniformly harder. action_std rose
  again 0.68 → 0.81 (same instability as v2.9). **v2.10 implemented**:
  revert training DR plus matching OOD ranges to v2.6; keep
  PLANNER-2a/2b plus REWARD-2 retune. Single-attribution test of "narrow DR was the
  dominant regression cause." Predicted training base_contact 6-8%,
  in-dist combined 25-30%. Ready to sync + launch.

---

## Project reference

### Stable invariants

- **Hardware:** Real Go2 + walking_bridge tested. Heartbeat + spacebar
  e-stops verified. `/utlidar/cloud` active.
- **Sim:** Isaac Lab on RTX 5090. Built on `Isaac-Velocity-Flat-Unitree-Go2-v0`
  as locomotion base.
- **Locomotion checkpoint:** off-the-shelf flat-trained
  `unitree_go2_flat/2026-04-15_19-24-45/exported/policy.pt`. Native
  fall rate 0.5%; in our CBF env at B0 ~20% due to 50 Hz command
  stream vs 10s held training commands. Treated as **fixed,
  off-the-shelf component** (drop-in safety filter philosophy — works
  with any user's locomotion). Frozen black box inside CBF env.
- **Action space (outer RL):** 5D `(α, φ, a, b, c)`; only 4 active
  (`b` reserved for SOCP, see `cbf_go2_env.py:195`).
  **Param-uncertainty mapping (audited 2026-05-07 very late against
  `cbf_go2_env.py:168-230`):**
  - α ∈ [0.1, 5.0] — class-K slope on shifted h `(h - c)`;
    **model/tracking error** (Molnar 2021).
  - φ ∈ [0.0, 5.0] — coefficient on `‖L_g h‖²` (RHS); **actuation
    uncertainty** (Kolathaya 2018 ISSf).
  - a ∈ [0.0, 1.0] — additive RHS slack; **state-indep measurement
    uncertainty** (Dean 2019).
  - c ∈ [0.0, 0.5] — shifts safe-set boundary inward via `h(x) − c`;
    **boundary correction** (when `h_lidar` underestimates the true
    safe boundary).
  - b — reserved for input-dep slack `b·‖u‖` (Dean 2019); requires
    SOCP solver, currently unused.
  Constraint: `L_g h · u_safe ≥ -α(h-c) + φ·‖L_g h‖² + a` (closed-
  form half-space projection on GPU). **v2.11 plan: widen `a` to
  [0, 3.0] and `c` to [0, 1.0] behind `WIDE_PARAM_RANGES` flag.**
- **Privileged obs (PRIV-2):** 8207D = 15D dynamics + 8192D occupancy
  grid (2 frames × 64 × 64 × 0.1m, ego-centric, 6.4m FOV).
- **Teacher network (CNN):** `priv → CNN (Conv 2→16 s=2, 16→32 s=2,
  Linear→64), dyn MLP (15→64) → concat → Linear 128 → 12 → Z(12) →
  π_teacher (128, 5)`. `get_z(obs)` exposed.
- **Student input (Wk3):** LiDAR + base_vel + history → Ẑ → frozen π_teacher.
- **CBF filter:** Eq. 19 multi-obstacle SDF + Eq. 20 exp smoothing +
  4-param robust QP. Per-shape analytical SDF (boxes, cylinders).
  Minkowski sum: footprint + 0.15m. Closed-form half-space projection.
  `b` slot unused.
- **Reward (v2.10 stack on disk = v2.9b reward stack unchanged; v2.6
  working ckpt was trained on the v2.6 stack — proximity -1.0, no
  base_contact_penalty, no stuck term):**
  - `collision -100` (terminal on obstacle_contact)
  - `base_contact_penalty -100` (terminal on fall — REWARD-2 NEW)
  - `stuck -2.0` per step when ‖v_xy‖<0.15 m/s (REWARD-2 NEW)
  - `infeasibility -10` per step
  - `u_safe_deviation -0.1·‖u_safe-u_des‖²` per step
  - `proximity -0.5·exp(-min_sdf/0.5)` per step (REWARD-2 halved from -1.0)
  - `action_rate -0.005·‖Δa‖²` per step (smooths CBF params)
  - `u_safe_rate` function in code but UNREGISTERED — kept for REWARD-3 if v2.10 needs follow-up.
  Termination on per-shape SDF < 0 (obstacle_contact).
- **PPO regularization (v2.6 — working recipe):** AdamW optimizer
  with `weight_decay=1e-5` applied via monkey-patch on
  `OnPolicyRunner.__init__` from `cbf_go2/__init__.py` (no upstream
  Isaac Lab changes; scoped by experiment_name from cfg dict). Constant
  `_CBF_WEIGHT_DECAY` at top of `__init__.py`. `entropy_coef=0.005` (5×
  v2.5's 0.001) is the load-bearing fix that prevented action-std
  collapse. Action-rate penalty `-0.005·‖Δa‖²` (gentler than v2.5's
  `-0.01`). Orthogonal smoothness mechanisms — wd smooths
  input→output map, action-rate smooths in time. Frozen unless future
  failure demands change.
- **Diagnostic scripts (paper-figure inputs):**
  - `scripts/diag_jerk_source.py` — DIAG-1, per-step trace logger.
  - `scripts/eval_baseline.py --no_obstacles` — DIAG-2, planner+loco
    isolation (B0 α=0.5 with K_actual=0 forced).
  - `scripts/diag_lipschitz.py` — DIAG-3, Lipschitz estimator (3
    methods: spectral product, local Jacobian, finite-difference).
    Standalone — pure PyTorch, no Isaac Sim required.
- **Multi-planner mix.** **Working baseline (v2.6 ckpt):** smooth_goal
  0.40 / waypoint 0.25 / mpc 0.20 / legacy_goal 0.05 / walk 0.05 /
  adversarial 0.05 (sum 1.00); `resampling_time_range = (10, 10)`
  → 1 mid-episode switch per 20s episode. **On-disk env_cfg (v2.10,
  inherited from v2.8 PLANNER changes — kept for v2.10):**
  smooth_goal 0.45 / waypoint 0.30 / mpc 0.20 / legacy_goal 0.05
  (PLANNER-2b dropped walk + adversarial); `resampling_time_range =
  (100, 100)` → locked per episode (PLANNER-2a). Locked planner is
  deployment-realistic; B-α' (revert to mid-switch) was rejected as
  artificial fix.
- **DR.** **v2.10 on-disk = v2.6 narrow DR (REVERTED 2026-05-07):**
  friction (0.30, 1.20) / (0.20, 1.00), force ±10N / torque ±2Nm,
  COM xy ±5cm / z ±3cm, obstacle max_speed 0.2 m/s. Per-reset
  K_actual ∈ [0, 20] uniform; spawn 5.5×5m with pair-wise min-sep
  `r[i]+r[j]+0.4m`. v2.7's aggressive widening underconverged at
  3000 iters; v2.8/v2.9/v2.9b's milder widening at 5000 iters
  uniformly raised absolute combined eval values; v2.10 reverts to
  v2.6's narrow DR to recover clean convergence while keeping
  PLANNER-2a/2b + REWARD-2 retune.
- **SCENE-4 pool:** K_MAX=20 (8 cubes, 6 cylinders, 4 walls, 2 rect
  boxes, 0.20-2.0m); K_actual unique indices uniform from {0..19}.
- **Moving obstacles:** ~50% drift constant velocity per episode
  (±0.5 m/s per axis at v2.8). Cap below robot's ~1 m/s cruise.
- **Sensor noise + perception pipeline (slated for v2.12; needs
  refresher before coding):** sim currently perfect on two axes —
  (a) the QP uses **privileged analytical SDF** in `_compute_h()`,
  not a grid-based variant; (b) the priv-obs grid is an ego-centric
  crop of perfect obstacle info, **not LiDAR-realistic** (no
  occlusion, no range falloff, no noise). Both will need to change
  for honest deploy parity. v2.12 four-layer plan:
  - **L1 (h-pipeline parity):** replace `_compute_h()` with
    distance-transform-on-grid variant; train QP and deploy QP use
    the same h(x) computation. Robot footprint via morphological
    dilation of grid (replaces analytical Minkowski expansion);
    L_g h via finite-diff Sobel on the SDF grid.
  - **L2 (censored grid):** raycast-built priv-obs grid with
    line-of-sight occlusion + range falloff. LiDAR-realistic without
    full sensor sim. `a` (measurement uncertainty) is the CBF-
    theoretic compensation for what censoring can't capture.
  - **L3 (noise):** Gaussian noise (σ ∈ [0, 0.1]) on the grid in
    training DR mix; dedicated `Isaac-CBF-Go2-NoisyPerception-v0`
    OOD env (higher σ).
  - **L4 (arbitrary obstacle shapes):** once L1 drops the analytical-
    SDF requirement, irregular meshes (chairs, tables, weird objects)
    become trivial — rasterize footprint onto grid and let distance
    transform handle them. ~1 day for asset library + grid
    rasterization. Strengthens deploy realism story (real obstacles
    aren't cylinders).
  Closes the `a`-is-dead-weight gap with a real physical role
  (Dean 2019 measurement uncertainty bridges train h_priv vs deploy
  h_grid). Estimated ~4-5 days of work for all 4 layers (vs original
  1-hour scope for noise-only). Most asymmetric axis available
  (teacher's CNN can denoise; baselines structurally cannot).
- **u_des never enters the network.** Sidecar to CBF; only in reward
  via `‖u_safe - u_des‖²`.

### Risks & pivots

| Risk | L | I | Mitigation |
| --- | --- | --- | --- |
| Adaptation claim doesn't land | L | H | v2.6 baseline still holds: 5 wins / 1 compound tie / +6.9pp in-dist (or +10.5pp with locked-planner eval). v2.7-v2.9b all regressed; B-α' rejected. v2.9b actually won compound (+1.7pp) where v2.6 tied — REWARD-2 fixes something on compositional stress. v2.10 (DR revert + REWARD-2 retune) launching to test "narrow DR was dominant regression cause." Fallback: ship v2.6 + locked-planner-eval as paper claim. |
| Sim-to-real gap on LiDAR | H | M | Inject matching noise during student training |
| Reward tuning hell | M | M | Bounded ablation budget |
| Teacher-student unstable | M | M | Fall back to end-to-end PPO |
| Sim-to-real gap on dynamics | M | M | Robust CBF absorbs via φ; sim as main story |
| Locomotion fails on real floors | M | M | Off-the-shelf flat-trained loco; sim-to-real gap absorbed by teacher's robust CBF |
| CBF-QP infeasible at deploy | L | L | Soft constraints; damping fallback |
| Hardware breaks | L | M | Sim-only paper still submittable |

**Drop list (cut in this order if timeline tightens):**

1. Hardware demo (Wk4 stretch) → sim-only paper.
2. **L_f h obstacle-drift term + HOCBF** → keep as honest future-work
   limitations. The L_f h term is a ~50-line constraint extension
   (no solver change); HOCBF only matters if we switch to lower-level
   control (not on roadmap).
3. **v2.12 NoisyPerception env** → ship without it; flag `a` as
   "currently inactive in training DR; future-work" in paper
   limitations. Loses the 4th ablation row but preserves α/φ/c rows
   of Table 2.
4. **v2.11 (DR-implicit shaping + B-fixed-X ablations)** → ship v2.10
   plus 5-axis OOD as headline (no Table 2 ablation, but
   combined-metric adaptation story still defensible).
5. REWARD-3 (u_safe_rate registration) → ship v2.10 with whatever fall
   rate it lands at, honest discussion in limitations.
6. Further reward weight-tune iterations after v2.10 → ship v2.10 result.
7. v2.10 retrain → ship v2.6 + locked-planner-eval as the headline
   (already +10.5pp in-dist).
8. Per-planner breakdown eval → headline + near-OOD covers the claim.
9. INFRA-1 non-deploy constants → `LOCOMOTION_CHECKPOINT` move stays mandatory.

**Default fallback if v2.10 stalls:** ship v2.6 + locked planner
(eval-time only) + the existing 5-axis OOD suite as the headline.
The +10.5pp in-dist combined with 5 OOD wins and an honest compound-tie
limitation is a defensible paper. Caveat: v2.6 itself relies partly
on training-time mid-switching to escape stuck (intrinsic recovery
uncertain at deploy); flag as a sim-to-real risk in the paper.

**Pivot if adaptation claim doesn't survive:** Drop "teacher
Pareto-dominates hand-tuned." Keep "student matches teacher — LiDAR
plus history sufficient for CBF tuning, no privileged obs at deploy."
Paper still submittable.
