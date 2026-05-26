#!/bin/bash
# Wk3 within-episode push (2026-05-16): strengthened mid-episode push event
# + symptom-based priv obs (tracking_err history + IMU ω, removed
# applied_force/torque). a/c stay frozen at 0.
#
# Motivation: n_rays=128 retrain landed at BR collision 0.671 vs best fixed
# baseline 0.455 (-21.6pp). Joint success (no collision AND no fall AND
# reached goal) ~0.29 for BR vs ~0.35 for best baseline — baseline still
# wins. Cleaner perception alone did not unblock the teacher.
#
# Diagnosis (with buddy): all training-DR axes (friction, COM, applied_force,
# σ_actuation) are sampled once per reset, so the encoder cannot learn
# WITHIN-episode α/φ modulation. PPO settles on near-constant high α/φ and
# games velocity_tracking. The teacher has no environmental REASON to dial
# params situationally.
#
# Two coordinated changes in this iteration:
#   1. CbfGo2EnvCfg_LAYER3_PUSH overrides push_robot (±1.0 m/s every 5-10 s,
#      vs parent's ±0.5 m/s every 10-15 s). Equivalent to ~30 N × 0.5 s
#      impulse on the 15 kg base. 2-4 pushes per 20 s episode.
#   2. TeacherPrivCfg / TeacherPrivLidarCfg full swap:
#        REMOVED applied_force, applied_torque (sim-only — no hardware sensor)
#        tracking_err   3 → 15 (history_length=5, flatten)
#        ADDED          base_ang_vel (3) — readable on the real Go2 IMU
#      Net priv dim 16 → 25. _PRIV_DIM in cbf_go2_teacher_rma.py bumped.
#
# Decision gates:
#   - joint_success = (no collision) ∧ (no fall) ∧ (reached goal) on the
#     Layer3-Push eval beats best fixed-(α, φ) baseline
#   - α_std and φ_std clearly > the n_rays=128 retrain levels (1.66, 0.98)
#   - Post-hoc Pearson(α, |tracking_err|) and Pearson(φ, |IMU ω|) > 0.15
#     (symptom couplings)
#   - Encoder R²(COM, σ) preserved
#
# Sync before launch (from local Mac) — Wk3 within-episode touches env_cfg,
# __init__, teacher_rma, and the new launch script:
#   bash ~/Desktop/safety-go2/scripts/sync_to_lab.sh
#
# Usage on lab box (in tmux):
#   tmux new -s wk3push
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_layer3_push.sh 2>&1 | tee logs/train_and_eval_wk3push.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

AUX_COEF=0.0

echo "================================================================"
echo "Wk3 within-episode push: LAYER3_PUSH + symptom-based priv obs"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# Pre-flight: confirm the within-episode push pieces are in place
echo ""
echo "Pre-flight checks"
test -f source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_rma.py \
  && echo "  ✓ cbf_go2_teacher_rma.py present" \
  || { echo "  ✗ cbf_go2_teacher_rma.py missing — sync first"; exit 1; }
grep -q "CbfGo2EnvCfg_LAYER3_PUSH" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ CbfGo2EnvCfg_LAYER3_PUSH config present" \
  || { echo "  ✗ LAYER3_PUSH config missing — sync env_cfg.py"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-Push-v0" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ Isaac-CBF-Go2-RMA-Layer3-Push-v0 task registered" \
  || { echo "  ✗ LAYER3_PUSH task not registered — sync __init__.py"; exit 1; }
grep -q "^_PRIV_DIM = 31" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_rma.py \
  && echo "  ✓ _PRIV_DIM bumped to 31" \
  || { echo "  ✗ _PRIV_DIM not bumped to 31 — sync cbf_go2_teacher_rma.py"; exit 1; }
grep -q "base_ang_vel" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ base_ang_vel priv obs present" \
  || { echo "  ✗ base_ang_vel priv obs missing — sync env_cfg.py"; exit 1; }

# ---------- PHASE 1: PPO TRAINING ----------
echo ""
echo "================================================================"
echo "[1/3] PPO TRAINING: ${CBF_ITERATIONS:-1500} iters, 4096 envs"
echo "      RMA split encoders, AUX_COEF=0, symptom-based priv obs"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_AUX_COEF=$AUX_COEF
unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER
unset CBF_PRETRAINED_PRIV
unset CBF_FREEZE_PRIV

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-v0 \
  --num_envs 4096 --max_iterations ${CBF_ITERATIONS:-1500} \
  --headless

unset CBF_AUX_COEF

echo ""
echo "[1/3] PPO TRAINING done at $(date '+%H:%M:%S')"

# ---------- LOCATE CHECKPOINT ----------
LATEST_DIR=$(ls -1td logs/rsl_rl/cbf_go2_teacher_rma/*/ | head -1)
LATEST_DIR=${LATEST_DIR%/}
CKPT=$(ls -1t "${LATEST_DIR}"/model_*.pt 2>/dev/null | head -1)

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found in ${LATEST_DIR}"
    exit 1
fi
echo "Using checkpoint: $CKPT"

# ---------- PHASE 2: DIAGNOSTICS ----------
echo ""
echo "================================================================"
echo "[2/3] φ-CORR + α-CORR + PROBE DIAGNOSTICS"
echo "      Started at $(date '+%H:%M:%S')"
echo "      NOTE: --priv_dim 31 (was 16 pre-symptom-swap)"
echo "================================================================"

export CBF_AUX_COEF=0.0

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --use_locked \
  --output diagnose_phi_corr_wk3push.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --priv_dim 31 \
  --output diagnose_alpha_corr_wk3push.json \
  --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3push.json \
  --headless

# ---------- PHASE 3: HEADLINE EVAL ----------
echo ""
echo "================================================================"
echo "[3/3] HEADLINE EVAL: B0 + B1 + B2 + BR vs LAYER3_PUSH in-dist"
echo "      Joint-success metric will be computed from the CSV post-hoc"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

EVAL_OUT="logs/baseline_eval_wk3push_indist"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-Push-v0 \
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
echo "Checkpoint:           $CKPT"
echo "α-corr JSON:          diagnose_alpha_corr_wk3push.json"
echo "φ-corr JSON:          diagnose_phi_corr_wk3push.json"
echo "Linear probe JSON:    probe_z_linear_wk3push.json"
echo "Eval CSV:             logs/baseline_eval_wk3push_indist/baseline.csv"
echo ""
echo "Decision gates for Wk3 within-episode push:"
echo "  - joint_success (no col ∧ no fall ∧ goal) BR > best fixed baseline"
echo "  - α_std and φ_std > n_rays=128 retrain (1.66 / 0.98)"
echo "  - Pearson(α, |tracking_err|) > 0.15 ; Pearson(φ, |IMU ω|) > 0.15"
echo "  - Encoder R²(COM, σ) preserved (linear probe)"
echo "================================================================"
