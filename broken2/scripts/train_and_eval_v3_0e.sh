#!/bin/bash
# v3.0e — pretrain encoder with supervised aux loss, then PPO with frozen encoder.
#
# Diagnostic story leading here:
#   v3.0a: gradient bottleneck (sparse ∂u_safe/∂α). FAIL.
#   v3.0b: cbf_state in obs. (combined with c into one run)
#   v3.0c: + dense LHS reward. Still no state-conditioning. FAIL.
#   Z diag: encoder healthy (12/12 active) BUT ‖μ_indist-μ_HeavyCOM‖=0.16
#           encoder ignored dyn path (1.3K params) for grid path (530K).
#   v3.0d: skip-connection routing raw 19-D dyn around encoder. α_mean
#          diff grew from 0.06 → 0.21 across tasks but still <0.3 threshold.
#          combined LOST on HeavyCOM (OOD test confound).
#   Alpha-corr diag: policy IS state-conditional, but on cbf_state
#          (slack r=+0.64, h r=+0.54, ‖Lgh‖² r=−0.62) NOT on DR features
#          (friction r=+0.03, COM r≈0, mass r=+0.15 weak).
#
# v3.0e fix:
#   PRETRAIN the encoder with a CLEAN supervised gradient on the 15-D
#   privileged feature target. This forces Z to encode env-class info
#   BEFORE PPO starts. Then PPO trains with encoder FROZEN, so Z stays
#   committed to env-class encoding throughout training.
#
# Code change for v3.0e (vs v3.0d):
#   cbf_go2_teacher_cnn.py:  loads pretrained encoder from env var
#                            CBF_PRETRAINED_ENCODER, freezes if
#                            CBF_FREEZE_ENCODER=1
#   rsl_rl_ppo_cfg.py:       z_dim 12 → 24 (more room in Z for env class)
#   scripts/pretrain_encoder.py: NEW — supervised pretrain script
#
# Pipeline:
#   Phase 1 (~30 min): pretrain encoder + aux_head on (obs → priv 15D).
#                      Output: pretrained_encoder_v3_0e.pt
#   Phase 2 (~3h):     PPO training with pretrained+frozen encoder.
#                      Skip-connection + cbf_state obs + LHS reward all
#                      stay active from v3.0b/c/d.
#   Phase 3 (~25 min): eval on indist + HeavyCOM, B0 sweep + BR.
#
# Decision criterion (locked):
#   PASS:  alpha CORRELATION |r| > 0.20 with friction OR |com_offset|
#          AND BR combined beats best-of-B0 by ≥3pp on ≥1 task.
#   AMBIG: |r| > 0.20 visible but combined metric doesn't beat fixed-α.
#          Adaptation emerged; tune reward/curriculum to make it useful.
#   FAIL:  |r| < 0.10 on all DR features. Pretrained Z carries env class
#          but π_teacher still ignores it. Need to widen DR or rethink.
#
# Total wall: ~4h.
#
# Sync command before launch:
#   rsync -av ~/Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/{cbf_go2_teacher_cnn.py,agents/rsl_rl_ppo_cfg.py} \\
#       chrisliang@130.64.84.163:Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/
#   rsync -av ~/Desktop/safety-go2/scripts/{pretrain_encoder.py,train_and_eval_v3_0e.sh,diagnose_alpha_corr.py} \\
#       chrisliang@130.64.84.163:Desktop/safety-go2/scripts/
#
# Usage on lab box (run under tmux to survive SSH drops):
#   tmux new -s v30e
#   cd ~/Desktop/safety-go2/IsaacLab && \\
#   ~/Desktop/safety-go2/scripts/train_and_eval_v3_0e.sh 2>&1 | tee logs/train_and_eval_v3_0e.log
#   # detach with Ctrl-B then D; reattach: tmux attach -t v30e

set -e

cd ~/Desktop/safety-go2/IsaacLab

PRETRAINED=$(pwd)/pretrained_encoder_v3_0e.pt
Z_DIM=24

echo "================================================================"
echo "v3.0e (pretrained encoder + frozen during PPO) pipeline"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# Pre-flight: confirm v3.0e changes are present
echo ""
echo "Pre-flight: confirm v3.0e changes are in place"
grep -q "z_dim: int = 24" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/agents/rsl_rl_ppo_cfg.py \
  && echo "  ✓ z_dim = 24 (v3.0e, rsl_rl_ppo_cfg)" \
  || { echo "  ✗ z_dim not bumped to 24 — sync v3.0e changes first!"; exit 1; }
grep -q "CBF_PRETRAINED_ENCODER" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_cnn.py \
  && echo "  ✓ pretrained-encoder loader in teacher_cnn (v3.0e)" \
  || { echo "  ✗ pretrained-encoder hook missing — sync v3.0e changes first!"; exit 1; }
test -f ~/Desktop/safety-go2/scripts/pretrain_encoder.py \
  && echo "  ✓ pretrain_encoder.py exists" \
  || { echo "  ✗ pretrain_encoder.py missing — sync first!"; exit 1; }

# ---------- PHASE 1: PRETRAIN ENCODER ----------
echo ""
echo "================================================================"
echo "[1/3] PRETRAIN ENCODER: supervised on priv features → Z"
echo "      1024 envs × 200 steps = 204800 obs; 50 epochs; z_dim=$Z_DIM"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/pretrain_encoder.py \
  --task Isaac-CBF-Go2-v0 \
  --num_envs 1024 --rollout_steps 200 \
  --num_epochs 50 --batch_size 4096 \
  --learning_rate 5e-4 --z_dim $Z_DIM \
  --output "$PRETRAINED" \
  --headless

if [ ! -f "$PRETRAINED" ]; then
    echo "ERROR: pretrained encoder file not produced: $PRETRAINED"
    exit 1
fi
echo ""
echo "[1/3] PRETRAIN done at $(date '+%H:%M:%S'). File: $PRETRAINED"

# ---------- PHASE 2: PPO TRAINING WITH FROZEN ENCODER ----------
echo ""
echo "================================================================"
echo "[2/3] PPO TRAINING: 3000 iters, 4096 envs, frozen encoder"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_PRETRAINED_ENCODER="$PRETRAINED"
export CBF_FREEZE_ENCODER=1

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-v0 \
  --num_envs 4096 --max_iterations 3000 \
  --headless

unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER

echo ""
echo "[2/3] PPO TRAINING done at $(date '+%H:%M:%S')"

# ---------- LOCATE CHECKPOINT ----------
echo ""
echo "================================================================"
echo "Locating most recent checkpoint..."
echo "================================================================"

LATEST_DIR=$(ls -1td logs/rsl_rl/cbf_go2_teacher/*/ | head -1)
LATEST_DIR=${LATEST_DIR%/}
CKPT=$(ls -1t "${LATEST_DIR}"/model_*.pt 2>/dev/null | head -1)

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found in ${LATEST_DIR}"
    exit 1
fi
echo "Using checkpoint: $CKPT"

# ---------- PHASE 3: EVAL ----------
echo ""
echo "================================================================"
echo "[3/3] HEADLINE EVAL: 2 tasks (in-dist + HeavyCOM)"
echo "      Modes: B0 (sweep 7 α values) + BR"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

# Eval also needs the pretrained encoder so the trained policy loads
# correctly (state_dict must match the model the policy was saved with).
export CBF_PRETRAINED_ENCODER="$PRETRAINED"
export CBF_FREEZE_ENCODER=1

EVAL_TASKS=("v0" "HeavyCOM-v0")

task_tag() {
  if [ "$1" = "v0" ]; then echo "indist"; else echo "${1%-v0}"; fi
}

for TASK in "${EVAL_TASKS[@]}"; do
  TAG=$(task_tag "$TASK")
  OUT="logs/baseline_eval_v3_0e_${TAG}"
  echo ""
  echo "  >>> [$TASK] -> $OUT  ($(date '+%H:%M:%S'))"
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
    --task "Isaac-CBF-Go2-${TASK}" \
    --num_envs 64 --steps_per_config 1500 \
    --modes B0,BR \
    --alpha_grid "0.1,0.5,1.0,2.0,3.0,4.0,5.0" \
    --checkpoint "$CKPT" \
    --output_dir "$OUT" \
    --headless > "${OUT}.stdout.log" 2>&1
  echo "  >>> [$TASK] done at $(date '+%H:%M:%S')"
done

# ---------- ALPHA-CORRELATION DIAGNOSTIC ----------
echo ""
echo "================================================================"
echo "Running alpha-correlation diagnostic on v3.0e checkpoint"
echo "================================================================"

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output diagnose_alpha_corr_v3_0e.json \
  --headless

unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER

echo ""
echo "================================================================"
echo "PIPELINE DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint:   $CKPT"
echo "Pretrained:   $PRETRAINED"
echo "Eval CSVs:    logs/baseline_eval_v3_0e_{indist,HeavyCOM}/"
echo "α-corr JSON:  diagnose_alpha_corr_v3_0e.json"
echo ""
echo "Decision criterion:"
echo "  PASS:  Pearson(α, friction) OR Pearson(α, |com_offset|) > 0.20"
echo "         AND BR combined beats best-of-B0 by ≥3pp on ≥1 task"
echo "  AMBIG: corr visible but combined metric doesn't beat fixed-α"
echo "  FAIL:  all DR-feature corrs < 0.10 → π_teacher ignores Z's env class"
echo "================================================================"
