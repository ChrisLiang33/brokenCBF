#!/bin/bash
# v3.0f — analytical aux loss on CBF slack + explicit forward-progress reward.
#
# Builds on v3.0e (pretrained+frozen encoder, z_dim=24, cbf_state in obs,
# raw-dyn skip, v3.0c LHS reward). v3.0f adds two things:
#
# 1. Auxiliary supervised loss inside PPO.update():
#       L_aux = AUX_COEF · softplus(−slack(α_new))
#    where slack is recomputed analytically from the policy's α output
#    on the current mini-batch obs (using cached h and L_g h·u_des from
#    cbf_state). The gradient ∂L_aux/∂α is DENSE — non-zero every step,
#    independent of QP idle/active. Provides the per-step push on α
#    that PPO's normal policy gradient can't deliver in the idle zone.
#
# 2. Explicit forward-progress reward:
#       r_progress = +0.5 · max(0, v_xy · u_des_xy / ‖u_des_xy‖)
#    Without this, the aux loss can dominate the policy toward "stand
#    still and stay safe." This reward pulls the equilibrium back
#    toward "actually moving toward the goal."
#
# Trade-off found by PPO equilibrium:
#   - aux loss: push α high (more safety margin)
#   - velocity_along_cmd: reward forward motion
#   - collision/proximity/base_contact: penalize getting too close
#   The right state-conditional α emerges at the equilibrium of all three.
#
# Decision criterion (locked):
#   PASS:  Pearson(α, h) > 0.5 (stronger than v3.0d's 0.54)
#          AND Pearson(α, friction OR |com_offset|) > 0.20 (DR feature)
#          AND BR combined beats best-of-B0 by ≥3pp on ≥1 task
#   AMBIG: alpha-corrs improve but combined metric ties
#   FAIL:  alpha distribution still uniformly high → aux dominated;
#          reduce CBF_AUX_COEF or raise velocity_along_cmd weight
#
# Time: ~30 min pretrain + ~3h PPO + ~25 min eval = ~4h.
#
# Sync before launch:
#   rsync -av ~/Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/{__init__.py,cbf_go2_teacher_cnn.py,cbf_go2_rewards.py,cbf_go2_env_cfg.py} \
#       chrisliang@130.64.84.163:Desktop/safety-go2/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/
#   rsync -av ~/Desktop/safety-go2/scripts/train_and_eval_v3_0f.sh \
#       chrisliang@130.64.84.163:Desktop/safety-go2/scripts/
#
# Usage on lab box (under tmux):
#   tmux new -s v30f
#   cd ~/Desktop/safety-go2/IsaacLab && \\
#   ~/Desktop/safety-go2/scripts/train_and_eval_v3_0f.sh 2>&1 | tee logs/train_and_eval_v3_0f.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

PRETRAINED=$(pwd)/pretrained_encoder_v3_0f.pt
Z_DIM=24
AUX_COEF=0.02    # softplus(-slack) coefficient. Reduced from initial 0.05
                 # after sanity check: six existing reward terms already
                 # push toward high α (less projection); aux at 0.02 is
                 # ~20% of v3.0c's reward scale, enough to provide dense
                 # gradient signal without overwhelming the equilibrium.

echo "================================================================"
echo "v3.0f (analytical aux loss + progress reward) pipeline"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# Pre-flight: confirm v3.0f changes are present
echo ""
echo "Pre-flight: confirm v3.0f changes are in place"
grep -q "def compute_aux_loss" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_teacher_cnn.py \
  && echo "  ✓ compute_aux_loss() on actor model" \
  || { echo "  ✗ compute_aux_loss missing — sync v3.0f changes first!"; exit 1; }
grep -q "def velocity_along_cmd" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_rewards.py \
  && echo "  ✓ velocity_along_cmd reward function" \
  || { echo "  ✗ velocity_along_cmd reward missing!"; exit 1; }
grep -q "velocity_along_cmd = RewTerm" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ velocity_along_cmd RewTerm registered" \
  || { echo "  ✗ velocity_along_cmd RewTerm not registered!"; exit 1; }
grep -q "_cbf_aux_patched" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ PPO.update aux-loss monkey-patch present" \
  || { echo "  ✗ PPO aux-loss patch missing!"; exit 1; }

# ---------- PHASE 1: PRETRAIN ENCODER (same as v3.0e) ----------
echo ""
echo "================================================================"
echo "[1/3] PRETRAIN ENCODER: supervised on priv features → Z"
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

# ---------- PHASE 2: PPO TRAINING WITH AUX LOSS + FROZEN ENCODER ----------
echo ""
echo "================================================================"
echo "[2/3] PPO TRAINING: 3000 iters, 4096 envs"
echo "      Frozen pretrained encoder + CBF_AUX_COEF=$AUX_COEF + v_along_cmd reward"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

export CBF_PRETRAINED_ENCODER="$PRETRAINED"
export CBF_FREEZE_ENCODER=1
export CBF_AUX_COEF=$AUX_COEF

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-v0 \
  --num_envs 4096 --max_iterations 3000 \
  --headless

unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER
unset CBF_AUX_COEF

echo ""
echo "[2/3] PPO TRAINING done at $(date '+%H:%M:%S')"

# ---------- LOCATE CHECKPOINT ----------
LATEST_DIR=$(ls -1td logs/rsl_rl/cbf_go2_teacher/*/ | head -1)
LATEST_DIR=${LATEST_DIR%/}
CKPT=$(ls -1t "${LATEST_DIR}"/model_*.pt 2>/dev/null | head -1)

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint found in ${LATEST_DIR}"
    exit 1
fi
echo "Using checkpoint: $CKPT"

# ---------- PHASE 3: EVAL + alpha-corr DIAGNOSTIC ----------
echo ""
echo "================================================================"
echo "[3/3] HEADLINE EVAL + alpha-corr diagnostic"
echo "      Started at $(date '+%H:%M:%S')"
echo "================================================================"

# Eval also needs the pretrained encoder env vars for state_dict load.
export CBF_PRETRAINED_ENCODER="$PRETRAINED"
export CBF_FREEZE_ENCODER=1
# Aux coef irrelevant for eval but set to 0 to skip the aux patch.
export CBF_AUX_COEF=0.0

EVAL_TASKS=("v0" "HeavyCOM-v0")

task_tag() {
  if [ "$1" = "v0" ]; then echo "indist"; else echo "${1%-v0}"; fi
}

for TASK in "${EVAL_TASKS[@]}"; do
  TAG=$(task_tag "$TASK")
  OUT="logs/baseline_eval_v3_0f_${TAG}"
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

echo ""
echo "Running alpha-correlation diagnostic..."
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 100 \
  --output diagnose_alpha_corr_v3_0f.json \
  --headless

unset CBF_PRETRAINED_ENCODER
unset CBF_FREEZE_ENCODER
unset CBF_AUX_COEF

echo ""
echo "================================================================"
echo "PIPELINE DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint:   $CKPT"
echo "Pretrained:   $PRETRAINED"
echo "Eval CSVs:    logs/baseline_eval_v3_0f_{indist,HeavyCOM}/"
echo "α-corr JSON:  diagnose_alpha_corr_v3_0f.json"
echo ""
echo "Decision criterion:"
echo "  PASS:  Pearson(α, friction) OR Pearson(α, |com_offset|) > 0.20"
echo "         AND BR combined beats best-of-B0 by ≥3pp on ≥1 task"
echo "  AMBIG: corr visible but combined metric doesn't beat fixed-α"
echo "  FAIL:  all DR-feature corrs < 0.10, OR aux dominated"
echo "         (α_mean near 5.0, α_std near 0 → reduce CBF_AUX_COEF)"
echo "================================================================"
