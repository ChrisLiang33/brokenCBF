#!/bin/bash
# V14 (2026-05-21) — V13.1 architecture + wider DR + adaptive c.
# Layered changes:
#   V13.1 (two-stream + proprio noise)
#   + V13.2 (σ_act 0.40, friction (0.10, 1.30))
#   + V14 (c_param_range = (-0.20, 0.20), δR DR ∈ (-0.15, 0.15))
#
# 3 of 5 cbf_params now adaptive: α, φ, c.  a stays frozen at 0.05, b unused.
#
# Bumped iterations 2500 → 3000 to give PPO room for the harder optimization.
#
# Usage on lab box:
#   tmux new -s wk3v14
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   CBF_ITERATIONS=3000 ~/Desktop/safety-go2/scripts/train_and_eval_twostream_v14.sh \
#     2>&1 | tee logs/train_and_eval_wk3v14.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-3000}
ENVS=${CBF_NUM_ENVS:-4096}

echo "================================================================"
echo "V14: V13.1 + wider DR + adaptive c"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')  iters=${ITERATIONS}"
echo "================================================================"

grep -q "CbfGo2EnvCfg_LAYER3_TWOSTREAM_V14" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ V14 train cfg present" \
  || { echo "  ✗ V14 train cfg missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-TwoStream-V14-v0" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ V14 train task registered" \
  || { echo "  ✗ V14 train task not registered"; exit 1; }

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER CBF_FREEZE_ENCODER CBF_PRETRAINED_PRIV CBF_FREEZE_PRIV

echo ""
echo "[1/4] PPO TRAINING at $(date '+%H:%M:%S')"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V14-v0 \
  --num_envs "${ENVS}" --max_iterations "${ITERATIONS}" \
  --headless

unset CBF_AUX_COEF
LATEST_DIR=$(ls -1td logs/rsl_rl/cbf_go2_teacher_rma/*/ | head -1)
LATEST_DIR=${LATEST_DIR%/}
CKPT=$(ls -1 "${LATEST_DIR}"/model_*.pt 2>/dev/null \
    | awk -F'model_|\\.pt' '{print $2, $0}' \
    | sort -n | awk '{print $2}' | tail -1)
[ -f "$CKPT" ] || { echo "ERROR: no ckpt"; exit 1; }
echo "Final ckpt: $CKPT"

echo ""
echo "[2/4] DIAGNOSTICS at $(date '+%H:%M:%S')"
export CBF_AUX_COEF=0.0

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V14-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --priv_layout v13 \
  --output diagnose_phi_corr_wk3v14.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V14-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --priv_layout v13 \
  --output diagnose_alpha_corr_wk3v14.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V14-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3v14.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_grad_sensitivity.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V14-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 50 --priv_dim 33 \
  --priv_layout v13 \
  --alpha_min 0.5 --alpha_max 3.0 \
  --output diagnose_grad_sensitivity_wk3v14.json --headless

echo ""
echo "[3/4] 4-EVAL (single seed, full sweep) at $(date '+%H:%M:%S')"

# 1. in-dist (V14 task — full wider DR + adaptive c at eval)
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V14-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v14_indist" --headless

# 2. trainmatch (adaptive c at deploy)
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-AdaptiveC-TrainMatch-V14-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v14_trainmatch" --headless

# 3. OOD (adaptive c)
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-AdaptiveC-V14-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v14_ood" --headless

# 4. STRESSOR (adaptive c)
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-AdaptiveC-Stressor-V14-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v14_stressor" --headless

echo ""
echo "================================================================"
echo "V14 TRAIN+EVAL DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt: $CKPT"
echo "================================================================"

# ─────────────────────────────────────────────────────────────────────
# [4/4] CHAINED MULTI-SEED SWEEP (V13.1 + V14 head-to-head)
# Reuses multi_seed_sweep.sh but with V14 as the second teacher.
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] CHAINED MULTI-SEED SWEEP at $(date '+%H:%M:%S')"

TEACHER_V13_1=${TEACHER_V13_1:-/home/chrisliang/Desktop/safety-go2/IsaacLab/logs/rsl_rl/cbf_go2_teacher_rma/2026-05-20_18-25-01/model_2499.pt}
if [ ! -f "$TEACHER_V13_1" ]; then
  echo "WARNING: V13.1 teacher not found at $TEACHER_V13_1; skipping V13.1 leg."
  TEACHER_V13_1=""
fi

# Variant of multi_seed_sweep.sh that uses V14's task IDs instead of V13.2's.
SEEDS=(42 123 7)
declare -A TASK_V13_1
TASK_V13_1[trainmatch]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V13-1-v0"
TASK_V13_1[ood]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-V13-1-v0"
TASK_V13_1[stressor]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-Stressor-V13-1-v0"
declare -A TASK_V14
TASK_V14[trainmatch]="Isaac-CBF-Go2-RMA-Deploy-Realistic-AdaptiveC-TrainMatch-V14-v0"
TASK_V14[ood]="Isaac-CBF-Go2-RMA-Deploy-Realistic-AdaptiveC-V14-v0"
TASK_V14[stressor]="Isaac-CBF-Go2-RMA-Deploy-Realistic-AdaptiveC-Stressor-V14-v0"

run_one() {
  local model=$1; local dist=$2; local seed=$3
  local task=$4; local teacher=$5
  echo ""
  echo "─── ${model} / ${dist} / seed=${seed} ───"
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
    --task "$task" \
    --num_envs 64 --steps_per_config 1000 \
    --modes B0,B1,B2,BR \
    --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
    --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
    --checkpoint "$teacher" --seed "$seed" \
    --output_dir "logs/multiseed_${model}_${dist}_seed${seed}" --headless
}

if [ -n "$TEACHER_V13_1" ]; then
  for dist in trainmatch ood stressor; do
    for seed in "${SEEDS[@]}"; do
      run_one "v13_1" "$dist" "$seed" "${TASK_V13_1[$dist]}" "$TEACHER_V13_1"
    done
  done
fi

for dist in trainmatch ood stressor; do
  for seed in "${SEEDS[@]}"; do
    run_one "v14" "$dist" "$seed" "${TASK_V14[$dist]}" "$CKPT"
  done
done

echo ""
echo "================================================================"
echo "V14 + MULTI-SEED TEACHER SWEEP DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# ─────────────────────────────────────────────────────────────────────
# [5/5] STUDENT RE-DISTILLATION + MULTI-SEED BS-A EVAL
# Trains a fresh student adapter against V14's z_env, then evaluates
# BS-A (student in the loop) across the 3 deploy dists × 3 seeds.
# Result: statistical confidence on "distilled student matches teacher."
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "[5/5] STUDENT RE-DISTILLATION + BS-A MULTI-SEED at $(date '+%H:%M:%S')"

STUDENT_DUMP=~/Desktop/safety-go2/dump_v14_for_student.npz
STUDENT_CKPT=~/Desktop/safety-go2/checkpoints/student_v14.pt
mkdir -p ~/Desktop/safety-go2/checkpoints

# 5a. Dump V14 teacher rollouts.
echo ""
echo "─── 5a. Dump V14 teacher rollouts ───"
./isaaclab.sh -p ~/Desktop/safety-go2/scripts/dump_teacher_rollout_for_student.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V14-v0 \
  --checkpoint "$CKPT" \
  --num_envs 256 --rollout_steps 1000 \
  --priv_dim 33 --priv_hidden_dim 14 \
  --output "$STUDENT_DUMP" --headless

# 5b. Train student offline (no Isaac needed, pure PyTorch).
echo ""
echo "─── 5b. Train student adapter ───"
~/miniconda3/envs/isaaclab/bin/python \
  ~/Desktop/safety-go2/scripts/train_student_v13.py \
  --dump "$STUDENT_DUMP" \
  --history_len 50 --batch_size 1024 --epochs 50 \
  --device cuda \
  --output "$STUDENT_CKPT"

# 5c. Multi-seed BS-A eval (BR vs BS-A head-to-head per seed).
echo ""
echo "─── 5c. Multi-seed BS-A eval (V14 student vs V14 teacher) ───"
declare -A TASK_V14_BS
TASK_V14_BS[trainmatch]="Isaac-CBF-Go2-RMA-Deploy-Realistic-AdaptiveC-TrainMatch-V14-v0"
TASK_V14_BS[ood]="Isaac-CBF-Go2-RMA-Deploy-Realistic-AdaptiveC-V14-v0"
TASK_V14_BS[stressor]="Isaac-CBF-Go2-RMA-Deploy-Realistic-AdaptiveC-Stressor-V14-v0"

run_bs() {
  local dist=$1; local seed=$2; local task=$3
  echo ""
  echo "─── BS-A / ${dist} / seed=${seed} ───"
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
    --task "$task" \
    --num_envs 64 --steps_per_config 1000 \
    --modes BR,BS-A \
    --alpha_grid "2.0" --phi_grid "0.5" \
    --epsilon0_grid "0.5" --lambda_grid "1.0" \
    --checkpoint "$CKPT" \
    --student_adapter_checkpoint "$STUDENT_CKPT" \
    --seed "$seed" \
    --output_dir "logs/bs_a_v14_${dist}_seed${seed}" --headless
}

for dist in trainmatch ood stressor; do
  for seed in 42 123 7; do
    run_bs "$dist" "$seed" "${TASK_V14_BS[$dist]}"
  done
done

echo ""
echo "================================================================"
echo "V14 + MULTI-SEED + STUDENT FULLY DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt:    $CKPT"
echo "student: $STUDENT_CKPT"
echo ""
echo "Headline checks vs V13.1 (single seed):"
echo "  trainmatch  V13.1 BR=0.861 (+7pp on 1 seed; revisit w/ multi-seed)"
echo "  OOD         V13.1 BR=0.797 (+4pp on 1 seed; revisit w/ multi-seed)"
echo "  STRESSOR    V13.1 BR=0.664 (lost 13pp — V14 wider DR should fix)"
echo ""
echo "Adaptive c diagnostic: check Pearson(c_param, δR) — non-trivial"
echo "  correlation would confirm c learned to track δR per episode."
echo ""
echo "Student multi-seed BS-A: aggregate logs/bs_a_v14_*/baseline.csv to"
echo "  see BS-A vs BR head-to-head with seed error bars."
echo "================================================================"
