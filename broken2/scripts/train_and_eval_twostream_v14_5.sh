#!/bin/bash
# V14.5 TRAIN + EVAL + ABLATION (2026-05-22).
#
# Pure-config staged predecessor to V15. Tests whether σ_act regime amplitude
# alone (not friction/push within-ep code) is what bottlenecks z_env utilization.
#
# Two deltas from V13.1:
#   actuation_noise_sigma_max  : 0.20 → 0.40
#   actuation_noise_curriculum : True → False
# dr_window_sigma_act stays True (already inherited).
#
# Success criterion: Δᾱ from CBF_ABLATE_Z_ENV=mean ≥ 0.20 (vs V13.1's ~0.04)
# AND physical composite within std of best-fixed on ≥3 of 4 dists.
#
# Sequence:
#   [0] Pre-flight locomotion sanity at σ_act=0.40 (V13.1 BR ckpt, short rollout)
#   [1] PPO training (2500 iters, ~5-6 hr on RTX 5090)
#   [2] Diagnostics (phi_corr, alpha_corr, probe_z_linear, grad_sensitivity)
#   [3] 4-eval (indist / trainmatch / ood / stressor)
#   [4] 3-axis ablation: z_env / proprio / z_grid muting (BR mode only)
#
# Usage on lab box:
#   tmux new -s wk3v14_5
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   CBF_ITERATIONS=2500 ~/Desktop/safety-go2/scripts/train_and_eval_twostream_v14_5.sh \
#     2>&1 | tee logs/train_and_eval_wk3v14_5.log

set -e
cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-2500}
ENVS=${CBF_NUM_ENVS:-4096}
V13_1_CKPT=${V13_1_CKPT:-/home/chrisliang/Desktop/safety-go2/IsaacLab/logs/rsl_rl/cbf_go2_teacher_rma/2026-05-20_18-25-01/model_2499.pt}

echo "================================================================"
echo "V14.5 TRAIN + EVAL + ABLATION"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')  iters=${ITERATIONS}"
echo "================================================================"

# Registration check
grep -q "CbfGo2EnvCfg_LAYER3_TWOSTREAM_V14_5" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
  && echo "  ✓ V14.5 train cfg present" \
  || { echo "  ✗ V14.5 train cfg missing"; exit 1; }
grep -q "Isaac-CBF-Go2-RMA-Layer3-TwoStream-V14_5-v0" \
  source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
  && echo "  ✓ V14.5 train task registered" \
  || { echo "  ✗ V14.5 train task not registered"; exit 1; }

# ─────────────────────────────────────────────────────────────────────
# [0/5] PRE-FLIGHT LOCOMOTION SANITY @ σ_act=0.40
# Run V13.1 BR ckpt on V14.5 trainmatch task. If locomotion survives
# (mean_v_xy reasonable, fall_rate < 10%), σ_act=0.40 is feasible.
# If not, abort and drop V14.5's σ_act_max to 0.30 before re-running.
# ─────────────────────────────────────────────────────────────────────

if [ -f "$V13_1_CKPT" ]; then
  echo ""
  echo "[0/5] PRE-FLIGHT SANITY @ $(date '+%H:%M:%S')"
  echo "      V13.1 BR ckpt on V14.5 trainmatch task (σ_act ∈ [0, 0.40])"
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
    --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V14_5-v0 \
    --num_envs 64 --steps_per_config 300 \
    --modes BR \
    --alpha_grid "2.0" --phi_grid "0.5" \
    --epsilon0_grid "0.5" --lambda_grid "1.0" \
    --checkpoint "$V13_1_CKPT" \
    --output_dir "logs/sanity_v14_5_pre_flight" --headless

  # Quick auto-check: if BR fall_rate > 0.10, the V14.5 σ_act range is too wide
  # for the frozen locomotion controller. Abort and let the user drop to 0.30.
  fall_rate=$(awk -F',' 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next}
    {if($h["mode"]=="BR") print $h["fall_rate"]}' \
    logs/sanity_v14_5_pre_flight/baseline.csv)
  echo "      fall_rate at σ_act=0.40: $fall_rate"
  pass=$(awk "BEGIN{print ($fall_rate <= 0.10) ? 1 : 0}")
  if [ "$pass" -ne "1" ]; then
    echo ""
    echo "  ✗ SANITY FAILED: fall_rate $fall_rate > 0.10"
    echo "    σ_act=0.40 destabilizes the frozen locomotion controller."
    echo "    Recommended action: edit cbf_go2_env_cfg.py line ~4396 and the"
    echo "    three deploy variants below to set actuation_noise_sigma_max=0.30,"
    echo "    then re-run this script."
    exit 1
  fi
  echo "  ✓ Sanity passed (fall_rate $fall_rate ≤ 0.10). Proceeding to training."
else
  echo "  ⚠ V13.1 ckpt not found at $V13_1_CKPT — skipping pre-flight sanity."
fi

export CBF_AUX_COEF=0.0
unset CBF_PRETRAINED_ENCODER CBF_FREEZE_ENCODER CBF_PRETRAINED_PRIV CBF_FREEZE_PRIV
unset CBF_ABLATE_Z_ENV CBF_ABLATE_PROPRIO CBF_ABLATE_Z_GRID

# ─────────────────────────────────────────────────────────────────────
# [1/5] PPO TRAINING
# ─────────────────────────────────────────────────────────────────────

echo ""
echo "[1/5] PPO TRAINING at $(date '+%H:%M:%S')"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V14_5-v0 \
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

# ─────────────────────────────────────────────────────────────────────
# [2/5] DIAGNOSTICS (phi_corr, alpha_corr, probe_z_linear, grad_sensitivity)
# Reminder: σ_act diagnostic correlations are still curriculum-stunted in
# the sense that training-task common_step_counter starts at 0 for each
# diagnostic run. But V14.5 has curriculum OFF, so σ_act is sampled from
# U(0, 0.40) regardless. Should NOT be stunted this time.
# ─────────────────────────────────────────────────────────────────────

echo ""
echo "[2/5] DIAGNOSTICS at $(date '+%H:%M:%S')"
export CBF_AUX_COEF=0.0

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V14_5-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --priv_layout v13 \
  --output diagnose_phi_corr_wk3v14_5.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V14_5-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --priv_dim 33 --priv_layout v13 \
  --output diagnose_alpha_corr_wk3v14_5.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V14_5-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 100 \
  --output probe_z_linear_wk3v14_5.json --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_grad_sensitivity.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V14_5-v0 \
  --checkpoint "$CKPT" --num_envs 256 --rollout_steps 50 --priv_dim 33 \
  --priv_layout v13 \
  --alpha_min 0.5 --alpha_max 3.0 \
  --output diagnose_grad_sensitivity_wk3v14_5.json --headless

# ─────────────────────────────────────────────────────────────────────
# [3/5] 4-EVAL
# ─────────────────────────────────────────────────────────────────────

echo ""
echo "[3/5] 4-EVAL at $(date '+%H:%M:%S')"

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Layer3-TwoStream-V14_5-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v14_5_indist" --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V14_5-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v14_5_trainmatch" --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-V14_5-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v14_5_ood" --headless

./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
  --task Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-Stressor-V14_5-v0 \
  --num_envs 64 --steps_per_config 1000 \
  --modes B0,B1,B2,BR \
  --alpha_grid "0.5,2.0,3.0" --phi_grid "0.5,2.0" \
  --epsilon0_grid "0.5" --lambda_grid "1.0,3.0" \
  --checkpoint "$CKPT" \
  --output_dir "logs/baseline_eval_wk3v14_5_stressor" --headless

# ─────────────────────────────────────────────────────────────────────
# [4/5] 3-AXIS ABLATION (the actual V14.5 success criterion)
# Δᾱ_{z_env_mute} ≥ 0.20 → V14.5 passes (vs V13.1's ~0.04)
# ─────────────────────────────────────────────────────────────────────

echo ""
echo "[4/5] 3-AXIS ABLATION at $(date '+%H:%M:%S')"

declare -A TASKS
TASKS[indist]="Isaac-CBF-Go2-RMA-Layer3-TwoStream-V14_5-v0"
TASKS[trainmatch]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-TrainMatch-V14_5-v0"
TASKS[ood]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-V14_5-v0"
TASKS[stressor]="Isaac-CBF-Go2-RMA-Deploy-Realistic-FrozenAC-Stressor-V14_5-v0"

run_ablate() {
  local tag=$1; local var=$2; local dist=$3
  echo ""
  echo "─── ablate=${tag} ${dist} ─── $(date '+%H:%M:%S')"
  unset CBF_ABLATE_Z_ENV CBF_ABLATE_PROPRIO CBF_ABLATE_Z_GRID
  export "$var=mean"
  ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
    --task "${TASKS[$dist]}" \
    --num_envs 64 --steps_per_config 1000 \
    --modes BR \
    --alpha_grid "2.0" --phi_grid "0.5" \
    --epsilon0_grid "0.5" --lambda_grid "1.0" \
    --checkpoint "$CKPT" \
    --output_dir "logs/ablate_${tag}_v14_5_${dist}" --headless
}

for dist in indist trainmatch ood stressor; do
  run_ablate z_env   CBF_ABLATE_Z_ENV   "$dist"
  run_ablate proprio CBF_ABLATE_PROPRIO "$dist"
  run_ablate z_grid  CBF_ABLATE_Z_GRID  "$dist"
done

unset CBF_ABLATE_Z_ENV CBF_ABLATE_PROPRIO CBF_ABLATE_Z_GRID

# ─────────────────────────────────────────────────────────────────────
# [5/5] DONE
# ─────────────────────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo "V14.5 DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo "ckpt: $CKPT"
echo ""
echo "To check V14.5 success criterion (Δᾱ_{z_env_mute} ≥ 0.20):"
echo "  for d in indist trainmatch ood stressor; do"
echo "    awk -F',' 'NR==1{for(i=1;i<=NF;i++)h[\$i]=i;next} \$h[\"mode\"]==\"BR\"{print \$h[\"avg_cbf_alpha_mean\"]}' \\"
echo "      logs/baseline_eval_wk3v14_5_\$d/baseline.csv logs/ablate_z_env_v14_5_\$d/baseline.csv"
echo "  done"
echo ""
echo "If V14.5 hits the bar: σ_act was the lever, skip V15 friction/push code."
echo "If V14.5 flat: V15 = V14.5 + ~25 lines new env code for friction/push within-ep flips."
echo "================================================================"
