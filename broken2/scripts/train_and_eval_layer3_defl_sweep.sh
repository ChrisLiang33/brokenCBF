#!/bin/bash
# L2 deflection penalty sweep — sequential overnight run (2026-05-17).
#
# Wk3 OMNI ceiling diagnostic revealed that both the locked-best and the
# omniscient teachers sit at the same high-deflection basin
# (deflection ≈ 0.73-0.79, α saturated near upper bound). The locked-best
# survives only because the perception cushion makes big QP corrections
# fire at safe true-distances; the OMNI teacher's identical regime causes
# a fall_rate explosion (0.045 → 0.441) once the cushion vanishes.
#
# This sweep tests whether an L2 tax on ‖u_safe − u_des‖ walks α off
# saturation in both regimes:
#
#   1. LAYER3_PUSH_A_C_DEFL (locked-best base + deflection penalty)
#      Goal: push joint_actual above the 0.724 locked-best ceiling.
#      Natural distillation teacher (perception channel matches
#      Mid-360 + cluster-fit at deploy).
#
#   2. LAYER3_PUSH_A_C_OMNI_DEFL (omniscient base + deflection penalty)
#      Goal: validate the mechanism — does the penalty actually fix
#      the OMNI panic-braking falls? Diagnostic only; not the
#      distillation teacher.
#
# Both use weight = -0.01 on cbf_deflection_l2, calibrated to peer with
# the existing cbf_a_l1 tax. If by ~iter 800 α has not budged off the
# saturated ~4.0 basin in either run, the weight needs doubling.
#
# Sync before launch:
#   bash ~/Desktop/safety-go2/scripts/sync_to_lab.sh
#
# Usage on lab box (in tmux):
#   tmux new -s wk3defl
#   cd ~/Desktop/safety-go2/IsaacLab && \
#   ~/Desktop/safety-go2/scripts/train_and_eval_layer3_defl_sweep.sh \
#     2>&1 | tee logs/train_and_eval_wk3defl.log

set -e

cd ~/Desktop/safety-go2/IsaacLab

ITERATIONS=${CBF_ITERATIONS:-1500}
ENVS=${CBF_NUM_ENVS:-4096}

run_one () {
    # Args: 1=label (e.g. "defl"), 2=task id, 3=cfg class name
    local label="$1"
    local task="$2"
    local cfg="$3"

    echo ""
    echo "================================================================"
    echo "[${label}] starting at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  task: ${task}"
    echo "  cfg:  ${cfg}"
    echo "================================================================"

    echo ""
    echo "Pre-flight checks"
    grep -q "${cfg}" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/cbf_go2_env_cfg.py \
      && echo "  ✓ ${cfg} config present" \
      || { echo "  ✗ config missing"; exit 1; }
    grep -q "${task}" source/isaaclab_tasks/isaaclab_tasks/manager_based/safety/cbf_go2/__init__.py \
      && echo "  ✓ task registered" \
      || { echo "  ✗ task not registered"; exit 1; }

    # ---------- PHASE 1: PPO TRAINING ----------
    echo ""
    echo "[${label} 1/3] PPO TRAINING: ${ITERATIONS} iters, ${ENVS} envs"
    echo "      Started at $(date '+%H:%M:%S')"

    export CBF_AUX_COEF=0.0
    unset CBF_PRETRAINED_ENCODER
    unset CBF_FREEZE_ENCODER
    unset CBF_PRETRAINED_PRIV
    unset CBF_FREEZE_PRIV

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task "${task}" \
      --num_envs "${ENVS}" --max_iterations "${ITERATIONS}" \
      --headless

    unset CBF_AUX_COEF

    echo ""
    echo "[${label} 1/3] PPO TRAINING done at $(date '+%H:%M:%S')"

    local latest_dir
    latest_dir=$(ls -1td logs/rsl_rl/cbf_go2_teacher_rma/*/ | head -1)
    latest_dir=${latest_dir%/}
    local ckpt
    ckpt=$(ls -1 "${latest_dir}"/model_*.pt 2>/dev/null \
        | awk -F'model_|\\.pt' '{print $2, $0}' \
        | sort -n | awk '{print $2}' | tail -1)
    if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
        echo "ERROR [${label}]: no checkpoint found in ${latest_dir}"
        exit 1
    fi
    echo "Using checkpoint: $ckpt"

    # ---------- PHASE 2: DIAGNOSTICS ----------
    echo ""
    echo "[${label} 2/3] DIAGNOSTICS at $(date '+%H:%M:%S')"

    export CBF_AUX_COEF=0.0

    ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_phi_corr.py \
      --task "${task}" \
      --checkpoint "$ckpt" \
      --num_envs 256 --rollout_steps 100 \
      --priv_dim 31 --use_locked \
      --output "diagnose_phi_corr_wk3${label}.json" --headless

    ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_alpha_corr.py \
      --task "${task}" \
      --checkpoint "$ckpt" \
      --num_envs 256 --rollout_steps 100 \
      --priv_dim 31 \
      --output "diagnose_alpha_corr_wk3${label}.json" --headless

    ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_a_corr.py \
      --task "${task}" \
      --checkpoint "$ckpt" \
      --num_envs 256 --rollout_steps 100 \
      --priv_dim 31 \
      --output "diagnose_a_corr_wk3${label}.json" --headless

    ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/diagnose_c_corr.py \
      --task "${task}" \
      --checkpoint "$ckpt" \
      --num_envs 256 --rollout_steps 100 \
      --priv_dim 31 --c_lo -0.20 --c_hi 0.20 \
      --output "diagnose_c_corr_wk3${label}.json" --headless

    ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/probe_z_linear.py \
      --task "${task}" \
      --checkpoint "$ckpt" \
      --num_envs 256 --rollout_steps 100 \
      --output "probe_z_linear_wk3${label}.json" --headless

    # ---------- PHASE 3: HEADLINE EVAL (with collision split) ----------
    echo ""
    echo "[${label} 3/3] HEADLINE EVAL at $(date '+%H:%M:%S')"

    local out_dir="logs/baseline_eval_wk3${label}_indist"
    ./isaaclab.sh -p ~/Desktop/safety-go2/scripts/eval_baseline.py \
      --task "${task}" \
      --num_envs 64 --steps_per_config 1000 \
      --modes B0,B1,B2,BR \
      --alpha_grid "0.5,2.0,4.0" \
      --phi_grid "0.5,2.0" \
      --epsilon0_grid "0.5" \
      --lambda_grid "1.0,3.0" \
      --checkpoint "$ckpt" \
      --output_dir "$out_dir" --headless

    unset CBF_AUX_COEF
    echo ""
    echo "[${label} 3/3] HEADLINE EVAL done at $(date '+%H:%M:%S')"
    echo "  ckpt: $ckpt"
    echo "  csv:  $out_dir/baseline.csv"
}

echo "================================================================"
echo "L2 deflection penalty sweep: locked-best then omniscient"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

run_one "defl"     "Isaac-CBF-Go2-RMA-Layer3-Push-A-C-Defl-v0"      "CbfGo2EnvCfg_LAYER3_PUSH_A_C_DEFL"
run_one "omnidefl" "Isaac-CBF-Go2-RMA-Layer3-Push-A-C-Omni-Defl-v0" "CbfGo2EnvCfg_LAYER3_PUSH_A_C_OMNI_DEFL"

echo ""
echo "================================================================"
echo "SWEEP DONE at $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Compare against ceilings (joint_actual):"
echo "  Locked best (PUSH_A_C):       0.724  (+1.6 pp vs best fixed)"
echo "  OMNI (regressed):             0.503  (mechanism diagnostic)"
echo ""
echo "Eval CSVs:"
echo "  logs/baseline_eval_wk3defl_indist/baseline.csv      (locked-best + defl)"
echo "  logs/baseline_eval_wk3omnidefl_indist/baseline.csv  (omni + defl)"
echo ""
echo "Gates (locked-best + defl):"
echo "  joint_actual > 0.724    (new ceiling)"
echo "  α population mean < 4.0 (walks off saturation)"
echo "  col_actual < 0.05       (no under-deflection)"
echo "  fall_rate < 0.05"
echo ""
echo "Gates (omni + defl, mechanism validation):"
echo "  fall_rate < 0.20        (panic-braking pathology fixed)"
echo "  col_actual < 0.10"
echo "  joint_actual > 0.50"
echo "================================================================"
