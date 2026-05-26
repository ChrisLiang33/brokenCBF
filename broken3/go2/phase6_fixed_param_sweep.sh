#!/usr/bin/env bash
# Overnight Phase 6 fixed (phi, alpha) sweep.
#
# Runs the sweep at three disturbance levels (0, 15, 30 N) with a
# denser grid (7 phi x 5 alpha = 35 combos) and longer rollouts
# (600 steps x 256 envs) for tight error bars in close-distance bins.
#
# Expected wall time: ~5-7 hours total on RTX 5090.
#
# Each disturbance run writes to its own out_dir:
#   phase6_fixed_param_sweep_outputs_d00/
#   phase6_fixed_param_sweep_outputs_d15/
#   phase6_fixed_param_sweep_outputs_d30/
#
# Verdicts are printed to console AND saved in each out_dir's JSON.
# Master log at phase6_fixed_param_sweep_outputs_master.log.
#
# Usage on labbox:
#     cd ~/Desktop/cbf_rl_mvp/go2
#     bash phase6_fixed_param_sweep.sh 2>&1 | tee phase6_fixed_param_sweep.log
#
# Or to run truly detached (survives ssh disconnect):
#     nohup bash phase6_fixed_param_sweep.sh > phase6_fixed_param_sweep.log 2>&1 &
#     disown
#
set -euo pipefail

CKPT="/home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt"
TASK="Isaac-CBF-Adaptive-Go2-RandObs-v0"
NUM_ENVS=256
STEPS_PER_COMBO=600
PHI_GRID="0.0,0.15,0.3,0.5,0.7,0.85,1.0"
ALPHA_GRID="1.0,1.75,2.5,3.25,4.0"
COLL_THR=0.05

MASTER_LOG="phase6_fixed_param_sweep_outputs_master.log"
: > "${MASTER_LOG}"

echo "=========================================================================" | tee -a "${MASTER_LOG}"
echo "  PHASE 6 OVERNIGHT FIXED (phi, alpha) SWEEP" | tee -a "${MASTER_LOG}"
echo "  task:       ${TASK}" | tee -a "${MASTER_LOG}"
echo "  num_envs:   ${NUM_ENVS}" | tee -a "${MASTER_LOG}"
echo "  steps/combo:${STEPS_PER_COMBO}" | tee -a "${MASTER_LOG}"
echo "  phi grid:   ${PHI_GRID}" | tee -a "${MASTER_LOG}"
echo "  alpha grid: ${ALPHA_GRID}" | tee -a "${MASTER_LOG}"
echo "  coll_thr:   ${COLL_THR}" | tee -a "${MASTER_LOG}"
echo "  started:    $(date)" | tee -a "${MASTER_LOG}"
echo "=========================================================================" | tee -a "${MASTER_LOG}"

run_one () {
    local d="$1"
    local label="$2"
    local outdir="phase6_fixed_param_sweep_outputs_${label}"

    echo "" | tee -a "${MASTER_LOG}"
    echo "-------------------------------------------------------------------------" | tee -a "${MASTER_LOG}"
    echo "  RUN: disturbance=${d}N  ->  ${outdir}" | tee -a "${MASTER_LOG}"
    echo "  started: $(date)" | tee -a "${MASTER_LOG}"
    echo "-------------------------------------------------------------------------" | tee -a "${MASTER_LOG}"

    ~/IsaacLab/isaaclab.sh -p phase6_fixed_param_sweep.py \
        --checkpoint "${CKPT}" \
        --task "${TASK}" \
        --num_envs "${NUM_ENVS}" \
        --steps_per_combo "${STEPS_PER_COMBO}" \
        --disturbance "${d}" \
        --phi_grid "${PHI_GRID}" \
        --alpha_grid "${ALPHA_GRID}" \
        --coll_rate_thr "${COLL_THR}" \
        --out_dir "${outdir}" \
        --headless 2>&1 | tee -a "${MASTER_LOG}"

    echo "  finished: $(date)" | tee -a "${MASTER_LOG}"

    # echo just the verdict line for quick scan
    if [[ -f "${outdir}/phase6_fixed_param_sweep.json" ]]; then
        echo "  VERDICT (d=${d}N):" | tee -a "${MASTER_LOG}"
        python3 -c "
import json, sys
with open('${outdir}/phase6_fixed_param_sweep.json') as f:
    d = json.load(f)
print('   ', d['verdict'])
" | tee -a "${MASTER_LOG}"
    fi
}

# ---- runs ----
run_one 0   "d00"
run_one 15  "d15"
run_one 30  "d30"

# ---- final summary ----
echo "" | tee -a "${MASTER_LOG}"
echo "=========================================================================" | tee -a "${MASTER_LOG}"
echo "  ALL RUNS COMPLETE" | tee -a "${MASTER_LOG}"
echo "  finished: $(date)" | tee -a "${MASTER_LOG}"
echo "=========================================================================" | tee -a "${MASTER_LOG}"
for label in d00 d15 d30; do
    outdir="phase6_fixed_param_sweep_outputs_${label}"
    if [[ -f "${outdir}/phase6_fixed_param_sweep.json" ]]; then
        echo "" | tee -a "${MASTER_LOG}"
        echo "  ${outdir}:" | tee -a "${MASTER_LOG}"
        python3 -c "
import json, sys
with open('${outdir}/phase6_fixed_param_sweep.json') as f:
    d = json.load(f)
print('    verdict:', d['verdict'])
print('    per-bin best:')
for r in d['per_bin_best']:
    if r['phi'] is None:
        continue
    row = r['row']
    print(f\"      {r['bin_label']:>25}:  phi={r['phi']:.2f}  alpha={r['alpha']:.2f}  prog={row['mean_prog']:+.5f}  coll={100*row['coll_rate']:.2f}%\")
" | tee -a "${MASTER_LOG}"
    else
        echo "  ${outdir}: NO OUTPUT" | tee -a "${MASTER_LOG}"
    fi
done
echo "=========================================================================" | tee -a "${MASTER_LOG}"
