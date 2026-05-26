#!/usr/bin/env bash
# Overnight pipeline: trains + evals 3 teachers vs baselines.
#
# Teachers:
#   T1 RMA-classic (4-priv visible)
#   T2 NoPriv      (4-priv slot, masked to zero)
#   T3 RMAHistory  (no priv, 50-step proprio history through 1D CNN)
#
# For each: train if missing, run input attribution.
# For T1 + T2: cross-scene eval on EvalNoPrivE1..E4 (shared 195-d obs).
#   - T1 will be evaluated with priv masked (small domain shift; the
#     attribution showed T1 barely used priv anyway).
#   - T2 evaluated as designed.
# T3 cross-scene eval is skipped (would need EvalHistory* scenes).
#
# Restartable: every step is skipped if its primary output already exists.
# All output streamed to a single log file + stdout.

set -u
trap 'echo "[$(date +%T)] aborted at line $LINENO"' ERR

ROOT=~/Desktop/cbf_rl_mvp/go2
LOCO=/home/chrisliang/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-05-23_08-47-44/model_299.pt
LOG=$ROOT/phase9_overnight_pipeline.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
cd "$ROOT"

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "  OVERNIGHT PIPELINE start: $(date)"                          | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# helper -------------------------------------------------------------
run_if_missing() {
  local name="$1"; local outfile="$2"; shift 2
  if [ -f "$outfile" ]; then
    echo "[$(date +%T)] SKIP   $name (exists: $outfile)" | tee -a "$LOG"
    return 0
  fi
  echo "[$(date +%T)] START  $name" | tee -a "$LOG"
  ( "$@" ) 2>&1 | tee -a "$LOG"
  if [ -f "$outfile" ]; then
    echo "[$(date +%T)] DONE   $name" | tee -a "$LOG"
  else
    echo "[$(date +%T)] FAIL   $name (output not produced; check log)" | tee -a "$LOG"
  fi
}

ISAACLAB=~/IsaacLab/isaaclab.sh

# ----- 1. Train T1 RMA-classic (skip if checkpoint exists) ----------
run_if_missing "train T1 RMA-classic" \
  "$ROOT/phase7_rma_static_teacher_outputs/rsl_rl/model_final.pt" \
  env PYTHONUNBUFFERED=1 "$ISAACLAB" -p phase5_train_teacher.py \
    --checkpoint "$LOCO" \
    --task Isaac-CBF-Adaptive-Go2-RMAStatic-v0 \
    --num_envs 2048 --max_iterations 3000 --diag_interval 500 \
    --out_dir phase7_rma_static_teacher_outputs --headless

# ----- 2. Train T2 NoPriv (skip if checkpoint exists) ---------------
run_if_missing "train T2 NoPriv" \
  "$ROOT/phase8_ablation_nopriv_outputs/rsl_rl/model_final.pt" \
  env PYTHONUNBUFFERED=1 "$ISAACLAB" -p phase5_train_teacher.py \
    --checkpoint "$LOCO" \
    --task Isaac-CBF-Adaptive-Go2-RMAStaticNoPriv-v0 \
    --num_envs 2048 --max_iterations 3000 --diag_interval 500 \
    --out_dir phase8_ablation_nopriv_outputs --headless

# ----- 3. Train T3 RMAHistory (new) ---------------------------------
run_if_missing "train T3 RMAHistory" \
  "$ROOT/phase9_rma_history_teacher_outputs/rsl_rl/model_final.pt" \
  env PYTHONUNBUFFERED=1 "$ISAACLAB" -p phase5_train_teacher.py \
    --checkpoint "$LOCO" \
    --task Isaac-CBF-Adaptive-Go2-RMAHistory-v0 \
    --num_envs 2048 --max_iterations 3000 --diag_interval 500 \
    --out_dir phase9_rma_history_teacher_outputs --headless

# ----- 4. Tune baselines on RMAStaticNoPriv env ---------------------
# Shared by T1 + T2 cross-scene evals. RMAStaticNoPriv is the right
# tuning env for the EvalNoPriv* scenes.
run_if_missing "tune baselines on RMAStaticNoPriv" \
  "$ROOT/phase9_nopriv_baselines_outputs/phase5_baselines_summary.json" \
  env PYTHONUNBUFFERED=1 "$ISAACLAB" -p phase5_baselines.py \
    --checkpoint "$LOCO" \
    --task Isaac-CBF-Adaptive-Go2-RMAStaticNoPriv-v0 \
    --num_envs 256 --eval_eps_per_cell 256 --eval_steps 1250 \
    --out_dir phase9_nopriv_baselines_outputs --headless

# ----- 5. Cross-scene eval T1 RMA-classic on EvalNoPrivE1..E4 -------
# Domain shift caveat: T1 was trained with priv visible; EvalNoPriv
# scenes mask priv. Attribution showed T1 barely used priv anyway,
# so the shift should be small.
run_if_missing "cross-scene eval T1 RMA-classic" \
  "$ROOT/phase9_t1_rma_classic_xscene_outputs/phase6_eval_scenes_summary.csv" \
  env PYTHONUNBUFFERED=1 "$ISAACLAB" -p phase6_eval_scenes.py \
    --teacher_ckpt "$ROOT/phase7_rma_static_teacher_outputs/rsl_rl/model_final.pt" \
    --locomotion_ckpt "$LOCO" \
    --baselines_summary "$ROOT/phase9_nopriv_baselines_outputs/phase5_baselines_summary.json" \
    --scene_prefix EvalNoPriv \
    --num_envs 256 --eval_eps_per_cell 256 --eval_steps 1250 \
    --out_dir phase9_t1_rma_classic_xscene_outputs --headless

# ----- 6. Cross-scene eval T2 NoPriv on EvalNoPrivE1..E4 ------------
run_if_missing "cross-scene eval T2 NoPriv" \
  "$ROOT/phase9_t2_nopriv_xscene_outputs/phase6_eval_scenes_summary.csv" \
  env PYTHONUNBUFFERED=1 "$ISAACLAB" -p phase6_eval_scenes.py \
    --teacher_ckpt "$ROOT/phase8_ablation_nopriv_outputs/rsl_rl/model_final.pt" \
    --locomotion_ckpt "$LOCO" \
    --baselines_summary "$ROOT/phase9_nopriv_baselines_outputs/phase5_baselines_summary.json" \
    --scene_prefix EvalNoPriv \
    --num_envs 256 --eval_eps_per_cell 256 --eval_steps 1250 \
    --out_dir phase9_t2_nopriv_xscene_outputs --headless

# ----- 7. Input attribution on each teacher -------------------------
# T3 (RMAHistory) uses the updated phase8 path that detects the
# history model architecture.
run_if_missing "attribution T1 RMA-classic" \
  "$ROOT/phase9_t1_attribution_outputs/input_attribution.json" \
  env PYTHONUNBUFFERED=1 "$ISAACLAB" -p phase8_input_attribution.py \
    --teacher_ckpt "$ROOT/phase7_rma_static_teacher_outputs/rsl_rl/model_final.pt" \
    --locomotion_ckpt "$LOCO" \
    --task Isaac-CBF-Adaptive-Go2-RMAStatic-v0 \
    --out_dir phase9_t1_attribution_outputs --headless

run_if_missing "attribution T2 NoPriv" \
  "$ROOT/phase9_t2_attribution_outputs/input_attribution.json" \
  env PYTHONUNBUFFERED=1 "$ISAACLAB" -p phase8_input_attribution.py \
    --teacher_ckpt "$ROOT/phase8_ablation_nopriv_outputs/rsl_rl/model_final.pt" \
    --locomotion_ckpt "$LOCO" \
    --task Isaac-CBF-Adaptive-Go2-RMAStaticNoPriv-v0 \
    --out_dir phase9_t2_attribution_outputs --headless

run_if_missing "attribution T3 RMAHistory" \
  "$ROOT/phase9_t3_attribution_outputs/input_attribution.json" \
  env PYTHONUNBUFFERED=1 "$ISAACLAB" -p phase8_input_attribution.py \
    --teacher_ckpt "$ROOT/phase9_rma_history_teacher_outputs/rsl_rl/model_final.pt" \
    --locomotion_ckpt "$LOCO" \
    --task Isaac-CBF-Adaptive-Go2-RMAHistory-v0 \
    --out_dir phase9_t3_attribution_outputs --headless

# ----- 8. Print summary --------------------------------------------
echo ""                                                          | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "  OVERNIGHT PIPELINE done: $(date)"                          | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo ""                                                          | tee -a "$LOG"
echo "Final checkpoints:"                                        | tee -a "$LOG"
for f in \
  "$ROOT/phase7_rma_static_teacher_outputs/rsl_rl/model_final.pt" \
  "$ROOT/phase8_ablation_nopriv_outputs/rsl_rl/model_final.pt" \
  "$ROOT/phase9_rma_history_teacher_outputs/rsl_rl/model_final.pt"
do
  if [ -f "$f" ]; then echo "  [OK]   $f"; else echo "  [MISS] $f"; fi
done | tee -a "$LOG"
echo ""                                                          | tee -a "$LOG"
echo "Eval summaries:"                                           | tee -a "$LOG"
for f in \
  "$ROOT/phase9_t1_rma_classic_xscene_outputs/phase6_eval_scenes_summary.csv" \
  "$ROOT/phase9_t2_nopriv_xscene_outputs/phase6_eval_scenes_summary.csv" \
  "$ROOT/phase9_t1_attribution_outputs/input_attribution.json" \
  "$ROOT/phase9_t2_attribution_outputs/input_attribution.json" \
  "$ROOT/phase9_t3_attribution_outputs/input_attribution.json"
do
  if [ -f "$f" ]; then echo "  [OK]   $f"; else echo "  [MISS] $f"; fi
done | tee -a "$LOG"
echo ""                                                          | tee -a "$LOG"
echo "Open log:  $LOG"                                           | tee -a "$LOG"
