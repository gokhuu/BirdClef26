#!/bin/bash
# =============================================================================
# SEResNeXt26d — full training pipeline.
# Smoke test (Phase 0) + focal pretrain (Phase 1) + finetune (Phase 2).
# Designed to run unattended overnight (~3.5h total).
#
# Usage: bash scripts/train_seresnext_all.sh
#
# After this completes successfully, proceed to:
#   bash scripts/export_seresnext_onnx.sh
#   then upload ONNX to a NEW Kaggle dataset (don't overwrite B0!)
# =============================================================================

set -euo pipefail

FOLDS="0 1 2 3 4"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Pre-flight: clean up any leftover smoke / partial dirs
echo "Pre-flight: cleaning stale SEResNeXt dirs..."
for d in experiments/seresnext_smoke_fold0 \
         experiments/seresnext_focal_fold{0,1,2,3,4} \
         experiments/seresnext_finetune_fold{0,1,2,3,4}; do
    if [ -d "$d" ]; then
        echo "  removing $d"
        rm -rf "$d"
    fi
done

# Sanity: verify configs exist
for cfg in configs/smoke_seresnext.yaml \
           configs/focal_pretrain_seresnext.yaml \
           configs/finetune_seresnext.yaml; do
    if [ ! -f "$cfg" ]; then
        echo "ABORT: missing config $cfg"
        exit 1
    fi
done

# =============================================================================
# PHASE 0: Smoke test (~3 min)
# Aborts the whole pipeline if SEResNeXt fails to learn focal data.
# =============================================================================
echo ""
echo "===================================================="
echo "PHASE 0: Smoke test"
echo "Started at: $(date)"
echo "===================================================="
python -m src.training.train configs/smoke_seresnext.yaml --fold 0 \
    2>&1 | tee "${LOG_DIR}/seresnext_smoke.log"

# Check the smoke result
smoke_csv="experiments/seresnext_smoke_fold0/training_log.csv"
if [ -f "$smoke_csv" ]; then
    smoke_auc=$(awk -F',' 'NR==3 {print $4}' "$smoke_csv")
    echo ""
    echo "Smoke test epoch-2 val_auc: $smoke_auc"

    # Compare to 0.65 threshold using awk (bash has no float compare)
    bad=$(awk -v auc="$smoke_auc" 'BEGIN { print (auc < 0.65) ? 1 : 0 }')
    if [ "$bad" = "1" ]; then
        echo ""
        echo "!!! ABORT: smoke val_auc $smoke_auc < 0.65."
        echo "!!! SEResNeXt failing same way V2-S did. Don't run Phase 1."
        echo "!!! Check ${LOG_DIR}/seresnext_smoke.log for diagnostics."
        exit 1
    fi
    echo "Smoke healthy ($smoke_auc >= 0.65). Proceeding to Phase 1."
else
    echo "!!! ABORT: smoke training_log.csv missing — smoke training failed."
    exit 1
fi

# =============================================================================
# PHASE 1: Focal pretrain, 5 folds (~2.3h)
# =============================================================================
echo ""
echo "===================================================="
echo "PHASE 1: Focal pretrain, 5 folds, ~2.3h total"
echo "Started at: $(date)"
echo "===================================================="
for f in $FOLDS; do
    echo ""
    echo "--- Fold $f focal pretrain  [$(date +%H:%M:%S)] ---"
    BIRDCLEF_RUN_NAME="seresnext_focal_fold${f}" \
        python -m src.training.train \
            configs/focal_pretrain_seresnext.yaml \
            --fold "$f" \
            2>&1 | tee "${LOG_DIR}/seresnext_focal_fold${f}.log"
done

echo ""
echo "===================================================="
echo "Phase 1 complete at: $(date)"
echo "Per-fold best focal val_auc:"
echo "===================================================="
for f in $FOLDS; do
    log_csv="experiments/seresnext_focal_fold${f}/training_log.csv"
    if [ -f "$log_csv" ]; then
        best=$(awk -F',' 'NR>1 {print $4}' "$log_csv" | sort -rn | head -1)
        echo "  fold ${f}: best val_auc = ${best}"
    else
        echo "  fold ${f}: NO LOG FOUND"
    fi
done

# =============================================================================
# PHASE 2: Soundscape finetune, 5 folds (~1.4h)
# =============================================================================
echo ""
echo "===================================================="
echo "PHASE 2: Soundscape finetune, 5 folds, ~1.4h total"
echo "Started at: $(date)"
echo "===================================================="
for f in $FOLDS; do
    ckpt="experiments/seresnext_focal_fold${f}/best_model.pt"
    if [ ! -f "$ckpt" ]; then
        echo ""
        echo "!!! SKIPPING fold ${f} finetune — Phase 1 checkpoint missing: $ckpt"
        continue
    fi

    echo ""
    echo "--- Fold $f finetune (warmstart from $ckpt)  [$(date +%H:%M:%S)] ---"
    BIRDCLEF_RUN_NAME="seresnext_finetune_fold${f}" \
        python -m src.training.finetune \
            configs/finetune_seresnext.yaml \
            --fold "$f" \
            --init-checkpoint "$ckpt" \
            2>&1 | tee "${LOG_DIR}/seresnext_finetune_fold${f}.log"
done

echo ""
echo "===================================================="
echo "Phase 2 complete at: $(date)"
echo "Per-fold best combined val_auc:"
echo "===================================================="
for f in $FOLDS; do
    log_csv="experiments/seresnext_finetune_fold${f}/training_log.csv"
    if [ -f "$log_csv" ]; then
        # finetune.py columns:
        # epoch,train_loss,focal_val_loss,focal_val_auc,soundscape_val_loss,
        # soundscape_val_auc,combined_val_loss,combined_val_auc,lr,time_s
        best_comb=$(awk -F',' 'NR>1 {print $8}' "$log_csv" | sort -rn | head -1)
        best_sc=$(awk -F',' 'NR>1 {print $6}' "$log_csv" | sort -rn | head -1)
        best_focal=$(awk -F',' 'NR>1 {print $4}' "$log_csv" | sort -rn | head -1)
        echo "  fold ${f}: combined=${best_comb}  soundscape=${best_sc}  focal=${best_focal}"
    else
        echo "  fold ${f}: NO LOG FOUND"
    fi
done

echo ""
echo "===================================================="
echo "ALL TRAINING DONE at: $(date)"
echo "===================================================="
echo ""
echo "Compare to your B0 v2 baseline (combined val):"
echo "  fold 0: 0.956   fold 1: 0.957   fold 2: 0.962   fold 3: 0.958   fold 4: 0.964"
echo ""
echo "Compare to V2-S finetune (combined val):"
echo "  fold 0: 0.949   fold 1: 0.955   fold 2: 0.954   fold 3: 0.959   fold 4: 0.957"
echo ""
echo "If SEResNeXt is within ~0.01 of B0: solid ensemble partner candidate."
echo ""
echo "Next steps:"
echo "  1. Export ONNX:    bash scripts/export_seresnext_onnx.sh"
echo "  2. Benchmark CPU:  python scripts/benchmark_cpu_inference.py \\"
echo "                       --models experiments/seresnext_finetune_fold*/model_int8_fp32.onnx \\"
echo "                       --num_threads 4 --remaining_budget_min 35"
echo "  3. Upload to NEW Kaggle dataset (do NOT overwrite B0!)"
