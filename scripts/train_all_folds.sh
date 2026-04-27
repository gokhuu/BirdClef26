#!/bin/bash
# =============================================================================
# Orchestrate V2-S full 5-fold training.
# Smoke test (Phase 0) already completed — this script runs Phase 1 + Phase 2.
# =============================================================================
# Uses train.py CLI:  python -m src.training.train <config> [--fold N]
# Uses BIRDCLEF_RUN_NAME env var to name per-fold output directories.
# Uses sed to substitute {FOLD} in finetune config's init_checkpoint path.
# =============================================================================

set -euo pipefail

FOLDS="0 1 2 3 4"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# =============================================================================
# PHASE 1: Focal pretrain, 5 folds, ~4.7h total
# =============================================================================
echo "===================================================="
echo "PHASE 1: Focal pretrain, 5 folds, ~4.7h total"
echo "Started at: $(date)"
echo "===================================================="
for f in $FOLDS; do
    echo ""
    echo "--- Fold $f focal pretrain  [$(date +%H:%M:%S)] ---"
    BIRDCLEF_RUN_NAME="effv2s_focal_fold${f}" \
        python -m src.training.train \
            configs/focal_pretrain_effv2s.yaml \
            --fold "$f" \
            2>&1 | tee "${LOG_DIR}/effv2s_focal_fold${f}.log"
done

echo ""
echo "===================================================="
echo "Phase 1 complete at: $(date)"
echo "Per-fold best focal val_auc:"
echo "===================================================="
for f in $FOLDS; do
    log_csv="experiments/effv2s_focal_fold${f}/training_log.csv"
    if [ -f "$log_csv" ]; then
        # Column 4 is val_auc (epoch,train_loss,val_loss,val_auc,lr,time_s)
        best=$(awk -F',' 'NR>1 {print $4}' "$log_csv" | sort -rn | head -1)
        echo "  fold ${f}: best val_auc = ${best}"
    else
        echo "  fold ${f}: NO LOG FOUND (training may have failed)"
    fi
done

# =============================================================================
# PHASE 2: Soundscape finetune, 5 folds, ~2.4h total
# =============================================================================
echo ""
echo "===================================================="
echo "PHASE 2: Soundscape finetune, 5 folds, ~2.4h total"
echo "Started at: $(date)"
echo "===================================================="
mkdir -p configs/_generated
for f in $FOLDS; do
    # Substitute {FOLD} in init_checkpoint path per-fold
    gen_config="configs/_generated/finetune_effv2s_fold${f}.yaml"
    sed "s/{FOLD}/${f}/g" configs/finetune_effv2s.yaml > "$gen_config"

    # Guard: verify warmstart checkpoint exists before launching
    ckpt="experiments/effv2s_focal_fold${f}/best_model.pt"
    if [ ! -f "$ckpt" ]; then
        echo ""
        echo "!!! SKIPPING fold ${f} finetune — checkpoint missing: $ckpt"
        echo "!!! (Phase 1 for this fold likely failed. Check logs/effv2s_focal_fold${f}.log)"
        continue
    fi

    echo ""
    echo "--- Fold $f finetune (warmstart from $ckpt)  [$(date +%H:%M:%S)] ---"
    BIRDCLEF_RUN_NAME="effv2s_finetune_fold${f}" \
        python -m src.training.train \
            "$gen_config" \
            --fold "$f" \
            2>&1 | tee "${LOG_DIR}/effv2s_finetune_fold${f}.log"
done

echo ""
echo "===================================================="
echo "Phase 2 complete at: $(date)"
echo "Per-fold best finetune val_auc:"
echo "===================================================="
for f in $FOLDS; do
    log_csv="experiments/effv2s_finetune_fold${f}/training_log.csv"
    if [ -f "$log_csv" ]; then
        best=$(awk -F',' 'NR>1 {print $4}' "$log_csv" | sort -rn | head -1)
        echo "  fold ${f}: best val_auc = ${best}"
    else
        echo "  fold ${f}: NO LOG FOUND"
    fi
done

echo ""
echo "===================================================="
echo "All training done. Total finished at: $(date)"
echo "===================================================="
echo ""
echo "Next steps (manual, after reviewing AUCs above):"
echo "  1. Export ONNX + int8 (per fold)"
echo "  2. Benchmark CPU inference budget"
