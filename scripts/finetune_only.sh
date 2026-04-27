#!/bin/bash
# =============================================================================
# Phase 2 only — re-run finetune using finetune.py (NOT train.py)
# Phase 1 (focal pretrain) already complete; checkpoints in experiments/effv2s_focal_fold{0..4}
# =============================================================================

set -euo pipefail

FOLDS="0 1 2 3 4"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Sanity check: clean up the broken Phase 2 outputs from the previous (wrong) run
echo "Pre-flight: removing stale finetune dirs from broken run..."
for f in $FOLDS; do
    if [ -d "experiments/effv2s_finetune_fold${f}" ]; then
        echo "  removing experiments/effv2s_finetune_fold${f}"
        rm -rf "experiments/effv2s_finetune_fold${f}"
    fi
done

# Verify Phase 1 checkpoints all exist before launching
echo ""
echo "Pre-flight: verifying Phase 1 checkpoints..."
for f in $FOLDS; do
    ckpt="experiments/effv2s_focal_fold${f}/best_model.pt"
    if [ ! -f "$ckpt" ]; then
        echo "  MISSING: $ckpt"
        echo "  ABORT — Phase 1 checkpoint missing for fold $f"
        exit 1
    fi
    echo "  OK: $ckpt"
done

echo ""
echo "===================================================="
echo "PHASE 2: Soundscape finetune via finetune.py"
echo "Started at: $(date)"
echo "===================================================="

for f in $FOLDS; do
    ckpt="experiments/effv2s_focal_fold${f}/best_model.pt"

    echo ""
    echo "--- Fold $f finetune  [$(date +%H:%M:%S)] ---"
    echo "    init_checkpoint: $ckpt"

    BIRDCLEF_RUN_NAME="effv2s_finetune_fold${f}" \
        python -m src.training.finetune \
            configs/finetune_effv2s.yaml \
            --fold "$f" \
            --init-checkpoint "$ckpt" \
            2>&1 | tee "${LOG_DIR}/effv2s_finetune_fold${f}.log"
done

echo ""
echo "===================================================="
echo "Phase 2 complete at: $(date)"
echo "Per-fold best combined val_auc:"
echo "===================================================="
for f in $FOLDS; do
    log_csv="experiments/effv2s_finetune_fold${f}/training_log.csv"
    if [ -f "$log_csv" ]; then
        # finetune.py CSV columns:
        # epoch,train_loss,focal_val_loss,focal_val_auc,soundscape_val_loss,
        # soundscape_val_auc,combined_val_loss,combined_val_auc,lr,time_s
        # combined_val_auc is column 8
        best_comb=$(awk -F',' 'NR>1 {print $8}' "$log_csv" | sort -rn | head -1)
        best_sc=$(awk -F',' 'NR>1 {print $6}' "$log_csv" | sort -rn | head -1)
        best_focal=$(awk -F',' 'NR>1 {print $4}' "$log_csv" | sort -rn | head -1)
        echo "  fold ${f}: combined=${best_comb}  soundscape=${best_sc}  focal=${best_focal}"
    else
        echo "  fold ${f}: NO LOG FOUND"
    fi
done

echo ""
echo "Compare to your B0 v2 baseline:"
echo "  fold 0: 0.956   fold 1: 0.957   fold 2: 0.962   fold 3: 0.958   fold 4: 0.964"
echo ""
echo "If V2-S combined AUCs are within ~0.01 of B0 v2: ensemble partner ready."
echo "If V2-S is significantly weaker but ABOVE 0.93: still useful in ensemble."
echo "If V2-S is below 0.90: something else wrong, investigate before ensembling."
