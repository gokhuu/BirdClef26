#!/bin/bash
# =============================================================================
# SEResNeXt26d — Phase 2 seed ensemble.
# Runs additional finetune passes with different seeds, reusing existing
# Phase 1 (focal pretrain) checkpoints. Does NOT touch your original
# seed=42 results — new runs go to differently-named directories.
#
# Prerequisites:
#   - Phase 1 checkpoints exist at experiments/seresnext_focal_fold{0..4}/best_model.pt
#   - Original seed=42 Phase 2 already done at experiments/seresnext_finetune_fold{0..4}
#   - finetune.py supports the --seed CLI flag (apply same edits as train.py)
#
# Usage: bash scripts/train_seresnext_seed_ensemble.sh
#
# Output dirs:
#   experiments/seresnext_finetune_seed${SEED}_fold${F}/
#
# Cost estimate: ~1.4h per seed (5 folds of finetune).
# =============================================================================

set -euo pipefail

FOLDS="0 1 2 3 4"
SEEDS="123 2024"   # Add 1-2 seeds; original run was seed=42
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# -----------------------------------------------------------------------------
# Pre-flight: verify Phase 1 checkpoints exist (do NOT delete anything)
# -----------------------------------------------------------------------------
echo "Pre-flight: verifying Phase 1 checkpoints..."
missing=0
for f in $FOLDS; do
    ckpt="experiments/seresnext_focal_fold${f}/best_model.pt"
    if [ ! -f "$ckpt" ]; then
        echo "  MISSING: $ckpt"
        missing=1
    else
        echo "  ok: $ckpt"
    fi
done

if [ "$missing" = "1" ]; then
    echo ""
    echo "ABORT: one or more Phase 1 checkpoints missing."
    echo "Run train_seresnext_all.sh first to produce focal pretrain checkpoints."
    exit 1
fi

# Verify finetune config exists
if [ ! -f "configs/finetune_seresnext.yaml" ]; then
    echo "ABORT: missing configs/finetune_seresnext.yaml"
    exit 1
fi

echo "All Phase 1 checkpoints present. Starting seed ensemble..."

# -----------------------------------------------------------------------------
# Run Phase 2 finetune for each (seed, fold) combination
# -----------------------------------------------------------------------------
for SEED in $SEEDS; do
    echo ""
    echo "===================================================="
    echo "SEED $SEED — Phase 2 finetune across all folds"
    echo "Started at: $(date)"
    echo "===================================================="

    for f in $FOLDS; do
        run_name="seresnext_finetune_seed${SEED}_fold${f}"
        exp_dir="experiments/${run_name}"

        # Skip if already done (allows resume after interruption)
        if [ -d "$exp_dir" ] && [ -f "${exp_dir}/best_model.pt" ]; then
            echo "  skipping ${run_name} (already trained)"
            continue
        fi

        ckpt="experiments/seresnext_focal_fold${f}/best_model.pt"
        echo ""
        echo "--- seed=${SEED} fold=${f} finetune  [$(date +%H:%M:%S)] ---"
        BIRDCLEF_RUN_NAME="$run_name" \
            python -m src.training.finetune \
                configs/finetune_seresnext.yaml \
                --fold "$f" \
                --seed "$SEED" \
                --init-checkpoint "$ckpt" \
                2>&1 | tee "${LOG_DIR}/${run_name}.log"
    done
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "===================================================="
echo "Seed ensemble complete at: $(date)"
echo "===================================================="
echo ""
echo "Results — original (seed=42):"
for f in $FOLDS; do
    log_csv="experiments/seresnext_finetune_fold${f}/training_log.csv"
    if [ -f "$log_csv" ]; then
        # finetune.py columns:
        # epoch,train_loss,focal_val_loss,focal_val_auc,soundscape_val_loss,
        # soundscape_val_auc,combined_val_loss,combined_val_auc,lr,time_s
        best_comb=$(awk -F',' 'NR>1 {print $8}' "$log_csv" | sort -rn | head -1)
        echo "  seed=42  fold=${f}: combined=${best_comb}"
    fi
done

for SEED in $SEEDS; do
    echo ""
    echo "Results — seed=${SEED}:"
    for f in $FOLDS; do
        log_csv="experiments/seresnext_finetune_seed${SEED}_fold${f}/training_log.csv"
        if [ -f "$log_csv" ]; then
            best_comb=$(awk -F',' 'NR>1 {print $8}' "$log_csv" | sort -rn | head -1)
            echo "  seed=${SEED}  fold=${f}: combined=${best_comb}"
        else
            echo "  seed=${SEED}  fold=${f}: NO LOG"
        fi
    done
done

echo ""
echo "===================================================="
echo "Next steps:"
echo "  1. Update your inference notebook to load and average across:"
echo "       experiments/seresnext_finetune_fold{0..4}/best_model.pt   (seed=42)"
for SEED in $SEEDS; do
    echo "       experiments/seresnext_finetune_seed${SEED}_fold{0..4}/best_model.pt"
done
echo "     Average the SIGMOID PROBABILITIES, not raw logits or hard labels."
echo "  2. Verify locally that the ensemble OOF score beats your single-seed"
echo "     OOF score before burning a Kaggle submission."
echo "  3. Submit. Expected lift: 0.002-0.005 on top of existing 5-fold ensemble."
echo "===================================================="
