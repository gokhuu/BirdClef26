#!/bin/bash
# =============================================================================
# B0 SED pseudo_v2 multi-seed finetune.
# Reuses existing focal pretrain checkpoints at experiments/sed_b0_fold{0..4}/.
# Original seed=42 already exists at experiments/sed_finetune_pseudo_v2_fold{0..4}/.
# This adds seeds 123 and 2024.
#
# Output dirs: experiments/sed_finetune_pseudo_v2_seed${SEED}_fold${F}/
# Total cost:  ~3.4h (2 seeds × 5 folds × 12 epochs × ~17 min/run)
# =============================================================================

set -euo pipefail

FOLDS="0 1 2 3 4"
SEEDS="123 2024"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Pre-flight: verify B0 focal pretrain checkpoints exist
echo "Pre-flight: verifying B0 focal pretrain checkpoints..."
missing=0
for f in $FOLDS; do
    ckpt="experiments/sed_b0_fold${f}/best_model.pt"
    if [ ! -f "$ckpt" ]; then
        echo "  MISSING: $ckpt"
        missing=1
    else
        echo "  ok: $ckpt"
    fi
done
if [ "$missing" = "1" ]; then
    echo "ABORT: missing B0 focal pretrain checkpoints."
    exit 1
fi

# Pre-flight: verify config and pseudo-labels CSV
if [ ! -f "configs/finetune_pseudo_v2.yaml" ]; then
    echo "ABORT: missing configs/finetune_pseudo_v2.yaml"
    exit 1
fi
if [ ! -f "data/pseudo/sed_b0_v2_pseudo_balanced.csv" ]; then
    echo "ABORT: missing pseudo-labels CSV. The 0.882 recipe depends on this."
    echo "       File expected at: data/pseudo/sed_b0_v2_pseudo_balanced.csv"
    exit 1
fi

echo "All prerequisites present. Starting multi-seed run..."

# -----------------------------------------------------------------------------
# Per-seed Phase 2 finetune
# -----------------------------------------------------------------------------
for SEED in $SEEDS; do
    echo ""
    echo "===================================================="
    echo "SEED $SEED — Phase 2 finetune across all folds"
    echo "Started at: $(date)"
    echo "===================================================="

    for f in $FOLDS; do
        run_name="sed_finetune_pseudo_v2_seed${SEED}_fold${f}"
        exp_dir="experiments/${run_name}"

        # Resume-friendly: skip if already trained successfully
        if [ -d "$exp_dir" ] && [ -f "${exp_dir}/best_model.pt" ]; then
            echo "  skipping ${run_name} (already trained)"
            continue
        fi

        ckpt="experiments/sed_b0_fold${f}/best_model.pt"
        echo ""
        echo "--- seed=${SEED} fold=${f} finetune  [$(date +%H:%M:%S)] ---"
        BIRDCLEF_RUN_NAME="$run_name" \
            python -m src.training.finetune \
                configs/finetune_pseudo_v2.yaml \
                --fold "$f" \
                --seed "$SEED" \
                --init-checkpoint "$ckpt" \
                2>&1 | tee "${LOG_DIR}/${run_name}.log"
    done
done

# -----------------------------------------------------------------------------
# Summary — comparing original seed=42 to new seeds
# -----------------------------------------------------------------------------
echo ""
echo "===================================================="
echo "Multi-seed run complete at: $(date)"
echo "===================================================="

echo ""
echo "Results — original (seed=42, your 0.882 ensemble):"
for f in $FOLDS; do
    log_csv="experiments/sed_finetune_pseudo_v2_fold${f}/training_log.csv"
    if [ -f "$log_csv" ]; then
        # finetune.py columns: epoch,train_loss,focal_val_loss,focal_val_auc,
        #                      soundscape_val_loss,soundscape_val_auc,
        #                      combined_val_loss,combined_val_auc,lr,time_s
        best_comb=$(awk -F',' 'NR>1 {print $8}' "$log_csv" | sort -rn | head -1)
        echo "  seed=42  fold=${f}: combined=${best_comb}"
    else
        echo "  seed=42  fold=${f}: NO LOG"
    fi
done

for SEED in $SEEDS; do
    echo ""
    echo "Results — seed=${SEED}:"
    for f in $FOLDS; do
        log_csv="experiments/sed_finetune_pseudo_v2_seed${SEED}_fold${f}/training_log.csv"
        if [ -f "$log_csv" ]; then
            best_comb=$(awk -F',' 'NR>1 {print $8}' "$log_csv" | sort -rn | head -1)
            echo "  seed=${SEED}  fold=${f}: combined=${best_comb}"
        else
            echo "  seed=${SEED}  fold=${f}: NO LOG"
        fi
    done
done

# Compute per-seed mean combined val_auc to show seed variance at a glance
echo ""
echo "===================================================="
echo "Per-seed mean combined val_auc (seed variance check):"
echo "===================================================="
for SEED in 42 $SEEDS; do
    if [ "$SEED" = "42" ]; then
        prefix="experiments/sed_finetune_pseudo_v2_fold"
    else
        prefix="experiments/sed_finetune_pseudo_v2_seed${SEED}_fold"
    fi
    sum=0
    count=0
    for f in $FOLDS; do
        log_csv="${prefix}${f}/training_log.csv"
        if [ -f "$log_csv" ]; then
            best=$(awk -F',' 'NR>1 {print $8}' "$log_csv" | sort -rn | head -1)
            sum=$(awk -v s="$sum" -v b="$best" 'BEGIN { print s + b }')
            count=$((count + 1))
        fi
    done
    if [ "$count" -gt 0 ]; then
        mean=$(awk -v s="$sum" -v c="$count" 'BEGIN { printf "%.5f", s / c }')
        echo "  seed=${SEED}: mean = ${mean} (over ${count} folds)"
    fi
done

echo ""
echo "Interpretation guide:"
echo "  If per-seed means span > 0.005:  seed variance is real → ensembling will help."
echo "  If per-seed means span < 0.002:  seed variance is tiny → ensembling marginal."
echo "                                     (This is what we saw with SEResNeXt.)"
echo ""
echo "===================================================="
echo "Next steps:"
echo "  1. Quantize all 10 new finetune checkpoints to int8 ONNX:"
echo "     bash scripts/export_b0_multiseed_onnx.sh"
echo "  2. Verify quantization fidelity (B0 quantizes cleanly, unlike V2-S)"
echo "  3. Upload as new Kaggle dataset: birdclef-2026-b0-multiseed-onnx"
echo "  4. Update inference notebook to add the 10 new B0 sessions"
echo "  5. Submit at 0.75 B0_combined / 0.25 SEResNeXt blend"
echo "===================================================="
