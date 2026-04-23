#!/usr/bin/env fish
# ============================================================
# Full fine-tune pipeline: train all 5 folds, export to ONNX,
# copy to submissions/. Aborts on first failure.
#
# Run from project root:
#   fish scripts/run_finetune_pipeline.fish
# ============================================================

set -l start_time (date +%s)

# ---- Stage 1: fine-tune all 5 folds ----
echo "================================================================"
echo "STAGE 1/3: Fine-tuning 5 folds (~70 min)"
echo "================================================================"
for f in 0 1 2 3 4
    echo
    echo ">>> Fold $f starting at "(date '+%H:%M:%S')
    python src/training/finetune.py configs/experiment_sed_b0_finetune.yaml \
        --fold $f \
        --init-checkpoint experiments/sed_b0_fold{$f}/best_model.pt
    or begin
        echo "✗ Fold $f failed. Aborting pipeline."
        exit 1
    end
end

# ---- Stage 2: export each best_model.pt to ONNX ----
echo
echo "================================================================"
echo "STAGE 2/3: Exporting 5 checkpoints to ONNX"
echo "================================================================"
for f in 0 1 2 3 4
    echo
    echo ">>> Exporting fold $f"
    python src/training/export_onnx.py experiments/sed_finetune_fold{$f}/best_model.pt
    or begin
        echo "✗ Export of fold $f failed. Aborting pipeline."
        exit 1
    end
end

# ---- Stage 3: copy & rename ONNX files into submissions/ ----
echo
echo "================================================================"
echo "STAGE 3/3: Copying ONNX files to submissions/"
echo "================================================================"
mkdir -p submissions
for f in 0 1 2 3 4
    cp experiments/sed_finetune_fold{$f}/best_model.onnx \
       submissions/best_model_{$f}.onnx
    or begin
        echo "✗ Copy of fold $f failed. Aborting pipeline."
        exit 1
    end
end

# ---- Final summary ----
set -l elapsed (math (date +%s) - $start_time)
set -l minutes (math --scale=1 $elapsed / 60)

echo
echo "================================================================"
echo "✓ Pipeline complete in $minutes minutes"
echo "======================================================