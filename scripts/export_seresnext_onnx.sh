#!/bin/bash
# =============================================================================
# Export all 5 SEResNeXt finetune checkpoints to ONNX (fp32 only).
# We learned from V2-S that int8 dynamic quantization hurts both accuracy
# AND speed for these architectures, so fp32 is the right call.
# =============================================================================

set -euo pipefail

FOLDS="0 1 2 3 4"

# Pre-flight: verify scripts/export_onnx_int8.py exists
if [ ! -f "scripts/export_onnx_int8.py" ]; then
    echo "ABORT: scripts/export_onnx_int8.py not found."
    echo "       Should be in your repo from the V2-S work."
    exit 1
fi

for f in $FOLDS; do
    ckpt="experiments/seresnext_finetune_fold${f}/best_model.pt"
    out="experiments/seresnext_finetune_fold${f}/model_int8.onnx"

    if [ ! -f "$ckpt" ]; then
        echo "SKIP fold ${f}: checkpoint missing ($ckpt)"
        continue
    fi

    echo ""
    echo "=== Exporting fold ${f} ==="
    python scripts/export_onnx_int8.py \
        --checkpoint "$ckpt" \
        --config configs/finetune_seresnext.yaml \
        --output "$out"

    # Spearman warning will appear if quantization breaks ranks. The fp32
    # file (out + "_fp32.onnx") is what we'll actually ship.
done

echo ""
echo "===================================================="
echo "Export complete. Files for Kaggle upload:"
echo "===================================================="
ls -lh experiments/seresnext_finetune_fold*/model_int8_fp32.onnx
