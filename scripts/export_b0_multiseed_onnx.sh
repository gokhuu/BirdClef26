#!/bin/bash
# =============================================================================
# Export the 10 new B0 multi-seed checkpoints to ONNX (int8 quantized).
# B0 quantizes cleanly (unlike V2-S), so we ship int8 to fit the inference
# budget when adding 10 more models to the ensemble.
# =============================================================================

set -euo pipefail

FOLDS="0 1 2 3 4"
SEEDS="123 2024"

if [ ! -f "scripts/export_onnx_int8.py" ]; then
    echo "ABORT: scripts/export_onnx_int8.py not found."
    exit 1
fi

for SEED in $SEEDS; do
    for f in $FOLDS; do
        run_name="sed_finetune_pseudo_v2_seed${SEED}_fold${f}"
        ckpt="experiments/${run_name}/best_model.pt"
        out="experiments/${run_name}/model_int8.onnx"

        if [ ! -f "$ckpt" ]; then
            echo "SKIP ${run_name}: checkpoint missing"
            continue
        fi

        echo ""
        echo "=== Exporting ${run_name} ==="
        python scripts/export_onnx_int8.py \
            --checkpoint "$ckpt" \
            --config configs/finetune_pseudo_v2.yaml \
            --output "$out"
    done
done

echo ""
echo "===================================================="
echo "Export complete. Files for Kaggle upload (int8):"
echo "===================================================="
ls -lh experiments/sed_finetune_pseudo_v2_seed*_fold*/model_int8.onnx 2>/dev/null

echo ""
echo "Stage for upload:"
cat <<'STAGE'
mkdir -p kaggle_upload/b0_multiseed
for SEED in 123 2024
    for f in 0 1 2 3 4
        cp experiments/sed_finetune_pseudo_v2_seed${SEED}_fold${f}/model_int8.onnx \
           kaggle_upload/b0_multiseed/b0_seed${SEED}_fold${f}.onnx
    end
end
ls -lh kaggle_upload/b0_multiseed/
STAGE
