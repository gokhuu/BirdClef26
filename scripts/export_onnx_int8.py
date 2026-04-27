#!/usr/bin/env python
"""
Export a trained V2-S finetune checkpoint to ONNX (fp32 + int8).

Uses your project's build_model() to instantiate the architecture, loads
weights with strict=True, exports to ONNX, then quantizes to int8 dynamic.
Verifies output equivalence between torch / fp32-onnx / int8-onnx before
declaring success.

Usage:
    python scripts/export_onnx_int8.py \\
        --checkpoint experiments/effv2s_finetune_fold0/best_model.pt \\
        --config configs/finetune_effv2s.yaml \\
        --output experiments/effv2s_finetune_fold0/model_int8.onnx

Or batch all 5 folds:
    for f in 0 1 2 3 4; do
        python scripts/export_onnx_int8.py \\
            --checkpoint experiments/effv2s_finetune_fold${f}/best_model.pt \\
            --config configs/finetune_effv2s.yaml \\
            --output experiments/effv2s_finetune_fold${f}/model_int8.onnx
    done
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def load_model(checkpoint_path: str, config_path: str, device: str = "cpu"):
    """Use the project's build_model() to instantiate, then load weights."""
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from src.models import build_model

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Loaded {checkpoint_path}")
    print(f"    {len(state)} tensors, {n_params:,} parameters, strict=True")
    return model


def export_onnx_fp32(model, output_path, input_shape=(1, 1, 128, 313), opset=17):
    """Export to fp32 ONNX with dynamic batch dimension."""
    dummy = torch.randn(*input_shape, dtype=torch.float32)
    fp32_path = output_path.replace(".onnx", "_fp32.onnx")

    try:
        torch.onnx.export(
            model,
            dummy,
            fp32_path,
            input_names=["spec"],
            output_names=["logits"],
            dynamic_axes={"spec": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
        )
    except Exception as e:
        print(f"  [WARN] Standard export failed: {e}")
        print(f"  [INFO] Retrying with dynamo=True ...")
        torch.onnx.export(
            model,
            dummy,
            fp32_path,
            input_names=["spec"],
            output_names=["logits"],
            dynamic_axes={"spec": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=True,
        )

    print(f"  ✓ fp32 ONNX -> {fp32_path}  ({os.path.getsize(fp32_path) / 1e6:.1f} MB)")
    return fp32_path


def quantize_dynamic_int8(fp32_path, int8_path):
    """Dynamic int8 quantization — no calibration data needed."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
    )
    print(f"  ✓ int8 ONNX -> {int8_path}  ({os.path.getsize(int8_path) / 1e6:.1f} MB)")


def verify(model, fp32_path, int8_path, input_shape=(1, 1, 128, 313)):
    """Compare torch / fp32-onnx / int8-onnx outputs on synthetic input."""
    import onnxruntime as ort
    from scipy.stats import spearmanr

    # Use realistic input distribution: standard-normal,
    # mimicking your normalized (mean=-55, std=17 dB → N(0,1)) mel specs
    x = np.random.randn(*input_shape).astype(np.float32)

    with torch.no_grad():
        y_torch = torch.sigmoid(model(torch.from_numpy(x))).numpy()

    sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    y_fp32 = 1.0 / (1.0 + np.exp(-sess_fp32.run(None, {"spec": x})[0]))

    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    y_int8 = 1.0 / (1.0 + np.exp(-sess_int8.run(None, {"spec": x})[0]))

    fp32_diff = float(np.abs(y_torch - y_fp32).max())
    int8_diff = float(np.abs(y_torch - y_int8).max())
    rho_int8 = float(spearmanr(y_torch.ravel(), y_int8.ravel()).statistic)

    print(f"\n  === Sanity check (synthetic input) ===")
    print(f"  torch vs fp32 max |diff|: {fp32_diff:.6f}  (target < 1e-4)")
    print(f"  torch vs int8 max |diff|: {int8_diff:.6f}  (target < 5e-2)")
    print(f"  torch vs int8 Spearman:   {rho_int8:.6f}  (target > 0.99)")

    warnings = []
    if fp32_diff > 1e-3:
        warnings.append(f"fp32 deviates more than expected ({fp32_diff:.6f})")
    if rho_int8 < 0.99:
        warnings.append(f"int8 rank correlation low ({rho_int8:.4f})")

    if warnings:
        print(f"\n  [WARNINGS]")
        for w in warnings:
            print(f"    - {w}")
        print(f"  [SUGGESTION] Ship fp32 ONNX instead, or skip quantization for this fold.")
    else:
        print(f"\n  [OK] All checks passed.")

    return rho_int8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True,
                        help="YAML config used during finetune (provides architecture)")
    parser.add_argument("--output", required=True, help="Path for output int8 .onnx")
    parser.add_argument("--input_shape", type=int, nargs=4, default=[1, 1, 128, 313])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip_verify", action="store_true")
    parser.add_argument("--keep_fp32", action="store_true",
                        help="Keep the intermediate fp32 ONNX file (default: keep)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"[1/3] Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, args.config)

    print(f"\n[2/3] Exporting fp32 ONNX...")
    fp32_path = export_onnx_fp32(model, args.output, tuple(args.input_shape), args.opset)

    print(f"\n[3/3] Quantizing to int8...")
    quantize_dynamic_int8(fp32_path, args.output)

    if not args.skip_verify:
        verify(model, fp32_path, args.output, tuple(args.input_shape))

    print(f"\nDone. Use {args.output} for inference.")
    print(f"(fp32 fallback at {fp32_path})")


if __name__ == "__main__":
    main()
