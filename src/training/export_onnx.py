#!/usr/bin/env python3
"""
Export a trained BirdCLEF PyTorch checkpoint to ONNX format for CPU inference.

Usage:
    python src/training/export_onnx.py experiments/baseline_effb0_fold0/best_model.pt
    python src/training/export_onnx.py experiments/baseline_effb0_fold0/best_model.pt --quantize
    python src/training/export_onnx.py experiments/sed_b0_fold0/best_model.pt --all-folds

Model type is auto-detected from config.yaml in the same directory as the checkpoint.
If no config is found, falls back to --model-type / --backbone CLI args.

This script:
  1. Loads a PyTorch checkpoint (best_model.pt)
  2. Exports to ONNX format
  3. Verifies ONNX output matches PyTorch output (within tolerance)
  4. Optionally applies dynamic quantization for faster CPU inference
  5. Prints model size comparison and inference timing
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    sys.exit("ERROR: timm is required. Install with: pip install timm")

try:
    import yaml
except ImportError:
    sys.exit("ERROR: pyyaml is required. Install with: pip install pyyaml")


# ---------------------------------------------------------------------------
# Model definitions (self-contained, mirroring src/models/)
#
# These are duplicated from src/models/{classifier,sed}.py on purpose so the
# export script stays usable even if the src layout changes later. Keep them
# in sync with the training-time definitions.
# ---------------------------------------------------------------------------


class BirdCLEFModel(nn.Module):
    """EfficientNet backbone + global avg pool + linear head (baseline)."""

    def __init__(self, backbone="tf_efficientnet_b0_ns", num_classes=234,
                 dropout=0.3, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained,
            in_chans=1, num_classes=0, global_pool="avg",
        )
        n_features = self.encoder.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, num_classes),
        )

    def forward(self, x):
        features = self.encoder(x)
        logits = self.head(features)
        return logits


class AttentionPooling(nn.Module):
    """Learned soft attention over the time axis."""

    def __init__(self, in_features, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_features, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, C, T)
        att_logits = self.attention(x)
        att_weights = torch.softmax(att_logits, dim=-1)
        pooled = torch.sum(x * att_weights, dim=-1)
        return pooled, att_weights


class BirdCLEFSED(nn.Module):
    """EfficientNet backbone + attention pooling over time (SED)."""

    def __init__(self, backbone="tf_efficientnet_b0_ns", num_classes=234,
                 dropout=0.3, pretrained=False, attention_hidden_dim=128):
        super().__init__()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained,
            in_chans=1, num_classes=0, global_pool="",
        )
        feature_dim = self.encoder.num_features
        self.attention_pool = AttentionPooling(
            in_features=feature_dim, hidden_dim=attention_hidden_dim,
        )
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        # Always return logits only — no return_attention branch —
        # so ONNX tracing is clean.
        features = self.encoder(x)            # (B, C, H', W')
        features = features.mean(dim=2)       # mean over frequency -> (B, C, W')
        pooled, _ = self.attention_pool(features)
        logits = self.head(pooled)
        return logits


# ---------------------------------------------------------------------------
# Spectrogram parameters (must match training exactly)
# ---------------------------------------------------------------------------
SPEC_PARAMS = {
    "sr": 32000,
    "n_mels": 128,
    "fmax": 16000,
    "hop_length": 512,
    "n_fft": 2048,
    "duration": 5.0,
}
INPUT_SHAPE = (1, 1, 128, 313)  # (batch, channels, n_mels, time_frames)


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def load_training_config(ckpt_path):
    """
    Look for config.yaml next to the checkpoint. Returns dict or None.
    """
    ckpt_path = Path(ckpt_path)
    cfg_path = ckpt_path.parent / "config.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def build_model_from_config(cfg, cli_defaults):
    """
    Construct the right architecture based on config + CLI fallbacks.

    cli_defaults: dict with keys {model_type, backbone, num_classes, dropout,
                                  attention_hidden_dim}
    """
    if cfg is None:
        cfg = {}

    model_type = cfg.get("model_type", cli_defaults["model_type"])
    backbone = cfg.get("backbone", cli_defaults["backbone"])
    num_classes = cfg.get("num_classes", cli_defaults["num_classes"])
    dropout = cfg.get("dropout", cli_defaults["dropout"])

    if model_type == "classifier":
        return BirdCLEFModel(
            backbone=backbone, num_classes=num_classes,
            dropout=dropout, pretrained=False,
        ), model_type, backbone
    elif model_type == "sed":
        attn_hidden = cfg.get(
            "attention_hidden_dim", cli_defaults["attention_hidden_dim"]
        )
        return BirdCLEFSED(
            backbone=backbone, num_classes=num_classes,
            dropout=dropout, pretrained=False,
            attention_hidden_dim=attn_hidden,
        ), model_type, backbone
    else:
        sys.exit(f"ERROR: Unknown model_type: {model_type!r}")


def load_checkpoint(ckpt_path, cli_defaults):
    """Load a trained PyTorch checkpoint, auto-detecting architecture."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        sys.exit(f"ERROR: Checkpoint not found: {ckpt_path}")

    cfg = load_training_config(ckpt_path)
    model, model_type, backbone = build_model_from_config(cfg, cli_defaults)

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if isinstance(state, dict):
        if "model_state_dict" in state:
            sd = state["model_state_dict"]
        elif "state_dict" in state:
            sd = state["state_dict"]
        else:
            sd = state
    else:
        sys.exit("ERROR: Unexpected checkpoint format")

    model.load_state_dict(sd)
    model.eval()

    cfg_source = "config.yaml" if cfg is not None else "CLI defaults"
    print(f"  Loaded checkpoint: {ckpt_path}")
    print(f"  Model type:        {model_type}  (from {cfg_source})")
    print(f"  Backbone:          {backbone}")
    print(f"  PyTorch file size: {file_size_mb(ckpt_path):.2f} MB")
    return model


def export_onnx(model, output_path, opset_version=17):
    """Export PyTorch model to ONNX format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*INPUT_SHAPE)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print(f"  ONNX model saved: {output_path} ({file_size_mb(output_path):.2f} MB)")
    return output_path


def verify_onnx(model, onnx_path, atol=1e-5, rtol=1e-4):
    """Verify ONNX model produces identical outputs to PyTorch model."""
    import onnxruntime as ort

    dummy_input = torch.randn(*INPUT_SHAPE)

    # PyTorch inference
    model.eval()
    with torch.no_grad():
        pt_output = model(dummy_input).numpy()

    # ONNX inference
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    ort_output = sess.run(None, {"input": dummy_input.numpy()})[0]

    # Compare
    max_diff = np.max(np.abs(pt_output - ort_output))
    mean_diff = np.mean(np.abs(pt_output - ort_output))
    all_close = np.allclose(pt_output, ort_output, atol=atol, rtol=rtol)

    print(f"  Verification: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, "
          f"all_close={all_close}")

    if not all_close:
        print("  WARNING: ONNX output differs from PyTorch beyond tolerance!")
        print(f"  Consider increasing atol (current: {atol}) or investigating.")
    else:
        print("  ✓ ONNX output matches PyTorch output within tolerance.")

    return all_close


def quantize_onnx(onnx_path, quantized_path=None):
    """Apply dynamic quantization to ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if quantized_path is None:
        quantized_path = str(onnx_path).replace(".onnx", "_quantized.onnx")

    quantize_dynamic(
        str(onnx_path),
        str(quantized_path),
        weight_type=QuantType.QUInt8,
    )
    print(f"  Quantized ONNX saved: {quantized_path} "
          f"({file_size_mb(quantized_path):.2f} MB)")
    return Path(quantized_path)


def benchmark_inference(model_or_path, n_runs=50, warmup=5, is_onnx=False):
    """Benchmark single-sample inference time."""
    import onnxruntime as ort

    dummy_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)

    if is_onnx:
        sess = ort.InferenceSession(str(model_or_path),
                                    providers=["CPUExecutionProvider"])
        for _ in range(warmup):
            sess.run(None, {"input": dummy_input})
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            sess.run(None, {"input": dummy_input})
            times.append(time.perf_counter() - t0)
    else:
        model = model_or_path
        model.eval()
        dummy_torch = torch.from_numpy(dummy_input)
        with torch.no_grad():
            for _ in range(warmup):
                model(dummy_torch)
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(dummy_torch)
                times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # ms
    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "median_ms": np.median(times),
        "p95_ms": np.percentile(times, 95),
    }


def process_single_checkpoint(ckpt_path, cli_defaults, do_quantize=True):
    """Full export pipeline for a single checkpoint."""
    ckpt_path = Path(ckpt_path)
    onnx_path = ckpt_path.parent / ckpt_path.name.replace(".pt", ".onnx")

    print(f"\n{'='*70}")
    print(f"Processing: {ckpt_path}")
    print(f"{'='*70}")

    print("\n[1/5] Loading PyTorch checkpoint...")
    model = load_checkpoint(ckpt_path, cli_defaults)

    print("\n[2/5] Exporting to ONNX...")
    export_onnx(model, onnx_path)

    print("\n[3/5] Verifying ONNX output...")
    verify_onnx(model, onnx_path)

    quantized_path = None
    if do_quantize:
        print("\n[4/5] Applying dynamic quantization...")
        quantized_path = quantize_onnx(onnx_path)
        print("  Verifying quantized model...")
        verify_onnx(model, quantized_path, atol=0.05, rtol=0.01)
    else:
        print("\n[4/5] Skipping quantization (use --quantize to enable)")

    print("\n[5/5] Benchmarking inference speed...")
    print(f"  {'Format':<25} {'Mean (ms)':>10} {'Std':>8} {'P95':>8} {'Size (MB)':>10}")
    print(f"  {'-'*65}")

    pt_times = benchmark_inference(model, is_onnx=False)
    pt_size = file_size_mb(ckpt_path)
    print(f"  {'PyTorch (.pt)':<25} {pt_times['mean_ms']:>10.2f} "
          f"{pt_times['std_ms']:>8.2f} {pt_times['p95_ms']:>8.2f} {pt_size:>10.2f}")

    onnx_times = benchmark_inference(onnx_path, is_onnx=True)
    onnx_size = file_size_mb(onnx_path)
    print(f"  {'ONNX (.onnx)':<25} {onnx_times['mean_ms']:>10.2f} "
          f"{onnx_times['std_ms']:>8.2f} {onnx_times['p95_ms']:>8.2f} {onnx_size:>10.2f}")

    if quantized_path:
        q_times = benchmark_inference(quantized_path, is_onnx=True)
        q_size = file_size_mb(quantized_path)
        print(f"  {'Quantized ONNX':<25} {q_times['mean_ms']:>10.2f} "
              f"{q_times['std_ms']:>8.2f} {q_times['p95_ms']:>8.2f} {q_size:>10.2f}")

    print(f"\n  Speedup vs PyTorch:")
    print(f"    ONNX:      {pt_times['mean_ms']/onnx_times['mean_ms']:.2f}x faster")
    if quantized_path:
        print(f"    Quantized: {pt_times['mean_ms']/q_times['mean_ms']:.2f}x faster")
    print(f"  Size reduction:")
    print(f"    ONNX:      {(1 - onnx_size/pt_size)*100:.1f}% smaller")
    if quantized_path:
        print(f"    Quantized: {(1 - q_size/pt_size)*100:.1f}% smaller")

    return {
        "onnx_path": onnx_path,
        "quantized_path": quantized_path,
        "pt_time_ms": pt_times["mean_ms"],
        "onnx_time_ms": onnx_times["mean_ms"],
        "q_time_ms": q_times["mean_ms"] if quantized_path else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export BirdCLEF PyTorch model to ONNX")
    parser.add_argument("checkpoint", type=str,
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--quantize", action="store_true", default=True,
                        help="Apply dynamic quantization (default: True)")
    parser.add_argument("--no-quantize", dest="quantize", action="store_false",
                        help="Skip quantization")
    parser.add_argument("--all-folds", action="store_true",
                        help="Export all 5 fold checkpoints (auto-detects paths)")
    parser.add_argument("--num-classes", type=int, default=234)
    parser.add_argument("--backbone", type=str, default="tf_efficientnet_b0_ns",
                        help="Fallback backbone if config.yaml not found")
    parser.add_argument("--model-type", type=str, default="classifier",
                        choices=["classifier", "sed"],
                        help="Fallback model type if config.yaml not found")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--attention-hidden-dim", type=int, default=128)

    args = parser.parse_args()

    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        sys.exit("ERROR: Install onnx and onnxruntime:\n"
                 "  pip install onnx onnxruntime onnxruntime-extensions")

    cli_defaults = {
        "model_type": args.model_type,
        "backbone": args.backbone,
        "num_classes": args.num_classes,
        "dropout": args.dropout,
        "attention_hidden_dim": args.attention_hidden_dim,
    }

    if args.all_folds:
        ckpt_path = Path(args.checkpoint)
        base_dir = ckpt_path.parent.parent
        prefix = ckpt_path.parent.name

        for suffix in ["_fold0", "_fold1", "_fold2", "_fold3", "_fold4"]:
            if suffix in prefix:
                base_prefix = prefix.replace(suffix, "")
                break
        else:
            sys.exit(f"ERROR: Cannot determine fold pattern from {prefix}")

        results = []
        for fold_id in range(5):
            fold_dir = base_dir / f"{base_prefix}_fold{fold_id}"
            fold_ckpt = fold_dir / ckpt_path.name
            if fold_ckpt.exists():
                result = process_single_checkpoint(
                    fold_ckpt, cli_defaults, do_quantize=args.quantize,
                )
                results.append(result)
            else:
                print(f"\nWARNING: Checkpoint not found: {fold_ckpt}")

        print(f"\n{'='*70}")
        print(f"SUMMARY: Exported {len(results)}/{5} fold models")
        print(f"{'='*70}")
        for r in results:
            path = r["quantized_path"] or r["onnx_path"]
            t = r["q_time_ms"] or r["onnx_time_ms"]
            print(f"  {path}  ({file_size_mb(path):.1f} MB, {t:.1f} ms/sample)")
    else:
        process_single_checkpoint(
            args.checkpoint, cli_defaults, do_quantize=args.quantize,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
