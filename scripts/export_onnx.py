#!/usr/bin/env python3
"""
Export BirdCLEF PyTorch checkpoints to ONNX (fp32 by default).

Replaces both export_onnx.py and export_onnx_int8.py. Architecture comes
from src.models.build_model(cfg) — the same function the training pipeline
uses — so this script can never drift out of sync with training. The two
old scripts each duplicated their own copy of BirdCLEFSED, which would
silently break the moment the canonical model gained a new __init__ param
(input_normalize, encoder_in_chans, frame-level logits, etc).

Default behavior is fp32-only. Dynamic int8 quantization runs ~3.57x
SLOWER than fp32 on Kaggle CPU due to an ORT op-dispatch pathology
(verified empirically: 113ms vs 402ms for 5xB0). Pass --quantize only if
you need int8 for somewhere other than Kaggle; verification will still
run with looser tolerances.

Usage:
    # Single fold
    python scripts/export_onnx.py \\
        experiments/sed_finetune_pseudo_v3_fold0/best_model.pt

    # All 5 folds (auto-detects fold pattern from the path)
    python scripts/export_onnx.py \\
        experiments/sed_finetune_pseudo_v3_fold0/best_model.pt --all-folds

    # Override architecture if config.yaml is missing (rare, fallback path)
    python scripts/export_onnx.py path/to/best_model.pt \\
        --model-type sed --backbone tf_efficientnet_b0_ns

    # Also produce an int8 ONNX (NOT recommended for Kaggle inference)
    python scripts/export_onnx.py path/to/best_model.pt --quantize
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make src.* importable. Walk up from script location to find the dir that
# contains 'src/'. Works whether the script lives in scripts/ or src/training/.
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve().parent
for _candidate in [_here, _here.parent, _here.parent.parent]:
    if (_candidate / "src").is_dir():
        sys.path.insert(0, str(_candidate))
        break
else:
    sys.exit("ERROR: Could not locate project root (no 'src/' dir found "
             "in script's parents). Run from repo root or move script to scripts/.")

try:
    import yaml
except ImportError:
    sys.exit("ERROR: pyyaml is required. Install with: pip install pyyaml")

try:
    from src.models import build_model
except ImportError as e:
    sys.exit(f"ERROR: Cannot import src.models.build_model: {e}\n"
             f"Make sure you're running from the repo root.")


# ---------------------------------------------------------------------------
# Spectrogram parameters (must match training exactly)
# ---------------------------------------------------------------------------
INPUT_SHAPE = (1, 1, 128, 313)  # (batch, channels, n_mels, time_frames)


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


# ---------------------------------------------------------------------------
# Config & checkpoint loading
# ---------------------------------------------------------------------------

def load_training_config(ckpt_path: Path) -> dict | None:
    """Look for config.yaml next to the checkpoint."""
    cfg_path = ckpt_path.parent / "config.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve_cfg(ckpt_path: Path, cli_overrides: dict) -> dict:
    """Build a config dict for build_model(), preferring config.yaml.

    Hard-fails if neither config.yaml nor sufficient CLI args are present —
    the old script's silent CLI fallback was a footgun because it didn't
    know about params like input_normalize / encoder_in_chans, so the
    exported model could have a *different* architecture than the trained
    one. Better to fail loudly.
    """
    cfg = load_training_config(ckpt_path)

    if cfg is None:
        if not cli_overrides.get("model_type") or not cli_overrides.get("backbone"):
            sys.exit(
                f"ERROR: No config.yaml at {ckpt_path.parent}/ and no "
                f"--model-type / --backbone CLI overrides given.\n"
                f"Either restore config.yaml next to the checkpoint, or "
                f"pass both --model-type and --backbone explicitly."
            )
        print(f"  [WARN] No config.yaml found — using CLI overrides only.")
        print(f"         If the trained model used non-default attention_hidden_dim,")
        print(f"         input_normalize, or encoder_in_chans, you must pass those too.")
        cfg = {}

    # CLI overrides win over config (rare, but lets you override a stale config)
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v

    # build_model needs at minimum these — fail early if missing
    for required in ("model_type", "backbone", "num_classes"):
        if required not in cfg:
            sys.exit(f"ERROR: cfg missing required key: {required!r}")

    return cfg


def load_checkpoint(ckpt_path: Path, cfg: dict) -> torch.nn.Module:
    """Load weights into a model built by build_model(cfg). strict=True."""
    if not ckpt_path.exists():
        sys.exit(f"ERROR: Checkpoint not found: {ckpt_path}")

    model = build_model(cfg)

    # weights_only=True: refuses to unpickle anything that isn't tensors/
    # primitives. Safer; also matches finetune.py's loader.
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)

    # Defensive: some older checkpoints wrapped state in another dict.
    # Bare state_dict is what train.py / finetune.py write today.
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Strip 'module.' prefix from any DataParallel-saved checkpoints
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # strict=True: any architecture mismatch raises immediately. Silent
    # partial loads (load_state_dict's default behavior) are the worst
    # failure mode because they look like success while reinitializing
    # part of the model.
    model.load_state_dict(state, strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded checkpoint: {ckpt_path}")
    print(f"  Architecture:      {cfg['model_type']} / {cfg['backbone']}")
    print(f"  Parameters:        {n_params:,}")
    print(f"  PyTorch file size: {file_size_mb(ckpt_path):.2f} MB")
    return model


# ---------------------------------------------------------------------------
# ONNX export & verification
# ---------------------------------------------------------------------------

def export_onnx(model: torch.nn.Module, output_path: Path, opset_version: int = 17):
    """Export to fp32 ONNX. Falls back to dynamo=True for tricky backbones."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*INPUT_SHAPE)
    common_kwargs = dict(
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    try:
        torch.onnx.export(model, dummy_input, str(output_path), **common_kwargs)
    except Exception as e:
        print(f"  [WARN] Standard exporter failed: {type(e).__name__}: {e}")
        print(f"  [INFO] Retrying with dynamo=True...")
        torch.onnx.export(model, dummy_input, str(output_path),
                          dynamo=True, **common_kwargs)

    print(f"  ONNX saved: {output_path} ({file_size_mb(output_path):.2f} MB)")
    return output_path


def verify_onnx(model: torch.nn.Module, onnx_path: Path,
                atol: float = 1e-4, label: str = "fp32") -> tuple[bool, float]:
    """Compare PyTorch and ONNX outputs on a synthetic input.

    Reports both max abs diff (for fp32 tightness) and Spearman rank
    correlation (for int8, where absolute values can drift but rank order
    is what AUC actually cares about).
    """
    import onnxruntime as ort
    try:
        from scipy.stats import spearmanr
    except ImportError:
        spearmanr = None

    # Standard-normal input mimics your normalized log-mel statistics.
    # Earlier int8 export script noted: cached specs land near (mean=-55,
    # std=17 dB) before normalization; after the SED's input_normalize=False
    # path they're roughly N(0,1)-shaped on average. Synthetic ~ N(0,1) is
    # close enough to surface real divergence.
    dummy = torch.randn(*INPUT_SHAPE)

    with torch.no_grad():
        pt_out = model(dummy).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]

    max_diff = float(np.max(np.abs(pt_out - onnx_out)))
    mean_diff = float(np.mean(np.abs(pt_out - onnx_out)))

    rho_str = ""
    rho = float("nan")
    if spearmanr is not None:
        rho_res = spearmanr(pt_out.ravel(), onnx_out.ravel())
        rho = float(rho_res.statistic)
        rho_str = f", spearman={rho:.6f}"

    passed = max_diff < atol and (np.isnan(rho) or rho > 0.99)
    status = "OK" if passed else "WARN"
    print(f"  Verify [{label}]: max_diff={max_diff:.2e}, "
          f"mean_diff={mean_diff:.2e}{rho_str}  ->  [{status}]")

    if not passed:
        if max_diff >= atol:
            print(f"    max_diff exceeded tolerance ({atol:.0e})")
        if not np.isnan(rho) and rho <= 0.99:
            print(f"    rank correlation {rho:.4f} below 0.99 threshold")

    return passed, max_diff


# ---------------------------------------------------------------------------
# Quantization (kept available, defaulted off)
# ---------------------------------------------------------------------------

def quantize_onnx(fp32_path: Path, int8_path: Path) -> Path:
    """Apply ONNX dynamic int8 quantization.

    NOT recommended for Kaggle CPU — verified to be 3.57x slower than fp32
    due to an ORT op-dispatch pathology on this competition's models.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    print(f"  Quantized:  {int8_path} ({file_size_mb(int8_path):.2f} MB)")
    return int8_path


# ---------------------------------------------------------------------------
# Benchmarking (local sanity check; Kaggle has a separate benchmark script)
# ---------------------------------------------------------------------------

def benchmark_inference(model_or_path, *, is_onnx: bool, n_runs: int = 50, warmup: int = 5):
    """Per-sample latency for either a torch model or an ONNX path."""
    import onnxruntime as ort

    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)

    if is_onnx:
        sess = ort.InferenceSession(str(model_or_path), providers=["CPUExecutionProvider"])
        for _ in range(warmup):
            sess.run(None, {"input": dummy})
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            sess.run(None, {"input": dummy})
            times.append(time.perf_counter() - t0)
    else:
        model = model_or_path
        model.eval()
        x = torch.from_numpy(dummy)
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(x)
                times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000
    return {
        "mean_ms":   float(np.mean(times_ms)),
        "std_ms":    float(np.std(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "p95_ms":    float(np.percentile(times_ms, 95)),
    }


# ---------------------------------------------------------------------------
# Per-checkpoint pipeline
# ---------------------------------------------------------------------------

def process_single_checkpoint(ckpt_path: Path, cli_overrides: dict,
                              do_quantize: bool = False,
                              do_benchmark: bool = True) -> dict:
    """Full export pipeline for one checkpoint."""
    ckpt_path = Path(ckpt_path)
    onnx_path = ckpt_path.parent / ckpt_path.name.replace(".pt", ".onnx")
    int8_path = ckpt_path.parent / ckpt_path.name.replace(".pt", "_quantized.onnx")

    print(f"\n{'='*70}")
    print(f"Processing: {ckpt_path}")
    print(f"{'='*70}")

    print("\n[1/4] Loading checkpoint...")
    cfg = resolve_cfg(ckpt_path, cli_overrides)
    model = load_checkpoint(ckpt_path, cfg)

    print("\n[2/4] Exporting to ONNX (fp32)...")
    export_onnx(model, onnx_path)
    fp32_ok, _ = verify_onnx(model, onnx_path, atol=1e-4, label="fp32")

    int8_ok = None
    if do_quantize:
        print("\n[3/4] Quantizing to int8 (NOT recommended for Kaggle)...")
        quantize_onnx(onnx_path, int8_path)
        int8_ok, _ = verify_onnx(model, int8_path, atol=5e-2, label="int8")
    else:
        print("\n[3/4] Skipping quantization (--quantize not set)")

    result = {
        "ckpt_path":     ckpt_path,
        "onnx_path":     onnx_path,
        "quantized_path": int8_path if do_quantize else None,
        "fp32_verified": fp32_ok,
        "int8_verified": int8_ok,
    }

    if not do_benchmark:
        return result

    print("\n[4/4] Benchmarking (local only — Kaggle has its own benchmark)...")
    print(f"  {'Format':<22} {'Mean (ms)':>10} {'Std':>8} {'P95':>8} {'Size (MB)':>10}")
    print(f"  {'-'*62}")

    pt_t = benchmark_inference(model, is_onnx=False)
    pt_size = file_size_mb(ckpt_path)
    print(f"  {'PyTorch (.pt)':<22} {pt_t['mean_ms']:>10.2f} "
          f"{pt_t['std_ms']:>8.2f} {pt_t['p95_ms']:>8.2f} {pt_size:>10.2f}")

    onnx_t = benchmark_inference(onnx_path, is_onnx=True)
    onnx_size = file_size_mb(onnx_path)
    print(f"  {'ONNX fp32':<22} {onnx_t['mean_ms']:>10.2f} "
          f"{onnx_t['std_ms']:>8.2f} {onnx_t['p95_ms']:>8.2f} {onnx_size:>10.2f}")

    result["pt_time_ms"] = pt_t["mean_ms"]
    result["onnx_time_ms"] = onnx_t["mean_ms"]

    if do_quantize:
        q_t = benchmark_inference(int8_path, is_onnx=True)
        q_size = file_size_mb(int8_path)
        print(f"  {'ONNX int8':<22} {q_t['mean_ms']:>10.2f} "
              f"{q_t['std_ms']:>8.2f} {q_t['p95_ms']:>8.2f} {q_size:>10.2f}")
        result["q_time_ms"] = q_t["mean_ms"]
        # Local-machine warning. On Kaggle this ratio inverts to ~3.57x slower.
        if q_t["mean_ms"] > onnx_t["mean_ms"]:
            print(f"\n  [NOTE] int8 is slower than fp32 on this hardware too. "
                  f"Definitely don't ship int8 to Kaggle.")

    return result


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------

def expand_all_folds(seed_ckpt: Path) -> list[Path]:
    """Given experiments/run_foldN/best_model.pt, return all 5 fold paths."""
    parent_name = seed_ckpt.parent.name
    base_dir = seed_ckpt.parent.parent

    base_prefix = None
    for fold_id in range(5):
        suffix = f"_fold{fold_id}"
        if parent_name.endswith(suffix):
            base_prefix = parent_name[: -len(suffix)]
            break

    if base_prefix is None:
        sys.exit(f"ERROR: --all-folds expected the checkpoint dir to end in "
                 f"_fold{{0..4}}, got: {parent_name}")

    paths = []
    for fold_id in range(5):
        candidate = base_dir / f"{base_prefix}_fold{fold_id}" / seed_ckpt.name
        if candidate.exists():
            paths.append(candidate)
        else:
            print(f"  [WARN] Missing: {candidate}")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Export BirdCLEF PyTorch model(s) to ONNX",
    )
    parser.add_argument("checkpoint", type=str,
                        help="Path to best_model.pt")
    parser.add_argument("--all-folds", action="store_true",
                        help="Process all 5 folds (auto-detects fold pattern)")
    parser.add_argument("--quantize", action="store_true",
                        help="Also produce int8 ONNX. NOT recommended for "
                             "Kaggle CPU (verified 3.57x slower).")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip local timing benchmark")
    parser.add_argument("--opset", type=int, default=17)

    # Fallback overrides for missing config.yaml. Prefer to leave these
    # unset and let resolve_cfg() pick up config.yaml.
    parser.add_argument("--model-type", default=None, choices=["classifier", "sed"])
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--attention-hidden-dim", type=int, default=None)
    parser.add_argument("--input-normalize", type=lambda s: s.lower() == "true",
                        default=None,
                        help="(SED) Set true for ConvNeXt/LayerNorm backbones")
    parser.add_argument("--encoder-in-chans", type=int, default=None,
                        help="(SED) 1 for B0/SX, 3 for ConvNeXt")

    args = parser.parse_args()

    # Lazy import check — fail fast with a helpful message
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        sys.exit("ERROR: Install onnx + onnxruntime:\n"
                 "  pip install onnx onnxruntime")

    cli_overrides = {
        "model_type":           args.model_type,
        "backbone":             args.backbone,
        "num_classes":          args.num_classes,
        "dropout":              args.dropout,
        "attention_hidden_dim": args.attention_hidden_dim,
        "input_normalize":      args.input_normalize,
        "encoder_in_chans":     args.encoder_in_chans,
    }

    seed_ckpt = Path(args.checkpoint)
    targets = expand_all_folds(seed_ckpt) if args.all_folds else [seed_ckpt]

    if args.all_folds:
        print(f"Found {len(targets)} fold checkpoint(s) to export.")

    results = []
    for ckpt in targets:
        try:
            r = process_single_checkpoint(
                ckpt, cli_overrides,
                do_quantize=args.quantize,
                do_benchmark=not args.no_benchmark,
            )
            results.append(r)
        except SystemExit:
            raise
        except Exception as e:
            print(f"\n[ERROR] Failed on {ckpt}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    if args.all_folds:
        print(f"\n{'='*70}")
        print(f"SUMMARY: exported {len(results)}/{len(targets)} folds")
        print(f"{'='*70}")
        all_ok = True
        for r in results:
            ok = r["fp32_verified"] and (r["int8_verified"] in (None, True))
            all_ok = all_ok and ok
            tag = "OK  " if ok else "WARN"
            path = r["quantized_path"] or r["onnx_path"]
            print(f"  [{tag}] {path}")
        if not all_ok:
            sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()