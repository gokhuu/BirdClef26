#!/usr/bin/env python
"""Export trained BirdCLEF models to ONNX.

For BirdCLEFSED: dual-output ONNX with named outputs
    'logits_pool' (attention-pooled, identical to legacy single-output export)
    'logits_max'  (per-frame head + max over T)

For BirdCLEFModel and other non-SED architectures: single-output ONNX,
output name 'logits' (unchanged behaviour).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import build_model
from src.models.sed import BirdCLEFSED

# ADAPT: swap in whichever config loader you currently use.
from src.utils.config import load_config


ATOL = 1e-4
SPEARMAN_MIN = 0.99


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a.ravel(), b.ravel()).correlation)


class _SEDDualOutputWrapper(nn.Module):
    """Wraps BirdCLEFSED so its forward returns (logits_pool, logits_max).
    Pure plumbing — no learnable params, no extra ops vs forward_dual."""

    def __init__(self, model: BirdCLEFSED):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_dual(x)


def _make_dummy(cfg, batch_size: int = 1) -> torch.Tensor:
    # ADAPT: align field names with your cfg schema.
    n_mels = getattr(cfg, "n_mels", 128)
    n_frames = getattr(cfg, "n_frames", 313)
    in_chans = getattr(cfg, "in_chans", 1)
    return torch.randn(batch_size, in_chans, n_mels, n_frames)


def export_to_onnx(
    model: nn.Module, dummy: torch.Tensor, output_path: str, *, opset: int = 17
) -> bool:
    """Returns True iff exported as dual-output (i.e. SED model)."""
    model = model.eval().cpu()
    dummy = dummy.cpu()
    is_sed = isinstance(model, BirdCLEFSED)

    if is_sed:
        export_module = _SEDDualOutputWrapper(model)
        output_names = ["logits_pool", "logits_max"]
        dynamic_axes = {
            "input": {0: "batch"},
            "logits_pool": {0: "batch"},
            "logits_max": {0: "batch"},
        }
    else:
        export_module = model
        output_names = ["logits"]
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        export_module,
        dummy,
        output_path,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    return is_sed


def verify_onnx(
    model: nn.Module, dummy: torch.Tensor, output_path: str, is_sed: bool
) -> None:
    """Verify ONNX matches PyTorch on every named output (atol + Spearman)."""
    import onnxruntime as ort

    model.eval()
    with torch.no_grad():
        if is_sed:
            tp, tm = model.forward_dual(dummy)
            torch_outs = {"logits_pool": tp.numpy(), "logits_max": tm.numpy()}
        else:
            torch_outs = {"logits": model(dummy).numpy()}

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    onnx_names = [o.name for o in sess.get_outputs()]
    onnx_vals = sess.run(None, {"input": dummy.numpy()})
    onnx_outs = dict(zip(onnx_names, onnx_vals))

    if set(onnx_names) != set(torch_outs.keys()):
        raise RuntimeError(f"ONNX outputs {onnx_names} != expected {list(torch_outs)}")

    for name, t in torch_outs.items():
        o = onnx_outs[name]
        max_abs = float(np.max(np.abs(t - o)))
        rho = _spearman(t, o)
        print(f"  [{name}]  max|Δ|={max_abs:.2e}  spearman={rho:.6f}")
        if max_abs > ATOL:
            raise RuntimeError(f"{name}: max abs diff {max_abs:.2e} > atol {ATOL:.2e}")
        if rho < SPEARMAN_MIN:
            raise RuntimeError(f"{name}: spearman {rho:.4f} < {SPEARMAN_MIN}")
    print("  verification: OK")


def _load_state_dict(model: nn.Module, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    _load_state_dict(model, args.checkpoint)
    model.eval()

    dummy = _make_dummy(cfg, batch_size=args.batch_size)

    print(f"Exporting {type(model).__name__}  ->  {args.output}")
    is_sed = export_to_onnx(model, dummy, args.output, opset=args.opset)
    print(f"  mode: {'dual-output (SED)' if is_sed else 'single-output (non-SED)'}")
    verify_onnx(model, dummy, args.output, is_sed)
    print("Done.")


if __name__ == "__main__":
    main()
