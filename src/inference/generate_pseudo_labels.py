"""
BirdCLEF+ 2026 — Pseudo-label Generation
========================================

Runs the 5-fold fine-tuned ensemble over unlabeled soundscape spectrograms
and writes per-window soft labels (averaged sigmoid probabilities) to CSV.

Pipeline:
  1. Load 5 fold checkpoints (defaults to experiments/sed_finetune_fold{f}/best_model.pt).
  2. Iterate .npy spec files in cfg['unlabeled_soundscape_spec_dir'], batched.
  3. For each window: average sigmoid(logits) across 5 models.
  4. Optional temperature sharpening: probs ** T  (T > 1 pushes ambiguous → 0).
  5. Optional confidence gate: drop windows where max(probs) < min_confidence.
  6. Write CSV with columns: row_id, <species_0>, ..., <species_233>.

Usage:
    python src/inference/generate_pseudo_labels.py configs/experiment_sed_b0_finetune_pseudo.yaml

Required config keys (in addition to standard model/data keys):
    unlabeled_soundscape_spec_dir : str   # dir of *.npy spec files (unlabeled)
    pseudo_labels_csv             : str   # output CSV path
Optional:
    pseudo_source_checkpoint_pattern : str   # default "experiments/sed_finetune_fold{fold}/best_model.pt"
    pseudo_inference_batch_size      : int   # default 64
    pseudo_sharpening_T              : float # default 1.5  (1.0 disables)
    pseudo_min_confidence            : float # default 0.3  (0.0 disables)
"""

import sys
import os
import yaml
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.getcwd())

from src.models import build_model
from src.training.train import get_target_species

# Match SoundscapeDataset's spec geometry.
TARGET_FRAMES = 313
N_MELS = 128


# ===================================================================
# Helpers
# ===================================================================

def load_ensemble(cfg: dict, device) -> list[torch.nn.Module]:
    """Load 5 fold checkpoints into eval-mode models on device."""
    pattern = cfg.get(
        "pseudo_source_checkpoint_pattern",
        "experiments/sed_finetune_fold{fold}/best_model.pt",
    )
    models = []
    for f in range(5):
        ckpt_path = pattern.format(fold=f)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Source checkpoint not found: {ckpt_path}\n"
                f"Adjust 'pseudo_source_checkpoint_pattern' in config or train fold {f} first."
            )
        model = build_model(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)
        print(f"  ✓ Loaded fold {f}: {ckpt_path}")
    return models


def load_and_pad_spec(path: str) -> np.ndarray:
    """Load a (128, T) spec and pad/trim to (128, TARGET_FRAMES)."""
    spec = np.load(path)
    n = spec.shape[1]
    if n == TARGET_FRAMES:
        return spec
    if n > TARGET_FRAMES:
        start = (n - TARGET_FRAMES) // 2
        return spec[:, start:start + TARGET_FRAMES]
    return np.pad(spec, ((0, 0), (0, TARGET_FRAMES - n)),
                  mode="constant", constant_values=spec.min())


def iter_unlabeled_specs(spec_dir: str) -> list[Path]:
    """Return sorted list of .npy files in spec_dir."""
    spec_dir = Path(spec_dir)
    if not spec_dir.exists():
        raise FileNotFoundError(f"unlabeled_soundscape_spec_dir does not exist: {spec_dir}")
    files = sorted(spec_dir.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy files found in {spec_dir}")
    return files


@torch.no_grad()
def predict_batch(models: list, batch: torch.Tensor, use_amp: bool) -> np.ndarray:
    """Run all models on a (B, 1, 128, 313) batch, return mean sigmoid (B, 234)."""
    accum = None
    for m in models:
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = m(batch)
        probs = torch.sigmoid(logits.float())
        accum = probs if accum is None else accum + probs
    return (accum / len(models)).cpu().numpy()


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base_path = cfg.pop("base_config", None)
    if base_path:
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(cfg)   # child overrides parent
        cfg = base

    # Required keys
    for key in ("unlabeled_soundscape_spec_dir", "pseudo_labels_csv"):
        if key not in cfg:
            raise KeyError(f"Required config key missing: {key}")

    # Knobs
    batch_size      = cfg.get("pseudo_inference_batch_size", 64)
    sharpening_T    = float(cfg.get("pseudo_sharpening_T", 1.5))
    min_confidence  = float(cfg.get("pseudo_min_confidence", 0.3))
    out_csv         = cfg["pseudo_labels_csv"]
    spec_dir        = cfg["unlabeled_soundscape_spec_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  amp={use_amp}")
    print(f"Sharpening T={sharpening_T}  min_confidence={min_confidence}  batch={batch_size}")

    # Setup
    target_species = get_target_species(cfg)
    print(f"Classes: {len(target_species)}")

    print("\nLoading 5-fold ensemble...")
    models = load_ensemble(cfg, device)

    print(f"\nScanning {spec_dir}...")
    spec_files = iter_unlabeled_specs(spec_dir)
    print(f"  {len(spec_files):,} unlabeled windows")

    # Inference loop — batch by batch, accumulate row_ids + probs
    all_row_ids = []
    all_probs = []
    n_kept = 0
    n_dropped = 0
    t0 = time.time()

    for start in range(0, len(spec_files), batch_size):
        chunk = spec_files[start:start + batch_size]
        specs = np.stack([load_and_pad_spec(str(p)) for p in chunk])     # (B, 128, 313)
        batch = torch.from_numpy(specs).unsqueeze(1).float().to(device)  # (B, 1, 128, 313)

        probs = predict_batch(models, batch, use_amp)  # (B, 234)

        # Sharpening (in float32, on numpy)
        if sharpening_T != 1.0:
            probs = probs ** sharpening_T

        # Confidence gate — drop windows the ensemble has nothing to say about
        max_per_row = probs.max(axis=1)
        keep_mask = max_per_row >= min_confidence

        for i, p in enumerate(chunk):
            if keep_mask[i]:
                all_row_ids.append(p.stem)
                all_probs.append(probs[i])
                n_kept += 1
            else:
                n_dropped += 1

        if (start // batch_size) % 20 == 0:
            elapsed = time.time() - t0
            done = start + len(chunk)
            rate = done / max(elapsed, 1e-6)
            eta = (len(spec_files) - done) / max(rate, 1e-6)
            print(f"  [{done:>6}/{len(spec_files)}]  "
                  f"kept={n_kept}  dropped={n_dropped}  "
                  f"{rate:.0f} win/s  eta {eta/60:.1f}min")

    elapsed = time.time() - t0
    print(f"\nInference done in {elapsed/60:.1f}min")
    print(f"  kept    : {n_kept:,}")
    print(f"  dropped : {n_dropped:,}  (max_prob < {min_confidence})")

    if n_kept == 0:
        raise RuntimeError(
            "No windows passed the confidence gate. Lower pseudo_min_confidence "
            "or check that the source checkpoints actually predict positives."
        )

    # Build & write CSV
    probs_arr = np.stack(all_probs).astype(np.float32)  # (N, 234)
    df = pd.DataFrame(probs_arr, columns=target_species)
    df.insert(0, "row_id", all_row_ids)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Wrote {len(df):,} rows × {len(df.columns)} cols → {out_csv}")

    # Quick distribution sanity check — helps catch generation bugs early
    print("\nSanity check on output:")
    max_probs = probs_arr.max(axis=1)
    print(f"  max_prob distribution: "
          f"p50={np.percentile(max_probs, 50):.3f}  "
          f"p90={np.percentile(max_probs, 90):.3f}  "
          f"p99={np.percentile(max_probs, 99):.3f}")
    pos_per_window = (probs_arr > 0.5).sum(axis=1)
    print(f"  species with prob>0.5 per window: "
          f"mean={pos_per_window.mean():.2f}  max={pos_per_window.max()}")


if __name__ == "__main__":
    main()