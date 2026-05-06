"""
scripts/build_ambience_pool.py

Build a real-ambience pool from the unlabeled soundscape spectrograms
for use by augmentations.add_real_ambience_spec.

Method:
  1. Load the existing 5-fold ensemble (the one that produced 0.883).
  2. Run inference over every .npy in unlabeled_soundscape_spec_dir.
  3. For each window, compute max_prob = max over species of avg sigmoid.
  4. Keep windows where max_prob < threshold (default 0.10) — the model
     thinks no species is present, so this is most likely pure ambience:
     insects, wind, water, distant traffic. The Pantanal background.
  5. Stack kept specs into one (N, n_mels, T) float32 array and save.

Usage:
    python scripts/build_ambience_pool.py configs/finetune_pseudo_v4.yaml

The config must contain:
    unlabeled_soundscape_spec_dir : path to *.npy specs
    ambience_pool_path            : output .npy path
    pseudo_source_checkpoint_pattern : 5-fold checkpoint pattern
Optional:
    ambience_max_prob_threshold   : float, default 0.10
    ambience_max_pool_size        : int,   default 30000 (cap for RAM safety)
"""

import sys
import os
import yaml
import time
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from src.models import build_model
from src.training.train import get_target_species

TARGET_FRAMES = 313
N_MELS = 128


def load_ensemble(cfg: dict, device) -> list:
    pattern = cfg.get(
        "pseudo_source_checkpoint_pattern",
        "experiments/sed_finetune/sed_finetune_fold{fold}/best_model.pt",
    )
    models = []
    for f in range(5):
        ckpt_path = pattern.format(fold=f)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Source checkpoint not found: {ckpt_path}")
        m = build_model(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        m.load_state_dict(state, strict=True)
        m.eval()
        models.append(m)
        print(f"  ✓ Loaded fold {f}: {ckpt_path}")
    return models


def load_and_pad_spec(path: str) -> np.ndarray:
    spec = np.load(path)
    n = spec.shape[1]
    if n == TARGET_FRAMES:
        return spec.astype(np.float32, copy=False)
    if n > TARGET_FRAMES:
        start = (n - TARGET_FRAMES) // 2
        return spec[:, start:start + TARGET_FRAMES].astype(np.float32, copy=False)
    return np.pad(
        spec, ((0, 0), (0, TARGET_FRAMES - n)),
        mode="constant", constant_values=spec.min(),
    ).astype(np.float32, copy=False)


@torch.no_grad()
def predict_batch(models, batch, use_amp):
    accum = None
    for m in models:
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = m(batch)
        probs = torch.sigmoid(logits.float())
        accum = probs if accum is None else accum + probs
    return (accum / len(models)).cpu().numpy()


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
        base.update(cfg)
        cfg = base

    for key in ("unlabeled_soundscape_spec_dir", "ambience_pool_path"):
        if key not in cfg:
            raise KeyError(f"Required config key missing: {key}")

    threshold = float(cfg.get("ambience_max_prob_threshold", 0.10))
    max_pool_size = int(cfg.get("ambience_max_pool_size", 30_000))
    batch_size = int(cfg.get("pseudo_inference_batch_size", 64))
    spec_dir = cfg["unlabeled_soundscape_spec_dir"]
    out_path = Path(cfg["ambience_pool_path"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  amp={use_amp}")
    print(f"Threshold: max_prob < {threshold}  (lower = stricter, smaller pool)")
    print(f"Pool size cap: {max_pool_size:,}")
    print(f"Output: {out_path}\n")

    target_species = get_target_species(cfg)
    print(f"Classes: {len(target_species)}\n")

    print("Loading 5-fold ensemble...")
    models = load_ensemble(cfg, device)

    print(f"\nScanning {spec_dir}...")
    spec_files = sorted(Path(spec_dir).glob("*.npy"))
    if not spec_files:
        raise RuntimeError(f"No .npy files in {spec_dir}")
    print(f"  {len(spec_files):,} unlabeled windows\n")

    kept_specs: list[np.ndarray] = []
    kept_paths: list[str] = []
    n_seen = 0
    n_kept = 0
    n_dropped = 0
    max_prob_log = []  # for distribution stats at end

    t0 = time.time()
    early_stop = False

    for start in range(0, len(spec_files), batch_size):
        chunk = spec_files[start:start + batch_size]
        specs_np = np.stack([load_and_pad_spec(str(p)) for p in chunk])  # (B, 128, T)
        batch = torch.from_numpy(specs_np).unsqueeze(1).float().to(device)

        probs = predict_batch(models, batch, use_amp)        # (B, 234)
        max_per_row = probs.max(axis=1)                       # (B,)
        max_prob_log.extend(max_per_row.tolist())

        for i in range(len(chunk)):
            n_seen += 1
            if max_per_row[i] < threshold:
                kept_specs.append(specs_np[i])
                kept_paths.append(str(chunk[i]))
                n_kept += 1
                if n_kept >= max_pool_size:
                    early_stop = True
                    break
            else:
                n_dropped += 1

        if (start // batch_size) % 20 == 0:
            elapsed = time.time() - t0
            rate = n_seen / max(elapsed, 1e-6)
            eta = (len(spec_files) - n_seen) / max(rate, 1e-6)
            print(f"  [{n_seen:>6}/{len(spec_files)}]  "
                  f"kept={n_kept}  dropped={n_dropped}  "
                  f"{rate:.0f} win/s  eta {eta/60:.1f}min")

        if early_stop:
            print(f"\n  Reached pool size cap ({max_pool_size:,}). Stopping early.")
            break

    elapsed = time.time() - t0
    print(f"\nScan done in {elapsed/60:.1f}min")
    print(f"  Seen   : {n_seen:,}")
    print(f"  Kept   : {n_kept:,}")
    print(f"  Dropped: {n_dropped:,}")

    if n_kept == 0:
        raise RuntimeError(
            f"No windows passed the ambience filter (max_prob < {threshold}).\n"
            f"Either raise the threshold or check that the ensemble checkpoints "
            f"and the unlabeled spec dir match the dataset you intend."
        )

    # Distribution sanity check
    mp = np.array(max_prob_log)
    print("\nMax-prob distribution across the unlabeled set:")
    for q in (10, 25, 50, 75, 90, 99):
        print(f"  p{q:>2}: {np.percentile(mp, q):.4f}")
    print(f"  (kept everything below {threshold})")

    # Stack & save
    pool = np.stack(kept_specs).astype(np.float32)
    print(f"\nPool array shape: {pool.shape}  dtype={pool.dtype}")
    print(f"Estimated size:   {pool.nbytes / 1e9:.2f} GB")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, pool)
    print(f"✓ Wrote {out_path}")

    # Also write a sidecar with the file paths used, for debugging
    sidecar = out_path.with_suffix(".sources.txt")
    with open(sidecar, "w") as f:
        for p in kept_paths:
            f.write(p + "\n")
    print(f"✓ Wrote source list -> {sidecar}")

    # Final spec-stats sanity (helps confirm Pantanal-vs-other domain)
    print("\nKept-pool spectrogram statistics (sanity check vs. focal stats):")
    print(f"  min:  {pool.min():.2f}  (expected ~ -80 for log-mel)")
    print(f"  max:  {pool.max():.2f}  (expected ~ 0)")
    print(f"  mean: {pool.mean():.2f}")
    print(f"  std:  {pool.std():.2f}")


if __name__ == "__main__":
    main()
