"""
BirdCLEF+ 2026 — Fine-tune Training Script (Step 4)
====================================================

Warm-starts from a focal-trained SED checkpoint and continues training on a
mixed focal+soundscape batch distribution. Reports three separate validation
AUCs per epoch (focal-only, soundscape-only, combined) so we can detect
catastrophic forgetting while measuring soundscape adaptation.

Artifacts written to experiments/{run_name}/:
  - best_model.pt             (selected on combined val AUC)
  - config.yaml               (dump of the fine-tune config)
  - training_log.csv          (per-epoch losses/AUCs for all three val sets)
  - oof_preds.csv             (focal val OOF, same shape as sed_b0)
  - oof_preds_soundscape.csv  (soundscape val OOF, new artifact)

Usage:
    python src/training/finetune.py configs/experiment_sed_b0_finetune.yaml
    python src/training/finetune.py configs/experiment_sed_b0_finetune.yaml --fold 2
    python src/training/finetune.py configs/experiment_sed_b0_finetune.yaml \
        --fold 2 --init-checkpoint experiments/sed_b0_fold2/best_model.pt
    python src/training/finetune.py configs/experiment_sed_b0_finetune.yaml \
        --fold 2 --seed 123      # for seed ensembling

v2.1 changes (seed ensembling support):
  - --seed CLI flag overrides config seed
  - random.seed() added (covers any stdlib random usage)
  - DataLoader workers seeded via worker_init_fn (imported from train.py)
  - Train DataLoader shuffle uses a seeded torch.Generator
"""

import sys
import os
import yaml
import time
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

# Project root on path
sys.path.insert(0, os.getcwd())

from src.data.dataset import BirdCLEFDataset
from src.data.soundscape_dataset import SoundscapeDataset
from src.data.mixed_dataset import MixedDataset
from src.data.pseudo_soundscape_dataset import PseudoSoundscapeDataset
from src.training.losses import build_loss
from src.models import build_model

# Reuse proven helpers from train.py rather than copying them.
from src.training.train import (
    get_target_species,
    build_aug_config,
    build_scheduler,
    compute_macro_auc,
    train_one_epoch,
    validate,
    seed_worker,
)


# ===================================================================
# Data loaders — focal train (mixed) + 3-way val
# ===================================================================

def build_finetune_loaders(cfg: dict, target_species: list[str],
                           generator: torch.Generator | None = None):
    """Return: train_loader, (focal_val_ds, focal_val_loader),
                              (sc_val_ds,    sc_val_loader),
                              combined_val_loader

    Train: MixedDataset over focal + labeled soundscape (+ pseudo-labeled
           soundscape, if `pseudo_labels_csv` is set in cfg).
    Val: three independent loaders — focal only, soundscape only, and
         the concatenation. Pseudo-labels never enter validation.

    The optional `generator` is used by the train loader's shuffle so that
    batch ordering is deterministic given a fixed seed. Workers (train and
    val) are seeded via worker_init_fn imported from train.py.
    """
    aug_config = build_aug_config(cfg)

    # --- Focal datasets (existing class, unchanged) ---
    focal_common = dict(
        folds_csv=cfg["folds_csv"],
        fold=cfg["fold"],
        spec_dir=cfg["spec_dir"],
        target_species=target_species,
        train_meta_csv=cfg["train_meta_csv"],
        audio_dir=cfg["audio_dir"],
    )
    focal_aug = {"aug_config": aug_config}
    if cfg.get("spec_augment", False):
        focal_aug["aug_spec_p"] = cfg.get("aug_spec_p", 0.5)
    if cfg.get("mixup_alpha", 0) > 0:
        focal_aug["aug_mixup_p"] = cfg.get("aug_mixup_p", 0.5)
    if cfg.get("gaussian_noise_std", 0) > 0:
        focal_aug["aug_noise_p"] = cfg.get("aug_noise_p", 0.5)

    focal_train = BirdCLEFDataset(mode="train", **focal_common, **focal_aug)
    focal_val   = BirdCLEFDataset(mode="val",   **focal_common)

    # --- Labeled soundscape datasets ---
    soundscape_train = SoundscapeDataset(
        folds_csv=cfg["soundscape_folds_csv"],
        fold=cfg["fold"],
        mode="train",
        spec_dir=cfg["soundscape_spec_dir"],
        target_species=target_species,
        aug_config=aug_config,
    )
    soundscape_val = SoundscapeDataset(
        folds_csv=cfg["soundscape_folds_csv"],
        fold=cfg["fold"],
        mode="val",
        spec_dir=cfg["soundscape_spec_dir"],
        target_species=target_species,
    )

    # --- Mixed train dataset (focal + sc [+ pseudo]) ---
    sc_ratio = cfg.get("soundscape_ratio", 0.3)
    sources = [focal_train, soundscape_train]
    ratios  = [1.0 - sc_ratio, sc_ratio]
    pseudo_train = None

    pseudo_csv = cfg.get("pseudo_labels_csv")
    if pseudo_csv:
        if not os.path.exists(pseudo_csv):
            raise FileNotFoundError(
                f"pseudo_labels_csv set but not found: {pseudo_csv}"
            )
        pseudo_train = PseudoSoundscapeDataset(
            pseudo_csv=pseudo_csv,
            spec_dir=cfg.get("unlabeled_soundscape_spec_dir", cfg["soundscape_spec_dir"]),
            target_species=target_species,
            aug_config=aug_config,
            label_smoothing=cfg.get("pseudo_label_smoothing", 0.0),
        )
        pseudo_ratio = cfg.get("pseudo_ratio", 0.3)
        if not 0.0 < pseudo_ratio < 1.0:
            raise ValueError(f"pseudo_ratio must be in (0, 1), got {pseudo_ratio}")
        # Carve pseudo_ratio out of the total; preserve focal:sc proportion
        keep = 1.0 - pseudo_ratio
        ratios = [r * keep for r in ratios] + [pseudo_ratio]
        sources.append(pseudo_train)

    mixed_train = MixedDataset(sources, ratios)

    # --- Loaders ---
    bs = cfg["batch_size"]
    nw = cfg.get("num_workers", 4)

    train_loader = DataLoader(
        mixed_train,
        batch_size=bs, shuffle=True, num_workers=nw,
        pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    focal_val_loader = DataLoader(
        focal_val,
        batch_size=bs * 2, shuffle=False, num_workers=nw, pin_memory=True,
        worker_init_fn=seed_worker,
    )
    sc_val_loader = DataLoader(
        soundscape_val,
        batch_size=bs * 2, shuffle=False, num_workers=nw, pin_memory=True,
        worker_init_fn=seed_worker,
    )
    combined_val_loader = DataLoader(
        ConcatDataset([focal_val, soundscape_val]),
        batch_size=bs * 2, shuffle=False, num_workers=nw, pin_memory=True,
        worker_init_fn=seed_worker,
    )

    print(f"  focal train: {len(focal_train)}  | soundscape train: {len(soundscape_train)}")
    if pseudo_train is not None:
        print(f"  pseudo train: {len(pseudo_train)}")
    print(f"  focal val:   {len(focal_val)}    | soundscape val:   {len(soundscape_val)}")
    print(f"  mixed epoch length: {len(mixed_train)}  ratios={[round(r, 3) for r in ratios]}")

    return (train_loader,
            focal_val, focal_val_loader,
            soundscape_val, sc_val_loader,
            combined_val_loader)


# ===================================================================
# Checkpoint loading
# ===================================================================

def load_checkpoint_strict(model: nn.Module, checkpoint_path: str, device):
    """Load a state dict with strict=True and print a brief summary.

    strict=True means any architecture mismatch (missing keys, unexpected
    keys, shape mismatches) raises immediately. This is the behavior we
    want for fine-tuning: silent partial loads are the worst failure
    mode because they look like successful warm-starts while actually
    reinitializing part of the model.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"init_checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Tolerate 'module.' prefix if someone ever trains with DataParallel
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Loaded {checkpoint_path}")
    print(f"    {len(state)} tensors, {n_params:,} parameters, strict=True")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument("--fold", type=int, default=None, help="Override fold number")
    parser.add_argument("--init-checkpoint", default=None,
                        help="Override init_checkpoint path (e.g. for different fold)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed from config (use for seed ensembling)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base_path = cfg.pop("base_config", None)
    if base_path:
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(cfg)   # child overrides parent
        cfg = base

    if args.fold is not None:
        cfg["fold"] = args.fold
    if args.init_checkpoint is not None:
        cfg["init_checkpoint"] = args.init_checkpoint
    if args.seed is not None:
        cfg["seed"] = args.seed

    # Seed everything
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Generator for train DataLoader shuffle (deterministic batch ordering per seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    print(f"Seed: {seed}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", end="")
    if device.type == "cuda":
        print(f"  ({torch.cuda.get_device_name(0)})")
    else:
        print()

    # Data
    target_species = get_target_species(cfg)
    (train_loader,
     focal_val_ds, focal_val_loader,
     sc_val_ds, sc_val_loader,
     combined_val_loader) = build_finetune_loaders(cfg, target_species, generator=loader_generator)
    print(f"Data: fold {cfg['fold']}  classes={len(target_species)}")

    # Model — architecture from cfg, weights from init_checkpoint
    model = build_model(cfg).to(device)
    load_checkpoint_strict(model, cfg["init_checkpoint"], device)

    # Loss / optimizer / scheduler — fresh, not resumed
    criterion = build_loss(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    # AMP
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Experiment dir
    base_name = cfg.get("experiment_name", "sed_finetune")
    run_name = os.environ.get("BIRDCLEF_RUN_NAME") \
            or f"{base_name}_fold{cfg['fold']}"
    exp_dir = Path("experiments") / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    log_rows = []
    best_comb_auc = 0.0
    best_epoch = -1

    print(f"\n{'='*64}")
    print(f"Fine-tune: {run_name}")
    print(f"{'='*64}\n")

    for epoch in range(cfg["epochs"]):
        t0 = time.time()

        # Train on mixed focal+soundscape (reuse train.py's loop)
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
        )

        # Three-way validation
        focal_loss, focal_labels, focal_preds = validate(
            model, focal_val_loader, criterion, device,
        )
        sc_loss, sc_labels, sc_preds = validate(
            model, sc_val_loader, criterion, device,
        )
        comb_loss, comb_labels, comb_preds = validate(
            model, combined_val_loader, criterion, device,
        )

        focal_auc = compute_macro_auc(focal_labels, focal_preds)
        sc_auc    = compute_macro_auc(sc_labels, sc_preds)
        comb_auc  = compute_macro_auc(comb_labels, comb_preds)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        log_rows.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "focal_val_loss": focal_loss,
            "focal_val_auc":  focal_auc,
            "soundscape_val_loss": sc_loss,
            "soundscape_val_auc":  sc_auc,
            "combined_val_loss": comb_loss,
            "combined_val_auc":  comb_auc,
            "lr": lr_now,
            "time_s": elapsed,
        })

        is_best = comb_auc > best_comb_auc
        if is_best:
            best_comb_auc = comb_auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), exp_dir / "best_model.pt")

            # Focal OOF (matches sed_b0 convention)
            focal_oof = pd.DataFrame(focal_preds, columns=target_species)
            focal_oof.insert(0, "filename",
                             focal_val_ds.df["filename"].values[:len(focal_oof)])
            focal_oof.to_csv(exp_dir / "oof_preds.csv", index=False)

            # Soundscape OOF (new artifact, for separate inspection)
            sc_oof = pd.DataFrame(sc_preds, columns=target_species)
            sc_oof.insert(0, "row_id",
                          sc_val_ds.df["row_id"].values[:len(sc_oof)])
            sc_oof.to_csv(exp_dir / "oof_preds_soundscape.csv", index=False)

        marker = " ★ BEST" if is_best else ""
        print(
            f"  Epoch {epoch+1:>2}/{cfg['epochs']}: "
            f"train={train_loss:.4f}  "
            f"focal_auc={focal_auc:.4f}  "
            f"sc_auc={sc_auc:.4f}  "
            f"comb_auc={comb_auc:.4f}  "
            f"lr={lr_now:.2e}  ({elapsed:.0f}s){marker}"
        )

    pd.DataFrame(log_rows).to_csv(exp_dir / "training_log.csv", index=False)

    print(f"\n{'='*64}")
    print(f"Done: {run_name}")
    print(f"Best epoch: {best_epoch}  |  best combined val AUC: {best_comb_auc:.4f}")
    print(f"Outputs: {exp_dir}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
