"""
BirdCLEF+ 2026 — Training Script
==================================

Single-fold training with configurable backbone, loss, and augmentations.

Usage:
    python src/training/train.py configs/experiment_v2_augment.yaml
    python src/training/train.py configs/experiment_v2_augment.yaml --fold 2

v2 changes:
  - Focal loss support (loss: focal in config)
  - Waveform augmentation config forwarded to dataset
  - Train soundscape integration
  - Augmentation config dict passed to dataset
"""

import sys
import os
import yaml
import time
import copy
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import timm

# Ensure project root is on path
sys.path.insert(0, os.getcwd())

from src.data.dataset import BirdCLEFDataset
from src.training.losses import build_loss
from src.models import build_model

# ===================================================================
# Model
# ===================================================================

class BirdCLEFModel(nn.Module):
    """timm backbone + classification head."""

    def __init__(self, backbone: str, num_classes: int, dropout: float, pretrained: bool):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.encoder.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


# ===================================================================
# Helpers
# ===================================================================

def get_target_species(cfg: dict) -> list[str]:
    """Derive the ordered species list from taxonomy / sample_submission / train CSV."""
    data_dir = Path(cfg["train_meta_csv"]).parent

    tax_path = data_dir / "taxonomy.csv"
    ss_path = data_dir / "sample_submission.csv"

    if tax_path.exists():
        tax = pd.read_csv(tax_path)
        col = "primary_label" if "primary_label" in tax.columns else tax.columns[0]
        return sorted(tax[col].unique().tolist())

    if ss_path.exists():
        ss = pd.read_csv(ss_path)
        return sorted(c for c in ss.columns if c not in ("row_id", "filename"))

    meta = pd.read_csv(cfg["train_meta_csv"])
    return sorted(meta["primary_label"].unique().tolist())


def build_aug_config(cfg: dict) -> dict:
    """Build augmentation config dict from YAML config.

    Maps both legacy keys and new v2 keys into a single dict
    that gets passed to the dataset and augmentation functions.
    """
    aug = {}

    # Legacy spectrogram augmentations
    if cfg.get("spec_augment", False):
        aug["aug_spec_p"] = cfg.get("aug_spec_p", 0.5)
        aug["time_mask_max"] = cfg.get("time_mask_max", 50)
        aug["freq_mask_max"] = cfg.get("freq_mask_max", 20)

    if cfg.get("gaussian_noise_std", 0) > 0:
        aug["aug_noise_p"] = cfg.get("aug_noise_p", 0.5)
        aug["gaussian_noise_std"] = cfg["gaussian_noise_std"]

    # v2: Waveform augmentations
    if cfg.get("bg_noise", False):
        aug["aug_bg_noise_p"] = cfg.get("aug_bg_noise_p", 0.5)
        aug["snr_min_db"] = cfg.get("snr_min_db", 5.0)
        aug["snr_max_db"] = cfg.get("snr_max_db", 20.0)

    if cfg.get("random_gain", False):
        aug["aug_gain_p"] = cfg.get("aug_gain_p", 0.5)
        aug["gain_range_db"] = cfg.get("gain_range_db", 6.0)

    if cfg.get("random_filter", False):
        aug["aug_filter_p"] = cfg.get("aug_filter_p", 0.4)

    if cfg.get("silence_embed", False):
        aug["aug_silence_p"] = cfg.get("aug_silence_p", 0.3)
        aug["window_samples"] = int(cfg.get("sample_rate", 32000) * cfg.get("window_seconds", 5.0))

    aug["sample_rate"] = cfg.get("sample_rate", 32000)
    aug["mixup_alpha"] = cfg.get("mixup_alpha", 0.4)

    return aug


def build_loaders(cfg: dict, target_species: list[str]):
    """Return train and val DataLoaders for the configured fold."""
    common = dict(
        folds_csv=cfg["folds_csv"],
        fold=cfg["fold"],
        spec_dir=cfg["spec_dir"],
        target_species=target_species,
        train_meta_csv=cfg["train_meta_csv"],
        audio_dir=cfg["audio_dir"],
    )

    # Build augmentation config
    aug_config = build_aug_config(cfg)

    # Augmentation probability flags
    aug_kwargs = {"aug_config": aug_config}
    if cfg.get("spec_augment", False):
        aug_kwargs["aug_spec_p"] = cfg.get("aug_spec_p", 0.5)
    if cfg.get("mixup_alpha", 0) > 0:
        aug_kwargs["aug_mixup_p"] = cfg.get("aug_mixup_p", 0.5)
    if cfg.get("gaussian_noise_std", 0) > 0:
        aug_kwargs["aug_noise_p"] = cfg.get("aug_noise_p", 0.5)

    # Soundscape support
    soundscape_kwargs = {}
    if cfg.get("soundscape_dir") and cfg.get("soundscape_labels_csv"):
        soundscape_kwargs["soundscape_dir"] = cfg["soundscape_dir"]
        soundscape_kwargs["soundscape_labels_csv"] = cfg["soundscape_labels_csv"]
        soundscape_kwargs["soundscape_ratio"] = cfg.get("soundscape_ratio", 0.3)

    ds_train = BirdCLEFDataset(mode="train", **common, **aug_kwargs, **soundscape_kwargs)
    ds_val = BirdCLEFDataset(mode="val", **common)

    loader_train = DataLoader(
        ds_train,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )
    return loader_train, loader_val, ds_val


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """CosineAnnealingLR with optional linear warmup."""
    total_epochs = cfg["epochs"]
    warmup_epochs = cfg.get("warmup_epochs", 0)

    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-7)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-7)


def compute_macro_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    """Compute macro-averaged ROC-AUC, skipping classes with no positives."""
    from sklearn.metrics import roc_auc_score

    aucs = []
    for i in range(labels.shape[1]):
        col_labels = labels[:, i]
        if col_labels.sum() > 0 and col_labels.sum() < len(col_labels):
            try:
                auc = roc_auc_score(col_labels, preds[:, i])
                aucs.append(auc)
            except ValueError:
                pass
    return np.mean(aucs) if aucs else 0.0


# ===================================================================
# Training loop
# ===================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (specs, labels) in enumerate(loader):
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits = model(specs)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device) -> tuple[float, np.ndarray, np.ndarray]:
    """Validate and return (loss, all_labels, all_preds)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    for specs, labels in loader:
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(specs)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_loss = total_loss / max(n_batches, 1)

    return avg_loss, all_labels, all_preds


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--fold", type=int, default=None, help="Override fold number")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.fold is not None:
        cfg["fold"] = args.fold

    # Seed
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", end="")
    if device.type == "cuda":
        print(f"  ({torch.cuda.get_device_name(0)})")
    else:
        print()

    # Data
    target_species = get_target_species(cfg)
    loader_train, loader_val, ds_val = build_loaders(cfg, target_species)
    print(f"Data: fold {cfg['fold']} — {len(loader_train.dataset)} train, "
          f"{len(loader_val.dataset)} val, {len(target_species)} classes")

    # Model
    model = build_model(cfg).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['backbone']}  params={n_params:,}")

    # Loss — now configurable
    criterion = build_loss(cfg)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    # Scheduler
    scheduler = build_scheduler(optimizer, cfg, len(loader_train))

    # AMP scaler
    use_amp = device.type == "cuda" and "convnext" not in cfg["backbone"].lower()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Experiment directory
    run_name = os.environ.get("BIRDCLEF_RUN_NAME") \
            or cfg.get("run_name") \
            or f"{cfg['backbone']}_fold{cfg['fold']}"
    exp_dir = Path("experiments") / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config copy
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Training log
    log_rows = []
    best_auc = 0.0
    best_epoch = -1

    print(f"\n{'='*60}")
    print(f"Training: {run_name}")
    print(f"{'='*60}\n")

    for epoch in range(cfg["epochs"]):
        t0 = time.time()

        train_loss = train_one_epoch(model, loader_train, criterion, optimizer, scaler, device)
        val_loss, val_labels, val_preds = validate(model, loader_val, criterion, device)
        val_auc = compute_macro_auc(val_labels, val_preds)

        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        log_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "lr": lr_now,
            "time_s": elapsed,
        }
        log_rows.append(log_row)

        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), exp_dir / "best_model.pt")
            # Save OOF predictions
            oof_df = pd.DataFrame(val_preds, columns=target_species)
            oof_df.insert(0, "filename",
                          ds_val.df["filename"].values[:len(oof_df)])
            oof_df.to_csv(exp_dir / "oof_preds.csv", index=False)

        marker = " ★ BEST" if is_best else ""
        print(f"  Epoch {epoch+1:>2}/{cfg['epochs']}: "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_auc={val_auc:.4f}  lr={lr_now:.2e}  "
              f"({elapsed:.0f}s){marker}")

    # Save training log
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(exp_dir / "training_log.csv", index=False)

    print(f"\n{'='*60}")
    print(f"Done: {run_name}")
    print(f"Best epoch: {best_epoch}  |  Best validation AUC: {best_auc:.4f}")
    print(f"Outputs saved to: {exp_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
