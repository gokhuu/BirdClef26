"""
BirdCLEF+ 2026 — Pseudo-labeled Soundscape Dataset
==================================================

Reads soft labels (per-class probabilities in [0, 1]) from a CSV produced by
generate_pseudo_labels.py, alongside cached spectrograms in spec_dir.

Mirrors SoundscapeDataset's interface so MixedDataset can compose them
transparently. Differences:
  - Labels are floats from CSV columns, not parsed from a primary_label string.
  - No fold split — pseudo-labels are unlabeled-source data, used train-only.
  - Optional label smoothing pulls soft targets toward 0.5 (regularization).

CSV schema (produced by generate_pseudo_labels.py):
    row_id,<species_0>,<species_1>,...,<species_233>
where row_id matches the .npy filename in spec_dir (without extension).

Returns (spec_tensor, label_tensor) in the same shape and dtype as
SoundscapeDataset: (1, 128, 313) float32 and (234,) float32.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional

from src.data.augmentations import apply_spec_augmentations


class PseudoSoundscapeDataset(Dataset):
    TARGET_FRAMES = 313
    N_MELS = 128

    def __init__(
        self,
        pseudo_csv: str,
        spec_dir: str,
        target_species: list[str],
        aug_config: Optional[dict] = None,
        label_smoothing: float = 0.0,
    ):
        if not 0.0 <= label_smoothing <= 0.5:
            raise ValueError(
                f"label_smoothing must be in [0, 0.5], got {label_smoothing}"
            )

        self.spec_dir = spec_dir
        self.num_classes = len(target_species)
        self.aug_config = aug_config or {}
        self.label_smoothing = label_smoothing

        df = pd.read_csv(pseudo_csv)

        missing = [sp for sp in target_species if sp not in df.columns]
        if missing:
            raise ValueError(
                f"PseudoSoundscapeDataset: {len(missing)} species missing from "
                f"{pseudo_csv}. First few: {missing[:5]}"
            )
        if "row_id" not in df.columns:
            raise ValueError(f"PseudoSoundscapeDataset: 'row_id' column missing from {pseudo_csv}")
        if len(df) == 0:
            raise ValueError(f"PseudoSoundscapeDataset: {pseudo_csv} is empty")

        # Pre-extract once — avoids per-sample pandas indexing during training.
        self.row_ids = df["row_id"].values
        self.labels  = df[target_species].values.astype(np.float32)  # (N, 234)

        # Sanity-clip in case any prob was written slightly outside [0, 1]
        # (e.g. from sharpening rounding); BCE handles 0/1 fine but anything
        # outside that range is undefined behavior for the loss.
        np.clip(self.labels, 0.0, 1.0, out=self.labels)

    def __len__(self):
        return len(self.row_ids)

    def _load_spec(self, row_id: str) -> np.ndarray:
        path = os.path.join(self.spec_dir, row_id + ".npy")
        spec = np.load(path)

        # Same pad/trim convention as SoundscapeDataset for shape safety.
        n_frames = spec.shape[1]
        if n_frames == self.TARGET_FRAMES:
            return spec
        if n_frames > self.TARGET_FRAMES:
            start = (n_frames - self.TARGET_FRAMES) // 2
            return spec[:, start:start + self.TARGET_FRAMES]
        pad_width = self.TARGET_FRAMES - n_frames
        return np.pad(spec, ((0, 0), (0, pad_width)),
                      mode="constant", constant_values=spec.min())

    def __getitem__(self, idx):
        spec = self._load_spec(self.row_ids[idx])
        label = self.labels[idx].copy()

        if self.label_smoothing > 0:
            label = label * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Always apply augmentations — pseudo-data is train-only.
        spec = apply_spec_augmentations(spec, self.aug_config)

        spec_tensor  = torch.from_numpy(spec).unsqueeze(0).float()  # (1, 128, 313)
        label_tensor = torch.from_numpy(label).float()              # (234,)
        return spec_tensor, label_tensor

    def __repr__(self):
        return (f"PseudoSoundscapeDataset(rows={len(self.row_ids)}, "
                f"classes={self.num_classes}, label_smoothing={self.label_smoothing})")