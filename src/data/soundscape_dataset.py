"""
BirdCLEF+ 2026 — Soundscape Dataset (Step 3)
============================================

Reads per-window cached spectrograms from data/processed_soundscapes/
(produced by preprocess_soundscapes.py) and the accompanying folds CSV.

Differences vs BirdCLEFDataset:
  - One .npy per 5-second window (not per full file). No crop needed.
  - Labels are semicolon-separated multi-hot strings, not single primary_label.
  - No on-the-fly audio fallback; the cache is the sole source.
  - No mixup (kept to focal-focal interactions inside BirdCLEFDataset).

Returns (spec_tensor, label_tensor) in the same shape and dtype as
BirdCLEFDataset: (1, 128, 313) float32 and (234,) float32. This symmetry
is what lets MixedDataset compose the two transparently.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional

from src.data.augmentations import apply_spec_augmentations


class SoundscapeDataset(Dataset):
    TARGET_FRAMES = 313
    N_MELS = 128

    def __init__(
        self,
        folds_csv: str,
        fold: int,
        mode: str,
        spec_dir: str,
        target_species: list[str],
        aug_config: Optional[dict] = None,
    ):
        assert mode in ("train", "val"), f"mode must be 'train' or 'val', got {mode}"

        self.mode = mode
        self.spec_dir = spec_dir
        self.species_to_idx = {sp: i for i, sp in enumerate(target_species)}
        self.num_classes = len(target_species)
        self.aug_config = aug_config or {}

        df = pd.read_csv(folds_csv)
        # Match BirdCLEFDataset convention exactly:
        #   train = out-of-fold rows, val = in-fold rows
        mask = (df["fold"] != fold) if mode == "train" else (df["fold"] == fold)
        self.df = df[mask].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(
                f"SoundscapeDataset: no rows found for fold={fold}, mode={mode}. "
                f"Check that {folds_csv} contains fold values 0..4."
            )

    def __len__(self):
        return len(self.df)

    def _load_spec(self, row_id: str) -> np.ndarray:
        path = os.path.join(self.spec_dir, row_id + ".npy")
        spec = np.load(path)

        # Guard against edge cases (last window of file slightly short, etc.)
        # Pad or trim to exactly TARGET_FRAMES. Pad value = spec.min() to
        # match BirdCLEFDataset._crop's pad convention.
        n_frames = spec.shape[1]
        if n_frames == self.TARGET_FRAMES:
            return spec
        if n_frames > self.TARGET_FRAMES:
            # Deterministic center trim (we're working with fixed windows, not crops)
            start = (n_frames - self.TARGET_FRAMES) // 2
            return spec[:, start:start + self.TARGET_FRAMES]
        pad_width = self.TARGET_FRAMES - n_frames
        return np.pad(spec, ((0, 0), (0, pad_width)),
                      mode="constant", constant_values=spec.min())

    def _build_label(self, label_str) -> np.ndarray:
        """Parse semicolon-separated species codes into a multi-hot vector."""
        label = np.zeros(self.num_classes, dtype=np.float32)
        if not isinstance(label_str, str) or not label_str:
            return label
        for sp in label_str.split(";"):
            sp = sp.strip()
            if sp and sp in self.species_to_idx:
                label[self.species_to_idx[sp]] = 1.0
        return label

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spec = self._load_spec(row["row_id"])
        label = self._build_label(row["primary_label"])

        if self.mode == "train":
            spec = apply_spec_augmentations(spec, self.aug_config)

        spec_tensor = torch.from_numpy(spec).unsqueeze(0).float()  # (1, 128, 313)
        label_tensor = torch.from_numpy(label).float()             # (234,)
        return spec_tensor, label_tensor

    def __repr__(self):
        return (f"SoundscapeDataset(mode={self.mode}, fold_rows={len(self.df)}, "
                f"classes={self.num_classes})")