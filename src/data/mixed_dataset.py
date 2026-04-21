"""
BirdCLEF+ 2026 — Mixed Focal+Soundscape Dataset (Step 3)
=========================================================

Combines a focal dataset and a soundscape dataset at a configurable per-sample
mixing ratio. Used only for training; validation uses the component datasets
separately (one val loader per domain, plus one for the union).

Design: per-sample random mixing, not per-batch alternation.
  - Each __getitem__ call decides independently whether to pull from focal
    or soundscape, with probability `soundscape_ratio`.
  - This gives each batch a stochastic mix (batch of 32 at ratio=0.3 has
    E[soundscape]=9.6, range ~4-16), which is better for BatchNorm running
    stats than alternating pure-focal and pure-soundscape batches.

Oversampling semantics:
  - focal:  ~28k samples       soundscape: ~580 (fold 4 train = larger, fold 2 train = smaller)
  - At ratio=0.3, expected soundscape draws per epoch = 0.3 * len(focal) ≈ 8400.
  - Each unique soundscape window is thus drawn ~14x per epoch — real
    oversampling, no duplication in memory.
  - Epoch length stays at len(focal) so LR schedule / warmup / logging
    keep their focal-relative meaning.
"""

import numpy as np
from torch.utils.data import Dataset
from typing import Optional


class MixedDataset(Dataset):
    def __init__(
        self,
        focal_dataset: Dataset,
        soundscape_dataset: Dataset,
        soundscape_ratio: float,
        epoch_size: Optional[int] = None,
    ):
        if not 0.0 <= soundscape_ratio <= 1.0:
            raise ValueError(f"soundscape_ratio must be in [0, 1], got {soundscape_ratio}")
        if len(soundscape_dataset) == 0:
            raise ValueError("soundscape_dataset is empty — check folds assignment")

        self.focal = focal_dataset
        self.soundscape = soundscape_dataset
        self.ratio = soundscape_ratio
        self._len = epoch_size if epoch_size is not None else len(focal_dataset)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # Per-sample Bernoulli draw. Uses numpy (not torch) to match the
        # idiom used inside BirdCLEFDataset (np.random.random for mixup gate).
        if np.random.random() < self.ratio:
            sc_idx = np.random.randint(len(self.soundscape))
            return self.soundscape[sc_idx]
        # Focal path — modulo in case epoch_size > len(focal), though we
        # default to epoch_size == len(focal), so this is a no-op usually.
        focal_idx = idx % len(self.focal)
        return self.focal[focal_idx]

    def __repr__(self):
        return (f"MixedDataset(focal={len(self.focal)}, "
                f"soundscape={len(self.soundscape)}, "
                f"ratio={self.ratio}, epoch_size={self._len})")