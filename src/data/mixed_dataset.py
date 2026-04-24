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


# src/data/mixed_dataset.py
class MixedDataset(Dataset):
    def __init__(self, datasets, ratios, epoch_size=None):
        if len(datasets) != len(ratios):
            raise ValueError(f"got {len(datasets)} datasets, {len(ratios)} ratios")
        if not abs(sum(ratios) - 1.0) < 1e-6:
            raise ValueError(f"ratios must sum to 1, got {sum(ratios)}")
        if any(len(d) == 0 for d in datasets):
            raise ValueError("one of the datasets is empty")
        
        self.datasets = list(datasets)
        self.ratios = np.asarray(ratios, dtype=np.float64)
        self._cum = np.cumsum(self.ratios)
        self._len = epoch_size if epoch_size is not None else len(datasets[0])
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        which = int(np.searchsorted(self._cum, np.random.random()))
        ds = self.datasets[which]
        if which == 0:                          # focal anchors the index
            return ds[idx % len(ds)]
        return ds[np.random.randint(len(ds))]   # others sampled uniformly