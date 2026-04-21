"""
BirdCLEF+ 2026 — PyTorch Dataset
=================================

Supports:
  - Cached .npy spectrograms from Xeno-canto / iNat focal recordings
  - On-the-fly spectrogram computation with waveform augmentations (v2)
  - Train soundscape windows (Part B) if available
  - Configurable mixing ratio between focal and soundscape data
  - Multi-label targets for soundscape windows

v2 changes:
  - Waveform augmentations applied BEFORE spectrogram in train mode
  - Augmentation config passed as dict, not individual kwargs
  - Optional train soundscape integration
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional

from src.data.augmentations import (
    apply_spec_augmentations,
    apply_mixup,
)

class BirdCLEFDataset(Dataset):
    """Dataset for BirdCLEF+ 2026 with domain-shift augmentation support.

    In train mode:
      1. Load cached spectrogram OR load raw audio → apply waveform augs → compute spec
      2. Random 5s crop
      3. Apply spectrogram augmentations
      4. Optionally mixup with another sample

    In val mode:
      1. Load cached spectrogram
      2. Center 5s crop
      3. No augmentations
    """

    # Standard spectrogram dimensions
    TARGET_FRAMES = 313  # ~5s at hop=512, sr=32000
    N_MELS = 128

    def __init__(
        self,
        folds_csv: str,
        fold: int,
        mode: str,
        spec_dir: str,
        target_species: list[str],
        train_meta_csv: str = "data/raw/train.csv",
        audio_dir: str = "data/raw/train_audio",
        # Augmentation probabilities (train mode only)
        aug_spec_p: float = 0.0,
        aug_mixup_p: float = 0.0,
        aug_noise_p: float = 0.0,
        # v2: augmentation config dict for new augmentations
        aug_config: Optional[dict] = None,
        # v2: train soundscape support
        soundscape_dir: Optional[str] = None,
        soundscape_labels_csv: Optional[str] = None,
        soundscape_ratio: float = 0.0,  # fraction of batch from soundscapes
    ):
        assert mode in ("train", "val"), f"mode must be 'train' or 'val', got {mode}"

        self.mode = mode
        self.spec_dir = spec_dir
        self.audio_dir = audio_dir
        self.target_species = target_species
        self.species_to_idx = {sp: i for i, sp in enumerate(target_species)}
        self.num_classes = len(target_species)

        # Build augmentation config
        self.aug_config = aug_config or {}
        # Merge legacy probability flags into config
        if aug_spec_p > 0:
            self.aug_config.setdefault("aug_spec_p", aug_spec_p)
        if aug_noise_p > 0:
            self.aug_config.setdefault("aug_noise_p", aug_noise_p)
        self.aug_mixup_p = aug_mixup_p if mode == "train" else 0.0

        # Load fold data
        folds_df = pd.read_csv(folds_csv)
        meta = pd.read_csv(train_meta_csv)

        if mode == "train":
            fold_mask = folds_df["fold"] != fold
        else:
            fold_mask = folds_df["fold"] == fold

        fold_files = set(folds_df.loc[fold_mask, "filename"].tolist())
        self.df = meta[meta["filename"].isin(fold_files)].reset_index(drop=True)

        # v2: Load train soundscapes if available
        self.soundscape_samples = []
        self.soundscape_ratio = soundscape_ratio if mode == "train" else 0.0

        if (mode == "train" and soundscape_dir and soundscape_labels_csv
                and os.path.exists(soundscape_dir) and os.path.exists(soundscape_labels_csv)):
            self._load_soundscapes(soundscape_dir, soundscape_labels_csv)

    def _load_soundscapes(self, soundscape_dir: str, labels_csv: str):
        """Load train soundscape windows and their multi-label annotations."""
        labels_df = pd.read_csv(labels_csv)

        # Expected format: row_id, species columns (or 'species' + 'start_time' etc.)
        # Handle common BirdCLEF soundscape label formats
        if "row_id" in labels_df.columns:
            species_cols = [c for c in labels_df.columns
                           if c not in ("row_id", "filename", "soundscape_id",
                                        "start_time", "end_time", "seconds")]
            for _, row in labels_df.iterrows():
                label_vec = np.zeros(self.num_classes, dtype=np.float32)
                for sp in species_cols:
                    if sp in self.species_to_idx and row.get(sp, 0) > 0:
                        label_vec[self.species_to_idx[sp]] = 1.0

                if label_vec.sum() > 0:  # skip windows with no target species
                    self.soundscape_samples.append({
                        "row_id": row["row_id"],
                        "label": label_vec,
                        "dir": soundscape_dir,
                    })
        elif "species" in labels_df.columns:
            # Alternative format: one row per species per window
            grouped = labels_df.groupby(
                [c for c in ["filename", "start_time", "end_time"]
                 if c in labels_df.columns]
            )
            for group_key, group_df in grouped:
                label_vec = np.zeros(self.num_classes, dtype=np.float32)
                for _, row in group_df.iterrows():
                    sp = row["species"]
                    if sp in self.species_to_idx:
                        label_vec[self.species_to_idx[sp]] = 1.0

                if label_vec.sum() > 0:
                    self.soundscape_samples.append({
                        "filename": group_key[0] if isinstance(group_key, tuple) else group_key,
                        "start_time": group_key[1] if isinstance(group_key, tuple) and len(group_key) > 1 else 0,
                        "label": label_vec,
                        "dir": soundscape_dir,
                    })

        if self.soundscape_samples:
            print(f"  Loaded {len(self.soundscape_samples)} soundscape windows")

    def __len__(self):
        n = len(self.df)
        if self.soundscape_ratio > 0 and self.soundscape_samples:
            # Inflate length to account for soundscape samples
            n = int(n / (1.0 - self.soundscape_ratio))
        return n

    def _load_spec(self, filename: str) -> np.ndarray:
        """Load cached spectrogram or compute on-the-fly with waveform augmentations."""
        base = os.path.splitext(filename)[0]
        cache_path = os.path.join(self.spec_dir, base + ".npy")

        # Fall back to cached spectrogram
        if os.path.exists(cache_path):
            return np.load(cache_path)

        # Last resort: compute on-the-fly without augmentations
        audio_path = os.path.join(self.audio_dir, filename)
        if os.path.exists(audio_path):
            waveform = load_audio(audio_path)
            return compute_melspec_from_waveform(waveform)

        raise FileNotFoundError(
            f"No cached spectrogram at {cache_path} and no audio fallback for {filename}"
        )

    def _crop(self, spec: np.ndarray) -> np.ndarray:
        """Crop spectrogram to TARGET_FRAMES along time axis."""
        n_frames = spec.shape[1]

        if n_frames >= self.TARGET_FRAMES:
            if self.mode == "train":
                start = np.random.randint(0, n_frames - self.TARGET_FRAMES + 1)
            else:
                start = (n_frames - self.TARGET_FRAMES) // 2
            return spec[:, start:start + self.TARGET_FRAMES]
        else:
            # Pad with minimum value
            pad_width = self.TARGET_FRAMES - n_frames
            return np.pad(spec, ((0, 0), (0, pad_width)),
                          mode="constant", constant_values=spec.min())

    def _get_label(self, row) -> np.ndarray:
        """Build label vector from metadata row."""
        label = np.zeros(self.num_classes, dtype=np.float32)

        primary = row.get("primary_label", "")
        if primary in self.species_to_idx:
            label[self.species_to_idx[primary]] = 1.0

        # Handle secondary labels
        secondary = row.get("secondary_labels", "")
        if isinstance(secondary, str) and secondary not in ("", "[]"):
            # Parse list format: "['sp1', 'sp2']" or "sp1;sp2"
            secondary = secondary.strip("[]' ").replace("'", "").replace('"', '')
            for sp in secondary.split(","):
                sp = sp.strip().strip("'\" ")
                if sp and sp in self.species_to_idx:
                    label[self.species_to_idx[sp]] = 1.0

        return label

    def _get_soundscape_item(self, sc_idx: int):
        """Load a train soundscape window."""
        sample = self.soundscape_samples[sc_idx]
        label = sample["label"]

        # Try to load audio
        if "row_id" in sample:
            # Find the audio file
            row_id = sample["row_id"]
            parts = row_id.rsplit("_", 1)
            soundscape_id = parts[0]
            end_sec = int(parts[1]) if len(parts) > 1 else 5
            start_sec = end_sec - 5

            # Search for the audio file
            audio_path = None
            for ext in (".ogg", ".wav", ".flac", ".mp3"):
                candidate = os.path.join(sample["dir"], soundscape_id + ext)
                if os.path.exists(candidate):
                    audio_path = candidate
                    break

            if audio_path:
                waveform = load_audio(audio_path, offset=start_sec, duration=5.0)
                waveform = apply_waveform_augmentations(waveform, self.aug_config)
                spec = compute_melspec_from_waveform(waveform)
            else:
                # Fallback: zeros
                spec = np.full((self.N_MELS, self.TARGET_FRAMES), -80.0, dtype=np.float32)
        else:
            filename = sample.get("filename", "")
            start_time = sample.get("start_time", 0)
            audio_path = os.path.join(sample["dir"], filename)

            if os.path.exists(audio_path):
                waveform = load_audio(audio_path, offset=start_time, duration=5.0)
                waveform = apply_waveform_augmentations(waveform, self.aug_config)
                spec = compute_melspec_from_waveform(waveform)
            else:
                spec = np.full((self.N_MELS, self.TARGET_FRAMES), -80.0, dtype=np.float32)

        spec = self._crop(spec)
        return spec, label

    def __getitem__(self, idx):
        # Decide whether to return a soundscape sample
        if (self.soundscape_ratio > 0 and self.soundscape_samples
                and np.random.random() < self.soundscape_ratio):
            sc_idx = np.random.randint(len(self.soundscape_samples))
            spec, label = self._get_soundscape_item(sc_idx)

            # Apply spec augmentations
            if self.mode == "train":
                spec = apply_spec_augmentations(spec, self.aug_config)

            spec_tensor = torch.from_numpy(spec).unsqueeze(0).float()
            label_tensor = torch.from_numpy(label).float()
            return spec_tensor, label_tensor

        # Normal focal recording sample
        actual_idx = idx % len(self.df)
        row = self.df.iloc[actual_idx]

        spec = self._load_spec(row["filename"])
        spec = self._crop(spec)
        label = self._get_label(row)

        # Spectrogram-domain augmentations (train mode only)
        if self.mode == "train":
            spec = apply_spec_augmentations(spec, self.aug_config)

        # Mixup (train mode only)
        if self.mode == "train" and np.random.random() < self.aug_mixup_p:
            mix_idx = np.random.randint(len(self.df))
            mix_row = self.df.iloc[mix_idx]
            spec2 = self._load_spec(mix_row["filename"])
            spec2 = self._crop(spec2)
            label2 = self._get_label(mix_row)

            if self.mode == "train":
                spec2 = apply_spec_augmentations(spec2, self.aug_config)

            alpha = self.aug_config.get("mixup_alpha", 0.4)
            spec, label = apply_mixup(spec, label, spec2, label2, alpha=alpha)

        spec_tensor = torch.from_numpy(spec).unsqueeze(0).float()  # (1, n_mels, T)
        label_tensor = torch.from_numpy(label).float()

        return spec_tensor, label_tensor

    def __repr__(self):
        sc_info = ""
        if self.soundscape_samples:
            sc_info = f", soundscapes={len(self.soundscape_samples)}"
        return (f"BirdCLEFDataset(mode={self.mode}, fold samples={len(self.df)}, "
                f"classes={self.num_classes}{sc_info})")
