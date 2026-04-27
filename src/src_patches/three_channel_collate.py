"""
Optional: 3-channel mel-spec input for EfficientNet-V2-S.

Background: timm averages RGB conv stem weights when you pass `in_chans=1`.
This works but typically underperforms the 3-channel-repeat approach by
~0.002-0.005 macro AUC on BirdCLEF-style tasks. With `in21k_ft_in1k` pretrained
weights the 3-channel path seems to matter slightly more because the stem
was trained on very rich ImageNet-21k color distribution.

TWO WAYS TO USE THIS:

Option A — Repeat at dataset/collate time (recommended, no model change):
    Set encoder_in_chans=3 in your config, and use the collate_fn below in your
    DataLoader. The spec arrays on disk remain 1-channel; repetition happens at load.

Option B — Let timm do it:
    Set encoder_in_chans=1 as normal. timm handles single-channel input internally.
    Simpler, no code changes, but ~0.002-0.005 AUC weaker. Fine for the first run.

START WITH OPTION B. Only switch to Option A if your focal val AUC plateaus
below 0.90 and you're looking for a small boost. Don't bake in the complexity
before you know you need it.
"""

from typing import Sequence, Tuple

import torch


def three_channel_collate(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Drop-in replacement for your default collate if using Option A above.
    Assumes each batch item is (spec, label) where spec has shape (1, n_mels, n_frames).

    Usage:
        loader = DataLoader(dataset, batch_size=32, collate_fn=three_channel_collate, ...)
    """
    specs, labels = zip(*batch)
    specs = torch.stack(specs, dim=0)      # (B, 1, mels, frames)

    if specs.shape[1] == 1:
        specs = specs.repeat(1, 3, 1, 1)   # (B, 3, mels, frames)
    elif specs.shape[1] == 3:
        pass                                # already 3-channel
    else:
        raise ValueError(f"Unexpected channel count: {specs.shape[1]}")

    labels = torch.stack(labels, dim=0)
    return specs, labels


# If you're feeling fancy: different per-channel augmentations can actually help.
# E.g. channel 0 = raw spec, channel 1 = spec + small time shift, channel 2 = spec + small freq shift.
# Skip this — it's a 0.001 trick and adds complexity. The simple repeat is fine.
