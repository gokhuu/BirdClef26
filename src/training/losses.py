"""
BirdCLEF+ 2026 — Loss Functions
================================

Configurable loss functions for training:
  - BCE (binary cross-entropy with logits) — baseline
  - Focal loss — downweights easy/confident predictions, focuses on hard
    examples. Helps with rare species and noisy negatives.

Both operate on multi-label logits (no sigmoid applied beforehand).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Sigmoid focal loss for multi-label classification.

    Focal loss = -alpha * (1 - p_t)^gamma * log(p_t)

    where p_t = sigmoid(logit) if target=1, else 1-sigmoid(logit).

    This downweights well-classified examples (high p_t) and focuses
    the model on hard/ambiguous cases — particularly helpful when:
    - Some species are rare and easily overwhelmed by easy negatives
    - Soundscape recordings have noisy labels
    - There's severe class imbalance (234 classes, most negative)

    Args:
        gamma: Focusing parameter (default 2.0). Higher = more focus on hard.
        alpha: Weighting for positive class (default 0.25).
        reduction: 'mean' (default), 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (B, C), no sigmoid applied.
            targets: Multi-label target (B, C), values in [0, 1].

        Returns:
            Scalar loss (if reduction='mean' or 'sum') or per-element loss.
        """
        # Numerically stable focal loss using logsigmoid
        # BCE = -target * log(sigmoid(x)) - (1 - target) * log(1 - sigmoid(x))
        # Use log_sigmoid for numerical stability
        log_p = F.logsigmoid(logits)          # log(sigmoid(x))
        log_1_minus_p = F.logsigmoid(-logits)  # log(1 - sigmoid(x))

        p = torch.sigmoid(logits)

        # Focal weight: (1 - p_t)^gamma
        # For positive targets: p_t = p, weight = (1-p)^gamma
        # For negative targets: p_t = 1-p, weight = p^gamma
        focal_pos = (1.0 - p) ** self.gamma
        focal_neg = p ** self.gamma

        # Combine: target selects between pos and neg terms
        loss = (
            -self.alpha * targets * focal_pos * log_p
            - (1.0 - self.alpha) * (1.0 - targets) * focal_neg * log_1_minus_p
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(cfg: dict) -> nn.Module:
    """Factory function to build loss from config.

    Config keys:
        loss: 'bce' (default) or 'focal'
        focal_gamma: focusing parameter (default 2.0)
        focal_alpha: positive class weight (default 0.25)

    Returns:
        Loss module.
    """
    loss_type = cfg.get("loss", "bce")

    if loss_type == "focal":
        gamma = cfg.get("focal_gamma", 2.0)
        alpha = cfg.get("focal_alpha", 0.25)
        print(f"  Loss: FocalLoss(gamma={gamma}, alpha={alpha})")
        return FocalLoss(gamma=gamma, alpha=alpha)
    else:
        print("  Loss: BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss()
