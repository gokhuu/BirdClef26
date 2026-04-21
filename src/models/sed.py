"""
Sound Event Detection (SED) model for BirdCLEF.

Replaces global average pooling with learned attention over time.
Rationale: bird calls occupy ~0.5s of a 5s window; GAP dilutes
that signal with silence. Attention pooling lets the model
learn to focus on frames with vocal activity.

Pipeline:
  spectrogram (B, 1, n_mels, T)
    -> EfficientNet forward_features -> (B, C, H', W')
    -> mean over frequency (H')      -> (B, C, W')
    -> attention pooling over time   -> (B, C)
    -> dropout + linear              -> (B, num_classes) logits
"""
import timm
import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """
    Learned soft attention over the time axis.

    Computes one attention weight per time frame via a tiny 1D CNN,
    softmax-normalizes across time, and returns the weighted sum
    of feature vectors.

    Input:  (B, C, T)
    Output: pooled features (B, C), attention weights (B, 1, T)
    """

    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_features, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, C, T)
        att_logits = self.attention(x)                 # (B, 1, T)
        att_weights = torch.softmax(att_logits, dim=-1)  # normalize across time
        pooled = torch.sum(x * att_weights, dim=-1)    # (B, C)
        return pooled, att_weights


class BirdCLEFSED(nn.Module):
    """
    EfficientNet backbone with attention temporal pooling.

    Output shape is (B, num_classes) logits, identical to BirdCLEFModel,
    so the existing training loop, loss, and ONNX export work unchanged.
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        dropout: float = 0.3,
        pretrained: bool = True,
        attention_hidden_dim: int = 128,
    ):
        super().__init__()
        # global_pool="" + num_classes=0 -> returns raw spatial feature maps
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="",
        )
        feature_dim = self.encoder.num_features

        self.attention_pool = AttentionPooling(
            in_features=feature_dim,
            hidden_dim=attention_hidden_dim,
        )
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: (B, 1, n_mels, T)
        features = self.encoder(x)           # (B, C, H', W')  e.g. (B, 1280, 4, 10)
        features = features.mean(dim=2)      # mean over frequency -> (B, C, W')
        pooled, att_weights = self.attention_pool(features)  # (B, C), (B, 1, W')
        logits = self.head(pooled)           # (B, num_classes)

        if return_attention:
            return logits, att_weights
        return logits

    @torch.no_grad()
    def extract_attention(self, x: torch.Tensor):
        """Helper for debugging / visualization. Not used during training."""
        self.eval()
        _, att = self.forward(x, return_attention=True)
        return att