"""
Sound Event Detection (SED) model for BirdCLEF.

Replaces global average pooling with learned attention over time.
Rationale: bird calls occupy ~0.5s of a 5s window; GAP dilutes
that signal with silence. Attention pooling lets the model
learn to focus on frames with vocal activity.

Pipeline:
  spectrogram (B, 1, n_mels, T)
    -> optional per-sample standardization (input_normalize=True):
       required for LayerNorm-based backbones like ConvNeXt; not
       needed for BatchNorm backbones like EfficientNet.
    -> optional mono->3ch broadcast (encoder_in_chans=3):
       required for ConvNeXt because timm's in_chans=1 collapse of
       the 4x4 stride-4 stem kills ~30% of pretrained filters on
       log-mel input. B0's 3x3 stem tolerates in_chans=1 fine.
    -> backbone forward_features -> (B, C, H', W')
    -> mean over frequency (H')   -> (B, C, W')
    -> attention pooling over time -> (B, C)
    -> dropout + linear            -> (B, num_classes) logits
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def _normalize_and_expand(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_normalize:
            x_mean = x.mean(dim=(2, 3), keepdim=True)
            x_std = x.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            x = (x - x_mean) / x_std
        if self.encoder_in_chans == 3:
            x = x.expand(-1, 3, -1, -1)
        return x

    def _encode_to_features(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1|3, F, T_in) -> (B, C, T)
        x = self._normalize_and_expand(x)
        features = self.encoder(x)  # (B, C, H', W')
        features = features.mean(dim=2)  # mean over freq -> (B, C, T)
        return features

    def _head_linear(self) -> nn.Linear:
        # Defensive: head is currently Sequential(Dropout, Linear); locate the Linear
        # by type so this still works if the head structure is ever tweaked.
        linears = [m for m in self.head.modules() if isinstance(m, nn.Linear)]
        if len(linears) != 1:
            raise RuntimeError(
                f"BirdCLEFSED.head must contain exactly one nn.Linear, found {len(linears)}"
            )
        return linears[0]

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: (B, 1, n_mels, T)
        features = self._encode_to_features(x)               # (B, C, T)
        pooled, att_weights = self.attention_pool(features)  # (B, C), (B, 1, T)
        logits = self.head(pooled)                           # (B, num_classes)
        if return_attention:
            return logits, att_weights
        return logits

def forward_dual(self, x: torch.Tensor):
    """Inference-only: returns (logits_pool, logits_max), each (B, num_classes).
    Reuses the head Linear; no new parameters."""
    features = self._encode_to_features(x)               # (B, C, T)
    pooled, _ = self.attention_pool(features)            # (B, C)
    logits_pool = self.head(pooled)                      # (B, num_classes)

    linear = self._head_linear()
    # (B, C, T) -> (B, T, C) -> Linear -> (B, T, num_classes) -> max over T
    frame_logits = F.linear(features.transpose(1, 2), linear.weight, linear.bias)
    logits_max = frame_logits.amax(dim=1)                # (B, num_classes)
    return logits_pool, logits_max


class BirdCLEFSED(nn.Module):
    """
    Backbone with attention temporal pooling.

    Output shape is (B, num_classes) logits, identical to BirdCLEFModel,
    so the existing training loop, loss, and ONNX export work unchanged.

    Args:
        backbone: timm model name (e.g. "tf_efficientnet_b0_ns", "convnext_tiny")
        num_classes: output classes
        dropout: head dropout prob
        pretrained: load ImageNet pretrained weights
        attention_hidden_dim: hidden dim of attention MLP
        input_normalize: if True, per-sample standardize input to ~(0, 1).
            REQUIRED for ConvNeXt and other LayerNorm-based backbones.
            Leave False for B0 to preserve checkpoint compatibility.
        encoder_in_chans: input channels passed to the timm encoder. Default
            is 1 (matches B0 training). Set to 3 for ConvNeXt — the mono
            spec is broadcast to 3 identical channels inside forward().
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        dropout: float = 0.3,
        pretrained: bool = True,
        attention_hidden_dim: int = 128,
        input_normalize: bool = False,
        encoder_in_chans: int = 1,
    ):
        super().__init__()
        self.input_normalize = input_normalize
        self.encoder_in_chans = encoder_in_chans

        # global_pool="" + num_classes=0 -> returns raw spatial feature maps
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=encoder_in_chans,
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

        # Per-sample standardization (ConvNeXt needs this; B0 does not).
        if self.input_normalize:
            x_mean = x.mean(dim=(2, 3), keepdim=True)
            x_std = x.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            x = (x - x_mean) / x_std

        # Broadcast mono -> 3ch if encoder was built with in_chans=3.
        if self.encoder_in_chans == 3:
            x = x.expand(-1, 3, -1, -1)

        features = self.encoder(x)           # (B, C, H', W')
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
