# Patch: pass `drop_path_rate` through to timm

Your `BirdCLEFSED` takes `backbone`, `input_normalize`, `encoder_in_chans`.
To set `drop_path_rate=0.1` (critical for V2-S on sparse labels), you need
to forward kwargs to `timm.create_model`.

If your current code looks roughly like:

```python
class BirdCLEFSED(nn.Module):
    def __init__(self, backbone, input_normalize=True, encoder_in_chans=1, num_classes=234, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=encoder_in_chans,
            num_classes=0,
            global_pool="",
        )
        # ... pooling + head
```

Change to:

```python
class BirdCLEFSED(nn.Module):
    def __init__(
        self,
        backbone,
        input_normalize=True,
        encoder_in_chans=1,
        num_classes=234,
        pretrained=True,
        timm_kwargs=None,          # <-- NEW
    ):
        super().__init__()
        timm_kwargs = timm_kwargs or {}
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=encoder_in_chans,
            num_classes=0,
            global_pool="",
            **timm_kwargs,          # <-- NEW
        )
        # ... rest unchanged
```

That's it. All your existing B0 experiments still work because `timm_kwargs`
defaults to `None`. New V2-S experiments pass `timm_kwargs={"drop_path_rate": 0.1}`.

## Also — confirm the head feature dim is auto-inferred

Your SED head does `mean-over-freq -> AttentionPooling-over-time -> Linear`.
The Linear layer needs to know the backbone feature dim. Most implementations do:

```python
with torch.no_grad():
    dummy = torch.zeros(1, encoder_in_chans, 128, 313)
    feat = self.encoder(dummy)
    self.feat_dim = feat.shape[1]  # channels after global_pool=""
self.head = nn.Linear(self.feat_dim, num_classes)
```

If yours hardcodes `1280` anywhere, you need to make this dynamic — otherwise
SEResNeXt26d fallback (feat_dim=2048) will fail at init. V2-S is still 1280
like B0 so it works either way, but SEResNeXt won't.

## One more thing — Attention pooling input dim

Same logic for your AttentionPooling module if it has a learnable projection.
Confirm it also uses the inferred `feat_dim` rather than hardcoded value.

## Quick test the patch works

```python
import timm
import torch
from src.models.sed import BirdCLEFSED

# Should not raise
m = BirdCLEFSED(
    backbone="tf_efficientnetv2_s.in21k_ft_in1k",
    encoder_in_chans=1,
    pretrained=True,
    timm_kwargs={"drop_path_rate": 0.1},
)
x = torch.randn(2, 1, 128, 313)
y = m(x)
assert y.shape == (2, 234), f"unexpected output shape {y.shape}"
print("OK, output:", y.shape)
```
