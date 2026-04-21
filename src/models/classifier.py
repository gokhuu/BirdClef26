"""Baseline classifier: EfficientNet backbone + global average pooling + linear head."""
import timm
import torch.nn as nn


class BirdCLEFModel(nn.Module):
    def __init__(self, backbone, num_classes, dropout=0.3, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.encoder.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        features = self.encoder(x)       # (B, feature_dim)
        return self.head(features)       # (B, num_classes)