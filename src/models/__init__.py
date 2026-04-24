"""Model factory. Selects architecture from config['model_type']."""
from .classifier import BirdCLEFModel
from .sed import BirdCLEFSED

__all__ = ["BirdCLEFModel", "BirdCLEFSED", "build_model"]


def build_model(config):
    """
    Build a model from a config dict.

    Supports:
      model_type: "classifier"  (default, backward-compatible with baseline)
      model_type: "sed"         (attention pooling over time)

    For "sed": reads optional input_normalize and encoder_in_chans flags.
    Defaults preserve B0 checkpoint compatibility (False and 1). ConvNeXt
    configs must set input_normalize: true and encoder_in_chans: 3.
    """
    model_type = config.get("model_type", "classifier")
    backbone = config["backbone"]
    num_classes = config["num_classes"]
    dropout = config.get("dropout", 0.3)
    pretrained = config.get("pretrained", True)

    if model_type == "classifier":
        return BirdCLEFModel(
            backbone=backbone,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
        )
    elif model_type == "sed":
        return BirdCLEFSED(
            backbone=backbone,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
            attention_hidden_dim=config.get("attention_hidden_dim", 128),
            input_normalize=config.get("input_normalize", False),
            encoder_in_chans=config.get("encoder_in_chans", 1),
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. Expected 'classifier' or 'sed'."
        )
