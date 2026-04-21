# scripts/test_sed_model.py
import torch
import yaml
from src.models import build_model

with open("configs/experiment_sed_b0.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["pretrained"] = False  # skip ImageNet download for the test

model = build_model(cfg).eval()

# Forward pass
x = torch.randn(2, 1, 128, 313)
with torch.no_grad():
    logits = model(x)
    logits_with_att, att = model(x, return_attention=True)

print(f"logits shape:     {logits.shape}")              # (2, 234)
print(f"attention shape:  {att.shape}")                 # (2, 1, 10) approx
print(f"attention sums:   {att.sum(dim=-1).squeeze()}") # ~[1.0, 1.0]
assert logits.shape == (2, 234)
assert torch.allclose(logits, logits_with_att)
assert torch.allclose(att.sum(dim=-1), torch.ones_like(att.sum(dim=-1)), atol=1e-5)

# ONNX export
torch.onnx.export(
    model, x, "/tmp/sed_test.onnx",
    opset_version=17,
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
)
print("ONNX export OK")

# Parameter count vs baseline
n_params = sum(p.numel() for p in model.parameters())
print(f"total params:     {n_params/1e6:.2f}M")         # ~4.1M (baseline is ~4.0M)