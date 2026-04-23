import numpy as np
import torch
import onnxruntime as ort
import sys, os
sys.path.insert(0, os.getcwd())
from src.models import build_model
import yaml

# Load both versions
cfg = yaml.safe_load(open("experiments/sed_finetune_fold0/config.yaml"))
pt_model = build_model(cfg)
pt_model.load_state_dict(torch.load(
    "experiments/sed_finetune_fold0/best_model.pt",
    map_location="cpu", weights_only=True
))
pt_model.eval()

ort_session = ort.InferenceSession(
    "experiments/sed_finetune_fold0/best_model.onnx",
    providers=["CPUExecutionProvider"],
)

# Run a dummy input through both
x = np.random.randn(1, 1, 128, 313).astype(np.float32)

with torch.no_grad():
    pt_out = pt_model(torch.from_numpy(x)).numpy()

ort_in_name = ort_session.get_inputs()[0].name
ort_out = ort_session.run(None, {ort_in_name: x})[0]

# Compare
diff = np.abs(pt_out - ort_out).max()
print(f"PyTorch out range: [{pt_out.min():.4f}, {pt_out.max():.4f}]")
print(f"ONNX out range:    [{ort_out.min():.4f}, {ort_out.max():.4f}]")
print(f"Max abs diff:      {diff:.2e}")
print(f"Verdict: {'OK' if diff < 1e-3 else 'WARNING — significant divergence'}")