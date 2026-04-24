import sys, os
sys.path.insert(0, os.getcwd())
import torch, yaml, numpy as np
from src.models import build_model
from src.training.train import build_loaders, get_target_species
from sklearn.metrics import roc_auc_score

with open("configs/experiment_sed_convnext.yaml") as f:
    cfg = yaml.safe_load(f)
# Force mixup off for this test
cfg["mixup_alpha"] = 0.0
cfg["aug_mixup_p"] = 0.0

target_species = get_target_species(cfg)
loader, _, _ = build_loaders(cfg, target_species)
specs, labels = next(iter(loader))
specs, labels = specs.cuda(), labels.cuda()
print(f"With mixup off, unique labels: {len(labels.unique())}")
print(f"Label range: [{labels.min():.3f}, {labels.max():.3f}]")

def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    p = torch.sigmoid(logits)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).mean()

def batch_auc(logits, labels):
    probs = torch.sigmoid(logits).cpu().numpy()
    y = labels.cpu().numpy()
    aucs = [roc_auc_score(y[:, i], probs[:, i])
            for i in range(y.shape[1]) if 0 < y[:, i].sum() < len(y)]
    return np.mean(aucs) if aucs else float('nan')

model = build_model(cfg).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-2)

print("\nFocal + NO mixup on fixed batch:")
for step in range(101):
    model.train()
    opt.zero_grad()
    logits = model(specs)
    loss = focal_loss(logits, labels)
    loss.backward()
    opt.step()
    if step % 10 == 0:
        model.eval()
        with torch.no_grad():
            auc = batch_auc(model(specs), labels)
        print(f"  step {step:>3}: loss={loss.item():.6f}  "
              f"logit_std={logits.std().item():.3f}  auc={auc:.4f}")