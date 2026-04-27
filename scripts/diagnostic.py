import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Load OOFs
b0 = pd.read_csv("experiments/sed_finetune_pseudo_v2_fold0/oof_preds_soundscape.csv").sort_values("row_id").reset_index(drop=True)
v2s = pd.read_csv("experiments/effv2s_finetune_fold0/oof_preds_soundscape.csv").sort_values("row_id").reset_index(drop=True)

# Load fold 0 soundscape val labels
folds_sc = pd.read_csv("data/folds/folds_soundscapes.csv")
val = folds_sc[folds_sc["fold"] == 0].sort_values("row_id").reset_index(drop=True)

print(f"OOF rows: B0={len(b0)}, V2S={len(v2s)}, val rows: {len(val)}")
print(f"Row IDs match val: {set(b0['row_id']) == set(val['row_id'])}")

# Build label matrix from primary_label column
species_cols = [c for c in b0.columns if c != "row_id"]
print(f"Species columns: {len(species_cols)}")

# Construct labels: one-hot from primary_label
# (your soundscape val might be single-label per window; check this matches your loss setup)
label_to_idx = {sp: i for i, sp in enumerate(species_cols)}
y = np.zeros((len(val), len(species_cols)), dtype=np.float32)
for i, lbl in enumerate(val["primary_label"].values):
    if lbl in label_to_idx:
        y[i, label_to_idx[lbl]] = 1.0

# Compute macro AUC (skipping classes with no positives, like training does)
def macro_auc(y_true, y_pred):
    aucs = []
    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() > 0 and y_true[:, j].sum() < len(y_true):
            try:
                aucs.append(roc_auc_score(y_true[:, j], y_pred[:, j]))
            except ValueError:
                pass
    return np.mean(aucs), len(aucs)

b0_auc, b0_n = macro_auc(y, b0[species_cols].values)
v2s_auc, v2s_n = macro_auc(y, v2s[species_cols].values)

print(f"\nB0 soundscape AUC from OOF: {b0_auc:.4f}  ({b0_n} classes scored)")
print(f"V2S soundscape AUC from OOF: {v2s_auc:.4f}  ({v2s_n} classes scored)")
print(f"\nTraining reported: B0 sc_auc≈0.93 (your v2),  V2S sc_auc=0.857")