import pandas as pd
import numpy as np

# Load OOF predictions for fold 0 from both models
b0 = pd.read_csv("experiments/sed_finetune_pseudo_v2_fold0/oof_preds_soundscape.csv")
v2s = pd.read_csv("experiments/effv2s_finetune_fold0/oof_preds_soundscape.csv")

# Sort both by row_id to align rows
b0 = b0.sort_values("row_id").reset_index(drop=True)
v2s = v2s.sort_values("row_id").reset_index(drop=True)

# Pull just the species probability columns (drop row_id)
species_cols = [c for c in b0.columns if c != "row_id"]
b0_probs = b0[species_cols].values.flatten()
v2s_probs = v2s[species_cols].values.flatten()

# Pearson correlation across all (sample × class) probabilities
from scipy.stats import pearsonr, spearmanr
pearson = pearsonr(b0_probs, v2s_probs).statistic
spearman = spearmanr(b0_probs, v2s_probs).statistic
print(f"Pearson: {pearson:.4f}")
print(f"Spearman: {spearman:.4f}")