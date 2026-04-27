import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Load OOFs
b0 = pd.read_csv("experiments/sed_finetune_pseudo_v2_fold0/oof_preds_soundscape.csv")
sx = pd.read_csv("experiments/seresnext_finetune_fold0/oof_preds_soundscape.csv")

b0 = b0.sort_values("row_id").reset_index(drop=True)
sx = sx.sort_values("row_id").reset_index(drop=True)

species_cols = [c for c in b0.columns if c != "row_id"]
b0_p = b0[species_cols].values.flatten()
sx_p = sx[species_cols].values.flatten()

print(f"Spearman: {spearmanr(b0_p, sx_p).statistic:.4f}")
print(f"V2-S baseline correlation was: 0.6069")