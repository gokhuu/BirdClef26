# scripts/rebalance_pseudo_labels.py
import pandas as pd
import numpy as np

SRC = "data/pseudo/sed_b0_v1_pseudo.csv"
DST = "data/pseudo/sed_b0_v2_pseudo_balanced.csv"
MAX_POS_PER_SPECIES = 2000     # cap top species
THRESHOLD = 0.5                # what counts as "positive" for the cap

df = pd.read_csv(SRC)
species = [c for c in df.columns if c != "row_id"]
print(f"Input: {len(df):,} windows")

# For each species that has > MAX_POS_PER_SPECIES positive windows, find those rows
# and randomly suppress the excess — set that species' prob to 0 for chosen rows.
# This keeps the row (other species' signals remain) but caps any one species' dominance.
rng = np.random.default_rng(42)
suppressed_total = 0
for sp in species:
    pos_mask = df[sp] > THRESHOLD
    n_pos = pos_mask.sum()
    if n_pos <= MAX_POS_PER_SPECIES:
        continue
    pos_idx = np.where(pos_mask)[0]
    drop_idx = rng.choice(pos_idx, size=n_pos - MAX_POS_PER_SPECIES, replace=False)
    df.loc[drop_idx, sp] = 0.0
    suppressed_total += len(drop_idx)
    print(f"  {sp}: {n_pos:,} → {MAX_POS_PER_SPECIES:,} ({len(drop_idx):,} suppressed)")

# Drop rows that now have no species above threshold — they're useless
max_per_row = df[species].max(axis=1)
keep_mask = max_per_row > 0.3
print(f"\nRows before low-confidence drop: {len(df):,}")
df = df[keep_mask].reset_index(drop=True)
print(f"Rows after: {len(df):,}")
print(f"Total positive-labels suppressed: {suppressed_total:,}")

df.to_csv(DST, index=False)
print(f"\n✓ Wrote {DST}")

# Re-verify distribution
pos = (df[species] > 0.5).sum(axis=0).sort_values(ascending=False)
print(f"\nTop 10 species after rebalancing:")
print(pos.head(10))
print(f"\nSpecies with >100 positive windows: {(pos > 100).sum()}")
print(f"Species with 0 positive windows:    {(pos == 0).sum()}")