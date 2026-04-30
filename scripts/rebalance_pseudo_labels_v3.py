"""
scripts/rebalance_pseudo_labels_v3.py

v3 rebalance: DROP dominant-species rows entirely instead of zeroing them.

Difference from rebalance_pseudo_labels.py (v2):
  v2 set the dominant species' prob to 0.0 on excess rows. Other species
  in those rows kept their probs. Net effect: the model was trained to
  suppress its strongest signals on a fraction of windows for the most
  common species — precisely the species most likely to matter on Pantanal.
  v3 drops those rows entirely. No label corruption; slightly smaller
  dataset.

Same RNG seed (42), same per-species random selection as v2 — so the
rows v2 zeroed are exactly the rows v3 drops. Clean A/B comparison.

Output: data/pseudo/sed_b0_v3_pseudo_dropped.csv
"""
import pandas as pd
import numpy as np

SRC = "data/pseudo/sed_b0_v1_pseudo.csv"
DST = "data/pseudo/sed_b0_v3_pseudo_dropped.csv"
MAX_POS_PER_SPECIES = 2000
THRESHOLD = 0.5
LOW_CONF_DROP = 0.3   # match v2's secondary filter for parity

df = pd.read_csv(SRC)
species = [c for c in df.columns if c != "row_id"]
print(f"Input: {len(df):,} windows x {len(species)} species")

rng = np.random.default_rng(42)
rows_to_drop: set = set()

for sp in species:
    pos_mask = df[sp] > THRESHOLD
    n_pos = int(pos_mask.sum())
    if n_pos <= MAX_POS_PER_SPECIES:
        continue
    pos_idx = np.where(pos_mask.values)[0]
    drop_idx = rng.choice(
        pos_idx, size=n_pos - MAX_POS_PER_SPECIES, replace=False
    )
    rows_to_drop.update(int(i) for i in drop_idx)
    print(f"  {sp:>20s}: {n_pos:>5,} -> <={MAX_POS_PER_SPECIES:,} "
          f"({n_pos - MAX_POS_PER_SPECIES:,} queued)")

print(f"\nUnique rows queued for drop: {len(rows_to_drop):,}")
df = df.drop(index=list(rows_to_drop)).reset_index(drop=True)
print(f"After cap-drop: {len(df):,} rows")

# Defensive: in v2 this catches rows whose only strong species got zeroed.
# In v3 it should be a no-op (every surviving row's max prob is unchanged
# from v1, which itself was generated at min_confidence=0.3). Kept for
# pipeline parity with v2.
max_per_row = df[species].max(axis=1)
n_low = int((max_per_row <= LOW_CONF_DROP).sum())
df = df[max_per_row > LOW_CONF_DROP].reset_index(drop=True)
print(f"Low-confidence rows dropped: {n_low}  (expected ~0 in v3)")
print(f"Final: {len(df):,} rows")

df.to_csv(DST, index=False)
print(f"\nWrote {DST}")

# Sanity check
pos = (df[species] > 0.5).sum(axis=0).sort_values(ascending=False)
print(f"\nTop 10 species by positive count after rebalancing:")
for sp, n in pos.head(10).items():
    print(f"  {sp:>20s}: {n:,}")
print(f"\nSpecies with >100 positives: {(pos > 100).sum()}")
print(f"Species with 0 positives:    {(pos == 0).sum()}")
print(f"Total positive labels:       {int((df[species] > 0.5).sum().sum()):,}")
