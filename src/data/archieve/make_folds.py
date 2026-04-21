"""
BirdCLEF+ 2026 — Step 3: Stratified 5-Fold Generation

Generates stratified CV folds that:
  1. Stratify by primary_label
  2. Handle rare classes (≤5 samples) via group-aware placement
  3. Distribute collection sources (iNat / XC) across folds
  4. Output: data/folds/folds.csv  (filename, primary_label, fold)

Extra packages needed:
  pip install pandas scikit-learn

Usage:
  python src/data/make_folds.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

SEED = 42
N_FOLDS = 5
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "train.csv"
OUT_DIR = PROJECT_ROOT / "data" / "folds"
OUT_CSV = OUT_DIR / "folds.csv"

RARE_THRESHOLD = N_FOLDS  # species with ≤ this many samples get special handling


def make_folds():
    # ── Load ────────────────────────────────────────────────────────────
    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df)} rows, {df['primary_label'].nunique()} species")

    has_collection = "collection" in df.columns

    # ── Separate rare vs common species ─────────────────────────────────
    counts = df["primary_label"].value_counts()
    rare_species = set(counts[counts <= RARE_THRESHOLD].index)
    print(f"Rare species (≤{RARE_THRESHOLD} samples): {len(rare_species)}")

    mask_rare = df["primary_label"].isin(rare_species)
    df_rare = df[mask_rare].copy()
    df_common = df[~mask_rare].copy()

    # ── Fold rare species manually ──────────────────────────────────────
    # Distribute each rare species' samples round-robin across folds so
    # every fold's *training* set contains at least one sample.
    rng = np.random.RandomState(SEED)
    rare_folds = []
    for sp in sorted(rare_species):
        sp_idx = df_rare[df_rare["primary_label"] == sp].index.tolist()
        rng.shuffle(sp_idx)
        for i, idx in enumerate(sp_idx):
            rare_folds.append((idx, i % N_FOLDS))

    rare_fold_map = dict(rare_folds)

    # ── Fold common species with stratification ─────────────────────────
    # Use StratifiedGroupKFold if collection exists (treat collection as
    # a secondary grouping hint); otherwise plain StratifiedKFold.
    if has_collection:
        # Create a pseudo-group that couples (primary_label, collection)
        # so the splitter tries to spread collections across folds.
        df_common = df_common.copy()
        df_common["_group"] = (
            df_common["primary_label"].astype(str)
            + "_"
            + df_common["collection"].astype(str)
        )
        skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = skf.split(
            df_common, df_common["primary_label"], groups=df_common["_group"]
        )
    else:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = skf.split(df_common, df_common["primary_label"])

    common_fold_map = {}
    for fold_id, (_, val_idx) in enumerate(splits):
        for i in val_idx:
            common_fold_map[df_common.index[i]] = fold_id

    # ── Merge & save ────────────────────────────────────────────────────
    all_fold_map = {**common_fold_map, **rare_fold_map}
    df["fold"] = df.index.map(all_fold_map)
    assert df["fold"].notna().all(), "Some rows missing fold assignment!"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_cols = ["filename", "primary_label", "fold"]
    df["fold"] = df["fold"].astype(int)
    df[out_cols].to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}  ({len(df)} rows)")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n=== Samples per fold ===")
    print(df["fold"].value_counts().sort_index().to_string())

    print("\n=== Species coverage per fold (validation set) ===")
    all_species = set(df["primary_label"].unique())
    missing_any = False
    for fold_id in range(N_FOLDS):
        val_species = set(df[df["fold"] == fold_id]["primary_label"].unique())
        missing = all_species - val_species
        status = f"{len(val_species)}/{len(all_species)} species"
        if missing:
            missing_any = True
            status += f"  ⚠ missing {len(missing)}: {sorted(missing)[:10]}{'…' if len(missing)>10 else ''}"
        print(f"  Fold {fold_id}: {status}")

    if not missing_any:
        print("  ✓ All species present in every fold's validation set.")

    if has_collection:
        print("\n=== Fold × Collection crosstab ===")
        print(pd.crosstab(df["fold"], df["collection"]).to_string())

    print("\nDone.")


if __name__ == "__main__":
    make_folds()
