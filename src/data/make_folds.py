"""
BirdCLEF+ 2026 — Step 3: Stratified 5-Fold Generation (v2)

Strategy:
  - Rare species (< N_FOLDS samples): round-robin so every fold's
    training set has ≥1 sample. These species will be missing from
    some validation folds — unavoidable and logged.
  - All other species: plain StratifiedKFold on primary_label.
    This naturally distributes collection sources proportionally
    since each species' collection mix is split evenly.

Output: data/folds/folds.csv  (filename, primary_label, fold)

Usage:
  python src/data/make_folds.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 5
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "train.csv"
OUT_DIR = PROJECT_ROOT / "data" / "folds"
OUT_CSV = OUT_DIR / "folds.csv"


def make_folds():
    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df)} rows, {df['primary_label'].nunique()} species")

    has_collection = "collection" in df.columns
    counts = df["primary_label"].value_counts()

    # Species with fewer samples than folds can't appear in every val set
    rare_species = set(counts[counts < N_FOLDS].index)
    okay_species = set(counts[(counts >= N_FOLDS) & (counts < 2 * N_FOLDS)].index)
    print(f"Rare (< {N_FOLDS} samples, round-robin): {len(rare_species)}")
    print(f"Low  ({N_FOLDS}-{2*N_FOLDS-1} samples, stratified): {len(okay_species)}")

    mask_rare = df["primary_label"].isin(rare_species)
    df_rare = df[mask_rare].copy()
    df_main = df[~mask_rare].copy()

    # -- Rare: round-robin ------------------------------------------------
    rng = np.random.RandomState(SEED)
    rare_fold_map = {}
    for sp in sorted(rare_species):
        idx = df_rare[df_rare["primary_label"] == sp].index.tolist()
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            rare_fold_map[ix] = i % N_FOLDS

    # -- Main: StratifiedKFold --------------------------------------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    main_fold_map = {}
    for fold_id, (_, val_idx) in enumerate(skf.split(df_main, df_main["primary_label"])):
        for i in val_idx:
            main_fold_map[df_main.index[i]] = fold_id

    # -- Merge & save -----------------------------------------------------
    all_fold_map = {**main_fold_map, **rare_fold_map}
    df["fold"] = df.index.map(all_fold_map).astype(int)
    assert df["fold"].notna().all(), "Missing fold assignments!"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df[["filename", "primary_label", "fold"]].to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}  ({len(df)} rows)")

    # -- Summary ----------------------------------------------------------
    print("\n=== Samples per fold ===")
    print(df["fold"].value_counts().sort_index().to_string())

    all_sp = set(df["primary_label"].unique())
    print(f"\n=== Species coverage per fold (validation set) ===")
    total_missing = 0
    for fid in range(N_FOLDS):
        val_sp = set(df[df["fold"] == fid]["primary_label"].unique())
        miss = sorted(all_sp - val_sp)
        total_missing += len(miss)
        tag = f"  -> missing {len(miss)}: {miss}" if miss else ""
        print(f"  Fold {fid}: {len(val_sp)}/{len(all_sp)}{tag}")

    if total_missing == 0:
        print("  All species in every validation fold.")
    else:
        all_missing = set()
        for fid in range(N_FOLDS):
            val_sp = set(df[df["fold"] == fid]["primary_label"].unique())
            all_missing |= (all_sp - val_sp)
        unexpected = all_missing - rare_species
        if unexpected:
            print(f"\n  Non-rare species missing from some folds: {sorted(unexpected)}")
        else:
            print(f"\n  Only rare species (n<{N_FOLDS}) are missing — expected & acceptable.")

    if has_collection:
        print("\n=== Fold x Collection ===")
        ct = pd.crosstab(df["fold"], df["collection"])
        print(ct)
        pct = ct.div(ct.sum(axis=1), axis=0) * 100
        print("\nProportions (%):")
        print(pct.round(1))
        spread = pct.max().max() - pct.min().min()
        print(f"\nMax spread within a source: {spread:.1f}pp")

    print("\nDone.")


if __name__ == "__main__":
    make_folds()
