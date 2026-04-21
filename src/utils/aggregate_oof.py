"""
BirdCLEF+ 2026 — Aggregate OOF Predictions
=============================================

Combines OOF predictions from all 5 folds into a single full-dataset
evaluation and computes per-species and macro AUC.

Usage:
    python src/utils/aggregate_oof.py experiments/v2_augment

Expects folder structure:
    experiments/v2_augment_fold0/oof_preds.csv
    experiments/v2_augment_fold1/oof_preds.csv
    ...
    experiments/v2_augment_fold4/oof_preds.csv

Or:
    experiments/v2_augment/fold0/oof_preds.csv
    ...
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score


def find_oof_files(base_path: str, n_folds: int = 5) -> list[str]:
    """Find OOF prediction files across fold directories."""
    base = Path(base_path)
    patterns = [
        # Pattern 1: experiments/name_fold0/oof_preds.csv
        [str(base) + f"_fold{i}/oof_preds.csv" for i in range(n_folds)],
        # Pattern 2: experiments/name/fold0/oof_preds.csv
        [str(base / f"fold{i}" / "oof_preds.csv") for i in range(n_folds)],
        # Pattern 3: experiments/name_fold0_*/oof_preds.csv (with extra suffixes)
    ]

    for pattern in patterns:
        found = [p for p in pattern if os.path.exists(p)]
        if len(found) == n_folds:
            return found

    # Glob fallback
    all_oof = sorted(glob.glob(str(base.parent / f"{base.name}*" / "oof_preds.csv")))
    if len(all_oof) >= n_folds:
        return all_oof[:n_folds]

    return all_oof


def aggregate_oof(base_path: str, train_meta_csv: str = "data/raw/train.csv",
                  n_folds: int = 5) -> dict:
    """Aggregate OOF predictions and compute metrics.

    Returns:
        Dict with keys: macro_auc, per_species_auc, n_evaluable,
        n_species, oof_df
    """
    oof_files = find_oof_files(base_path, n_folds)

    if not oof_files:
        print(f"⚠ No OOF files found at {base_path}")
        return {}

    print(f"Found {len(oof_files)} OOF files:")
    for f in oof_files:
        print(f"  {f}")

    # Concatenate all folds
    dfs = []
    for f in oof_files:
        df = pd.read_csv(f)
        dfs.append(df)
    oof_df = pd.concat(dfs, ignore_index=True)

    # Load metadata to get ground truth
    meta = pd.read_csv(train_meta_csv)
    species_cols = [c for c in oof_df.columns if c != "filename"]

    # Build ground truth labels
    merged = oof_df[["filename"]].merge(meta[["filename", "primary_label"]], on="filename", how="left")

    labels = np.zeros((len(oof_df), len(species_cols)), dtype=np.float32)
    for i, sp in enumerate(species_cols):
        labels[:, i] = (merged["primary_label"] == sp).astype(float)

    preds = oof_df[species_cols].values.astype(np.float32)

    # Per-species AUC
    per_species = {}
    evaluable = 0
    for i, sp in enumerate(species_cols):
        col_labels = labels[:, i]
        n_pos = col_labels.sum()
        if n_pos > 0 and n_pos < len(col_labels):
            try:
                auc = roc_auc_score(col_labels, preds[:, i])
                per_species[sp] = {"auc": auc, "n_positive": int(n_pos)}
                evaluable += 1
            except ValueError:
                per_species[sp] = {"auc": None, "n_positive": int(n_pos)}
        else:
            per_species[sp] = {"auc": None, "n_positive": int(n_pos)}

    # Macro AUC
    valid_aucs = [v["auc"] for v in per_species.values() if v["auc"] is not None]
    macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0

    print(f"\n{'='*50}")
    print(f"Full OOF Macro AUC: {macro_auc:.4f}")
    print(f"Evaluable species:  {evaluable}/{len(species_cols)}")
    print(f"Total samples:      {len(oof_df)}")
    print(f"{'='*50}")

    # Save per-species AUC
    output_dir = Path(base_path).parent / (Path(base_path).name + "_allfolds")
    output_dir.mkdir(exist_ok=True)

    species_df = pd.DataFrame([
        {"species": sp, "auc": v["auc"], "n_positive": v["n_positive"]}
        for sp, v in per_species.items()
    ]).sort_values("auc", ascending=True, na_position="first")

    species_df.to_csv(output_dir / "per_species_auc.csv", index=False)

    # Distribution summary
    if valid_aucs:
        auc_arr = np.array(valid_aucs)
        print(f"\nAUC distribution:")
        print(f"  ≥ 0.99: {(auc_arr >= 0.99).sum()}")
        print(f"  ≥ 0.95: {(auc_arr >= 0.95).sum()}")
        print(f"  ≥ 0.90: {(auc_arr >= 0.90).sum()}")
        print(f"  < 0.80: {(auc_arr < 0.80).sum()}")
        print(f"  < 0.50: {(auc_arr < 0.50).sum()}")

        # Worst species
        print(f"\nWorst 10 species:")
        for _, row in species_df.head(10).iterrows():
            auc_str = f"{row['auc']:.4f}" if pd.notna(row['auc']) else "N/A"
            print(f"  {row['species']:>20s}: AUC={auc_str}  (n={int(row['n_positive'])})")

    return {
        "macro_auc": macro_auc,
        "per_species": per_species,
        "n_evaluable": evaluable,
        "n_species": len(species_cols),
        "oof_df": oof_df,
        "labels": labels,
        "preds": preds,
        "species_cols": species_cols,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/utils/aggregate_oof.py <experiment_base_path>")
        print("Example: python src/utils/aggregate_oof.py experiments/v2_augment")
        sys.exit(1)

    base_path = sys.argv[1]
    train_meta = sys.argv[2] if len(sys.argv) > 2 else "data/raw/train.csv"

    result = aggregate_oof(base_path, train_meta_csv=train_meta)

    if result:
        print(f"\nPer-species AUC saved to: {Path(base_path).parent / (Path(base_path).name + '_allfolds')}/per_species_auc.csv")


if __name__ == "__main__":
    main()
