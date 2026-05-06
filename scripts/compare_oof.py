"""
scripts/compare_oof.py

Compare two or more OOF prediction CSVs. Replaces corr_check.py, parity.py,
and diagnostic.py — same operations, one CLI.

Operations:
  - Pairwise Pearson + Spearman correlation across all (sample x class) probs
    (use this to gauge ensemble diversity: lower correlation = more diverse =
    bigger expected ensemble lift).
  - Optional macro AUC per file, computed against either:
      * --focal-meta CSV  (rebuilds labels from primary_label + secondary_labels)
      * --soundscape-folds CSV  (rebuilds one-hot from primary_label of fold-N rows)
    The label source must match the OOF type (focal vs soundscape).

Usage:
    # Just diversity (Pearson + Spearman) — replaces corr_check.py / parity.py
    python scripts/compare_oof.py \\
        experiments/sed_finetune_pseudo_v2/sed_finetune_pseudo_v2_fold0/oof_preds_soundscape.csv \\
        experiments/effv2s_finetune/effv2s_finetune_fold0/oof_preds_soundscape.csv

    # Diversity + macro AUC for soundscape OOFs — replaces diagnostic.py
    python scripts/compare_oof.py \\
        experiments/sed_finetune_pseudo_v2/sed_finetune_pseudo_v2_fold0/oof_preds_soundscape.csv \\
        experiments/effv2s_finetune/effv2s_finetune_fold0/oof_preds_soundscape.csv \\
        --soundscape-folds data/folds/folds_soundscapes.csv --fold 0

    # Macro AUC for focal OOFs
    python scripts/compare_oof.py \\
        experiments/seresnext_finetune/seresnext_finetune_fold0/oof_preds.csv \\
        --focal-meta data/train_metadata.csv

The OOF format:
  - First column: 'row_id' for soundscape OOFs, 'filename' for focal OOFs.
  - Remaining columns: per-species probabilities. Same column ordering across
    files is NOT required — we align on the intersection of species columns.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# OOF loading
# ---------------------------------------------------------------------------

def detect_key_col(df: pd.DataFrame) -> str:
    if "row_id" in df.columns:
        return "row_id"
    if "filename" in df.columns:
        return "filename"
    sys.exit(f"ERROR: OOF csv has neither 'row_id' nor 'filename' column. "
             f"Got: {list(df.columns)[:5]}...")


def load_oof(path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    """Returns (df_sorted_by_key, key_col, species_cols)."""
    df = pd.read_csv(path)
    key_col = detect_key_col(df)
    df = df.sort_values(key_col).reset_index(drop=True)
    species_cols = [c for c in df.columns if c != key_col]
    return df, key_col, species_cols


# ---------------------------------------------------------------------------
# Diversity (correlation across all flattened probs)
# ---------------------------------------------------------------------------

def pairwise_correlations(oofs: list[tuple[Path, pd.DataFrame, str, list[str]]]
                          ) -> None:
    """For every pair of OOFs, compute Pearson + Spearman over the flattened
    (row x species) probability matrix. Aligns on the intersection of species
    columns and on shared keys."""
    from scipy.stats import pearsonr, spearmanr

    n = len(oofs)
    if n < 2:
        print("(only one OOF supplied — skipping pairwise correlation)\n")
        return

    print(f"\n=== Pairwise correlation (Pearson / Spearman) ===")
    for i in range(n):
        for j in range(i + 1, n):
            p_i, df_i, key_i, sp_i = oofs[i]
            p_j, df_j, key_j, sp_j = oofs[j]

            if key_i != key_j:
                print(f"  [SKIP] {p_i.name} vs {p_j.name}: key mismatch "
                      f"({key_i} vs {key_j})")
                continue

            # Align on shared keys (handles partial overlap gracefully)
            shared_keys = set(df_i[key_i]) & set(df_j[key_j])
            if not shared_keys:
                print(f"  [SKIP] {p_i.name} vs {p_j.name}: no shared {key_i}s")
                continue
            shared_species = [c for c in sp_i if c in sp_j]
            if not shared_species:
                print(f"  [SKIP] {p_i.name} vs {p_j.name}: no shared species cols")
                continue

            mask_i = df_i[key_i].isin(shared_keys)
            mask_j = df_j[key_j].isin(shared_keys)
            a = df_i.loc[mask_i].sort_values(key_i)[shared_species].values.flatten()
            b = df_j.loc[mask_j].sort_values(key_j)[shared_species].values.flatten()

            pe = pearsonr(a, b).statistic
            sp = spearmanr(a, b).statistic
            print(f"  {p_i.name:>50s} vs {p_j.name}")
            print(f"    Pearson : {pe:.4f}    Spearman: {sp:.4f}    "
                  f"({len(shared_keys):,} keys x {len(shared_species)} species)")


# ---------------------------------------------------------------------------
# Macro AUC (optional, requires labels)
# ---------------------------------------------------------------------------

def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int]:
    """Macro AUC over species columns, skipping all-positive / all-negative
    columns the way training does."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for j in range(y_true.shape[1]):
        col_sum = y_true[:, j].sum()
        if 0 < col_sum < len(y_true):
            try:
                aucs.append(roc_auc_score(y_true[:, j], y_pred[:, j]))
            except ValueError:
                pass
    return (float(np.mean(aucs)) if aucs else float("nan"), len(aucs))


def build_focal_labels(focal_meta_csv: Path, species_cols: list[str],
                       filenames: list[str]) -> np.ndarray:
    """Build a multi-label matrix from train_metadata.csv. Picks up both
    primary_label and secondary_labels."""
    meta = pd.read_csv(focal_meta_csv).set_index("filename")
    sp2idx = {sp: i for i, sp in enumerate(species_cols)}
    Y = np.zeros((len(filenames), len(species_cols)), dtype=np.float32)
    for i, fn in enumerate(filenames):
        if fn not in meta.index:
            continue
        row = meta.loc[fn]
        primary = row.get("primary_label", "")
        if isinstance(primary, str) and primary in sp2idx:
            Y[i, sp2idx[primary]] = 1.0
        secondary = row.get("secondary_labels", "")
        if isinstance(secondary, str) and secondary not in ("", "[]"):
            cleaned = secondary.strip("[]' ").replace("'", "").replace('"', '')
            for sp in cleaned.split(","):
                sp = sp.strip().strip("'\" ")
                if sp and sp in sp2idx:
                    Y[i, sp2idx[sp]] = 1.0
    return Y


def build_soundscape_labels_from_folds(folds_csv: Path, fold: int,
                                       species_cols: list[str],
                                       row_ids: list[str]) -> np.ndarray:
    """Build a one-hot label matrix from data/folds/folds_soundscapes.csv,
    using primary_label of rows where fold == --fold."""
    folds_df = pd.read_csv(folds_csv)
    val = folds_df[folds_df["fold"] == fold].set_index("row_id")
    sp2idx = {sp: i for i, sp in enumerate(species_cols)}
    Y = np.zeros((len(row_ids), len(species_cols)), dtype=np.float32)
    for i, rid in enumerate(row_ids):
        if rid not in val.index:
            continue
        lbl = val.loc[rid, "primary_label"]
        if isinstance(lbl, str) and lbl in sp2idx:
            Y[i, sp2idx[lbl]] = 1.0
    return Y


def report_aucs(oofs, args) -> None:
    """Compute and print macro AUC for each OOF, given a label source."""
    if not (args.focal_meta or args.soundscape_folds):
        return

    print(f"\n=== Macro AUC ===")
    for path, df, key_col, species_cols in oofs:
        if args.focal_meta:
            if key_col != "filename":
                print(f"  [SKIP AUC] {path.name}: --focal-meta given but key is '{key_col}'")
                continue
            Y = build_focal_labels(Path(args.focal_meta), species_cols,
                                   df[key_col].tolist())
        else:
            if key_col != "row_id":
                print(f"  [SKIP AUC] {path.name}: --soundscape-folds given but key is '{key_col}'")
                continue
            Y = build_soundscape_labels_from_folds(
                Path(args.soundscape_folds), args.fold, species_cols,
                df[key_col].tolist())

        preds = df[species_cols].values.astype(np.float32)
        auc, n_scored = macro_auc(Y, preds)
        print(f"  {path.name:>60s}  AUC: {auc:.4f}  ({n_scored} classes scored)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("oofs", nargs="+", type=Path,
                        help="Two or more OOF prediction CSVs to compare. "
                             "Single file is allowed if you only want AUC.")
    parser.add_argument("--focal-meta", type=str, default=None,
                        help="Path to train_metadata.csv to compute focal AUC. "
                             "Use this when OOFs have a 'filename' key.")
    parser.add_argument("--soundscape-folds", type=str, default=None,
                        help="Path to folds_soundscapes.csv to compute soundscape AUC. "
                             "Use this when OOFs have a 'row_id' key.")
    parser.add_argument("--fold", type=int, default=0,
                        help="Which fold's val rows to use for soundscape labels (default 0)")
    args = parser.parse_args()

    if args.focal_meta and args.soundscape_folds:
        sys.exit("ERROR: pass --focal-meta OR --soundscape-folds, not both.")

    # Load all OOFs up front so we can fail fast on missing files
    oofs = []
    for p in args.oofs:
        if not p.exists():
            sys.exit(f"ERROR: OOF file not found: {p}")
        df, key_col, species_cols = load_oof(p)
        oofs.append((p, df, key_col, species_cols))
        print(f"  loaded {p}: {len(df):,} rows x {len(species_cols)} species "
              f"(key={key_col})")

    pairwise_correlations(oofs)
    report_aucs(oofs, args)


if __name__ == "__main__":
    main()
