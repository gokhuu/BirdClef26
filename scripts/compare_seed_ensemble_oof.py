"""
Compare single-seed vs multi-seed ensemble OOF performance.

Reads oof_preds.csv (focal val) and oof_preds_soundscape.csv from each
(seed, fold) experiment directory, averages probabilities across seeds
within each fold, and reports:

  - Per-fold single-seed AUCs (one row per seed)
  - Per-fold ensemble AUC (averaged across seeds)
  - Per-fold lift over best-single-seed and mean-single-seed
  - Overall mean lift

Run from project root:
    python scripts/compare_seed_ensemble_oof.py

Assumes experiment dir layout:
    experiments/seresnext_finetune_fold{F}/                 (seed=42)
    experiments/seresnext_finetune_seed{S}_fold{F}/         (seed!=42)
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Project root on path so we can reuse the AUC helper
sys.path.insert(0, os.getcwd())
from src.training.train import compute_macro_auc, get_target_species


# ---------------------------------------------------------------------------
# Config — edit these if your seeds / folds / dir naming differ
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 2024]
FOLDS = [0, 1, 2, 3, 4]
EXP_ROOT = Path("experiments")

# Path to any one finetune config (used to recover target_species ordering
# and to locate the labels CSVs). Any seed/fold's saved config works.
REFERENCE_CONFIG = EXP_ROOT / "seresnext_finetune_fold0" / "config.yaml"


def exp_dir(seed: int, fold: int) -> Path:
    """Return the experiment directory for a given (seed, fold)."""
    if seed == 42:
        return EXP_ROOT / f"seresnext_finetune_fold{fold}"
    return EXP_ROOT / f"seresnext_finetune_seed{seed}_fold{fold}"


# ---------------------------------------------------------------------------
# Label reconstruction
# ---------------------------------------------------------------------------

def build_focal_labels(focal_meta_csv: str, target_species: list[str],
                       filenames: list[str]) -> np.ndarray:
    """Rebuild the multi-label matrix for the focal val rows, in the order
    they appear in oof_preds.csv (which matches ds_val.df["filename"]).
    """
    meta = pd.read_csv(focal_meta_csv).set_index("filename")
    species_to_idx = {sp: i for i, sp in enumerate(target_species)}
    n = len(filenames)
    Y = np.zeros((n, len(target_species)), dtype=np.float32)

    for i, fn in enumerate(filenames):
        row = meta.loc[fn]
        primary = row.get("primary_label", "")
        if isinstance(primary, str) and primary in species_to_idx:
            Y[i, species_to_idx[primary]] = 1.0

        secondary = row.get("secondary_labels", "")
        if isinstance(secondary, str) and secondary not in ("", "[]"):
            cleaned = secondary.strip("[]' ").replace("'", "").replace('"', '')
            for sp in cleaned.split(","):
                sp = sp.strip().strip("'\" ")
                if sp and sp in species_to_idx:
                    Y[i, species_to_idx[sp]] = 1.0
    return Y


def build_soundscape_labels(labels_csv: str, target_species: list[str],
                            row_ids: list[str]) -> np.ndarray:
    """Rebuild the soundscape multi-label matrix in row_id order."""
    labels_df = pd.read_csv(labels_csv).set_index("row_id")
    species_to_idx = {sp: i for i, sp in enumerate(target_species)}
    n = len(row_ids)
    Y = np.zeros((n, len(target_species)), dtype=np.float32)

    species_cols = [c for c in labels_df.columns
                    if c in species_to_idx]

    for i, rid in enumerate(row_ids):
        if rid not in labels_df.index:
            continue
        row = labels_df.loc[rid]
        for sp in species_cols:
            if row.get(sp, 0) > 0:
                Y[i, species_to_idx[sp]] = 1.0
    return Y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_oof(path: Path, key_col: str) -> tuple[list[str], np.ndarray, list[str]]:
    """Load an OOF csv. Returns (key_values, prob_matrix, species_columns)."""
    df = pd.read_csv(path)
    keys = df[key_col].tolist()
    species_cols = [c for c in df.columns if c != key_col]
    probs = df[species_cols].values.astype(np.float32)
    return keys, probs, species_cols


def evaluate_fold(fold: int, target_species: list[str],
                  focal_meta_csv: str,
                  soundscape_labels_csv: str | None) -> dict:
    """Load OOF preds for all seeds on this fold, compute single-seed and
    ensemble AUCs over the combined (focal + soundscape) val set.

    Returns dict with per-seed and ensemble AUCs for this fold.
    """
    focal_preds_per_seed = []
    sc_preds_per_seed = []
    focal_keys = None
    sc_keys = None

    for seed in SEEDS:
        d = exp_dir(seed, fold)
        focal_path = d / "oof_preds.csv"
        sc_path = d / "oof_preds_soundscape.csv"

        if not focal_path.exists():
            raise FileNotFoundError(f"missing {focal_path}")

        keys_f, probs_f, _ = load_oof(focal_path, key_col="filename")
        if focal_keys is None:
            focal_keys = keys_f
        elif keys_f != focal_keys:
            raise ValueError(
                f"focal OOF row order mismatch between seeds for fold {fold}. "
                f"This shouldn't happen if val ordering is deterministic."
            )
        focal_preds_per_seed.append(probs_f)

        if sc_path.exists():
            keys_s, probs_s, _ = load_oof(sc_path, key_col="row_id")
            if sc_keys is None:
                sc_keys = keys_s
            elif keys_s != sc_keys:
                raise ValueError(
                    f"soundscape OOF row order mismatch between seeds for fold {fold}."
                )
            sc_preds_per_seed.append(probs_s)

    # Build label matrices once per fold
    focal_Y = build_focal_labels(focal_meta_csv, target_species, focal_keys)

    sc_Y = None
    if sc_keys and soundscape_labels_csv:
        sc_Y = build_soundscape_labels(soundscape_labels_csv, target_species, sc_keys)

    # Per-seed combined AUC
    per_seed_auc = {}
    for seed, fp in zip(SEEDS, focal_preds_per_seed):
        if sc_Y is not None and sc_preds_per_seed:
            sp = sc_preds_per_seed[SEEDS.index(seed)]
            preds = np.concatenate([fp, sp], axis=0)
            labels = np.concatenate([focal_Y, sc_Y], axis=0)
        else:
            preds = fp
            labels = focal_Y
        per_seed_auc[seed] = compute_macro_auc(labels, preds)

    # Ensemble: average probs across seeds
    focal_ens = np.mean(np.stack(focal_preds_per_seed, axis=0), axis=0)
    if sc_preds_per_seed and sc_Y is not None:
        sc_ens = np.mean(np.stack(sc_preds_per_seed, axis=0), axis=0)
        ens_preds = np.concatenate([focal_ens, sc_ens], axis=0)
        ens_labels = np.concatenate([focal_Y, sc_Y], axis=0)
    else:
        ens_preds = focal_ens
        ens_labels = focal_Y
    ens_auc = compute_macro_auc(ens_labels, ens_preds)

    return {"per_seed": per_seed_auc, "ensemble": ens_auc}


def main():
    # Load reference config to find label CSV paths and species ordering
    if not REFERENCE_CONFIG.exists():
        raise FileNotFoundError(
            f"Reference config not found: {REFERENCE_CONFIG}. "
            f"Edit REFERENCE_CONFIG at top of script if needed."
        )
    with open(REFERENCE_CONFIG) as f:
        cfg = yaml.safe_load(f)

    target_species = get_target_species(cfg)
    print(f"Loaded {len(target_species)} target species from reference config")

    focal_meta_csv = cfg["train_meta_csv"]
    soundscape_labels_csv = cfg.get("soundscape_labels_csv")

    if not soundscape_labels_csv or not Path(soundscape_labels_csv).exists():
        print(f"  (soundscape labels not found at {soundscape_labels_csv} — "
              f"falling back to focal-only AUC)")
        soundscape_labels_csv = None

    # Per-fold evaluation
    print(f"\n{'='*72}")
    header = f"{'fold':>4} | " + " | ".join(f"seed={s:>4}" for s in SEEDS) + \
             f" | {'mean':>7} | {'best':>7} | {'ensemble':>8} | {'lift_v_mean':>11} | {'lift_v_best':>11}"
    print(header)
    print('-' * len(header))

    all_results = []
    lifts_v_mean = []
    lifts_v_best = []

    for fold in FOLDS:
        try:
            r = evaluate_fold(fold, target_species, focal_meta_csv,
                              soundscape_labels_csv)
        except FileNotFoundError as e:
            print(f"  fold {fold}: SKIPPED — {e}")
            continue

        per_seed = r["per_seed"]
        ens = r["ensemble"]
        seed_vals = [per_seed[s] for s in SEEDS]
        mean_seed = float(np.mean(seed_vals))
        best_seed = float(np.max(seed_vals))
        lift_mean = ens - mean_seed
        lift_best = ens - best_seed
        lifts_v_mean.append(lift_mean)
        lifts_v_best.append(lift_best)

        seed_strs = " | ".join(f"  {per_seed[s]:.4f}" for s in SEEDS)
        print(f"{fold:>4} | {seed_strs} | {mean_seed:.4f} | {best_seed:.4f} | "
              f"  {ens:.4f} |    {lift_mean:+.4f} |    {lift_best:+.4f}")

        all_results.append({"fold": fold, "ensemble_auc": ens,
                            "mean_seed_auc": mean_seed, "best_seed_auc": best_seed,
                            **{f"seed_{s}_auc": per_seed[s] for s in SEEDS}})

    # Overall summary
    if lifts_v_mean:
        print('-' * len(header))
        print(f"\nOverall ensemble lift:")
        print(f"  vs mean single-seed:   {np.mean(lifts_v_mean):+.4f}  "
              f"(per-fold: {[f'{x:+.4f}' for x in lifts_v_mean]})")
        print(f"  vs best single-seed:   {np.mean(lifts_v_best):+.4f}  "
              f"(per-fold: {[f'{x:+.4f}' for x in lifts_v_best]})")

        print(f"\nInterpretation:")
        mean_lift = np.mean(lifts_v_mean)
        if mean_lift >= 0.003:
            print(f"  Solid lift ({mean_lift:+.4f}). Ensemble is worth submitting.")
        elif mean_lift >= 0.001:
            print(f"  Modest lift ({mean_lift:+.4f}). Submit if you have spare attempts;")
            print(f"  consider also adding architecture diversity for bigger gains.")
        else:
            print(f"  Minimal lift ({mean_lift:+.4f}). Seeds aren't producing diverse")
            print(f"  enough errors. Skip pure seed ensemble; pivot to architecture diversity.")

    # Save detailed CSV
    out_csv = Path("ensemble_oof_comparison.csv")
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    print(f"\nDetailed results written to: {out_csv}")


if __name__ == "__main__":
    main()
