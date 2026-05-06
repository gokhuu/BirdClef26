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
    # SEResNeXt (default)
    python scripts/compare_seed_ensemble_oof.py

    # B0 v2 multi-seed
    python scripts/compare_seed_ensemble_oof.py \\
        --base-name sed_finetune_pseudo_v2 \\
        --seeds 42,123,2024

Assumes experiment dir layout (post-reorg):
    experiments/{base_name}/{base_name}_fold{F}/                 (default seed)
    experiments/{base_name}_seed{S}/{base_name}_seed{S}_fold{F}/ (other seeds)

Edge case: B0's per-seed groups in your tree are `sed_finetune_pseudo_seed2024/`
(no "_v2_" in the parent name) while the inner runs are `sed_finetune_pseudo_v2_seed2024_fold0`.
Use --seed-dir-pattern to override the default mapping if needed.
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Project root on path so we can reuse the AUC helper
sys.path.insert(0, os.getcwd())
from src.training.train import compute_macro_auc, get_target_species


EXP_ROOT = Path("experiments")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def exp_dir(base_name: str, seed: int, fold: int,
            default_seed: int,
            seed_dir_pattern: str | None) -> Path:
    """Resolve the experiment dir for (base_name, seed, fold) under the
    nested layout: experiments/{group}/{run_name}/.

    For the default seed the group is `base_name` and the run is
    `{base_name}_fold{F}`. For other seeds the group is `{base_name}_seed{S}`
    (overridable via seed_dir_pattern) and the run is `{base_name}_seed{S}_fold{F}`.
    """
    if seed == default_seed:
        run_name = f"{base_name}_fold{fold}"
        return EXP_ROOT / base_name / run_name

    if seed_dir_pattern:
        group = seed_dir_pattern.format(base=base_name, seed=seed)
        run_name = f"{group}_fold{fold}"
    else:
        group = f"{base_name}_seed{seed}"
        run_name = f"{group}_fold{fold}"
    return EXP_ROOT / group / run_name


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
# OOF loading & per-fold evaluation
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
                  soundscape_labels_csv: str | None,
                  base_name: str, seeds: list[int], default_seed: int,
                  seed_dir_pattern: str | None) -> dict:
    """Load OOF preds for all seeds on this fold, compute single-seed and
    ensemble AUCs over the combined (focal + soundscape) val set."""
    focal_preds_per_seed = []
    sc_preds_per_seed = []
    focal_keys = None
    sc_keys = None

    for seed in seeds:
        d = exp_dir(base_name, seed, fold, default_seed, seed_dir_pattern)
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

    focal_Y = build_focal_labels(focal_meta_csv, target_species, focal_keys)

    sc_Y = None
    if sc_keys and soundscape_labels_csv:
        sc_Y = build_soundscape_labels(soundscape_labels_csv, target_species, sc_keys)

    per_seed_auc = {}
    for seed, fp in zip(seeds, focal_preds_per_seed):
        if sc_Y is not None and sc_preds_per_seed:
            sp = sc_preds_per_seed[seeds.index(seed)]
            preds = np.concatenate([fp, sp], axis=0)
            labels = np.concatenate([focal_Y, sc_Y], axis=0)
        else:
            preds = fp
            labels = focal_Y
        per_seed_auc[seed] = compute_macro_auc(labels, preds)

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-name", default="seresnext_finetune",
                        help="Run-name prefix without _fold suffix (default: seresnext_finetune)")
    parser.add_argument("--seeds", default="42,123,2024",
                        help="Comma-separated seed list (default: 42,123,2024)")
    parser.add_argument("--default-seed", type=int, default=42,
                        help="Seed treated as the 'no suffix' baseline run (default: 42)")
    parser.add_argument("--folds", default="0,1,2,3,4",
                        help="Comma-separated fold list (default: 0,1,2,3,4)")
    parser.add_argument("--seed-dir-pattern", default=None,
                        help="Override group naming for non-default seeds. "
                             "Use {base} and {seed} placeholders. "
                             "Example: '{base}_seed{seed}'. "
                             "Use this if your seed-ensemble groups don't match "
                             "the default '{base_name}_seed{seed}' convention.")
    parser.add_argument("--reference-config", default=None,
                        help="Path to a config.yaml containing train_meta_csv "
                             "and soundscape_labels_csv. Defaults to fold0's config.")
    parser.add_argument("--output-csv", default="ensemble_oof_comparison.csv",
                        help="Where to write the detailed per-fold table")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    folds = [int(f) for f in args.folds.split(",")]

    # Default reference config: fold0 of the default-seed run
    if args.reference_config is None:
        ref_dir = exp_dir(args.base_name, args.default_seed, folds[0],
                          args.default_seed, args.seed_dir_pattern)
        reference_config = ref_dir / "config.yaml"
    else:
        reference_config = Path(args.reference_config)

    if not reference_config.exists():
        raise FileNotFoundError(
            f"Reference config not found: {reference_config}. "
            f"Pass --reference-config to override."
        )
    with open(reference_config) as f:
        cfg = yaml.safe_load(f)

    target_species = get_target_species(cfg)
    print(f"Loaded {len(target_species)} target species from {reference_config}")
    print(f"base_name={args.base_name}  seeds={seeds}  default_seed={args.default_seed}")

    focal_meta_csv = cfg["train_meta_csv"]
    soundscape_labels_csv = cfg.get("soundscape_labels_csv")

    if not soundscape_labels_csv or not Path(soundscape_labels_csv).exists():
        print(f"  (soundscape labels not found at {soundscape_labels_csv} — "
              f"falling back to focal-only AUC)")
        soundscape_labels_csv = None

    print(f"\n{'='*72}")
    header = (f"{'fold':>4} | "
              + " | ".join(f"seed={s:>4}" for s in seeds)
              + f" | {'mean':>7} | {'best':>7} | {'ensemble':>8} "
              + f"| {'lift_v_mean':>11} | {'lift_v_best':>11}")
    print(header)
    print('-' * len(header))

    all_results = []
    lifts_v_mean = []
    lifts_v_best = []

    for fold in folds:
        try:
            r = evaluate_fold(fold, target_species, focal_meta_csv,
                              soundscape_labels_csv,
                              args.base_name, seeds, args.default_seed,
                              args.seed_dir_pattern)
        except FileNotFoundError as e:
            print(f"  fold {fold}: SKIPPED — {e}")
            continue

        per_seed = r["per_seed"]
        ens = r["ensemble"]
        seed_vals = [per_seed[s] for s in seeds]
        mean_seed = float(np.mean(seed_vals))
        best_seed = float(np.max(seed_vals))
        lift_mean = ens - mean_seed
        lift_best = ens - best_seed
        lifts_v_mean.append(lift_mean)
        lifts_v_best.append(lift_best)

        seed_strs = " | ".join(f"  {per_seed[s]:.4f}" for s in seeds)
        print(f"{fold:>4} | {seed_strs} | {mean_seed:.4f} | {best_seed:.4f} | "
              f"  {ens:.4f} |    {lift_mean:+.4f} |    {lift_best:+.4f}")

        all_results.append({"fold": fold, "ensemble_auc": ens,
                            "mean_seed_auc": mean_seed, "best_seed_auc": best_seed,
                            **{f"seed_{s}_auc": per_seed[s] for s in seeds}})

    if lifts_v_mean:
        print('-' * len(header))
        print(f"\nOverall ensemble lift:")
        print(f"  vs mean single-seed:   {np.mean(lifts_v_mean):+.4f}  "
              f"(per-fold: {[f'{x:+.4f}' for x in lifts_v_mean]})")
        print(f"  vs best single-seed:   {np.mean(lifts_v_best):+.4f}  "
              f"(per-fold: {[f'{x:+.4f}' for x in lifts_v_best]})")

        print(f"\nInterpretation:")
        mean_lift = float(np.mean(lifts_v_mean))
        if mean_lift >= 0.003:
            print(f"  Solid lift ({mean_lift:+.4f}). Ensemble is worth submitting.")
        elif mean_lift >= 0.001:
            print(f"  Modest lift ({mean_lift:+.4f}). Submit if you have spare attempts;")
            print(f"  consider also adding architecture diversity for bigger gains.")
        else:
            print(f"  Minimal lift ({mean_lift:+.4f}). Seeds aren't producing diverse")
            print(f"  enough errors. Skip pure seed ensemble; pivot to architecture diversity.")

    out_csv = Path(args.output_csv)
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    print(f"\nDetailed results written to: {out_csv}")


if __name__ == "__main__":
    main()
