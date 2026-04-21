"""
BirdCLEF+ 2026 — Run All Folds
================================

Convenience script to train all 5 folds sequentially from a single config.

Usage:
    python src/training/run_all_folds.py configs/experiment_v2_augment.yaml

The run_name in the config should end with _fold0 (or have no fold suffix).
This script will replace the fold number and run each fold.
"""

import sys
import os
import re
import yaml
import subprocess


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/training/run_all_folds.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base_run_name = cfg.get("run_name", "experiment")
    # Strip any existing fold suffix
    base_run_name = re.sub(r"_fold\d+$", "", base_run_name)

    n_folds = cfg.get("n_folds", 5)

    print(f"{'='*60}")
    print(f"Running {n_folds} folds for: {base_run_name}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")

    for fold in range(n_folds):
        fold_run_name = f"{base_run_name}_fold{fold}"
        print(f"\n{'─'*60}")
        print(f"FOLD {fold}/{n_folds-1}: {fold_run_name}")
        print(f"{'─'*60}\n")

        # Run train.py with fold override
        cmd = [
            sys.executable, "src/training/train.py",
            config_path,
            "--fold", str(fold),
        ]

        # We need to temporarily modify the run_name in the config
        # Use environment variable or just pass the fold override
        env = os.environ.copy()
        env["BIRDCLEF_RUN_NAME"] = fold_run_name

        result = subprocess.run(cmd, env=env)

        if result.returncode != 0:
            print(f"\n⚠ Fold {fold} failed with return code {result.returncode}")
            print("Continuing with next fold...\n")

    print(f"\n{'='*60}")
    print(f"All folds complete for: {base_run_name}")
    print(f"Next: python src/utils/aggregate_oof.py experiments/{base_run_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
