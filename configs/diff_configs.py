"""
Print keys that are in baseline.yaml but missing from experiment_sed_b0.yaml.

Usage:
    python diff_configs.py

Run from the project root (birdclef-2026/).
"""
import yaml
from pathlib import Path

BASELINE = Path("configs/baseline.yaml")
SED = Path("configs/experiment_sed_b0.yaml")


def main():
    if not BASELINE.exists():
        print(f"ERROR: {BASELINE} not found. Run this from the project root.")
        return
    if not SED.exists():
        print(f"ERROR: {SED} not found. Run this from the project root.")
        return

    with open(BASELINE) as f:
        baseline = yaml.safe_load(f)
    with open(SED) as f:
        sed = yaml.safe_load(f)

    missing = set(baseline) - set(sed)

    if not missing:
        print("No missing keys. SED config has everything baseline has.")
        return

    print(f"Keys in {BASELINE.name} but missing from {SED.name}:")
    print("-" * 60)
    for k in sorted(missing):
        print(f"  {k}: {baseline[k]!r}")
    print("-" * 60)
    print(f"\n{len(missing)} missing key(s). Copy these into {SED}.")

    # Also show keys that differ between the two (helpful for catching
    # overrides like experiment_name that should differ intentionally).
    shared = set(baseline) & set(sed)
    differing = {k: (baseline[k], sed[k]) for k in shared if baseline[k] != sed[k]}
    if differing:
        print(f"\nKeys present in both but with different values:")
        print("-" * 60)
        for k, (b, s) in sorted(differing.items()):
            print(f"  {k}:")
            print(f"    baseline: {b!r}")
            print(f"    sed:      {s!r}")


if __name__ == "__main__":
    main()
