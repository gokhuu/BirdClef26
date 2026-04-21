"""
BirdCLEF+ 2026 — Soundscape Preprocessing (Step 2)
==================================================

Processes the competition-provided labeled soundscape audio into the same
.npy cache format as focal training, but at PER-WINDOW granularity because
each 5-second window has its own label set (unlike focal files, which carry
one label for the whole recording).

Inputs:
  - data/raw/train_soundscapes_labels.csv  (24 rows/file: 12 windows × 2 annotators)
  - data/raw/train_soundscapes/*.ogg       (66 files, 60s each)
  - data/raw/taxonomy.csv                  (234 focal species)

Outputs:
  - data/processed_soundscapes/{stem}_{start:03d}.npy   (1 file per window, ~792 files)
  - data/folds/folds_soundscapes.csv                    (window metadata + fold column)

Run once from project root:
  python src/data/preprocess_soundscapes.py
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path for `src.data.preprocess` import
sys.path.insert(0, os.getcwd())
from src.data.preprocess import load_audio, compute_melspec_from_waveform


# -----------------------------------------------------------------
# Fold assignment
# -----------------------------------------------------------------

def fold_of(filename: str, n_folds: int = 5) -> int:
    """Deterministic fold assignment via md5 of filename stem.

    Same file → same fold, reproducibly, independent of CSV row order or
    dataset location on disk. Uses the stem so moving the files doesn't
    reassign folds.
    """
    stem = Path(filename).stem
    h = hashlib.md5(stem.encode("utf-8")).hexdigest()
    return int(h, 16) % n_folds


# -----------------------------------------------------------------
# Label parsing
# -----------------------------------------------------------------

def parse_label_string(label_str) -> set[str]:
    """Parse a semicolon-separated species-code string into a set.

    Handles NaN / empty / whitespace-only entries gracefully so that
    fully-negative windows yield an empty set rather than crashing.
    """
    if pd.isna(label_str):
        return set()
    codes = [c.strip() for c in str(label_str).split(";")]
    return {c for c in codes if c}


def union_annotator_labels(group: pd.DataFrame) -> set[str]:
    """Union of species codes across all annotator rows for one window.

    Normally 2 rows per (filename, start, end) group; robust to 1 or 3+
    in case the labels CSV has occasional annotation artefacts.
    """
    labels = set()
    for lbl in group["primary_label"]:
        labels |= parse_label_string(lbl)
    return labels

def parse_time_seconds(v) -> float:
    """Parse a time value that might be float seconds or 'HH:MM:SS' string.

    The competition CSV stores start/end as clock strings like '00:00:05'
    rather than raw seconds. We normalize to float seconds at load time so
    all downstream code (groupby, filename formatting, audio offset) can
    treat them as plain numbers.
    """
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:      # HH:MM:SS(.ss)
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        if len(parts) == 2:      # MM:SS(.ss)
            m, sec = parts
            return int(m) * 60 + float(sec)
    return float(s)


# -----------------------------------------------------------------
# Pre-processing validation (fail fast, before spending compute)
# -----------------------------------------------------------------

def validate_inputs(
    labels_df: pd.DataFrame,
    audio_dir: Path,
    taxonomy: set[str],
) -> None:
    required_cols = {"filename", "start", "end", "primary_label"}
    missing = required_cols - set(labels_df.columns)
    if missing:
        raise ValueError(f"Labels CSV missing columns: {missing}")

    # Every filename must have audio on disk
    unique_files = labels_df["filename"].unique()
    missing_audio = [f for f in unique_files if not (audio_dir / f).exists()]
    if missing_audio:
        raise FileNotFoundError(
            f"{len(missing_audio)} audio files in labels CSV missing from "
            f"{audio_dir}. First 5: {missing_audio[:5]}"
        )

    # Row-count sanity check (expected: 24 rows/file = 12 windows × 2 annotators)
    counts = labels_df.groupby("filename").size()
    odd_files = counts[counts != 24]
    if len(odd_files) > 0:
        print(f"  ℹ {len(odd_files)} files have !=24 annotation rows "
              f"(typical is 12 windows × 2 annotators = 24). "
              f"Single-annotator windows are fine — union logic handles them.")
        for fname, n in odd_files.head(10).items():
            print(f"      {fname}: {n} rows")

    # Every species code must exist in the focal taxonomy
    all_codes = set()
    for s in labels_df["primary_label"].dropna():
        all_codes |= parse_label_string(s)
    orphan = all_codes - taxonomy
    if orphan:
        raise ValueError(
            f"{len(orphan)} soundscape species not in focal taxonomy — "
            f"cannot map to classifier head indices. Orphans: {sorted(orphan)[:10]}"
        )

    print(f"  ✓ {len(unique_files)} files, {len(labels_df)} annotation rows, "
          f"{len(all_codes)} unique species — all present in taxonomy")


# -----------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------

def process_soundscapes(
    labels_csv: Path,
    audio_dir: Path,
    taxonomy_csv: Path,
    spec_out_dir: Path,
    folds_out_csv: Path,
    n_folds: int = 5,
    window_seconds: float = 5.0,
) -> None:
    print("Loading inputs...")
    labels_df = pd.read_csv(labels_csv)
    taxonomy = set(pd.read_csv(taxonomy_csv)["primary_label"].astype(str))

    # Normalize start/end to float seconds. The CSV stores them as 'HH:MM:SS'.
    labels_df["start"] = labels_df["start"].apply(parse_time_seconds)
    labels_df["end"]   = labels_df["end"].apply(parse_time_seconds)

    print("Validating inputs...")
    validate_inputs(labels_df, audio_dir, taxonomy)

    spec_out_dir.mkdir(parents=True, exist_ok=True)
    folds_out_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- Group by window, union labels across annotators ---
    print("Grouping windows and unioning annotator labels...")
    grouped = labels_df.groupby(["filename", "start", "end"], sort=True)
    windows = []
    for (filename, start, end), group in grouped:
        species = union_annotator_labels(group)
        windows.append({
            "filename": filename,
            "start": start,
            "end": end,
            "primary_label": ";".join(sorted(species)),
            "n_species": len(species),
        })
    windows_df = pd.DataFrame(windows)
    n_files = labels_df["filename"].nunique()
    print(f"  {len(windows_df)} unique windows (expected ~{n_files * 12})")

    # --- Per-window: load 5s audio slice, compute spec, cache ---
    print("Computing and caching per-window spectrograms...")
    stats = {"processed": 0, "skipped": 0, "failed": 0}
    rows = []
    failures = []

    for i, w in enumerate(windows_df.itertuples(index=False)):
        stem = Path(w.filename).stem
        start_int = int(round(float(w.start)))
        row_id = f"{stem}_{start_int:03d}"
        npy_path = spec_out_dir / f"{row_id}.npy"

        rows.append({
            "row_id": row_id,
            "filename": w.filename,
            "start": float(w.start),
            "end": float(w.end),
            "primary_label": w.primary_label,
            "n_species": w.n_species,
            "fold": fold_of(w.filename, n_folds),
        })

        if npy_path.exists():
            stats["skipped"] += 1
            continue

        try:
            waveform = load_audio(
                str(audio_dir / w.filename),
                offset=float(w.start),
                duration=window_seconds,
            )
            spec = compute_melspec_from_waveform(waveform)
            np.save(npy_path, spec)
            stats["processed"] += 1
        except Exception as e:
            stats["failed"] += 1
            failures.append((w.filename, float(w.start), str(e)))

        if (i + 1) % 100 == 0 or i == len(windows_df) - 1:
            print(f"  [{i+1:>4}/{len(windows_df)}] "
                  f"processed={stats['processed']} "
                  f"skipped={stats['skipped']} "
                  f"failed={stats['failed']}")

    if failures:
        print(f"\n  ⚠ {len(failures)} failures. First 5:")
        for f, s, e in failures[:5]:
            print(f"      {f} @ {s}s: {e}")

    # --- Write folds CSV ---
    folds_df = pd.DataFrame(rows)
    folds_df.to_csv(folds_out_csv, index=False)
    print(f"\n  Wrote {folds_out_csv} ({len(folds_df)} rows)")

    # --- Post-processing report ---
    print("\n" + "=" * 60)
    print("Post-processing report")
    print("=" * 60)

    n_zero = int((folds_df["n_species"] == 0).sum())
    print(f"Total windows:    {len(folds_df)}")
    print(f"Zero-label:       {n_zero} ({100*n_zero/max(len(folds_df),1):.1f}%)")
    print(f"Mean species/win: {folds_df['n_species'].mean():.2f}")
    print(f"Max species/win:  {folds_df['n_species'].max()}")

    print("\nPer-fold file and species coverage:")
    for fold in range(n_folds):
        sub = folds_df[folds_df["fold"] == fold]
        n_files_f = sub["filename"].nunique()
        species_in_fold = set()
        for s in sub["primary_label"]:
            species_in_fold |= parse_label_string(s)
        print(f"  fold {fold}: {n_files_f:>3} files, {len(sub):>4} windows, "
              f"{len(species_in_fold):>2} unique species")


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv",   default="data/raw/train_soundscapes_labels.csv")
    parser.add_argument("--audio-dir",    default="data/raw/train_soundscapes")
    parser.add_argument("--taxonomy-csv", default="data/raw/taxonomy.csv")
    parser.add_argument("--spec-out-dir", default="data/processed_soundscapes")
    parser.add_argument("--folds-out-csv",default="data/folds/folds_soundscapes.csv")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    process_soundscapes(
        labels_csv=Path(args.labels_csv),
        audio_dir=Path(args.audio_dir),
        taxonomy_csv=Path(args.taxonomy_csv),
        spec_out_dir=Path(args.spec_out_dir),
        folds_out_csv=Path(args.folds_out_csv),
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()