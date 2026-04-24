"""
BirdCLEF+ 2026 — Unlabeled Soundscape Preprocessing
====================================================

Walks every .ogg in --audio-dir, slices into 5-second windows, and writes
one .npy spec per window into --spec-out-dir. No labels needed — the output
is consumed by generate_pseudo_labels.py, which produces soft labels via
the fine-tuned ensemble.

Output naming matches preprocess_soundscapes.py exactly:
    {stem}_{start_seconds:03d}.npy
so labeled row_ids (e.g. BC2026_Train_0001_S08_20250606_030007_000) collide
deterministically with their unlabeled counterparts. We use this for the
exclusion gate in generate_pseudo_labels.py.

Run from project root:
    python src/data/preprocess_unlabeled_soundscapes.py
    python src/data/preprocess_unlabeled_soundscapes.py --max-files 1000  # smoke test
    python src/data/preprocess_unlabeled_soundscapes.py --n-workers 8
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.getcwd())
from src.data.preprocess import (
    load_audio,
    compute_melspec_from_waveform,
    SAMPLE_RATE,
    WINDOW_SECONDS,
)


# -----------------------------------------------------------------
# Per-file worker
# -----------------------------------------------------------------

def process_one_file(
    audio_path: Path,
    spec_out_dir: Path,
    sr: int,
    window_seconds: float,
) -> tuple[str, int, int]:
    """Slice one .ogg into windows, save each as .npy. Returns (status, n_written, n_skipped).

    Status: "ok" | "skipped_all" | "load_failed" | "empty"
    """
    stem = audio_path.stem
    window_samples = int(sr * window_seconds)

    # Quick skip: if every expected output already exists, don't even load audio.
    # We don't yet know the file's true length, so we use a generous upper bound
    # of 12 windows (60s files). Files shorter than that re-trigger load below.
    expected_starts = [i * int(window_seconds) for i in range(12)]
    expected_paths = [spec_out_dir / f"{stem}_{s:03d}.npy" for s in expected_starts]
    if all(p.exists() for p in expected_paths):
        return ("skipped_all", 0, len(expected_paths))

    try:
        waveform = load_audio(str(audio_path), sr=sr)
    except Exception as e:
        return ("load_failed", 0, 0)

    n_samples = len(waveform)
    if n_samples < window_samples:
        return ("empty", 0, 0)

    n_written = 0
    n_skipped = 0
    for start in range(0, n_samples, window_samples):
        end = start + window_samples
        if end > n_samples:
            # Pad final partial window to exact length (matches segment_waveform behavior).
            segment = np.pad(waveform[start:], (0, end - n_samples), mode="constant")
        else:
            segment = waveform[start:end]

        start_seconds = start // sr
        out_path = spec_out_dir / f"{stem}_{start_seconds:03d}.npy"
        if out_path.exists():
            n_skipped += 1
            continue

        spec = compute_melspec_from_waveform(segment, sr=sr)
        np.save(out_path, spec)
        n_written += 1

    return ("ok", n_written, n_skipped)


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir",    default="data/raw/train_soundscapes")
    parser.add_argument("--spec-out-dir", default="data/processed_unlabeled_soundscapes")
    parser.add_argument("--sr",            type=int,   default=SAMPLE_RATE)
    parser.add_argument("--window-seconds",type=float, default=WINDOW_SECONDS)
    parser.add_argument("--n-workers",    type=int,   default=4,
                        help="Parallel processes (CPU cores). Use 1 to debug.")
    parser.add_argument("--max-files",    type=int,   default=None,
                        help="Random-sample at most N files. Useful for smoke tests "
                             "or capping disk usage.")
    parser.add_argument("--seed",         type=int,   default=42,
                        help="Seed for --max-files sampling")
    args = parser.parse_args()

    audio_dir    = Path(args.audio_dir)
    spec_out_dir = Path(args.spec_out_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"--audio-dir does not exist: {audio_dir}")
    spec_out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(audio_dir.glob("*.ogg"))
    if not files:
        raise RuntimeError(f"No .ogg files found in {audio_dir}")
    print(f"Found {len(files):,} .ogg files in {audio_dir}")

    if args.max_files is not None and args.max_files < len(files):
        random.Random(args.seed).shuffle(files)
        files = files[:args.max_files]
        print(f"Sampled {len(files):,} files (--max-files={args.max_files}, seed={args.seed})")

    print(f"Output dir:  {spec_out_dir}")
    print(f"Workers:     {args.n_workers}")
    print(f"Spec params: sr={args.sr}, window={args.window_seconds}s, "
          f"~{int(args.sr * args.window_seconds)} samples/window")
    print()

    worker = partial(
        process_one_file,
        spec_out_dir=spec_out_dir,
        sr=args.sr,
        window_seconds=args.window_seconds,
    )

    stats = {"ok": 0, "skipped_all": 0, "load_failed": 0, "empty": 0}
    n_written_total = 0
    n_skipped_total = 0
    t0 = time.time()

    if args.n_workers <= 1:
        results = (worker(f) for f in files)
    else:
        pool = Pool(args.n_workers)
        results = pool.imap_unordered(worker, files, chunksize=8)

    try:
        for i, (status, n_written, n_skipped) in enumerate(results, 1):
            stats[status] += 1
            n_written_total += n_written
            n_skipped_total += n_skipped

            if i % 200 == 0 or i == len(files):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                eta = (len(files) - i) / max(rate, 1e-6)
                print(f"  [{i:>6}/{len(files)}] "
                      f"ok={stats['ok']} skipped_all={stats['skipped_all']} "
                      f"failed={stats['load_failed']} empty={stats['empty']}  "
                      f"specs_written={n_written_total:,}  "
                      f"{rate:.1f} files/s  eta {eta/60:.1f}min")
    finally:
        if args.n_workers > 1:
            pool.close()
            pool.join()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} min")
    print(f"  files processed       : {stats['ok']:,}")
    print(f"  files fully cached    : {stats['skipped_all']:,}")
    print(f"  load failures         : {stats['load_failed']:,}")
    print(f"  files too short       : {stats['empty']:,}")
    print(f"  total specs written   : {n_written_total:,}")
    print(f"  total specs skipped   : {n_skipped_total:,}")

    # Disk-usage estimate
    if n_written_total > 0:
        sample = next(spec_out_dir.glob("*.npy"), None)
        if sample:
            bytes_per = sample.stat().st_size
            total_mb = (n_written_total * bytes_per) / (1024 ** 2)
            print(f"  approx disk usage now : {total_mb:,.0f} MB "
                  f"({bytes_per/1024:.0f} KB/spec)")


if __name__ == "__main__":
    main()