# scripts/filter_ambience_pool.py
import argparse
import numpy as np
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="data/ambience_pool/pantanal_ambience.npy")
ap.add_argument("--output", default="data/ambience_pool/pantanal_ambience_clean.npy")
ap.add_argument(
    "--max-peak-mean-db",
    type=float,
    default=40.0,
    help="Drop clips whose (max - mean) exceeds this. "
    "Lower = stricter. Pantanal noise should be <30 dB.",
)
args = ap.parse_args()

pool = np.load(args.input, mmap_mode="r")
n = pool.shape[0]
print(f"Input: {pool.shape}  ({n} clips)")

keep_mask = np.zeros(n, dtype=bool)
peak_means = np.empty(n, dtype=np.float32)
for i in range(n):
    clip = np.array(pool[i])
    pm = float(clip.max() - clip.mean())
    peak_means[i] = pm
    keep_mask[i] = pm < args.max_peak_mean_db

n_keep = int(keep_mask.sum())
print(
    f"Kept {n_keep} / {n} clips ({100 * n_keep / n:.1f}%) "
    f"with peak-mean < {args.max_peak_mean_db} dB"
)
print(
    f"Dropped peak-mean range: "
    f"min={peak_means[~keep_mask].min():.1f}, "
    f"median={np.median(peak_means[~keep_mask]):.1f}, "
    f"max={peak_means[~keep_mask].max():.1f}"
)

filtered = np.stack([np.array(pool[i]) for i in range(n) if keep_mask[i]])
print(f"Output shape: {filtered.shape}  ({filtered.nbytes / 1024 / 1024:.0f} MB)")

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
np.save(args.output, filtered)
print(f"Saved: {args.output}")
