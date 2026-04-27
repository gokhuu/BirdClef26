#!/usr/bin/env python
"""
Benchmark CPU inference for a set of ONNX models. Answers the question:
"Can I afford to add these V2-S folds to my 5x B0 ensemble inside the 90min Kaggle budget?"

Kaggle CPU notebooks run on something approximating a 4-core Xeon at ~2.2GHz. Locally,
set --num_threads 4 and a thread-count-limited process to get a close approximation.
Your local CPU may be faster or slower than Kaggle — aim for a ~20% headroom over
the measurement.

Usage:
    python scripts/benchmark_cpu_inference.py \\
        --models experiments/effv2s_finetune_fold0/model_int8.onnx \\
                 experiments/effv2s_finetune_fold1/model_int8.onnx \\
        --num_test_clips 50 \\
        --clips_per_audio 12 \\
        --num_threads 4
"""

import argparse
import os
import time
from typing import List

import numpy as np


def build_session(model_path: str, num_threads: int):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])


def benchmark_model(
    model_path: str,
    num_clips: int,
    spec_shape=(1, 128, 313),
    batch_size: int = 1,
    num_threads: int = 4,
    warmup: int = 5,
):
    sess = build_session(model_path, num_threads)
    input_name = sess.get_inputs()[0].name

    # Generate synthetic specs — same dtype & shape as your preprocessed data
    specs = np.random.randn(num_clips, *spec_shape).astype(np.float32)
    # Normalize to your distribution roughly
    specs = specs * 17.0 + (-55.0)
    specs = (specs - (-55.0)) / 17.0  # no-op but documents the expected normalization

    # Warmup
    for i in range(min(warmup, num_clips)):
        sess.run(None, {input_name: specs[i:i+batch_size]})

    # Timed run
    start = time.perf_counter()
    for i in range(0, num_clips, batch_size):
        batch = specs[i:i + batch_size]
        if batch.shape[0] < batch_size:
            # pad last partial batch
            pad = np.zeros((batch_size - batch.shape[0], *spec_shape), dtype=np.float32)
            batch = np.concatenate([batch, pad], axis=0)
        _ = sess.run(None, {input_name: batch})
    elapsed = time.perf_counter() - start

    return {
        "model": os.path.basename(model_path),
        "size_mb": os.path.getsize(model_path) / 1e6,
        "elapsed_s": elapsed,
        "clips": num_clips,
        "ms_per_clip": elapsed / num_clips * 1000.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="ONNX model paths")
    parser.add_argument("--num_test_clips", type=int, default=60,
                        help="Number of 5-second clips to process per model (synthetic)")
    parser.add_argument("--clips_per_audio", type=int, default=12,
                        help="Kaggle test set: ~60s soundscapes at 5s windows = 12 clips each")
    parser.add_argument("--expected_num_files", type=int, default=700,
                        help="Approximate number of soundscape files in Kaggle hidden test set")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Try 1 (default) and 4 — Kaggle CPU usually best at 1 or 2 per fold")
    parser.add_argument("--num_threads", type=int, default=4,
                        help="ONNXRuntime intra-op threads; Kaggle CPU has ~4 cores")
    parser.add_argument("--remaining_budget_min", type=float, default=55.0,
                        help="Minutes of CPU budget left after your current 5x B0 ensemble")
    args = parser.parse_args()

    print(f"Benchmarking {len(args.models)} models with {args.num_test_clips} synthetic clips each")
    print(f"Threads: {args.num_threads}, batch: {args.batch_size}")
    print("-" * 80)

    results = []
    for m in args.models:
        r = benchmark_model(
            m,
            num_clips=args.num_test_clips,
            batch_size=args.batch_size,
            num_threads=args.num_threads,
        )
        results.append(r)
        print(f"{r['model']:50s}  {r['size_mb']:6.1f} MB  "
              f"{r['ms_per_clip']:6.1f} ms/clip")

    print("-" * 80)
    total_ms_per_clip = sum(r["ms_per_clip"] for r in results)
    total_clips = args.expected_num_files * args.clips_per_audio
    projected_s = total_ms_per_clip * total_clips / 1000.0
    projected_min = projected_s / 60.0

    # Apply a 20% conservatism factor for Kaggle CPU being typically slower than dev machines
    kaggle_projected_min = projected_min * 1.2

    print(f"\n=== Projection ===")
    print(f"Total ms/clip across all models:       {total_ms_per_clip:.1f} ms")
    print(f"Expected clips (files x clips/file):    {total_clips}")
    print(f"Projected runtime (local):              {projected_min:.1f} min")
    print(f"Projected runtime (Kaggle, +20% slower): {kaggle_projected_min:.1f} min")
    print(f"Remaining CPU budget:                   {args.remaining_budget_min:.1f} min")

    headroom = args.remaining_budget_min - kaggle_projected_min
    print(f"Headroom:                               {headroom:+.1f} min")

    print(f"\n=== Decision ===")
    if headroom >= 5:
        print(f"  [SHIP]  Comfortable fit. Ship all {len(args.models)} models.")
    elif headroom >= 0:
        print(f"  [RISKY] Tight fit (<5 min headroom). Consider dropping 1 fold or testing")
        print(f"          with --batch_size 2 which may be faster.")
    else:
        print(f"  [CUT]   Will overflow budget. Options in order:")
        print(f"            1. Ship top 3 folds only (by focal val AUC)")
        print(f"            2. Re-train single 'all-data' V2-S, ship that instead of 5 folds")
        print(f"            3. Drop V2-S partner, stick with 5x B0 at 0.882")


if __name__ == "__main__":
    main()
