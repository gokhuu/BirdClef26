"""
BirdCLEF+ 2026 — Inference Pipeline
=====================================

CPU-only inference for Kaggle submission with post-processing.

Post-processing applied:
  1. Prediction clipping (floor to 0.001)
  2. Temperature scaling (if fitted on OOF data)

Usage (local testing):
    python submissions/inference.py

On Kaggle: This is embedded in the submission notebook.
"""

import os
import sys
import time
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────
SAMPLE_RATE = 32000
N_MELS = 128
FMAX = 16000
HOP_LENGTH = 512
N_FFT = 2048
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
NUM_CLASSES = 234
SPEC_TIME_FRAMES = 313

# Post-processing parameters (update after fitting on OOF)
CLIP_MIN = 0.001        # Minimum prediction floor
CLIP_MAX = 0.999        # Maximum prediction ceiling
TEMPERATURE = 1.0       # Temperature scaling (1.0 = no scaling)
                        # Set this from fit_temperature() results

# ── Environment detection ──────────────────────────────────────────
ON_KAGGLE = os.path.exists("/kaggle/input")

if ON_KAGGLE:
    BASE_DIR = "/kaggle/input/birdclef-2026"
    # Update MODEL_DIR to point to your uploaded ONNX models dataset
    MODEL_DIR = "/kaggle/input/datasets/brandonkhuu/birdclef-2026-baseline-effb0-onnx"
    TEST_DIR = "/kaggle/input/competitions/birdclef-2026/test_soundscapes"
    SAMPLE_SUB = "/kaggle/input/competitions/birdclef-2026/sample_submission.csv"
else:
    BASE_DIR = "data/raw"
    MODEL_DIR = "experiments"
    TEST_DIR = os.path.join(BASE_DIR, "test_soundscapes")
    SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")


def setup_onnxruntime():
    """Import or install onnxruntime."""
    try:
        import onnxruntime as ort
        return ort
    except ImportError:
        import subprocess
        whl_pattern = "/kaggle/input/**/onnxruntime*.whl"
        whls = glob.glob(whl_pattern, recursive=True)
        if whls:
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "--no-deps", "-q", whls[0]])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "-q", "onnxruntime"])
        import onnxruntime as ort
        return ort


def find_onnx_models(model_dir, n_folds=5):
    """Find ONNX model files."""
    paths = []

    for fold_id in range(n_folds):
        candidates = [
            os.path.join(model_dir, f"baseline_effb0_fold{fold_id}", "best_model.onnx"),
            os.path.join(model_dir, f"v2_augment_fold{fold_id}", "best_model.onnx"),
            os.path.join(model_dir, f"v3_effb1_fold{fold_id}", "best_model.onnx"),
            os.path.join(model_dir, f"best_model_{fold_id}.onnx"),
            os.path.join(model_dir, f"fold{fold_id}.onnx"),
        ]
        for c in candidates:
            if os.path.exists(c):
                paths.append(c)
                break

    if not paths:
        all_onnx = sorted(glob.glob(os.path.join(model_dir, "**", "*.onnx"),
                                     recursive=True))
        regular = [f for f in all_onnx if "quantized" not in f]
        paths = (regular if regular else all_onnx)[:n_folds]

    return paths


def compute_melspec(waveform):
    """Compute log-mel spectrogram matching training pipeline."""
    import librosa

    S = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmax=FMAX,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    if S_db.shape[1] >= SPEC_TIME_FRAMES:
        S_db = S_db[:, :SPEC_TIME_FRAMES]
    else:
        S_db = np.pad(S_db, ((0, 0), (0, SPEC_TIME_FRAMES - S_db.shape[1])),
                      mode="constant", constant_values=S_db.min())
    return S_db


def predict_window(sessions, spec):
    """Run ensemble inference on a single window. Returns post-processed probs."""
    x = spec[np.newaxis, np.newaxis, :, :].astype(np.float32)

    all_logits = []
    for sess in sessions:
        logits = sess.run(None, {"input": x})[0][0]
        all_logits.append(logits)

    mean_logits = np.mean(all_logits, axis=0)

    # Post-processing: temperature scaling
    if TEMPERATURE != 1.0:
        mean_logits = mean_logits / TEMPERATURE

    # Sigmoid
    probs = 1.0 / (1.0 + np.exp(-mean_logits))

    # Post-processing: clipping
    probs = np.clip(probs, CLIP_MIN, CLIP_MAX)

    return probs


def main():
    import librosa

    ort = setup_onnxruntime()
    print(f"Environment: {'Kaggle' if ON_KAGGLE else 'Local'}")
    print(f"onnxruntime: {ort.__version__}")

    # Load species list
    sample_sub = pd.read_csv(SAMPLE_SUB)
    species_list = [c for c in sample_sub.columns if c != "row_id"]
    assert len(species_list) == NUM_CLASSES

    # Load models
    model_paths = find_onnx_models(MODEL_DIR)
    print(f"Found {len(model_paths)} ONNX models")

    sessions = []
    for p in model_paths:
        sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        sessions.append(sess)
        size_mb = os.path.getsize(p) / 1024 / 1024
        print(f"  {os.path.basename(p)}: {size_mb:.1f} MB")

    print(f"\nPost-processing: clip_min={CLIP_MIN}, temperature={TEMPERATURE}")

    # Find test audio
    audio_files = []
    for ext in ("*.ogg", "*.wav", "*.flac", "*.mp3"):
        audio_files.extend(glob.glob(os.path.join(TEST_DIR, ext)))
    audio_files = sorted(audio_files)
    print(f"Test soundscapes: {len(audio_files)}")

    # Process
    total_start = time.time()
    rows = []

    for file_idx, audio_path in enumerate(audio_files):
        soundscape_id = os.path.splitext(os.path.basename(audio_path))[0]
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        n_windows = int(np.ceil(len(y) / WINDOW_SAMPLES))

        for win_idx in range(n_windows):
            start = win_idx * WINDOW_SAMPLES
            segment = y[start:start + WINDOW_SAMPLES]

            if len(segment) < WINDOW_SAMPLES:
                segment = np.pad(segment, (0, WINDOW_SAMPLES - len(segment)),
                                 mode="constant")

            end_time = (win_idx + 1) * int(WINDOW_SECONDS)
            row_id = f"{soundscape_id}_{end_time}"

            spec = compute_melspec(segment)
            probs = predict_window(sessions, spec)

            row = {"row_id": row_id}
            for sp_idx, sp in enumerate(species_list):
                row[sp] = float(probs[sp_idx])
            rows.append(row)

        if (file_idx + 1) % 10 == 0 or file_idx == 0:
            elapsed = time.time() - total_start
            rate = (file_idx + 1) / elapsed
            remaining = (len(audio_files) - file_idx - 1) / rate if rate > 0 else 0
            print(f"  [{file_idx+1}/{len(audio_files)}] {soundscape_id} "
                  f"({elapsed/60:.1f}min elapsed, ~{remaining/60:.1f}min remaining)")

    # Build submission
    expected_cols = list(sample_sub.columns)
    if rows:
        submission = pd.DataFrame(rows)
        for col in expected_cols:
            if col not in submission.columns:
                submission[col] = CLIP_MIN  # Use clip_min as default
        submission = submission[expected_cols]
    else:
        submission = sample_sub.copy()
        # Apply clip_min to placeholder
        for col in species_list:
            submission[col] = CLIP_MIN
        print("⚠ No audio files found — using placeholder submission")

    submission.to_csv("submission.csv", index=False)
    total = time.time() - total_start
    print(f"\nDone! {len(rows)} rows in {total:.0f}s ({total/60:.1f}min)")
    print(f"Submission shape: {submission.shape}")


if __name__ == "__main__":
    main()
