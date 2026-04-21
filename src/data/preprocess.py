"""
BirdCLEF+ 2026 — Audio Preprocessing
======================================

Core functions for loading audio, computing mel-spectrograms,
and caching processed data.
"""

import os
import numpy as np
import librosa
from pathlib import Path
from typing import Optional


# Default spectrogram parameters (must match inference pipeline)
SAMPLE_RATE = 32000
N_MELS = 128
FMAX = 16000
HOP_LENGTH = 512
N_FFT = 2048
WINDOW_SECONDS = 5.0


def load_audio(
    path: str,
    sr: int = SAMPLE_RATE,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> np.ndarray:
    """Load audio file and resample to target sample rate.

    Args:
        path: Path to audio file (.ogg, .wav, .flac, .mp3).
        sr: Target sample rate.
        offset: Start time in seconds.
        duration: Duration in seconds (None = load full file).

    Returns:
        Mono waveform as float32 numpy array.
    """
    y, _ = librosa.load(path, sr=sr, mono=True, offset=offset, duration=duration)
    return y.astype(np.float32)


def compute_melspec_from_waveform(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    fmax: int = FMAX,
    hop_length: int = HOP_LENGTH,
    n_fft: int = N_FFT,
) -> np.ndarray:
    """Compute log-mel spectrogram from a waveform.

    Args:
        waveform: Mono audio waveform (1D float array).
        sr: Sample rate.
        n_mels: Number of mel frequency bins.
        fmax: Maximum frequency for mel filterbank.
        hop_length: Hop length in samples.
        n_fft: FFT window size.

    Returns:
        Log-mel spectrogram of shape (n_mels, T) in dB scale.
    """
    S = librosa.feature.melspectrogram(
        y=waveform, sr=sr,
        n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmax=fmax,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def segment_waveform(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    window_seconds: float = WINDOW_SECONDS,
    overlap: float = 0.0,
) -> list[np.ndarray]:
    """Segment a waveform into fixed-length windows.

    Args:
        waveform: Mono audio waveform.
        sr: Sample rate.
        window_seconds: Length of each window in seconds.
        overlap: Overlap fraction between windows (0.0 to 0.5).

    Returns:
        List of waveform segments, each zero-padded to window length.
    """
    window_samples = int(sr * window_seconds)
    step = int(window_samples * (1.0 - overlap))
    step = max(step, 1)

    segments = []
    total_samples = len(waveform)

    if total_samples <= window_samples:
        # Pad short clips
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[:total_samples] = waveform
        return [padded]

    for start in range(0, total_samples, step):
        end = start + window_samples
        segment = waveform[start:end]

        if len(segment) < window_samples:
            segment = np.pad(segment, (0, window_samples - len(segment)),
                             mode='constant')
        segments.append(segment)

    return segments


def cache_spectrograms(
    audio_dir: str,
    output_dir: str,
    file_list: list[str],
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    fmax: int = FMAX,
    hop_length: int = HOP_LENGTH,
    n_fft: int = N_FFT,
) -> dict:
    """Process all training audio and cache spectrograms as .npy files.

    Args:
        audio_dir: Root directory containing audio files.
        output_dir: Root directory for cached .npy spectrograms.
        file_list: List of relative file paths (e.g., 'species/XC123.ogg').
        sr, n_mels, fmax, hop_length, n_fft: Spectrogram parameters.

    Returns:
        Dict with processing statistics.
    """
    stats = {"processed": 0, "skipped": 0, "failed": 0, "total": len(file_list)}

    for i, filename in enumerate(file_list):
        base = os.path.splitext(filename)[0]
        out_path = os.path.join(output_dir, base + ".npy")

        # Skip if already cached
        if os.path.exists(out_path):
            stats["skipped"] += 1
            continue

        audio_path = os.path.join(audio_dir, filename)
        if not os.path.exists(audio_path):
            stats["failed"] += 1
            continue

        try:
            waveform = load_audio(audio_path, sr=sr)
            spec = compute_melspec_from_waveform(
                waveform, sr=sr, n_mels=n_mels, fmax=fmax,
                hop_length=hop_length, n_fft=n_fft,
            )

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, spec)
            stats["processed"] += 1

        except Exception as e:
            stats["failed"] += 1
            if stats["failed"] <= 5:
                print(f"  ⚠ Failed: {filename}: {e}")

        if (i + 1) % 1000 == 0 or i == len(file_list) - 1:
            print(f"  [{i+1:>6}/{len(file_list)}] "
                  f"processed={stats['processed']} "
                  f"skipped={stats['skipped']} "
                  f"failed={stats['failed']}")

    return stats
