"""
BirdCLEF+ 2026 — Spectrogram & Waveform Augmentations
=====================================================

Baseline augmentations (SpecAugment, Mixup, Gaussian noise) plus
domain-shift augmentations designed to close the gap between clean
focal recordings (Xeno-canto / iNat) and noisy passive soundscapes
(Pantanal test distribution).

New in v2:
  - Background noise injection (pink, brown, random clip mixing)
  - Random gain / volume scaling
  - Low-pass and high-pass filtering (simulates distance & mic variation)
  - Silence embedding (embeds vocalization in a mostly-silent window)

All augmentations are toggleable via probability parameters (0.0 = off).
Waveform-domain augmentations run BEFORE spectrogram computation.
Spectrogram-domain augmentations run AFTER.
"""

import numpy as np
from typing import Optional


# ===================================================================
# Spectrogram-domain augmentations (existing baseline)
# ===================================================================

def spec_augment(
    spec: np.ndarray,
    time_mask_max: int = 50,
    freq_mask_max: int = 20,
    n_time_masks: int = 2,
    n_freq_masks: int = 2,
) -> np.ndarray:
    """Apply SpecAugment: random time and frequency masking.

    Args:
        spec: Mel spectrogram array of shape (n_mels, T).
        time_mask_max: Maximum width of each time mask.
        freq_mask_max: Maximum width of each frequency mask.
        n_time_masks: Number of time masks to apply.
        n_freq_masks: Number of frequency masks to apply.

    Returns:
        Augmented spectrogram with masked regions set to the spec minimum.
    """
    spec = spec.copy()
    n_mels, n_frames = spec.shape
    fill_val = spec.min()

    for _ in range(n_time_masks):
        t = np.random.randint(0, max(time_mask_max, 1))
        t = min(t, n_frames)
        t0 = np.random.randint(0, max(n_frames - t, 1))
        spec[:, t0:t0 + t] = fill_val

    for _ in range(n_freq_masks):
        f = np.random.randint(0, max(freq_mask_max, 1))
        f = min(f, n_mels)
        f0 = np.random.randint(0, max(n_mels - f, 1))
        spec[f0:f0 + f, :] = fill_val

    return spec


def add_gaussian_noise(
    spec: np.ndarray,
    std: float = 0.01,
) -> np.ndarray:
    """Add Gaussian noise to a spectrogram.

    Args:
        spec: Mel spectrogram array.
        std: Standard deviation of the noise.

    Returns:
        Noisy spectrogram.
    """
    noise = np.random.randn(*spec.shape).astype(spec.dtype) * std
    return spec + noise


def apply_mixup(
    spec1: np.ndarray,
    label1: np.ndarray,
    spec2: np.ndarray,
    label2: np.ndarray,
    alpha: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Mixup: blend two spectrograms and their labels.

    Args:
        spec1, spec2: Mel spectrograms of the same shape.
        label1, label2: One-hot label vectors (234-dim).
        alpha: Beta distribution parameter.

    Returns:
        (blended_spec, blended_label) tuple.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    spec_mix = lam * spec1 + (1.0 - lam) * spec2
    label_mix = lam * label1 + (1.0 - lam) * label2
    return spec_mix, label_mix


# ===================================================================
# Waveform-domain augmentations (NEW — domain-shift)
# ===================================================================

def _generate_colored_noise(n_samples: int, color: str = "pink") -> np.ndarray:
    """Generate colored noise (pink or brown) via spectral shaping.

    Args:
        n_samples: Number of audio samples.
        color: 'pink' (1/f) or 'brown' (1/f^2) noise.

    Returns:
        Noise waveform normalized to unit RMS.
    """
    white = np.random.randn(n_samples)
    freqs = np.fft.rfftfreq(n_samples, d=1.0)
    freqs[0] = 1.0  # avoid division by zero

    spectrum = np.fft.rfft(white)

    if color == "pink":
        spectrum /= np.sqrt(freqs)
    elif color == "brown":
        spectrum /= freqs
    else:
        pass  # white noise — no modification

    noise = np.fft.irfft(spectrum, n=n_samples)
    rms = np.sqrt(np.mean(noise ** 2)) + 1e-9
    return (noise / rms).astype(np.float32)


def add_background_noise(
    waveform: np.ndarray,
    snr_min_db: float = 5.0,
    snr_max_db: float = 20.0,
    noise_type: str = "auto",
) -> np.ndarray:
    """Mix background noise into a waveform at a random SNR level.

    Simulates the ambient noise present in passive soundscape recordings.

    Args:
        waveform: Clean audio waveform (1D float array).
        snr_min_db: Minimum signal-to-noise ratio in dB.
        snr_max_db: Maximum signal-to-noise ratio in dB.
        noise_type: 'pink', 'brown', 'white', or 'auto' (random choice).

    Returns:
        Noisy waveform.
    """
    n = len(waveform)

    if noise_type == "auto":
        noise_type = np.random.choice(["pink", "brown", "white"])

    noise = _generate_colored_noise(n, color=noise_type)

    # Compute signal RMS (avoid silence causing issues)
    sig_rms = np.sqrt(np.mean(waveform ** 2)) + 1e-9
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-9

    # Random SNR
    snr_db = np.random.uniform(snr_min_db, snr_max_db)
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))

    noise = noise * (target_noise_rms / noise_rms)
    return (waveform + noise).astype(np.float32)


def random_gain(
    waveform: np.ndarray,
    gain_range_db: float = 6.0,
) -> np.ndarray:
    """Apply random gain/volume scaling to a waveform.

    Prevents the model from relying on absolute amplitude, which varies
    greatly between focal recordings (close mic) and passive soundscapes
    (distant birds, variable recorder sensitivity).

    Args:
        waveform: Audio waveform.
        gain_range_db: Maximum gain adjustment in dB (±).

    Returns:
        Gain-adjusted waveform.
    """
    gain_db = np.random.uniform(-gain_range_db, gain_range_db)
    gain_linear = 10.0 ** (gain_db / 20.0)
    return (waveform * gain_linear).astype(np.float32)


def _biquad_coefficients(fc: float, fs: float, filter_type: str, Q: float = 0.707):
    """Compute biquad filter coefficients for lowpass or highpass.

    Uses the Audio-EQ Cookbook formulas (Robert Bristow-Johnson).

    Args:
        fc: Cutoff frequency in Hz.
        fs: Sample rate in Hz.
        filter_type: 'lowpass' or 'highpass'.
        Q: Quality factor (0.707 = Butterworth).

    Returns:
        (b, a) coefficient arrays for scipy-style filtering.
    """
    w0 = 2.0 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)

    if filter_type == "lowpass":
        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
    elif filter_type == "highpass":
        b0 = (1.0 + cos_w0) / 2.0
        b1 = -(1.0 + cos_w0)
        b2 = (1.0 + cos_w0) / 2.0
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def _apply_filter(waveform: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Apply a biquad IIR filter using direct-form II transposed.

    Pure numpy implementation — no scipy dependency needed.
    """
    n = len(waveform)
    out = np.zeros(n, dtype=np.float64)

    # Direct-form II transposed
    z1 = 0.0
    z2 = 0.0
    for i in range(n):
        x = float(waveform[i])
        y = b[0] * x + z1
        z1 = b[1] * x - a[1] * y + z2
        z2 = b[2] * x - a[2] * y
        out[i] = y

    return out.astype(np.float32)


def random_filter(
    waveform: np.ndarray,
    sr: int = 32000,
    lowpass_min_hz: float = 4000.0,
    lowpass_max_hz: float = 14000.0,
    highpass_min_hz: float = 50.0,
    highpass_max_hz: float = 500.0,
    apply_lowpass_p: float = 0.5,
    apply_highpass_p: float = 0.5,
) -> np.ndarray:
    """Apply random low-pass and/or high-pass filtering.

    Simulates:
    - Distance: distant birds lose high-frequency content → lowpass
    - Microphone/recorder variability → highpass to remove low rumble

    Args:
        waveform: Audio waveform.
        sr: Sample rate.
        lowpass_min_hz: Minimum lowpass cutoff frequency.
        lowpass_max_hz: Maximum lowpass cutoff frequency.
        highpass_min_hz: Minimum highpass cutoff frequency.
        highpass_max_hz: Maximum highpass cutoff frequency.
        apply_lowpass_p: Probability of applying lowpass filter.
        apply_highpass_p: Probability of applying highpass filter.

    Returns:
        Filtered waveform.
    """
    result = waveform.copy()

    if np.random.random() < apply_lowpass_p:
        fc = np.random.uniform(lowpass_min_hz, lowpass_max_hz)
        # Clamp to Nyquist
        fc = min(fc, sr / 2 - 100)
        b, a = _biquad_coefficients(fc, sr, "lowpass")
        result = _apply_filter(result, b, a)

    if np.random.random() < apply_highpass_p:
        fc = np.random.uniform(highpass_min_hz, highpass_max_hz)
        b, a = _biquad_coefficients(fc, sr, "highpass")
        result = _apply_filter(result, b, a)

    return result


def embed_in_silence(
    waveform: np.ndarray,
    target_length: int,
    silence_floor_db: float = -60.0,
) -> np.ndarray:
    """Embed a short vocalization in a longer window of near-silence.

    Soundscapes are mostly quiet with brief bursts of bird activity.
    This augmentation teaches the model to handle sparse activations
    amidst silence.

    Args:
        waveform: Short audio clip (potentially shorter than target_length).
        target_length: Target number of samples for the output.
        silence_floor_db: Background noise floor in dB (relative to signal peak).

    Returns:
        Waveform of length target_length with the vocalization randomly placed.
    """
    clip_len = len(waveform)

    if clip_len >= target_length:
        # Just return a random crop — no silence embedding needed
        start = np.random.randint(0, clip_len - target_length + 1)
        return waveform[start:start + target_length].copy()

    # Create near-silent background
    output = np.zeros(target_length, dtype=np.float32)
    peak = np.abs(waveform).max() + 1e-9
    noise_amp = peak * (10.0 ** (silence_floor_db / 20.0))
    output += np.random.randn(target_length).astype(np.float32) * noise_amp

    # Place the vocalization at a random position
    max_start = target_length - clip_len
    start = np.random.randint(0, max_start + 1)
    output[start:start + clip_len] += waveform

    return output


# ===================================================================
# Orchestrator: apply all augmentations based on config
# ===================================================================

def add_bg_noise_spec(spec: np.ndarray, snr_min_db: float = 5.0, snr_max_db: float = 20.0) -> np.ndarray:
    """Add colored noise directly to a log-mel spectrogram.
    Approximates waveform noise injection without needing raw audio.
    """
    spec = spec.copy()
    n_mels, n_frames = spec.shape

    # Generate noise profile that's louder in low frequencies (pink-like)
    freq_weights = np.linspace(1.0, 0.3, n_mels).reshape(-1, 1)
    noise = np.random.randn(n_mels, n_frames).astype(spec.dtype) * freq_weights

    # Scale noise relative to signal level
    snr_db = np.random.uniform(snr_min_db, snr_max_db)
    signal_level = spec.max()
    noise_level = signal_level - snr_db
    noise = noise * 5.0 + noise_level  # shift noise to target dB level

    # Mix in log domain: log(signal + noise) ≈ max(signal, noise) for log-mel
    return np.logaddexp(spec / 10.0, noise / 10.0) * 10.0


def random_gain_spec(spec: np.ndarray, gain_range_db: float = 6.0) -> np.ndarray:
    """Random gain on log-mel spectrogram = adding a constant in dB."""
    gain_db = np.random.uniform(-gain_range_db, gain_range_db)
    return spec + gain_db


def random_filter_spec(
    spec: np.ndarray,
    lowpass_p: float = 0.5,
    highpass_p: float = 0.5,
) -> np.ndarray:
    """Simulate LP/HP filtering by attenuating frequency bands in the spectrogram."""
    spec = spec.copy()
    n_mels = spec.shape[0]

    if np.random.random() < lowpass_p:
        # Attenuate high frequencies (simulate distance)
        cutoff = np.random.randint(n_mels // 3, n_mels - 1)
        rolloff = np.linspace(0, -30, n_mels - cutoff)  # up to -30dB
        spec[cutoff:, :] += rolloff.reshape(-1, 1)

    if np.random.random() < highpass_p:
        # Attenuate low frequencies (simulate mic rumble removal)
        cutoff = np.random.randint(1, n_mels // 6)
        rolloff = np.linspace(-20, 0, cutoff)
        spec[:cutoff, :] += rolloff.reshape(-1, 1)

    return spec


def embed_silence_spec(spec: np.ndarray) -> np.ndarray:
    """Embed the active portion of a spectrogram in silence.
    Zeros out random left/right portions, keeping a random window of signal.
    """
    spec = spec.copy()
    n_mels, n_frames = spec.shape
    fill_val = spec.min()

    # Keep a random 30-70% of the time frames, silence the rest
    keep_frac = np.random.uniform(0.3, 0.7)
    keep_len = max(int(n_frames * keep_frac), 1)
    start = np.random.randint(0, n_frames - keep_len + 1)

    silenced = np.full_like(spec, fill_val)
    silenced[:, start:start + keep_len] = spec[:, start:start + keep_len]
    return silenced


def apply_spec_augmentations(spec: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply ALL augmentations in spectrogram domain. Train mode only."""

    # v2: Domain-shift augmentations (spectrogram approximations)
    if np.random.random() < cfg.get("aug_bg_noise_p", 0.0):
        spec = add_bg_noise_spec(
            spec,
            snr_min_db=cfg.get("snr_min_db", 5.0),
            snr_max_db=cfg.get("snr_max_db", 20.0),
        )

    if np.random.random() < cfg.get("aug_gain_p", 0.0):
        spec = random_gain_spec(
            spec, gain_range_db=cfg.get("gain_range_db", 6.0),
        )

    if np.random.random() < cfg.get("aug_filter_p", 0.0):
        spec = random_filter_spec(spec)

    if np.random.random() < cfg.get("aug_silence_p", 0.0):
        spec = embed_silence_spec(spec)

    # Original augmentations
    if np.random.random() < cfg.get("aug_spec_p", 0.0):
        spec = spec_augment(
            spec,
            time_mask_max=cfg.get("time_mask_max", 50),
            freq_mask_max=cfg.get("freq_mask_max", 20),
        )

    if np.random.random() < cfg.get("aug_noise_p", 0.0):
        spec = add_gaussian_noise(
            spec, std=cfg.get("gaussian_noise_std", 0.01),
        )

    return spec
