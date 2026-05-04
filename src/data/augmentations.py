"""
BirdCLEF+ 2026 — Spectrogram & Waveform Augmentations
=====================================================

Baseline augmentations (SpecAugment, Mixup, Gaussian noise) plus
domain-shift augmentations designed to close the gap between clean
focal recordings (Xeno-canto / iNat) and noisy passive soundscapes
(Pantanal test distribution).

v2:
  - Synthetic background noise injection (pink/brown/white)
  - Random gain / volume scaling
  - Low-pass and high-pass filtering
  - Silence embedding

v4:
  - Real-ambience mixing (add_real_ambience_spec). Mixes a real
    Pantanal-distribution ambience clip into the focal spec in log-mel
    dB space. The pool is built offline by scripts/build_ambience_pool.py
    from the unlabeled soundscapes by selecting windows the existing
    ensemble is highly uncertain about (max prob < 0.10).

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
    """Apply SpecAugment: random time and frequency masking."""
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


def add_gaussian_noise(spec: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to a spectrogram."""
    noise = np.random.randn(*spec.shape).astype(spec.dtype) * std
    return spec + noise


def apply_mixup(
    spec1: np.ndarray, label1: np.ndarray,
    spec2: np.ndarray, label2: np.ndarray,
    alpha: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Mixup: blend two spectrograms and their labels."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    spec_mix = lam * spec1 + (1.0 - lam) * spec2
    label_mix = lam * label1 + (1.0 - lam) * label2
    return spec_mix, label_mix


# ===================================================================
# Waveform-domain augmentations
# ===================================================================
# (Unchanged from v2 — these only fire on soundscape waveform path,
#  not on cached focal specs. See dataset.py for the firing rules.)

def _generate_colored_noise(n_samples: int, color: str = "pink") -> np.ndarray:
    white = np.random.randn(n_samples)
    freqs = np.fft.rfftfreq(n_samples, d=1.0)
    freqs[0] = 1.0
    spectrum = np.fft.rfft(white)
    if color == "pink":
        spectrum /= np.sqrt(freqs)
    elif color == "brown":
        spectrum /= freqs
    noise = np.fft.irfft(spectrum, n=n_samples)
    rms = np.sqrt(np.mean(noise ** 2)) + 1e-9
    return (noise / rms).astype(np.float32)


def add_background_noise(
    waveform: np.ndarray,
    snr_min_db: float = 5.0, snr_max_db: float = 20.0,
    noise_type: str = "auto",
) -> np.ndarray:
    n = len(waveform)
    if noise_type == "auto":
        noise_type = np.random.choice(["pink", "brown", "white"])
    noise = _generate_colored_noise(n, color=noise_type)
    sig_rms = np.sqrt(np.mean(waveform ** 2)) + 1e-9
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-9
    snr_db = np.random.uniform(snr_min_db, snr_max_db)
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    noise = noise * (target_noise_rms / noise_rms)
    return (waveform + noise).astype(np.float32)


def random_gain(waveform: np.ndarray, gain_range_db: float = 6.0) -> np.ndarray:
    gain_db = np.random.uniform(-gain_range_db, gain_range_db)
    gain_linear = 10.0 ** (gain_db / 20.0)
    return (waveform * gain_linear).astype(np.float32)


def _biquad_coefficients(fc: float, fs: float, filter_type: str, Q: float = 0.707):
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
    n = len(waveform)
    out = np.zeros(n, dtype=np.float64)
    z1, z2 = 0.0, 0.0
    for i in range(n):
        x = float(waveform[i])
        y = b[0] * x + z1
        z1 = b[1] * x - a[1] * y + z2
        z2 = b[2] * x - a[2] * y
        out[i] = y
    return out.astype(np.float32)


def random_filter(
    waveform: np.ndarray, sr: int = 32000,
    lowpass_min_hz: float = 4000.0, lowpass_max_hz: float = 14000.0,
    highpass_min_hz: float = 50.0, highpass_max_hz: float = 500.0,
    apply_lowpass_p: float = 0.5, apply_highpass_p: float = 0.5,
) -> np.ndarray:
    result = waveform.copy()
    if np.random.random() < apply_lowpass_p:
        fc = np.random.uniform(lowpass_min_hz, lowpass_max_hz)
        fc = min(fc, sr / 2 - 100)
        b, a = _biquad_coefficients(fc, sr, "lowpass")
        result = _apply_filter(result, b, a)
    if np.random.random() < apply_highpass_p:
        fc = np.random.uniform(highpass_min_hz, highpass_max_hz)
        b, a = _biquad_coefficients(fc, sr, "highpass")
        result = _apply_filter(result, b, a)
    return result


def embed_in_silence(
    waveform: np.ndarray, target_length: int,
    silence_floor_db: float = -60.0,
) -> np.ndarray:
    clip_len = len(waveform)
    if clip_len >= target_length:
        start = np.random.randint(0, clip_len - target_length + 1)
        return waveform[start:start + target_length].copy()
    output = np.zeros(target_length, dtype=np.float32)
    peak = np.abs(waveform).max() + 1e-9
    noise_amp = peak * (10.0 ** (silence_floor_db / 20.0))
    output += np.random.randn(target_length).astype(np.float32) * noise_amp
    max_start = target_length - clip_len
    start = np.random.randint(0, max_start + 1)
    output[start:start + clip_len] += waveform
    return output


# ===================================================================
# Spectrogram-domain domain-shift augmentations
# ===================================================================

def add_bg_noise_spec(spec: np.ndarray, snr_min_db: float = 5.0, snr_max_db: float = 20.0) -> np.ndarray:
    """Add SYNTHETIC colored noise to a log-mel spectrogram.

    NOTE (v4): empirically, this synthetic noise alone failed to move the
    LB (v3 = 0.883 = baseline). add_real_ambience_spec is the preferred
    domain-shift augmentation now. Kept for A/B comparison.
    """
    spec = spec.copy()
    n_mels, n_frames = spec.shape
    freq_weights = np.linspace(1.0, 0.3, n_mels).reshape(-1, 1)
    noise = np.random.randn(n_mels, n_frames).astype(spec.dtype) * freq_weights
    snr_db = np.random.uniform(snr_min_db, snr_max_db)
    signal_level = spec.max()
    noise_level = signal_level - snr_db
    noise = noise * 5.0 + noise_level
    return np.logaddexp(spec / 10.0, noise / 10.0) * 10.0


def random_gain_spec(spec: np.ndarray, gain_range_db: float = 6.0) -> np.ndarray:
    """Random gain on log-mel spectrogram = adding a constant in dB."""
    gain_db = np.random.uniform(-gain_range_db, gain_range_db)
    return spec + gain_db


def random_filter_spec(
    spec: np.ndarray, lowpass_p: float = 0.5, highpass_p: float = 0.5,
) -> np.ndarray:
    """Simulate LP/HP filtering by attenuating frequency bands in spec-domain."""
    spec = spec.copy()
    n_mels = spec.shape[0]
    if np.random.random() < lowpass_p:
        cutoff = np.random.randint(n_mels // 3, n_mels - 1)
        rolloff = np.linspace(0, -30, n_mels - cutoff)
        spec[cutoff:, :] += rolloff.reshape(-1, 1)
    if np.random.random() < highpass_p:
        cutoff = np.random.randint(1, n_mels // 6)
        rolloff = np.linspace(-20, 0, cutoff)
        spec[:cutoff, :] += rolloff.reshape(-1, 1)
    return spec


def embed_silence_spec(spec: np.ndarray) -> np.ndarray:
    """Zero out random left/right portions, keep a random window of signal."""
    spec = spec.copy()
    n_mels, n_frames = spec.shape
    fill_val = spec.min()
    keep_frac = np.random.uniform(0.3, 0.7)
    keep_len = max(int(n_frames * keep_frac), 1)
    start = np.random.randint(0, n_frames - keep_len + 1)
    silenced = np.full_like(spec, fill_val)
    silenced[:, start:start + keep_len] = spec[:, start:start + keep_len]
    return silenced


# ===================================================================
# Real-ambience mixing (NEW in v4)
# ===================================================================

# Module-level cache. Each DataLoader worker process has its own copy
# (workers fork from parent, but do not share post-fork module state by
# default). With mmap_mode='r' the actual spec data is paged in by the
# OS on demand and shared via the page cache, so additional workers
# don't multiply the RAM footprint of the pool.
_AMBIENCE_POOL_CACHE: dict = {}


def _load_ambience_pool(path: str) -> np.ndarray:
    """Lazy-load an ambience pool .npy file as a memory-mapped array.

    Cached per-path so multiple A/B pools can coexist in the same run.
    """
    if path not in _AMBIENCE_POOL_CACHE:
        _AMBIENCE_POOL_CACHE[path] = np.load(path, mmap_mode='r')
    return _AMBIENCE_POOL_CACHE[path]


def add_real_ambience_spec(
    spec: np.ndarray,
    ambience_pool_path: str,
    snr_min_db: float = -5.0,
    snr_max_db: float = 15.0,
) -> np.ndarray:
    """Mix a real Pantanal-distribution ambience clip into a focal spec.

    SNR convention: snr_db = (spec peak in dB) - (ambience peak in dB).
      snr_db = +15  : ambience is well below the bird signal
      snr_db =   0  : roughly equal peak levels
      snr_db =  -5  : ambience PEAK exceeds bird PEAK by 5 dB
                      (realistic for distant Pantanal recordings)

    Mixing is performed in log-mel dB space using logaddexp (matches the
    convention of add_bg_noise_spec; treats spec values as log-power and
    sums in linear-power before taking log again).

    Args:
        spec: focal log-mel of shape (n_mels, T_spec).
        ambience_pool_path: .npy file path; loaded lazily.
        snr_min_db, snr_max_db: SNR range to sample from.

    Returns:
        New (n_mels, T_spec) array with ambience mixed in.
    """
    pool = _load_ambience_pool(ambience_pool_path)

    # Pick one random ambience window from the pool.
    n_pool = len(pool)
    if n_pool == 0:
        return spec
    amb = np.array(pool[np.random.randint(n_pool)], dtype=spec.dtype)  # (n_mels, T)

    n_mels_spec, T_spec = spec.shape
    n_mels_amb,  T_amb  = amb.shape

    # Bail gracefully on shape mismatch — wrong pool was passed in.
    if n_mels_amb != n_mels_spec:
        return spec

    # Match time dim: random crop if longer, pad with min if shorter.
    if T_amb > T_spec:
        s = np.random.randint(0, T_amb - T_spec + 1)
        amb = amb[:, s:s + T_spec]
    elif T_amb < T_spec:
        amb = np.pad(amb, ((0, 0), (0, T_spec - T_amb)),
                     mode="constant", constant_values=amb.min())

    # Shift ambience so its peak is snr_db below the spec peak.
    snr_db = np.random.uniform(snr_min_db, snr_max_db)
    amb_shifted = amb - amb.max() + spec.max() - snr_db

    # Mix in log domain. Same convention as add_bg_noise_spec.
    return np.logaddexp(spec / 10.0, amb_shifted / 10.0) * 10.0


# ===================================================================
# Orchestrator: apply all augmentations based on config
# ===================================================================

def apply_spec_augmentations(spec: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply ALL spectrogram-domain augmentations. Train mode only."""

    # v4: Real-ambience mixing (preferred over synthetic add_bg_noise_spec).
    # Only fires if (a) probability > 0 AND (b) ambience_pool_path is set.
    # Both must come through cfg via build_aug_config.
    if (cfg.get("aug_real_ambience_p", 0.0) > 0
            and cfg.get("ambience_pool_path")
            and np.random.random() < cfg["aug_real_ambience_p"]):
        spec = add_real_ambience_spec(
            spec,
            ambience_pool_path=cfg["ambience_pool_path"],
            snr_min_db=cfg.get("real_ambience_snr_min_db", -5.0),
            snr_max_db=cfg.get("real_ambience_snr_max_db", 15.0),
        )

    # v2: Synthetic noise (kept for A/B; default off in v4).
    if np.random.random() < cfg.get("aug_bg_noise_p", 0.0):
        spec = add_bg_noise_spec(
            spec,
            snr_min_db=cfg.get("snr_min_db", 5.0),
            snr_max_db=cfg.get("snr_max_db", 20.0),
        )

    if np.random.random() < cfg.get("aug_gain_p", 0.0):
        spec = random_gain_spec(spec, gain_range_db=cfg.get("gain_range_db", 6.0))

    if np.random.random() < cfg.get("aug_filter_p", 0.0):
        spec = random_filter_spec(spec)

    if np.random.random() < cfg.get("aug_silence_p", 0.0):
        spec = embed_silence_spec(spec)

    # Original baseline augmentations.
    if np.random.random() < cfg.get("aug_spec_p", 0.0):
        spec = spec_augment(
            spec,
            time_mask_max=cfg.get("time_mask_max", 50),
            freq_mask_max=cfg.get("freq_mask_max", 20),
        )

    if np.random.random() < cfg.get("aug_noise_p", 0.0):
        spec = add_gaussian_noise(spec, std=cfg.get("gaussian_noise_std", 0.01))

    return spec
