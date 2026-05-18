"""Offline validation before submitting.

Three-stage feature parity diagnostic + smoke test.

  STAGE A — extract_features() correctness
    Load cached mel from data/processed/<file>.npy (what nb12 actually used),
    run extract_features() on it, compare to the parquet cache.
    If A fails: the extract_features function in inference.py differs from nb12.

  STAGE B — compute_melspec() vs training-time mel
    Compute mel from raw audio using inference.compute_melspec(),
    compare to the cached .npy mel directly.
    If B fails: the inference-time mel computation doesn't match src/data/preprocess.py
    (different power_to_db ref, normalization, fmin, padding, etc.)
    The .npy cache is the source of truth.

  STAGE C — end-to-end audio → features
    The combined pipeline as it runs at inference. Will fail if A or B fails.

  STAGE D — pipeline smoke test
    Load all artifacts, run predict_window() on one window. Confirms wiring.

Usage:
    python submissions/baseline_ensemble/validate.py
    python submissions/baseline_ensemble/validate.py --n-files 20
    python submissions/baseline_ensemble/validate.py --skip-feature-check
"""
import os, sys, argparse, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import inference as inf

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
META_CSV  = os.path.join(ROOT, 'data', 'raw', 'train.csv')
FOLDS_CSV = os.path.join(ROOT, 'data', 'folds', 'folds.csv')
FEAT_PQ   = os.path.join(ROOT, 'experiments', '_classical_features.parquet')
SPEC_DIR  = os.path.join(ROOT, config.SPEC_DIR)
TRAIN_AUDIO_DIR = os.path.join(ROOT, 'data', 'raw', 'train_audio')


def _find_cached_spec(filename):
    """Replicate nb12's load candidate paths."""
    from pathlib import Path
    stem = Path(filename).stem
    for cand in [os.path.join(SPEC_DIR, f'{stem}.npy'),
                 os.path.join(SPEC_DIR, filename.replace('.ogg', '.npy'))]:
        if os.path.exists(cand):
            return cand
    return None


def _load_spec_npy(path):
    """Replicate nb12's _load_one_spec."""
    a = np.load(path)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f'spec at {path} has shape {a.shape}, expected 2D')
    return a


def stage_a_cached_mel_to_features(fold0_rows, feat_cols, tolerance=1e-3):
    """Load cached mel, run extract_features on it, compare to parquet cache."""
    print(f'\n─── STAGE A: cached .npy mel → extract_features → parquet ─────────')
    print(f'  (isolates the feature extraction function)')

    n_ok, n_bad, n_skip = 0, 0, 0
    worst = []
    for _, row in fold0_rows.iterrows():
        fn = row['filename']
        spec_path = _find_cached_spec(fn)
        if spec_path is None:
            n_skip += 1
            continue
        mel = _load_spec_npy(spec_path)
        feats_now = inf.extract_classical_features(mel)
        feats_cached = row[feat_cols].values.astype(np.float32)
        max_diff = float(np.abs(feats_now - feats_cached).max())
        worst.append((max_diff, fn, mel.shape))
        if max_diff < tolerance:
            n_ok += 1
        else:
            n_bad += 1
            print(f'  ✗ {fn}: max|Δ|={max_diff:.4e}  mel shape {mel.shape}')

    worst.sort(reverse=True)
    print(f'\n  result: {n_ok}/{n_ok+n_bad} matched within tol={tolerance}, skipped {n_skip}')
    if worst:
        print(f'  worst diffs: {[f"{d:.2e} ({fn}, T={shp[1]})" for d, fn, shp in worst[:3]]}')
    if n_bad > 0:
        print(f'\n  ⚠ STAGE A FAILED — extract_features in inference.py differs from nb12')
        return False
    if n_ok == 0:
        print(f'\n  ⚠ STAGE A SKIPPED — no cached .npy files found at {SPEC_DIR}/')
        return None
    print(f'  ✓ STAGE A PASSED')
    return True


def stage_b_audio_to_mel(fold0_rows, tolerance=1e-3, max_files=5):
    """Compute mel from raw audio, compare to the cached .npy."""
    print(f'\n─── STAGE B: raw audio → compute_melspec → cached .npy ────────────')
    print(f'  (isolates the mel spectrogram computation)')
    import librosa

    n_ok, n_bad, n_skip = 0, 0, 0
    for _, row in fold0_rows.head(max_files).iterrows():
        fn = row['filename']
        spec_path = _find_cached_spec(fn)
        if spec_path is None:
            n_skip += 1
            continue
        audio_path = os.path.join(TRAIN_AUDIO_DIR, fn)
        if not os.path.exists(audio_path):
            audio_path = os.path.join(TRAIN_AUDIO_DIR, os.path.basename(fn))
        if not os.path.exists(audio_path):
            n_skip += 1
            continue

        cached_mel = _load_spec_npy(spec_path)
        y, _ = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
        seg = inf.extract_shifted_segment(y, 0, 0)
        live_mel = inf.compute_melspec(seg)

        if live_mel.shape != cached_mel.shape:
            n_bad += 1
            print(f'  ✗ {fn}: shape mismatch  live={live_mel.shape}  cached={cached_mel.shape}')
            continue

        max_diff = float(np.abs(live_mel - cached_mel).max())
        if max_diff < tolerance:
            n_ok += 1
        else:
            n_bad += 1
            print(f'  ✗ {fn}: max|Δ|={max_diff:.4e}')
            print(f'      cached mel: shape={cached_mel.shape}  range=[{cached_mel.min():.3f}, {cached_mel.max():.3f}]  mean={cached_mel.mean():.3f}')
            print(f'      live   mel: shape={live_mel.shape}  range=[{live_mel.min():.3f}, {live_mel.max():.3f}]  mean={live_mel.mean():.3f}')

    print(f'\n  result: {n_ok}/{n_ok+n_bad} matched within tol={tolerance}, skipped {n_skip}')
    if n_bad > 0:
        print(f'\n  ⚠ STAGE B FAILED — compute_melspec() does NOT match training-time mel')
        print(f'    The .npy cache is the source of truth. Check src/data/preprocess.py')
        print(f'    and update inference.compute_melspec() to match. Typical culprits:')
        print(f'    - librosa.power_to_db ref (np.max vs 1.0 vs no dB at all)')
        print(f'    - Normalization after power_to_db (z-score? min-max to [-1, 1]?)')
        print(f'    - fmin (defaults to 0 in librosa; some pipelines use 20-50 Hz)')
        print(f'    - Padding strategy at segment edges')
        print(f'\n  Quick mel-format inspection:')
        print(f'    python -c "import numpy as np; a=np.load(\\\'$(ls {SPEC_DIR}/*.npy | head -1)\\\'); '
              f'print(\\\'shape\\\', a.shape, \\\'range\\\', a.min(), a.max(), \\\'mean\\\', a.mean())"')
        return False
    if n_ok == 0:
        print(f'\n  ⚠ STAGE B SKIPPED — no audio files found in {TRAIN_AUDIO_DIR}/')
        return None
    print(f'  ✓ STAGE B PASSED')
    return True


def stage_c_end_to_end(fold0_rows, feat_cols, tolerance=1e-3):
    """Raw audio → compute_melspec → extract_features → parquet."""
    print(f'\n─── STAGE C: raw audio → end-to-end features → parquet ────────────')
    print(f'  (will fail if either STAGE A or STAGE B failed)')
    import librosa
    n_ok, n_bad, n_skip = 0, 0, 0
    for _, row in fold0_rows.iterrows():
        fn = row['filename']
        audio_path = os.path.join(TRAIN_AUDIO_DIR, fn)
        if not os.path.exists(audio_path):
            audio_path = os.path.join(TRAIN_AUDIO_DIR, os.path.basename(fn))
        if not os.path.exists(audio_path):
            n_skip += 1; continue
        y, _ = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
        seg = inf.extract_shifted_segment(y, 0, 0)
        mel = inf.compute_melspec(seg)
        feats_now = inf.extract_classical_features(mel)
        feats_cached = row[feat_cols].values.astype(np.float32)
        max_diff = float(np.abs(feats_now - feats_cached).max())
        if max_diff < tolerance:
            n_ok += 1
        else:
            n_bad += 1
    print(f'  result: {n_ok}/{n_ok+n_bad} matched within tol={tolerance}, skipped {n_skip}')
    return n_bad == 0 and n_ok > 0


def feature_parity(n_files=10, tolerance=1e-3):
    """Run all three stages in sequence."""
    if not os.path.exists(FEAT_PQ):
        print(f'⚠ feature cache not found at {FEAT_PQ} — skipping')
        return None
    if not os.path.exists(FOLDS_CSV):
        print(f'⚠ folds.csv not found at {FOLDS_CSV} — skipping')
        return None

    feat_df = pd.read_parquet(FEAT_PQ)
    folds = pd.read_csv(FOLDS_CSV)
    feat_df = feat_df.merge(folds[['filename', 'fold']], on='filename', how='left')
    fold0 = feat_df[feat_df['fold'] == 0].head(n_files).reset_index(drop=True)
    feat_cols = [c for c in feat_df.columns if c.startswith('f') and c[1:].isdigit()]
    print(f'comparing {len(fold0)} fold-0 files (tol={tolerance})')

    a = stage_a_cached_mel_to_features(fold0, feat_cols, tolerance)
    b = stage_b_audio_to_mel(fold0, tolerance)
    c = stage_c_end_to_end(fold0, feat_cols, tolerance) if a else None
    # Treat skipped stages as not-failing (None is informational)
    ok = (a is not False) and (b is not False) and (c is not False)
    return ok


def smoke_test():
    """Load all artifacts, run predict_window on one window."""
    print('\n─── STAGE D: pipeline smoke test ─────────────────────────────────')
    if not os.path.exists(TRAIN_AUDIO_DIR):
        print(f'⚠ {TRAIN_AUDIO_DIR} not found — skipping')
        return None
    import librosa, glob

    archs = inf.load_all_archs()
    if not archs:
        print('⚠ no NN archs loaded — smoke test cannot run')
        return False
    xgb_art     = inf.load_xgboost(config.XGB_DEPLOY_PATH)
    calibrators = inf.load_calibrators(config.CALIBRATOR_PATH)

    sample_sub   = pd.read_csv(config.SAMPLE_SUB)
    species_list = [c for c in sample_sub.columns if c != 'row_id']

    candidates = sorted(glob.glob(os.path.join(TRAIN_AUDIO_DIR, '**', '*.ogg'), recursive=True))
    if not candidates:
        return None
    audio_path = candidates[0]
    print(f'\nRunning predict_window on {audio_path}')

    y, _ = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
    t0 = time.perf_counter()
    probs = inf.predict_window(y, 0, archs, xgb_art, calibrators, species_list, n_tta=5)
    dt = time.perf_counter() - t0

    assert probs.shape == (len(species_list),)
    assert 0.0 <= probs.min() <= probs.max() <= 1.0

    top5 = np.argsort(probs)[::-1][:5]
    print(f'\n✓ shape={probs.shape}, range=[{probs.min():.4f}, {probs.max():.4f}], '
          f'latency={dt*1000:.0f}ms (5× TTA)')
    print(f'  Top-5 predictions:')
    for i in top5:
        print(f'    {species_list[i]:<12} {probs[i]:.4f}')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-files', type=int, default=10)
    parser.add_argument('--tolerance', type=float, default=1e-3)
    parser.add_argument('--skip-feature-check', action='store_true')
    parser.add_argument('--skip-smoke-test',    action='store_true')
    args = parser.parse_args()

    print(f'BASE_DIR        : {config.BASE_DIR}')
    print(f'SPEC_DIR        : {SPEC_DIR}')
    print(f'XGB_DEPLOY_PATH : {config.XGB_DEPLOY_PATH}')
    print(f'CALIBRATOR_PATH : {config.CALIBRATOR_PATH}')

    parity = None if args.skip_feature_check else feature_parity(args.n_files, args.tolerance)
    smoke  = None if args.skip_smoke_test    else smoke_test()

    print('\n' + '=' * 70)
    if (parity in (True, None)) and (smoke in (True, None)):
        print('✓ ALL CHECKS PASSED — ready to submit')
    else:
        print('✗ VALIDATION FAILED — fix issues before submitting')
        sys.exit(1)


if __name__ == '__main__':
    main()
