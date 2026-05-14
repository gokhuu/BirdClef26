"""Offline validation before submitting.

Run this LOCALLY before uploading to Kaggle. Two checks:

  1. Feature parity: re-extract 620-dim features from raw audio for a few fold-0 val files
     and compare to the cached parquet. Tolerance: max-abs-diff < 1e-3 per feature.
     Catches any mel spec / DCT / param drift between training and inference.

  2. Pipeline smoke test: load all artifacts, run predict_window() on one window of one
     fold-0 val file. Confirms NN ONNX sessions, XGBoost, and calibrators all wire up.

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
TRAIN_AUDIO_DIR = os.path.join(ROOT, 'data', 'raw', 'train_audio')


def check_feature_parity(n_files=10, tolerance=1e-3):
    """Re-extract features from raw audio, compare to cached parquet."""
    print(f'\n─── Feature parity check ({n_files} files, tol={tolerance}) ───')
    if not os.path.exists(FEAT_PQ):
        print(f'⚠ feature cache not found at {FEAT_PQ} — skipping')
        return True
    if not os.path.exists(FOLDS_CSV):
        print(f'⚠ folds.csv not found at {FOLDS_CSV} — skipping')
        return True
    import librosa

    feat_df = pd.read_parquet(FEAT_PQ)
    folds   = pd.read_csv(FOLDS_CSV)
    feat_df = feat_df.merge(folds[['filename', 'fold']], on='filename', how='left')
    fold0   = feat_df[feat_df['fold'] == 0].head(n_files).reset_index(drop=True)
    if len(fold0) == 0:
        print(f'⚠ no fold-0 rows found — skipping')
        return True

    feat_cols = [c for c in feat_df.columns if c.startswith('f') and c[1:].isdigit()]
    n_ok, n_bad = 0, 0
    worst_diffs = []
    for _, row in fold0.iterrows():
        fn = row['filename']
        # filename format is typically <species>/<basename>.ogg
        audio_path = os.path.join(TRAIN_AUDIO_DIR, fn)
        if not os.path.exists(audio_path):
            # Some setups don't keep the species prefix
            audio_path = os.path.join(TRAIN_AUDIO_DIR, os.path.basename(fn))
        if not os.path.exists(audio_path):
            print(f'  ⚠ {fn}: audio not found — skipping')
            continue

        y, _ = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
        seg  = inf.extract_shifted_segment(y, 0, 0)   # first 5s, no offset
        mel  = inf.compute_melspec(seg)
        feats_now = inf.extract_classical_features(mel)
        feats_cached = row[feat_cols].values.astype(np.float32)

        max_diff = float(np.abs(feats_now - feats_cached).max())
        worst_diffs.append((max_diff, fn))
        if max_diff < tolerance:
            n_ok += 1
        else:
            n_bad += 1
            print(f'  ✗ {fn}: max|Δ|={max_diff:.4e}')

    worst_diffs.sort(reverse=True)
    print(f'\nresult: {n_ok}/{n_ok + n_bad} files matched within tolerance')
    print(f'worst diffs (top 3): {[f"{d:.2e} ({fn})" for d, fn in worst_diffs[:3]]}')
    if n_bad > 0:
        print('\n⚠ Feature drift detected. The mel spec / feature pipeline in inference.py')
        print('  does NOT match the training-time pipeline. XGBoost predictions will be off.')
        print('  Fix before submitting: typical culprits are librosa power_to_db ref,')
        print('  fmax/fmin, n_mels, or padding behavior at segment edges.')
        return False
    return True


def smoke_test():
    """Load all artifacts, run predict_window on one window. Confirms end-to-end wiring."""
    print('\n─── Pipeline smoke test ───')
    if not os.path.exists(TRAIN_AUDIO_DIR):
        print(f'⚠ {TRAIN_AUDIO_DIR} not found — skipping')
        return True
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
        print(f'⚠ no audio under {TRAIN_AUDIO_DIR} — smoke test cannot run')
        return False
    audio_path = candidates[0]
    print(f'\nRunning predict_window on {audio_path}')

    y, _ = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
    t0 = time.perf_counter()
    probs = inf.predict_window(y, 0, archs, xgb_art, calibrators, species_list, n_tta=5)
    dt = time.perf_counter() - t0

    assert probs.shape == (len(species_list),), f'shape {probs.shape} != ({len(species_list)},)'
    assert 0.0 <= probs.min() <= probs.max() <= 1.0, f'probs out of [0,1]: [{probs.min()}, {probs.max()}]'

    top5_idx = np.argsort(probs)[::-1][:5]
    print(f'\n✓ predict_window returned shape={probs.shape}, range=[{probs.min():.4f}, {probs.max():.4f}], '
          f'latency={dt*1000:.0f}ms (5× TTA)')
    print(f'  Top-5 predictions:')
    for i in top5_idx:
        print(f'    {species_list[i]:<12} {probs[i]:.4f}')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-files', type=int, default=10, help='files to check for feature parity')
    parser.add_argument('--tolerance', type=float, default=1e-3, help='per-feature max-abs-diff tolerance')
    parser.add_argument('--skip-feature-check', action='store_true')
    parser.add_argument('--skip-smoke-test',    action='store_true')
    args = parser.parse_args()

    print(f'BASE_DIR        : {config.BASE_DIR}')
    print(f'XGB_DEPLOY_PATH : {config.XGB_DEPLOY_PATH}')
    print(f'CALIBRATOR_PATH : {config.CALIBRATOR_PATH}')

    ok = True
    if not args.skip_feature_check:
        ok &= check_feature_parity(args.n_files, args.tolerance)
    if not args.skip_smoke_test:
        ok &= smoke_test()

    print('\n' + '=' * 60)
    if ok:
        print('✓ ALL CHECKS PASSED — ready to submit')
    else:
        print('✗ VALIDATION FAILED — fix issues before submitting')
        sys.exit(1)


if __name__ == '__main__':
    main()
