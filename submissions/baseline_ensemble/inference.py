"""BirdCLEF 2026 baseline ensemble inference pipeline.

Per 5s window:
  1. For each TTA offset: shift segment → mel spec → run NN ONNX (per arch, per fold)
     - At offset=0, also extract 620-dim classical features for XGBoost
  2. Per arch: mean logits across folds + offsets → sigmoid → arch probability vector
  3. XGBoost on center features → 234-dim probability vector (0.5 fill for 28 missing species)
  4. Apply per-(model, species) isotonic calibrator to each arch + XGBoost output
  5. Mean-blend the calibrated probability vectors

Entry points:
  main()                            -- run end-to-end on TEST_DIR, write OUTPUT_PATH
  predict_window(...)               -- single-window predict (used by main and validate)
  load_all_archs(), load_xgboost(), load_calibrators()  -- artifact loaders
"""
import os, sys, time, glob, pickle, warnings
import numpy as np
import pandas as pd
import scipy.fftpack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════
# AUDIO + MEL SPEC + TTA OFFSET EXTRACTION
# ══════════════════════════════════════════════════════════════════════
def compute_melspec(waveform):
    """Log-mel spectrogram matching training preprocessing."""
    import librosa
    S = librosa.feature.melspectrogram(
        y=waveform, sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS, fmax=config.FMAX,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    if S_db.shape[1] >= config.SPEC_TIME_FRAMES:
        S_db = S_db[:, :config.SPEC_TIME_FRAMES]
    else:
        S_db = np.pad(S_db, ((0, 0), (0, config.SPEC_TIME_FRAMES - S_db.shape[1])),
                      mode='constant', constant_values=S_db.min())
    return S_db


def extract_shifted_segment(waveform, center_start, offset_samples):
    """5s segment shifted by offset; zero-pad at boundaries."""
    total = len(waveform)
    start = center_start + offset_samples
    end   = start + config.WINDOW_SAMPLES
    pad_left  = max(0, -start)
    pad_right = max(0, end - total)
    seg = waveform[max(0, start):min(total, end)]
    if pad_left or pad_right:
        seg = np.pad(seg, (pad_left, pad_right), mode='constant', constant_values=0.0)
    if len(seg) < config.WINDOW_SAMPLES:
        seg = np.pad(seg, (0, config.WINDOW_SAMPLES - len(seg)), mode='constant')
    elif len(seg) > config.WINDOW_SAMPLES:
        seg = seg[:config.WINDOW_SAMPLES]
    return seg


def select_tta_offsets(n_tta):
    o = config.TTA_OFFSETS_SAMPLES
    if n_tta >= 5: return o
    if n_tta == 4: return [o[0], o[1], o[3], o[4]]
    if n_tta == 3: return [o[0], o[2], o[4]]
    if n_tta == 2: return [o[2], o[4]]
    return [o[2]]


# ══════════════════════════════════════════════════════════════════════
# 620-DIM CLASSICAL FEATURES (matches notebook 12 §2 extract_features)
# ══════════════════════════════════════════════════════════════════════
def extract_classical_features(mel_db):
    """mel_db: (n_mels, T) log-mel in dB. Returns float32 vector of length 620."""
    n_mels, T = mel_db.shape

    band_mean = mel_db.mean(axis=1)
    band_std  = mel_db.std(axis=1)
    band_max  = mel_db.max(axis=1)
    band_p90  = np.percentile(mel_db, 90, axis=1)
    band_feats = np.concatenate([band_mean, band_std, band_max, band_p90])

    mfcc = scipy.fftpack.dct(mel_db, type=2, axis=0, norm='ortho')[:config.N_MFCC]
    mfcc_feats = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        mfcc.min(axis=1), mfcc.max(axis=1), np.percentile(mfcc, 50, axis=1),
    ])

    energy         = mel_db.sum()
    peak_band      = int(band_mean.argmax())
    peak_time_frac = float(mel_db.mean(axis=0).argmax()) / max(T - 1, 1)
    dyn_range      = mel_db.max() - mel_db.min()
    band_energy    = np.exp(band_mean - band_mean.max())
    spec_centroid  = (np.arange(n_mels) * band_energy).sum() / max(band_energy.sum(), 1e-9)
    band_sparsity  = (band_mean > band_mean.mean()).mean()
    time_energy    = mel_db.mean(axis=0)
    time_sparsity  = (time_energy > time_energy.mean()).mean()
    energy_top10   = np.sort(time_energy)[-max(T // 10, 1):].mean()
    energy_concentration = energy_top10 / max(time_energy.mean(), 1e-9)
    global_feats = np.array([
        energy, peak_band, peak_time_frac, dyn_range,
        spec_centroid, band_sparsity, time_sparsity, energy_concentration,
    ], dtype=np.float64)

    return np.concatenate([band_feats, mfcc_feats, global_feats]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# NN ONNX LOADING + INFERENCE
# ══════════════════════════════════════════════════════════════════════
def find_onnx_for_arch(arch_cfg, n_folds=5):
    """Find ONNX files for one architecture. Returns list of paths (may be <n_folds)."""
    paths = []
    for fold in range(n_folds):
        found = None
        for dir_pat in arch_cfg['dir_patterns']:
            for basename in arch_cfg['onnx_basenames']:
                p = os.path.join(arch_cfg['model_dir'], dir_pat.format(fold=fold), basename)
                if os.path.exists(p):
                    found = p; break
            if found: break
        if found:
            paths.append(found)
    if not paths:
        # Flat-layout fallback: model_dir/**/*.onnx
        all_onnx = sorted(glob.glob(os.path.join(arch_cfg['model_dir'], '**', '*.onnx'),
                                    recursive=True))
        regular = [f for f in all_onnx if 'quantized' not in f]
        paths = (regular if regular else all_onnx)[:n_folds]
    return paths


def load_nn_sessions(paths, label=''):
    """Load ONNX sessions; capture per-session input tensor name."""
    import onnxruntime as ort
    items = []
    for p in paths:
        sess = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
        inp_name = sess.get_inputs()[0].name
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f'  [{label}] {os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}: '
              f'{size_mb:.1f} MB, input="{inp_name}"')
        items.append((sess, inp_name))
    return items


def run_logits(sess, inp_name, x, alpha=None):
    """Dual-output SED ONNX: blend logits_pool + logits_max in logit space.
    Single-output ONNX falls through transparently."""
    if alpha is None:
        alpha = config.ALPHA
    out_names = [o.name for o in sess.get_outputs()]
    if 'logits_pool' in out_names and 'logits_max' in out_names:
        pool, mx = sess.run(['logits_pool', 'logits_max'], {inp_name: x})
        return ((1.0 - alpha) * pool + alpha * mx)[0]
    return sess.run(None, {inp_name: x})[0][0]


def load_all_archs():
    """Walk config.NN_ARCHS, load whatever ONNX files we find.
    Returns list of {name, calib_key, items}."""
    loaded = []
    for arch_cfg in config.NN_ARCHS:
        name = arch_cfg['name']
        print(f"\n{name} model search in: {arch_cfg['model_dir']}")
        paths = find_onnx_for_arch(arch_cfg)
        print(f'  Found {len(paths)} ONNX file(s)')
        items = load_nn_sessions(paths, label=name)
        if items:
            loaded.append({'name': name, 'calib_key': arch_cfg['calib_key'], 'items': items})
        else:
            print(f'  ⚠ no files matched — {name} excluded from ensemble')
    print(f'\nLoaded {len(loaded)} NN architecture(s): {[a["name"] for a in loaded]}')
    return loaded


# ══════════════════════════════════════════════════════════════════════
# XGBOOST DEPLOY ARTIFACT
# ══════════════════════════════════════════════════════════════════════
def load_xgboost(path):
    """Load the xgboost_deploy.pkl from notebook 15. Returns None if missing."""
    if not os.path.exists(path):
        print(f'⚠ XGBoost deploy artifact not found at {path} — XGB excluded')
        return None
    with open(path, 'rb') as f:
        artifact = pickle.load(f)
    n_trainable = int(np.asarray(artifact['trainable_mask']).sum())
    print(f'  XGBoost: {len(artifact["species_list"])} species '
          f'({n_trainable} trainable), xgb version {artifact["xgb_version"]}')
    return artifact


def xgb_predict_aligned(artifact, features, species_list_canonical):
    """Predict XGBoost probs, expand to canonical species set with 0.5 fill.

    features: (N, 620) float32
    species_list_canonical: list of 234 species codes (from sample_submission)
    Returns: (N, 234) float32
    """
    xgb_sp_to_idx = {sp: i for i, sp in enumerate(artifact['species_list'])}
    out = np.full((features.shape[0], len(species_list_canonical)), 0.5, dtype=np.float32)
    for canon_i, sp in enumerate(species_list_canonical):
        xgb_i = xgb_sp_to_idx.get(sp)
        if xgb_i is None:
            continue
        m = artifact['models'][xgb_i]
        if m is not None:
            out[:, canon_i] = m.predict_proba(features)[:, 1]
    return out


# ══════════════════════════════════════════════════════════════════════
# CALIBRATORS (per-(model, species) isotonic)
# ══════════════════════════════════════════════════════════════════════
def load_calibrators(path):
    """Load dict[(model_name, species_code) -> IsotonicRegression]."""
    if not os.path.exists(path):
        print(f'⚠ Calibrators not found at {path} — falling back to identity')
        return {}
    with open(path, 'rb') as f:
        calibrators = pickle.load(f)
    by_model = {}
    for (m, _) in calibrators.keys():
        by_model[m] = by_model.get(m, 0) + 1
    print(f'  Calibrators: {len(calibrators)} entries  ({by_model})')
    return calibrators


def apply_calibrator(model_name, probs, species_list, calibrators):
    """Apply per-species isotonic to one model's predictions.
    Missing (model_name, species_code) keys → identity remap.

    probs: (N, n_species), returns same shape.
    """
    out = probs.astype(np.float32, copy=True)
    for c, sp in enumerate(species_list):
        iso = calibrators.get((model_name, sp))
        if iso is not None:
            out[:, c] = iso.predict(probs[:, c]).astype(np.float32)
    return out


# ══════════════════════════════════════════════════════════════════════
# PER-WINDOW INFERENCE
# ══════════════════════════════════════════════════════════════════════
def predict_window(waveform, center_start, archs, xgb_artifact, calibrators,
                   species_list, n_tta, alpha=None):
    """Predict the mean-blended calibrated probability vector for one 5s window.

    archs:        list of {name, calib_key, items}  from load_all_archs()
    xgb_artifact: from load_xgboost() or None
    calibrators:  from load_calibrators()
    species_list: canonical species order (from sample_submission columns)
    n_tta:        1..5

    Returns: (n_species,) float32
    """
    if alpha is None:
        alpha = config.ALPHA
    offsets = select_tta_offsets(n_tta)

    arch_offset_logits = {a['name']: [] for a in archs}
    center_features    = None

    for offset in offsets:
        seg = extract_shifted_segment(waveform, center_start, offset)
        mel = compute_melspec(seg)
        if offset == 0 and center_features is None:
            center_features = extract_classical_features(mel)

        x = mel[np.newaxis, np.newaxis, :, :].astype(np.float32)
        for arch in archs:
            fold_logits = [run_logits(sess, inp, x, alpha) for sess, inp in arch['items']]
            arch_offset_logits[arch['name']].append(np.mean(fold_logits, axis=0))

    # Per-arch: mean over offsets → sigmoid → calibrate
    calibrated_stack = []
    for arch in archs:
        logits = np.mean(arch_offset_logits[arch['name']], axis=0)
        probs  = 1.0 / (1.0 + np.exp(-logits))
        cal    = apply_calibrator(arch['calib_key'], probs[None, :], species_list, calibrators)[0]
        calibrated_stack.append(cal)

    # XGBoost
    if xgb_artifact is not None:
        if center_features is None:
            seg = extract_shifted_segment(waveform, center_start, 0)
            center_features = extract_classical_features(compute_melspec(seg))
        xgb_probs = xgb_predict_aligned(xgb_artifact, center_features[None, :], species_list)
        cal_xgb   = apply_calibrator(config.XGB_CALIB_KEY, xgb_probs, species_list, calibrators)[0]
        calibrated_stack.append(cal_xgb)

    if not calibrated_stack:
        return np.zeros(len(species_list), dtype=np.float32)
    return np.mean(calibrated_stack, axis=0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# MAIN: PER-CLIP LOOP + CSV OUTPUT
# ══════════════════════════════════════════════════════════════════════
def run_directory(test_dir, sample_sub_path, output_path, archs, xgb_artifact, calibrators,
                  n_tta=None, time_budget=None, time_warn=None):
    """Process all audio in test_dir, write submission CSV. Returns DataFrame."""
    import librosa
    n_tta       = n_tta       or config.N_TTA_INITIAL
    time_budget = time_budget or config.TIME_BUDGET_SECONDS
    time_warn   = time_warn   or config.TIME_WARN_SECONDS

    sample_sub   = pd.read_csv(sample_sub_path)
    species_list = [c for c in sample_sub.columns if c != 'row_id']
    assert len(species_list) == config.NUM_CLASSES, \
        f'sample sub has {len(species_list)} species, expected {config.NUM_CLASSES}'

    audio_files = []
    for ext in ('*.ogg', '*.wav', '*.flac', '*.mp3'):
        audio_files.extend(glob.glob(os.path.join(test_dir, ext)))
    audio_files = sorted(audio_files)
    print(f'\nTest audio files: {len(audio_files)}  (test_dir={test_dir})')

    rows = []
    n_tta_cur = n_tta
    tta_downgrades = []
    timing = {'audio': 0.0, 'predict': 0.0, 'windows': 0}
    t_start = time.time()

    for file_idx, audio_path in enumerate(audio_files):
        soundscape_id = os.path.splitext(os.path.basename(audio_path))[0]

        t0 = time.perf_counter()
        y, _ = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
        timing['audio'] += time.perf_counter() - t0

        n_windows = int(np.ceil(len(y) / config.WINDOW_SAMPLES))
        for win_idx in range(n_windows):
            center_start = win_idx * config.WINDOW_SAMPLES
            end_sec      = (win_idx + 1) * int(config.WINDOW_SECONDS)
            row_id       = f'{soundscape_id}_{end_sec}'

            t0 = time.perf_counter()
            probs = predict_window(y, center_start, archs, xgb_artifact, calibrators,
                                   species_list, n_tta=n_tta_cur)
            timing['predict'] += time.perf_counter() - t0
            timing['windows'] += 1

            row = {'row_id': row_id}
            for sp_idx, sp in enumerate(species_list):
                row[sp] = float(probs[sp_idx])
            rows.append(row)

        # Time-budget fallback
        elapsed   = time.time() - t_start
        rate      = (file_idx + 1) / elapsed if elapsed > 0 else 1.0
        est_total = elapsed + (len(audio_files) - file_idx - 1) / rate
        if est_total > time_budget and n_tta_cur > 1:
            old = n_tta_cur
            n_tta_cur = max(1, n_tta_cur - 1)
            tta_downgrades.append({'idx': file_idx + 1, 'from': old, 'to': n_tta_cur,
                                   'elapsed_min': elapsed / 60})
            print(f'  ⚠ TIME-BUDGET FALLBACK at {file_idx+1}/{len(audio_files)}: '
                  f'est={est_total/60:.1f}min > {time_budget/60:.0f}min. '
                  f'TTA: {old}× → {n_tta_cur}×')

        if (file_idx + 1) % 10 == 0 or file_idx in (0, len(audio_files) - 1):
            warn = '⚠' if est_total > time_warn else ' '
            print(f'{warn} [{file_idx+1:>4}/{len(audio_files)}] {soundscape_id}: '
                  f'{n_windows} win | TTA={n_tta_cur}× | '
                  f'elapsed={elapsed/60:.1f}min | est={est_total/60:.1f}min')

    total = time.time() - t_start
    n_win = timing['windows']
    print(f'\nDone: {len(audio_files)} soundscapes, {n_win} windows in '
          f'{total:.1f}s ({total/60:.1f}min)')
    if n_win > 0:
        print(f'  Per window: audio={timing["audio"]/max(len(audio_files),1)*1000:.1f}ms/file, '
              f'predict={timing["predict"]/n_win*1000:.1f}ms/window')
    if tta_downgrades:
        print(f'\n⚠ TTA downgraded {len(tta_downgrades)} time(s):')
        for d in tta_downgrades:
            print(f'  at file {d["idx"]}: {d["from"]}× → {d["to"]}× '
                  f'(elapsed {d["elapsed_min"]:.1f}min)')
    else:
        print(f'\n✓ Completed with full {n_tta}× TTA throughout')

    # Build submission DataFrame with exact sample_sub column order
    expected_cols = list(sample_sub.columns)
    if rows:
        submission = pd.DataFrame(rows)
        for col in expected_cols:
            if col not in submission.columns:
                submission[col] = 0.0
        submission = submission[expected_cols]
    else:
        print('⚠ No audio files found — using sample submission as placeholder')
        submission = sample_sub.copy()

    assert list(submission.columns) == expected_cols, 'Column mismatch!'
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f'\nWrote: {output_path}  ({submission.shape[0]} rows × {submission.shape[1]} cols)')

    pred_vals = submission[species_list].values
    print(f'Prediction stats: mean={pred_vals.mean():.4f}  std={pred_vals.std():.4f}  '
          f'min={pred_vals.min():.4f}  max={pred_vals.max():.4f}')
    return submission


def main():
    """Run end-to-end on config.TEST_DIR, write config.OUTPUT_PATH."""
    print(f'Environment: {"Kaggle" if config.ON_KAGGLE else "Local"}')
    print(f'TEST_DIR    : {config.TEST_DIR}')
    print(f'SAMPLE_SUB  : {config.SAMPLE_SUB}')
    print(f'OUTPUT_PATH : {config.OUTPUT_PATH}')

    print('\n─── Loading NN architectures ───')
    archs = load_all_archs()
    if not archs:
        raise RuntimeError('No NN architectures loaded. Check NN_ARCHS paths in config.py')

    print('\n─── Loading XGBoost deploy ───')
    xgb_artifact = load_xgboost(config.XGB_DEPLOY_PATH)

    print('\n─── Loading calibrators ───')
    calibrators = load_calibrators(config.CALIBRATOR_PATH)

    print(f'\n─── Ensemble: {len(archs)} NN arch(s) + '
          f'{"XGBoost" if xgb_artifact else "no XGBoost"} '
          f'+ {"calibrated" if calibrators else "uncalibrated"} mean-blend ───')

    return run_directory(config.TEST_DIR, config.SAMPLE_SUB, config.OUTPUT_PATH,
                         archs, xgb_artifact, calibrators)


if __name__ == '__main__':
    main()
