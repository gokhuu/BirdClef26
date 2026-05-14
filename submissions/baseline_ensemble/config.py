"""Paths, hyperparameters, and the NN architecture registry for the baseline ensemble.

This is the only file that needs editing between Kaggle and local runs.
"""
import os

ON_KAGGLE = os.path.exists('/kaggle/input')

# ─── Paths ────────────────────────────────────────────────────────────
if ON_KAGGLE:
    BASE_DIR        = '/kaggle/input/birdclef-2026'
    TEST_DIR        = f'{BASE_DIR}/test_soundscapes'
    SAMPLE_SUB      = f'{BASE_DIR}/sample_submission.csv'

    # Each arch's ONNX exports typically live in their own Kaggle dataset
    B0_MODEL_DIR    = '/kaggle/input/birdclef-2026-baseline-effb0-onnx'
    SX_MODEL_DIR    = '/kaggle/input/datasets/brandonkhuu/birdclef-2026-effv2s-onnx'

    # XGBoost + calibrator pickles — upload as one dataset
    XGB_DEPLOY_PATH = '/kaggle/input/birdclef-2026-ensemble-artifacts/xgboost_deploy.pkl'
    CALIBRATOR_PATH = '/kaggle/input/birdclef-2026-ensemble-artifacts/isotonic_calibrators.pkl'

    OUTPUT_PATH     = '/kaggle/working/submission.csv'
else:
    BASE_DIR        = 'data/raw'
    TEST_DIR        = f'{BASE_DIR}/test_soundscapes'
    SAMPLE_SUB      = f'{BASE_DIR}/sample_submission.csv'

    B0_MODEL_DIR    = 'experiments'
    SX_MODEL_DIR    = 'experiments'

    XGB_DEPLOY_PATH = 'experiments/classical_xgboost/xgboost_deploy.pkl'
    CALIBRATOR_PATH = 'experiments/calibrators/isotonic_calibrators.pkl'

    OUTPUT_PATH     = 'submissions/baseline_ensemble/submission.csv'


# ─── Spectrogram (MUST match src/data/preprocess.py exactly) ─────────
SAMPLE_RATE      = 32000
N_MELS           = 128
HOP_LENGTH       = 512
N_FFT            = 2048
FMAX             = 16000
WINDOW_SECONDS   = 5.0
WINDOW_SAMPLES   = int(SAMPLE_RATE * WINDOW_SECONDS)   # 160000
SPEC_TIME_FRAMES = 313
NUM_CLASSES      = 234

# ─── Classical features (MUST match notebook 12 §2) ──────────────────
N_MFCC      = 20
FEATURE_DIM = N_MELS * 4 + N_MFCC * 5 + 8              # 620

# ─── TTA: 5 symmetric offsets across the 5s window ───────────────────
TTA_OFFSETS_SAMPLES = [
    -int(0.50 * WINDOW_SAMPLES),
    -int(0.25 * WINDOW_SAMPLES),
     0,
    +int(0.25 * WINDOW_SAMPLES),
    +int(0.50 * WINDOW_SAMPLES),
]
N_TTA_INITIAL = 5

# ─── Dual-head SED logit blend: 0=pool-only, 1=max-only ──────────────
ALPHA = 0.15

# ─── Time budget (Kaggle CPU 90-min cap, leave headroom) ─────────────
TIME_BUDGET_SECONDS = 80 * 60
TIME_WARN_SECONDS   = 75 * 60


# ─── NN architecture registry ────────────────────────────────────────
# `calib_key` MUST match the model_name used when fitting calibrators in
# notebook 16. If a calib_key isn't in the calibrator dict, that arch
# falls through to identity remap (usable but uncalibrated).
NN_ARCHS = [
    {
        'name':           'B0',
        'calib_key':      'effb0',
        'model_dir':      B0_MODEL_DIR,
        'dir_patterns':   ['baseline_effb0_fold{fold}'],
        'onnx_basenames': ['best_model.onnx', 'best_model_quantized.onnx'],
    },
    {
        'name':           'SX',
        'calib_key':      'seresnext',
        'model_dir':      SX_MODEL_DIR,
        'dir_patterns':   ['seresnext_finetune_fold{fold}', 'seresnext_fold{fold}'],
        'onnx_basenames': ['best_model.onnx', 'model_int8_fp32.onnx', 'best_model_quantized.onnx'],
    },
    # To add architectures (effv2s, convnext, effv2s_focal):
    #   1. Export 5 fold ONNX files for the arch
    #   2. Upload as a Kaggle dataset, set model_dir
    #   3. Add an entry here with calib_key matching the calibrator dict
    # The mean-blend automatically expands.
]

XGB_CALIB_KEY = 'xgboost'   # key used in notebook 16
