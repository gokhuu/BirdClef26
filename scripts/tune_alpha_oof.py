"""Sweep alpha for the pool/max blend using OOF predictions.
Runs each dual-output ONNX on its own fold's validation set,
computes macro-AUC at alphas in [0, 1], picks the best."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import onnxruntime as ort
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import BirdCLEFDataset

ALPHAS = np.linspace(0.0, 1.0, 21)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def macro_auc(logits, labels):
    """Macro AUC, skipping classes with no positives in the OOF split."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    valid = labels.sum(axis=0) > 0
    if not valid.any():
        return float("nan")
    return roc_auc_score(labels[:, valid], probs[:, valid], average="macro")


def predict_oof_fold(onnx_path, val_loader):
    """Run a fold's ONNX on its val set, return (pool, max, labels) arrays."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    assert "logits_pool" in out_names and "logits_max" in out_names, (
        f"{onnx_path} is not dual-output: outputs={out_names}"
    )

    pools, maxs, labels = [], [], []
    for spec, label in tqdm(
        val_loader, desc=os.path.basename(os.path.dirname(onnx_path))
    ):
        x = spec.numpy().astype(np.float32)  # (B, 1, n_mels, T)
        pool, mx = sess.run(["logits_pool", "logits_max"], {"input": x})
        pools.append(pool)
        maxs.append(mx)
        labels.append(label.numpy())
    return np.concatenate(pools), np.concatenate(maxs), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--exp-dir", required=True, help="e.g. experiments/sed_finetune_pseudo_v4"
    )
    ap.add_argument(
        "--exp-prefix",
        default="sed_finetune_pseudo_v4_fold",
        help="fold dir prefix; full name is {prefix}{N}",
    )
    ap.add_argument(
        "--onnx-name",
        default="best_model_dual.onnx",
        help="ONNX filename inside each fold dir",
    )
    ap.add_argument("--folds-csv", default="data/folds/folds.csv")
    ap.add_argument("--spec-dir", default="data/processed")
    ap.add_argument("--train-meta-csv", default="data/raw/train.csv")
    ap.add_argument("--sample-submission", default="data/raw/sample_submission.csv")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    # Species list defines column order — must match training
    sample_sub = pd.read_csv(args.sample_submission)
    target_species = [c for c in sample_sub.columns if c != "row_id"]
    print(f"Species: {len(target_species)}")

    all_pools, all_maxs, all_labels = [], [], []
    for fold in args.folds:
        val_ds = BirdCLEFDataset(
            folds_csv=args.folds_csv,
            fold=fold,
            mode="val",
            spec_dir=args.spec_dir,
            target_species=target_species,
            train_meta_csv=args.train_meta_csv,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=False,
        )

        onnx_path = os.path.join(
            args.exp_dir,
            f"{args.exp_prefix}{fold}",
            args.onnx_name,
        )
        if not os.path.exists(onnx_path):
            print(f"  fold {fold}: SKIPPING (no ONNX at {onnx_path})")
            continue

        pool, mx, labels = predict_oof_fold(onnx_path, val_loader)
        print(
            f"  fold {fold}: {len(labels)} clips, "
            f"pool AUC={macro_auc(pool, labels):.5f}, "
            f"max  AUC={macro_auc(mx, labels):.5f}"
        )
        all_pools.append(pool)
        all_maxs.append(mx)
        all_labels.append(labels)

    if not all_pools:
        print("No folds were evaluated. Check --exp-dir / --exp-prefix / --onnx-name.")
        return

    P = np.concatenate(all_pools)
    M = np.concatenate(all_maxs)
    Y = np.concatenate(all_labels)

    print(f"\nCombined OOF: {len(Y)} clips, {Y.shape[1]} classes")
    print(f"\n{'alpha':>6}  {'macro_auc':>10}")
    print("-" * 22)

    best_alpha, best_auc = 0.0, -1.0
    rows = []
    for a in ALPHAS:
        blended = (1.0 - a) * P + a * M
        auc = macro_auc(blended, Y)
        rows.append((a, auc))
        if auc > best_auc:
            best_alpha, best_auc = a, auc

    for a, auc in rows:
        marker = "  <-- best" if a == best_alpha else ""
        print(f"{a:>6.2f}  {auc:>10.5f}{marker}")

    print(f"\nBest alpha:  {best_alpha:.2f}  (macro_auc = {best_auc:.5f})")
    print(f"Pool only:   0.00  (macro_auc = {macro_auc(P, Y):.5f})")
    print(f"Max only:    1.00  (macro_auc = {macro_auc(M, Y):.5f})")


if __name__ == "__main__":
    main()
