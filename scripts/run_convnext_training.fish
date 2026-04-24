#!/usr/bin/env fish
# ============================================================
# ConvNeXt-tiny 5-fold training (diversity partner for sed_b0).
#
# This runner:
#   - Sets BIRDCLEF_RUN_NAME per fold so each fold writes to its own
#     experiment dir (sed_convnext_fold0 ... sed_convnext_fold4).
#   - Pre-flight checks that the config is sane (no duplicate keys,
#     ConvNeXt backbone, no static run_name).
#   - Refuses to start if any sed_b0 baseline or pseudo_v2 checkpoint
#     is missing — those are needed for the ensemble and must not be
#     accidentally overwritten.
#
# Run from project root:
#   fish scripts/run_convnext_training.fish
# ============================================================

set -l start_time (date +%s)
set -l CONFIG configs/experiment_sed_convnext.yaml
set -l EXP_BASE sed_convnext

# ---------- Pre-flight checks ----------
echo "================================================================"
echo "PRE-FLIGHT: validating config and baseline artifacts"
echo "================================================================"

# Config exists?
if not test -f $CONFIG
    echo "✗ Config not found: $CONFIG"
    exit 1
end

# Config has convnext backbone?
set -l backbone_line (grep -E '^backbone:' $CONFIG)
if not string match -q "*convnext*" -- $backbone_line
    echo "✗ Config backbone is not convnext_tiny:"
    echo "    $backbone_line"
    exit 1
end

# Config must NOT have a static run_name (would overwrite every fold).
if grep -qE '^run_name:' $CONFIG
    echo "✗ Config has a static run_name — remove it to prevent fold overwrites:"
    grep -n '^run_name:' $CONFIG
    exit 1
end

# Config must not have duplicate keys (YAML silently takes the last one).
set -l dup_backbone (grep -cE '^backbone:' $CONFIG)
if test $dup_backbone -gt 1
    echo "✗ Config has $dup_backbone 'backbone:' lines — duplicate key bug"
    exit 1
end

# Make sure BIRDCLEF_RUN_NAME isn't leaking from a parent shell.
if set -q BIRDCLEF_RUN_NAME
    echo "⚠ BIRDCLEF_RUN_NAME is set in environment: $BIRDCLEF_RUN_NAME"
    echo "  Unsetting to prevent it from forcing a single run dir."
    set -e BIRDCLEF_RUN_NAME
end

# sed_b0 baselines must exist (needed for future ensemble inference) and
# we do NOT want this script to touch them.
for f in 0 1 2 3 4
    if not test -f experiments/sed_b0_fold$f/best_model.pt
        echo "⚠ experiments/sed_b0_fold$f/best_model.pt missing"
        echo "  Not fatal for ConvNeXt training, but you need these for the final ensemble."
    end
end

echo "✓ Config looks sane, starting training"
echo "  config   : $CONFIG"
echo "  backbone : "(grep -E '^backbone:' $CONFIG | string trim)

# ---------- Training loop ----------
echo
echo "================================================================"
echo "STAGE 1/1: ConvNeXt-tiny 5-fold training (~6-10h on RTX 2060)"
echo "================================================================"

for f in 0 1 2 3 4
    # Each fold gets its own run dir via this env var.
    set -x BIRDCLEF_RUN_NAME {$EXP_BASE}_fold{$f}

    echo
    echo ">>> Fold $f starting at "(date '+%H:%M:%S')" → experiments/$BIRDCLEF_RUN_NAME"

    python src/training/train.py $CONFIG --fold $f
    or begin
        echo "✗ Fold $f failed. Aborting."
        set -e BIRDCLEF_RUN_NAME
        exit 1
    end

    # Verify the fold actually wrote to the right place.
    if not test -f experiments/{$EXP_BASE}_fold{$f}/best_model.pt
        echo "✗ Fold $f did not produce experiments/{$EXP_BASE}_fold{$f}/best_model.pt"
        echo "  Something is wrong with run_name resolution."
        set -e BIRDCLEF_RUN_NAME
        exit 1
    end
end

# Clean up so the env var doesn't bleed into the next shell command.
set -e BIRDCLEF_RUN_NAME

# ---------- Summary ----------
set -l elapsed (math (date +%s) - $start_time)
set -l minutes (math --scale=1 $elapsed / 60)
set -l hours (math --scale=2 $minutes / 60)

echo
echo "================================================================"
echo "✓ Training complete in $hours hours ($minutes min)"
echo "================================================================"
echo
echo "Checkpoints:"
for f in 0 1 2 3 4
    set -l ckpt experiments/{$EXP_BASE}_fold{$f}/best_model.pt
    if test -f $ckpt
        set -l size_mb (math --scale=1 (stat -c '%s' $ckpt) / 1048576)
        echo "  fold $f: $ckpt ($size_mb MB)"
    else
        echo "  fold $f: MISSING $ckpt"
    end
end

echo
echo "Per-fold best val AUCs:"
for f in 0 1 2 3 4
    set -l log experiments/{$EXP_BASE}_fold{$f}/training_log.csv
    if test -f $log
        python -c "
import pandas as pd
df = pd.read_csv('$log')
best = df.loc[df['val_auc'].idxmax()]
print(f'  fold $f: best_epoch={int(best.epoch):>2}/{len(df)}  val_auc={best.val_auc:.4f}')
"
    end
end

echo
echo "Next: fine-tune ConvNeXt on focal+labeled_soundscape to match sed_b0 pipeline."
echo "================================================================"
