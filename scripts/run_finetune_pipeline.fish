#!/usr/bin/env fish
# ============================================================
# Full fine-tune pipeline (pseudo-label round):
#   1. Fine-tune 5 folds with focal + labeled sc + pseudo
#   2. Export ONNX
#   3. Copy to submissions/
#   4. Verify submission integrity (md5, distinct weights, sanity)
# Aborts on first failure.
#
# Run from project root:
#   fish scripts/run_finetune_pipeline.fish
# ============================================================
set -l start_time (date +%s)
set -l EXP_NAME sed_finetune_pseudo_v2   # must match experiment_name in pseudo config

# ---- Stage 1: fine-tune ----
echo "================================================================"
echo "STAGE 1/4: Fine-tuning 5 folds with pseudo-labels (~90-110 min)"
echo "================================================================"
for f in 0 1 2 3 4
    echo
    echo ">>> Fold $f starting at "(date '+%H:%M:%S')
    python src/training/finetune.py configs/experiment_sed_b0_finetune_pseudo.yaml \
        --fold $f \
        --init-checkpoint experiments/sed_b0_fold{$f}/best_model.pt
    or begin
        echo "✗ Fold $f failed. Aborting pipeline."
        exit 1
    end
end

# ---- Stage 2: export ONNX ----
echo
echo "================================================================"
echo "STAGE 2/4: Exporting 5 checkpoints to ONNX"
echo "================================================================"
for f in 0 1 2 3 4
    echo
    echo ">>> Exporting fold $f"
    python src/training/export_onnx.py experiments/{$EXP_NAME}_fold{$f}/best_model.pt
    or begin
        echo "✗ Export of fold $f failed. Aborting pipeline."
        exit 1
    end
end

# ---- Stage 3: copy to submissions/ ----
echo
echo "================================================================"
echo "STAGE 3/4: Copying ONNX files to submissions/"
echo "================================================================"
mkdir -p submissions
for f in 0 1 2 3 4
    cp experiments/{$EXP_NAME}_fold{$f}/best_model.onnx \
       submissions/best_model_{$f}.onnx
    or begin
        echo "✗ Copy of fold $f failed. Aborting pipeline."
        exit 1
    end
end

# ---- Stage 4: verify submission integrity ----
echo
echo "================================================================"
echo "STAGE 4/4: Verifying submission/"
echo "================================================================"
set -l verify_ok 1

# 4a. Source and destination md5s must match per-fold
echo
echo "--- 4a. Source ↔ submission md5 parity ---"
for f in 0 1 2 3 4
    set -l src experiments/{$EXP_NAME}_fold{$f}/best_model.onnx
    set -l dst submissions/best_model_{$f}.onnx
    set -l src_md5 (md5sum $src | awk '{print $1}')
    set -l dst_md5 (md5sum $dst | awk '{print $1}')
    if test "$src_md5" = "$dst_md5"
        echo "  ✓ fold $f: $src_md5  ($EXP_NAME)"
    else
        echo "  ✗ fold $f MISMATCH"
        echo "      src ($src): $src_md5"
        echo "      dst ($dst): $dst_md5"
        set verify_ok 0
    end
end

# 4b. The 5 submission files must all be distinct (catches cp bugs where
#     one file got copied 5 times, a silent disaster).
echo
echo "--- 4b. All 5 submission files are distinct ---"
set -l n_unique (md5sum submissions/best_model_*.onnx | awk '{print $1}' | sort -u | wc -l)
if test "$n_unique" = "5"
    echo "  ✓ 5 distinct models in submissions/"
else
    echo "  ✗ Only $n_unique unique files among 5 submission ONNX files!"
    md5sum submissions/best_model_*.onnx
    set verify_ok 0
end

# 4c. Submission files must not match the warm-start source (sed_b0_fold*).
#     If they do, fine-tuning didn't change anything — training silently no-op'd.
echo
echo "--- 4c. Submission differs from warm-start baseline ---"
for f in 0 1 2 3 4
    set -l baseline experiments/sed_b0_fold{$f}/best_model_{$f}.onnx
    if not test -f $baseline
        echo "  ⚠ fold $f: baseline $baseline not found, skipping this check"
        continue
    end
    set -l sub_md5 (md5sum submissions/best_model_{$f}.onnx | awk '{print $1}')
    set -l base_md5 (md5sum $baseline | awk '{print $1}')
    if test "$sub_md5" != "$base_md5"
        echo "  ✓ fold $f: differs from sed_b0_fold$f (fine-tune changed weights)"
    else
        echo "  ✗ fold $f: IDENTICAL to sed_b0_fold$f warm-start — training was a no-op"
        set verify_ok 0
    end
end

# 4d. Submission files must not match old experiment dirs (stale-shipment guard).
#     Loops over any experiments/sed_finetune_* that isn't the current EXP_NAME.
echo
echo "--- 4d. Submission differs from other finetune runs (stale-shipment guard) ---"
for other_dir in experiments/sed_finetune_*
    set -l other_name (basename $other_dir)
    # Skip current run's dirs
    if string match -q "$EXP_NAME*" $other_name
        continue
    end
    # Skip per-fold dirs that aren't part of a 5-fold set (partial runs)
    if not test -f $other_dir/best_model.onnx
        continue
    end
    # Compare fold 0 as a spot check — a full collision across 5 folds has
    # ~zero probability, so one fold matching is enough to catch the bug.
    set -l fold_num (string match -r '_fold([0-9])$' -- $other_name)[2]
    if test -z "$fold_num"
        continue
    end
    set -l sub_md5 (md5sum submissions/best_model_{$fold_num}.onnx | awk '{print $1}')
    set -l other_md5 (md5sum $other_dir/best_model.onnx | awk '{print $1}')
    if test "$sub_md5" = "$other_md5"
        if string match -q "*$EXP_NAME*" $other_name
            # same experiment, fine
            continue
        end
        echo "  ✗ submissions/best_model_$fold_num.onnx matches $other_name — "\
             "you may be shipping the wrong run!"
        set verify_ok 0
    end
end
echo "  (checked against all experiments/sed_finetune_* dirs)"

# 4e. File size sanity — ONNX files should be ~17 MB; anything else means
#     export wrote a broken/empty/quantized file in the wrong place.
echo
echo "--- 4e. File size sanity (expect ~17 MB each) ---"
for f in 0 1 2 3 4
    set -l size_bytes (stat -c '%s' submissions/best_model_{$f}.onnx)
    set -l size_mb (math --scale=1 $size_bytes / 1048576)
    if test $size_bytes -lt 15000000 -o $size_bytes -gt 20000000
        echo "  ✗ fold $f: $size_mb MB — outside expected 15-20 MB range"
        set verify_ok 0
    else
        echo "  ✓ fold $f: $size_mb MB"
    end
end

# ---- Final summary ----
set -l elapsed (math (date +%s) - $start_time)
set -l minutes (math --scale=1 $elapsed / 60)
echo
echo "================================================================"
if test $verify_ok -eq 1
    echo "✓ Pipeline complete in $minutes minutes — submission verified"
    echo
    echo "Current submission/ md5s (source of truth for Kaggle upload):"
    md5sum submissions/best_model_*.onnx
    echo "================================================================"
    exit 0
else
    echo "✗ Pipeline finished training but VERIFICATION FAILED — DO NOT SUBMIT"
    echo "  Inspect stage 4 output above. submissions/ may contain wrong weights."
    echo "================================================================"
    exit 1
end