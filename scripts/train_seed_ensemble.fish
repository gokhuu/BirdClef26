#!/usr/bin/env fish
# ============================================================
# Run a Phase 2 finetune across additional seeds, reusing existing
# Phase 1 (focal pretrain) checkpoints. Writes per-(seed, fold) runs to
# their own group folders so the original default-seed run is never touched.
#
# Replaces train_b0_multiseed.sh + train_seresnext_seed_ensemble.sh.
#
# Usage examples:
#   # B0 v2 multi-seed (replaces train_b0_multiseed.sh)
#   fish scripts/train_seed_ensemble.fish \
#       --config configs/finetune_pseudo_v2.yaml \
#       --base-name sed_finetune_pseudo_v2 \
#       --pretrain-group sed_b0 \
#       --pretrain-name sed_b0 \
#       --seeds 123,2024 \
#       --extra-prereq data/pseudo/sed_b0_v2_pseudo_balanced.csv
#
#   # SEResNeXt multi-seed (replaces train_seresnext_seed_ensemble.sh)
#   fish scripts/train_seed_ensemble.fish \
#       --config configs/finetune_seresnext.yaml \
#       --base-name seresnext_finetune \
#       --pretrain-group seresnext_focal \
#       --pretrain-name seresnext_focal \
#       --seeds 123,2024
#
# Output dir layout (matches the rest of the repo's nested convention):
#   experiments/{base_name}_seed{S}/{base_name}_seed{S}_fold{F}/
# ============================================================

set -l FOLDS 0 1 2 3 4
set -l LOG_DIR logs
set -l CONFIG ""
set -l BASE_NAME ""
set -l PRETRAIN_GROUP ""
set -l PRETRAIN_NAME ""
set -l SEEDS_RAW "123,2024"
set -l DEFAULT_SEED "42"
set -l EXTRA_PREREQS

# ---------- Argument parsing ----------
set -l i 1
while test $i -le (count $argv)
    set -l flag $argv[$i]
    set -l val ""
    if test (math $i + 1) -le (count $argv)
        set val $argv[(math $i + 1)]
    end
    switch $flag
        case --config
            set CONFIG $val
            set i (math $i + 2)
        case --base-name
            set BASE_NAME $val
            set i (math $i + 2)
        case --pretrain-group
            set PRETRAIN_GROUP $val
            set i (math $i + 2)
        case --pretrain-name
            set PRETRAIN_NAME $val
            set i (math $i + 2)
        case --seeds
            set SEEDS_RAW $val
            set i (math $i + 2)
        case --default-seed
            set DEFAULT_SEED $val
            set i (math $i + 2)
        case --extra-prereq
            set EXTRA_PREREQS $EXTRA_PREREQS $val
            set i (math $i + 2)
        case '*'
            echo "Unknown argument: $flag"
            exit 1
    end
end

# ---------- Validate ----------
for required_var in CONFIG BASE_NAME PRETRAIN_GROUP PRETRAIN_NAME
    if test -z "$$required_var"
        echo "ABORT: --"(string lower (string replace -a _ - $required_var))" is required"
        exit 1
    end
end

set -l SEEDS (string split , -- $SEEDS_RAW)
mkdir -p $LOG_DIR

# ---------- Pre-flight: warm-start checkpoints ----------
echo "Pre-flight: verifying Phase 1 checkpoints in experiments/$PRETRAIN_GROUP/..."
set -l missing 0
for f in $FOLDS
    set -l ckpt experiments/$PRETRAIN_GROUP/{$PRETRAIN_NAME}_fold{$f}/best_model.pt
    if not test -f $ckpt
        echo "  MISSING: $ckpt"
        set missing 1
    else
        echo "  ok: $ckpt"
    end
end
if test $missing -eq 1
    echo "ABORT: missing Phase 1 checkpoints. Run the corresponding _all training first."
    exit 1
end

# ---------- Pre-flight: config and any extra prerequisites ----------
if not test -f $CONFIG
    echo "ABORT: missing config $CONFIG"
    exit 1
end
for prereq in $EXTRA_PREREQS
    if not test -f $prereq
        echo "ABORT: missing prerequisite file: $prereq"
        exit 1
    else
        echo "  ok: $prereq"
    end
end

echo "All prerequisites present. Starting seed ensemble..."
echo "  config         : $CONFIG"
echo "  base_name      : $BASE_NAME"
echo "  pretrain group : $PRETRAIN_GROUP (run name $PRETRAIN_NAME)"
echo "  seeds          : $SEEDS"

# ---------- Per-(seed, fold) Phase 2 finetune ----------
for SEED in $SEEDS
    echo
    echo "===================================================="
    echo "SEED $SEED — Phase 2 finetune across all folds"
    echo "Started at: "(date)
    echo "===================================================="

    set -l SEED_GROUP {$BASE_NAME}_seed{$SEED}

    for f in $FOLDS
        set -l RUN_NAME {$SEED_GROUP}_fold{$f}
        set -l EXP_DIR experiments/$SEED_GROUP/$RUN_NAME

        # Resume-friendly: skip if already trained successfully
        if test -d $EXP_DIR; and test -f $EXP_DIR/best_model.pt
            echo "  skipping $RUN_NAME (already trained)"
            continue
        end

        set -l CKPT experiments/$PRETRAIN_GROUP/{$PRETRAIN_NAME}_fold{$f}/best_model.pt

        echo
        echo "--- seed=$SEED fold=$f finetune  ["(date '+%H:%M:%S')"] ---"

        set -x BIRDCLEF_RUN_NAME $RUN_NAME
        python -m src.training.finetune \
            $CONFIG \
            --fold $f \
            --seed $SEED \
            --init-checkpoint $CKPT \
            2>&1 | tee $LOG_DIR/$RUN_NAME.log
        set -e BIRDCLEF_RUN_NAME
    end
end

# ---------- Summary ----------
echo
echo "===================================================="
echo "Seed ensemble complete at: "(date)
echo "===================================================="

# Default-seed (already-existing) run lives at experiments/{base_name}/{base_name}_fold{F}/
echo
echo "Results — original (seed=$DEFAULT_SEED):"
for f in $FOLDS
    set -l log_csv experiments/$BASE_NAME/{$BASE_NAME}_fold{$f}/training_log.csv
    if test -f $log_csv
        # finetune.py columns (combined_val_auc is column 8)
        set -l best_comb (awk -F',' 'NR>1 {print $8}' $log_csv | sort -rn | head -1)
        echo "  seed=$DEFAULT_SEED  fold=$f: combined=$best_comb"
    else
        echo "  seed=$DEFAULT_SEED  fold=$f: NO LOG ($log_csv)"
    end
end

for SEED in $SEEDS
    echo
    echo "Results — seed=$SEED:"
    set -l SEED_GROUP {$BASE_NAME}_seed{$SEED}
    for f in $FOLDS
        set -l log_csv experiments/$SEED_GROUP/{$SEED_GROUP}_fold{$f}/training_log.csv
        if test -f $log_csv
            set -l best_comb (awk -F',' 'NR>1 {print $8}' $log_csv | sort -rn | head -1)
            echo "  seed=$SEED  fold=$f: combined=$best_comb"
        else
            echo "  seed=$SEED  fold=$f: NO LOG"
        end
    end
end

# Per-seed mean
echo
echo "===================================================="
echo "Per-seed mean combined val_auc (seed variance check):"
echo "===================================================="
set -l ALL_SEEDS $DEFAULT_SEED $SEEDS
for SEED in $ALL_SEEDS
    if test "$SEED" = "$DEFAULT_SEED"
        set -l prefix experiments/$BASE_NAME/{$BASE_NAME}_fold
    else
        set -l SEED_GROUP {$BASE_NAME}_seed{$SEED}
        set -l prefix experiments/$SEED_GROUP/{$SEED_GROUP}_fold
    end
    set -l sum 0
    set -l count 0
    for f in $FOLDS
        set -l log_csv {$prefix}{$f}/training_log.csv
        if test -f $log_csv
            set -l best (awk -F',' 'NR>1 {print $8}' $log_csv | sort -rn | head -1)
            set sum (awk -v s="$sum" -v b="$best" 'BEGIN { print s + b }')
            set count (math $count + 1)
        end
    end
    if test $count -gt 0
        set -l mean (awk -v s="$sum" -v c="$count" 'BEGIN { printf "%.5f", s / c }')
        echo "  seed=$SEED: mean = $mean (over $count folds)"
    end
end

echo
echo "Interpretation:"
echo "  Per-seed mean range > 0.005 → seed variance is real, ensemble will help."
echo "  Per-seed mean range < 0.002 → seed variance tiny, ensemble lift marginal."
echo
echo "Next steps:"
echo "  python scripts/compare_seed_ensemble_oof.py \\"
echo "      --base-name $BASE_NAME --seeds $DEFAULT_SEED,"(string join , $SEEDS)
