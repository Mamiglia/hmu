#!/usr/bin/env bash

motion="violence"

echo "Processing motion: $motion"

bash scripts/utils/split_dataset.sh test "$motion"
# bash scripts/utils/split_dataset.sh train_val "$motion"

bash scripts/eval/eval_gen_baselines.sh "$motion"
bash scripts/eval/eval_gen_cp_combined.sh "$motion"
bash scripts/eval/eval_gen_rece.sh "$motion"

# bash scripts/eval/eval_gen_cp.sh "$motion" --no_tmr
# bash scripts/eval/eval_gen_baselines.sh "$motion" --no_tmr
# bash scripts/eval/eval_gen_rece.sh "$motion" --no_tmr

echo "Completed motion: $motion"
