#!/usr/bin/env bash

motions=(
    # "crawl_creep_slide"
    # "jump_leap_hop"
    # "kick"
    # "punch_jab_box_hit_beat"
    "zombie"
    # "none"
    # "duck_squat"
    # "throw"
    # "violence"
)

for motion in "${motions[@]}"; do
    echo "Processing motion: $motion"

    bash scripts/utils/split_dataset.sh test "$motion" HumanML3D 8 --no_tmr
    bash scripts/utils/split_dataset.sh train_val "$motion" HumanML3D 8 --no_tmr
    bash scripts/utils/split_dataset.sh train "$motion" HumanML3D 8 --no_tmr
    bash scripts/utils/split_dataset.sh val "$motion" HumanML3D 8 --no_tmr

    echo "Completed motion: $motion"
done
