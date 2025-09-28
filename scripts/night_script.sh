#!/usr/bin/env bash

motions=(
   "violence"
    "kick"
    "punch_jab_box_hit_beat"
    "crawl_creep_slide"
    "jump_leap_hop"
    "zombie"
    "duck_squat"
    "throw"
    # "none"
)
dataset="HumanML3D"

for motion in "${motions[@]}"; do
    echo "Processing motion: $motion"

    bash scripts/utils/split_dataset.sh \
        --main_split train_val \
        --split_name "$motion" \
        --dataset $dataset \
        --min_occurence 2
    
    bash scripts/utils/split_dataset.sh \
        --main_split test \
        --split_name "$motion" \
        --dataset $dataset \
        --min_occurence 2
    
    bash scripts/eval/lcr.sh "$motion" $dataset 
    
    bash scripts/eval/t2m_unlearn.sh \
        --method vanilla \
        --ckpt base.tar \
        --split_name "$motion"

    bash scripts/eval/t2m_unlearn.sh \
        --method dr \
        --ckpt dr_$motion.tar \
        --split_name "$motion"

    bash scripts/eval/rece_uce.sh "$motion" $dataset

    if [[ "$motion" == "violence" ]]; then
        bash scripts/eval/t2m_unlearn.sh \
            --method ft \
            --ckpt ft_$motion.tar \
            --split_name "$motion"

        # bash scripts/eval/t2m_unlearn.sh \
        #     --method esd \
        #     --ckpt esd.tar \
        #     --split_name "$motion"
    fi
    
    echo "Completed motion: $motion"
done;
