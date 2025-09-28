#!/usr/bin/env bash

motions=("violence" "kick" "punch_jab_box_hit_beat" "martial_arts" "gun_shoot" )
dataset="Motion-X"

for motion in "${motions[@]}"; do
    echo "Processing motion: $motion"

    bash scripts/utils/split_dataset.sh \
        --main_split train_val \
        --split_name "$motion" \
        --dataset $dataset \
        --min_occurence 1
    
    bash scripts/utils/split_dataset.sh \
        --main_split test \
        --split_name "$motion" \
        --dataset $dataset \
        --min_occurence 1

    bash scripts/eval/lcr.sh "$motion" $dataset 
    
    bash scripts/eval/t2m_unlearn.sh \
        --method vanilla \
        --ckpt base.tar \
        --split_name "$motion" \
        --dataset $dataset

    bash scripts/eval/rece_uce.sh "$motion" $dataset

    if [[ "$motion" == "violence" ]]; then
        bash scripts/eval/t2m_unlearn.sh \
            --method dr \
            --ckpt dr_$motion.tar \
            --split_name "$motion" \
            --dataset $dataset

        bash scripts/eval/t2m_unlearn.sh \
            --method ft \
            --ckpt ft_$motion.tar \
            --split_name "$motion" \
            --dataset $dataset

        bash scripts/eval/t2m_unlearn.sh \
            --method esd \
            --ckpt esd.tar \
            --split_name "$motion" \
            --dataset $dataset
    fi
    
    echo "Completed motion: $motion"
done;
