#!/usr/bin/env bash

source $CONDA_PATH/etc/profile.d/conda.sh
conda activate momask

split_name="$1"
echo ">>> Using split name: $split_name"

dataset="HumanML3D" # HumanML3D, Motion-X
if [ -n "$2" ]; then
    dataset="$2"
    echo ">>> Using dataset: $dataset"
fi

tmr=""
for arg in "$@"; do
    if [ "$arg" = "--tmr" ]; then
        tmr="-tmr"
        shift
    fi
done

epochs=1


# Set forget_texts based on split_name
forget_texts=$(jq --arg dataset "$dataset" --arg split_name "$split_name" -r '.splits[$dataset][$split_name].forget_texts[]' assets/splits.json | tr '\n' ' ')

retain_texts="walk walking run running jog step turn dance dancing move moving steps raise arm leg forward front backward sit dodge moves"
target_text=""

data_root="dataset/${dataset}"

seed=$RANDOM

for preserve_scale in 0.5 0.1; do
    echo ">>> Unlearning MoMask with UCE and RECE on $dataset with preserve_scale=$preserve_scale..."
    START=$(date +%s%3N)
    python -m src.methods.rece \
        --dataset_name $dataset \
        --name mtrans \
        --res_name rtrans \
        --vq_name rvq \
        --ext "UCE_RECE_${dataset}" \
        --seed $seed \
        --forget_text $forget_texts \
        --retain_text $retain_texts \
        --target_text "$target_text" \
        --ckpt base.tar \
        --epochs $epochs \
        --preserve_scale $preserve_scale

    END=$(date +%s%3N)
    ELAPSED=$((END - START))
    echo "Execution time: $ELAPSED milliseconds" 

    bash scripts/eval/t2m_unlearn.sh \
        --dataset "$dataset" \
        --split_name "$split_name" \
        --method RECE \
        --name "RECE${epochs}_$preserve_scale" \
        --ckpt "RECE${epochs}_$preserve_scale.tar"

    bash scripts/eval/t2m_unlearn.sh \
        --dataset "$dataset" \
        --split_name "$split_name" \
        --method UCE \
        --name "UCE_$preserve_scale" \
        --ckpt "UCE_$preserve_scale.tar"
done
