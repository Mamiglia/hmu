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

# Set forget_texts based on split_name
forget_texts=$(jq --arg dataset "$dataset" --arg split_name "$split_name" -r '.splits[$dataset][$split_name].forget_texts[]' assets/splits.json | tr '\n' ' ')

retainset_test="kw_splits/test-wo-${split_name}${tmr}" 
forgetset_test="kw_splits/test-w-${split_name}${tmr}"
retainset_train="kw_splits/train_val-wo-${split_name}${tmr}"
forgetset_train="kw_splits/train_val-w-${split_name}${tmr}"

data_root="dataset/${dataset}"

# Set forget_texts based on split_name
case $split_name in
    "violence")
        forget_texts=$(jq --arg dataset "$dataset" --arg split_name "$split_name" -r '.splits[$dataset][$split_name].forget_texts[]' assets/splits.json | tr '\n' ' ')
        split_names=("kick" "punch_jab_box_hit_beat")
        code_prunes=(16 8)
        # For Motion-X, use:
        # split_names=("kick" "punch_jab_box_hit_beat" "gun_shoot" "martial_arts" "weapons")
	    # code_prunes=(16 8 4 16 4)
        ;;
    *)
        echo "Error: Unknown split name '$split_name'" >&2
        exit 1
        ;;
esac

pruned_files=()
for i in "${!split_names[@]}"; do
    split="${split_names[$i]}"
    code_prune="${code_prunes[$i]}"
    forget_texts_split=$(jq --arg dataset "$dataset" --arg split_name "$split" -r '.splits[$dataset][$split_name].forget_texts[]' assets/splits.json | tr '\n' ' ')

    retainset_test_split="kw_splits/test-wo-${split}${tmr}"
    forgetset_test_split="kw_splits/test-w-${split}${tmr}"
    retainset_train_split="kw_splits/train_val-wo-${split}${tmr}"
    forgetset_train_split="kw_splits/train_val-w-${split}${tmr}"

    pruned_model_file="lcr${code_prune}_${dataset}_${split}${tmr}"
    ckpt="$pruned_model_file.tar"

    echo ">>> Pruning Model: $pruned_model_file"
    START=$(date +%s%3N)
    python -m src.methods.lcr \
        --ckpt base.tar \
        --dataset_name $dataset \
        --run_name $pruned_model_file \
        --split $data_root/$forgetset_train \
        --code_prune $code_prune \
        --codes_csv "assets/${dataset}_codes.csv" \
        --batch_size 512 \
        --repeat_times 20 \
        --just_select_codes

    pruned_files+=("assets/codes/$pruned_model_file.yaml")

done

pruned_model_combined_file="LCR_"
for i in "${!split_names[@]}"; do
    split="${split_names[$i]}"
    code_prune="${code_prunes[$i]}"
    pruned_model_combined_file+="${split}${code_prune}_"
done
pruned_model_combined_file+="combined${tmr}"
ckpt="$pruned_model_combined_file.tar"

echo ">>> Combining Pruned Models: $pruned_model_combined_file"

python -m src.methods.lcr \
    --ckpt base.tar \
    --dataset_name $dataset \
    --run_name $pruned_model_combined_file \
    --censor_codes_files "${pruned_files[@]}"

bash scripts/eval/t2m_unlearn.sh \
    --dataset "$dataset" \
    --split_name "$split_name" \
    --method "$pruned_model_combined_file" \
    --ckpt "$ckpt"