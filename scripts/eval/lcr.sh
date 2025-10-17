#!/usr/bin/env bash

# Show help message
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 <split_name> [dataset] [--tmr]"
    echo ""
    echo "Apply LCR (Latent Code Removal) unlearning method."
    echo ""
    echo "Arguments:"
    echo "  split_name             Concept split name (e.g., violence, kick)"
    echo "  dataset                Dataset name (default: HumanML3D)"
    echo ""
    echo "Options:"
    echo "  --tmr                  Use TMR-filtered splits"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 violence HumanML3D"
    exit 0
fi

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

if [ -z "$split_name" ]; then
    echo "Error: split_name argument is required."
    exit 1
fi
if [ -z "$dataset" ]; then
    echo "Error: dataset argument is required."
    exit 1
fi

# Set forget_texts based on split_name
forget_texts=$(jq --arg dataset "$dataset" --arg split_name "$split_name" -r '.splits[$dataset][$split_name].forget_texts[]' assets/splits.json | tr '\n' ' ')

retainset_test="kw_splits/test-wo-${split_name}${tmr}" 
forgetset_test="kw_splits/test-w-${split_name}${tmr}"
retainset_train="kw_splits/train_val-wo-${split_name}${tmr}"
forgetset_train="kw_splits/train_val-w-${split_name}${tmr}"

data_root="dataset/${dataset}"

for code_prune in 4 8 16 32 64;  # code_prune is the number of codes to prune
do
    # code_prune is the number of codes, not the layers
    pruned_model_file="lcr${code_prune}_${dataset}_${split_name}${tmr}"
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
        --repeat_times 20

    END=$(date +%s%3N)
    ELAPSED=$((END - START))
    echo "Execution time: $ELAPSED milliseconds" # TODO >> prune_time.txt

    # * momask baseline
    bash scripts/eval/t2m_unlearn.sh \
        --dataset "$dataset" \
        --split_name "$split_name" \
        --method "LCR" \
        --name "LCR$code_prune" \
        --ckpt "$ckpt"
done