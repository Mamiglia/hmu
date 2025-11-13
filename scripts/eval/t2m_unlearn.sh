#!/bin/bash

# Show help message
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Evaluate unlearning methods on text-to-motion models."
    echo ""
    echo "Options:"
    echo "  --dataset <name>       Dataset name (default: HumanML3D)"
    echo "  --split_name <name>    Concept split name (default: violence)"
    echo "  --ckpt <file>          Checkpoint file (default: base.tar)"
    echo "  --method <name>        Method name for logging (default: unnamed)"
    echo "  --name <name>          Run name override (default: same as method)"
    echo "  --max_rank <num>       Max rank for retrieval (default: 100)"
    echo "  --mtrans_name <name>   Motion transformer name (default: mtrans)"
    echo "  -h, --help             Show this help message"
    exit 0
fi

source $CONDA_PATH/etc/profile.d/conda.sh
conda activate momask

dataset="HumanML3D"  # Default dataset
split_name="violence"  # Default split name
ckpt="base.tar"  # Default checkpoint
tmr=""  # Default TMR suffix # KEEP DISABLE
max_rank=100  # Default max rank for TMR retrieval
method="unnamed"  # Default method name

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            dataset="$2"
            shift 2
            ;;
        --split_name)
            split_name="$2"
            shift 2
            ;;
        --ckpt)
            ckpt="$2"
            shift 2
            ;;
        # --max_rank)
        #     max_rank="$2"
        #     shift 2
        #     ;;
        # --tmr)
        #     tmr="-tmr" 
        #     shift
        #     ;;
        --method)
            method="$2"
            shift 2
            ;;
        --name)
            name="$2"
            shift 2
            ;;
        --mtrans_name)
            mtrans_name="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

name=${name:-$method}
mtrans_name=${mtrans_name:-"mtrans"}

seed=$RANDOM  # Random seed for unique run name
# ext="${method}_${dataset}_${split_name}${tmr}_$seed"
ext="${mtrans_name}_${name}_${dataset}_${split_name}${tmr}"

prompts_f="assets/qualitatives_${dataset}.txt"

retainset_test="kw_splits/test-wo-${split_name}${tmr}" 
forgetset_test="kw_splits/test-w-${split_name}${tmr}"

forget_texts=$(jq --arg dataset "$dataset" --arg split_name "$split_name" -r '.splits[$dataset][$split_name].forget_texts[]' assets/splits.json | tr '\n' ' ')

# export WANDB_MODE=online
export WANDB_PROJECT="hmu2"
export WANDB_NAME="$ext"
export WANDB_RUN_ID="$ext"
export WANDB_TAGS="$method,${split_name}${tmr},$dataset"
export WANDB_RUN_GROUP="$name-$dataset-$split_name"

echo ">>> Evaluating $ext on $dataset/$split_name..."

# --- NCS Evaluation --- DEPRECATED ---
# echo ">>> Processing Forget set for $ext"
# python -m src.methods.gen_t2m_batch \
#     --dataset_name $dataset \
#     --run_name $ext \
#     --ckpt $ckpt \
#     --skip_viz \
#     --repeat_times 10 \
#     --batch_size 512 \
#     --seed $seed \
#     --split $forgetset_test

# echo ">>> Running Detector on Forget set for $ext"
# out="$PWD/generation/${ext}"
# records="$out/records.json"
# conda activate TMR

# cd src/TMR
# python m2m_retrieval.py \
#     path=$records \
#     top_k=$max_rank
# cd ../../

# conda activate momask

# python -m src.eval.ncs_compute \
#     --file $records \
#     --run_name $ext \
#     --forget_kw $forget_texts

# --- T2M-U Evaluation ---

echo ">>> Generating qualitatives on $dataset..."
python -m src.momask_codes.gen_t2m \
    --dataset_name $dataset \
    --run_name $ext \
    --text_path $prompts_f \
    --repeat_times 1 \
    --seed $seed \
    --name $mtrans_name \
    --ckpt $ckpt

echo ">>> Evaluating $ext on Forget"
python -m src.eval.t2m_unlearn \
    --dataset_name $dataset \
    --ckpt $ckpt \
    --run_name $ext \
    --toxic_terms $forget_texts \
    --seed $seed \
    --split $forgetset_test \
    --name $mtrans_name \
    --method $method 


echo ">>> Evaluating $ext on Retain"
python -m src.eval.t2m_unlearn \
    --dataset_name $dataset \
    --ckpt $ckpt \
    --run_name $ext \
    --toxic_terms $forget_texts \
    --seed $seed \
    --name $mtrans_name \
    --split $retainset_test