#!/usr/bin/env bash

export WANDB_MODE=disabled
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate momask

gpu_id=0

tmr_folder="$(pwd)/TMR"
momask_folder="$(pwd)/momask-codes"
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --main_split)
            main_split="$2"
            shift 2
            ;;
        --split_name)
            split_name="$2"
            shift 2
            ;;
        --dataset|-d)
            dataset="$2"
            shift 2
            ;;
        --min_occurence)
            min_occurence="$2"
            shift 2
            ;;
        --tmr_min_occurence)
            tmr_min_occurence="$2"
            shift 2
            ;;
        --max_rank)
            max_rank="$2"
            shift 2
            ;;
        --repeat_times)
            repeat_times="$2"
            shift 2
            ;;
        --tmr)
            use_tmr=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 --main_split <split> --split_name <name> [options]"
            echo "Options:"
            echo "  --main_split <split>        Main dataset split (required)"
            echo "  --split_name <name>         Split name (required)"
            echo "  --dataset <name>            Dataset name (default: HumanML3D)"
            echo "  --tmr_min_occurence <num>     Minimum occurrences (default: 8)"
            echo "  --max_rank <num>            Maximum rank for TMR (default: 10)"
            echo "  --repeat_times <num>        Repeat times (default: 10)"
            echo "  --tmr                       Enable TMR filtering"
            echo "  --help, -h                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set defaults
dataset="${dataset:-HumanML3D}"
tmr_min_occurence="${tmr_min_occurence:-8}"
max_rank="${max_rank:-10}"
repeat_times="${repeat_times:-10}"
use_tmr="${use_tmr:-false}"
min_occurence="${min_occurence:-1}"

# Check required arguments
if [[ -z "$main_split" ]]; then
    echo "Error: --main_split is required"
    exit 1
fi

if [[ -z "$split_name" ]]; then
    echo "Error: --split_name is required"
    exit 1
fi

# Set model names
vq_name="rvq"
m_name="mtrans"
r_name="rtrans"

echo ">>> Using main split: $main_split"
echo ">>> Using split name: $split_name"
echo ">>> Using dataset: $dataset"
echo ">>> Using min occurrences: $tmr_min_occurence"
echo ">>> Using max rank: $max_rank"
echo ">>> Using repeat times: $repeat_times"
echo ">>> TMR filtering: $use_tmr"

forget_texts=$(jq --arg dataset "$dataset" --arg split_name "$split_name" -r '.splits[$dataset][$split_name].forget_texts[]' assets/splits.json | tr '\n' ',' | sed 's/,$//')

search_pattern=$(echo "$forget_texts" | sed 's/,/|/g')

base_path="dataset/$dataset"
retainset="kw_splits/$main_split-wo-$split_name" 
forgetset="kw_splits/$main_split-w-$split_name"
echo ">>> Naive splits for $main_split with forget texts: $forget_texts"
# This code block creates the splits based on keyword occurrence
# It's all implemented in bash!
# Only create splits if they don't already exist
if [[ ! -f "$base_path/$retainset.txt" || ! -f "$base_path/$forgetset.txt" ]]; then
    echo ">>> Split files don't exist, creating them now..."
    # Clear existing files if they exist but might be incomplete
    > $base_path/$forgetset.txt
    > $base_path/$retainset.txt

    total_files=$(wc -l < "$base_path/$main_split.txt")

    # Create base retainset:
    # Use pv to show progress while processing the file
    pv -l -s "$total_files" < "$base_path/$main_split.txt" | while IFS= read -r id; do
        matching_lines=$(grep -Ec "$search_pattern" "$base_path/texts/${id}.txt")
        total_lines=$(wc -l < "$base_path/texts/${id}.txt")
        if [ "$matching_lines" -ge "$min_occurence" ]; then
            echo "$id" >> $base_path/$forgetset.txt
        else
            echo "$id" >> $base_path/$retainset.txt
        fi
    done
    
    # Count and print the sizes
else
    echo ">>> Split files already exist, skipping creation"
fi
forget_count=$(wc -l < $base_path/$forgetset.txt)
retain_count=$(wc -l < $base_path/$retainset.txt)
echo ">>> Naive split files: $retainset.txt ($retain_count samples) and $forgetset.txt ($forget_count samples)"

# If TMR is not used, exit here
if [ "$use_tmr" = false ]; then
    echo ">>> TMR filtering is disabled, exiting"
    exit 0
fi

retainset_tmr="${retainset}-tmr"
forgetset_tmr="${forgetset}-tmr"
# Check if TMR split files already exist
if [[ -f "$base_path/$retainset_tmr.txt" && -f "$base_path/$forgetset_tmr.txt" ]]; then
    echo ">>> TMR split files already exist, exiting"
    exit 0
fi

ext="splitting_${dataset}_${main_split}_${split_name}_${RANDOM}"

echo ">>> Processing Forget set for vanilla model"
python gen_t2m_dataset.py \
    --res_name $r_name \
    --vq_name $vq_name \
    --dataset_name $dataset \
    --name $m_name \
    --gpu_id $gpu_id \
    --cond_scale 4 \
    --time_steps 10 \
    --ext $ext \
    --vq_model latest.tar  \
    --main_split $forgetset \
    --which_epoch latest.tar \
    --model_file latest.tar \
    --repeat_times $repeat_times \
    --skip_visualization

echo ">>> Running Retrieval on Forget set for vanilla model"
out="$momask_folder/generation/${ext}"
records="$out/records.json"
conda activate TMR

cd $tmr_folder
python batch_motion_retrieval.py \
    path=$records \
    top_k=$max_rank

cd $momask_folder
conda activate momask


# Add a new variable for the max rank to consider
echo ">>> Using max rank: $max_rank for TMR filtering on kw: $search_pattern"

# Then replace the hardcoded 10 with the variable
# Process the forget set - find motions related to specified keywords
jq --arg terms "$search_pattern" \
    '.records[] | 
     select(.retrieval | 
              any(.description | test($terms; "i"))) | 
     .name' $records | tr -d '"' | sort | uniq -c | awk -v tmr_min_occurence="$tmr_min_occurence" '$1 >= tmr_min_occurence {print $2}' >> $base_path/$retainset_tmr.txt >> $base_path/$forgetset_tmr.txt

echo ">>> Forget set processed and saved to $base_path/$forgetset_tmr.txt"

# Copy the base retain set file
cp $base_path/$retainset.txt $base_path/$retainset_tmr.txt

# Add to the retain set - motions NOT related to specified keywords
jq --arg terms "$search_pattern" \
    '.records[] | 
     select(.retrieval | 
              any(.description | test($terms; "i"))) | 
     .name' \
    $records | tr -d '"' | sort | uniq -c | awk -v tmr_min_occurence="$tmr_min_occurence" '$1 < tmr_min_occurence {print $2}' >> $base_path/$retainset_tmr.txt
echo ">>> Retain set processed and saved to $base_path/$retainset_tmr.txt"

# Count and print the final set sizes
forget_count_tmr=$(sort -u "$base_path/$forgetset_tmr.txt" | wc -l)
retain_count_tmr=$(sort -u "$base_path/$retainset_tmr.txt" | wc -l)

echo ">>> TMR filtered split sizes:"
echo ">>> Forget set ($forgetset_tmr): $forget_count_tmr samples"
echo ">>> Retain set ($retainset_tmr): $retain_count_tmr samples"

if [[ $forget_count_tmr -eq 0 || $retain_count_tmr -eq 0 ]]; then
    echo ">>> Error: TMR filtering resulted in empty sets. Please check the filtering criteria." >&2
    rm -f "$base_path/$forgetset_tmr.txt" "$base_path/$retainset_tmr.txt"
    exit 1
fi
