#!/usr/env/bin bash

# Show help message
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0"
    echo ""
    echo "Fine-tune text-to-motion models on cleaned dataset."
    echo ""
    echo "Note: Edit script to configure dataset, model names, and parameters."
    exit 0
fi

export WANDB_MODE=disabled
python train_t2m_transformer.py --name mtrans_t2m_ft --gpu_id 0 --dataset_name t2m_clean --batch_size 64 --vq_name rvq_name --is_continue
# python train_res_transformer.py --name rtrans_t2m_ft  --gpu_id 0 --dataset_name t2m_clean --batch_size 64 --vq_name rvq_name --cond_drop_prob 0.2 --share_weight --is_continue