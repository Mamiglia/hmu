#!/usr/env/bin bash
export WANDB_MODE=disabled
python train_t2m_transformer.py --name mtrans_t2m_ft --gpu_id 0 --dataset_name t2m_clean --batch_size 64 --vq_name rvq_name --is_continue
# python train_res_transformer.py --name rtrans_t2m_ft  --gpu_id 0 --dataset_name t2m_clean --batch_size 64 --vq_name rvq_name --cond_drop_prob 0.2 --share_weight --is_continue