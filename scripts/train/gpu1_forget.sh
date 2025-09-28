#!/usr/bin/env bash

# only retain
gpu_id=1
forget_kw="kick,punch,hit,beat,box"
set=forget
set_test="kw_splits/test-w-${forget_kw//,/_}"

# baseline
# python eval_t2m_trans_res.py \
#     --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw  \
#     --dataset_name HumanML3D \
#     --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns \
#     --gpu_id $gpu_id \
#     --cond_scale 4 \
#     --time_steps 10 \
#     --ext momask_t2m_$set \
#     --dataset_opt opt.txt \
#     --metric general \
#     --main_split $set_test \
#     --which_epoch net_best_fid.tar

# # uce
# python eval_t2m_trans_res.py \
#     --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw  \
#     --dataset_name HumanML3D \
#     --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns \
#     --gpu_id $gpu_id \
#     --cond_scale 4 \
#     --time_steps 10 \
#     --ext uce_t2m_$set \
#     --dataset_opt opt.txt \
#     --metric general \
#     --main_split $set_test \
#     --which_epoch UCE_toxic_hml3d.tar

# # rece
# python eval_t2m_trans_res.py \
#     --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw  \
#     --dataset_name HumanML3D \
#     --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns \
#     --gpu_id $gpu_id \
#     --cond_scale 4 \
#     --time_steps 10 \
#     --ext rece_t2m_$set \
#     --dataset_opt opt.txt \
#     --metric general \
#     --main_split $set_test \
#     --which_epoch RECE_toxic_hml3d.tar

# lcr
for n in 4 8 16 32
do
    python eval_t2m_trans_res.py \
    --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw  \
    --dataset_name HumanML3D \
    --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns \
    --gpu_id $gpu_id \
    --cond_scale 4 \
    --time_steps 10 \
    --ext lcr${n}_t2m_${set} \
    --dataset_opt opt.txt \
    --metric general \
    --main_split $set_test \
    --which_epoch net_best_fid.tar \
    --vq_model lcr${n}_kick_punch_hit_beat_box.tar
done

