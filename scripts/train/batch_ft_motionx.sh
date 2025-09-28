#! /usr/bin/env bash

#SBATCH --account=IscrC_MU4M
#SBATCH --partition boost_usr_prod
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --job-name=r_mx_ft
#SBATCH --error=cineca/r_mx_ft.err
#SBATCH --output=cineca/r_mx_ft.out         

module load profile/deeplrn cuda
source .venv/bin/activate
export WANDB_MODE=offline
# ! --is_continue is used for fine-tuning
python train_t2m_transformer.py --name mtrans_motionx_ft --gpu_id 0 --dataset_name motionx_clean --batch_size 64 --vq_name rvq_motionx --is_continue --max_epoch 1000
python train_res_transformer.py --name rtrans_motionx_ft  --gpu_id 0 --dataset_name motionx_clean --batch_size 64 --vq_name rvq_motionx --cond_drop_prob 0.2 --share_weight --is_continue --max_epoch 1000

# * Run
# srun -A IscrC_MU4M -p boost_usr_prod -N 1 -G 2 --pty /bin/bash