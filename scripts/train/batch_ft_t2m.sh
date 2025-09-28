#! /usr/bin/env bash

#SBATCH --account=IscrC_MU4M
#SBATCH --partition boost_usr_prod
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --job-name=r_t2m_ft
#SBATCH --error=cineca/r_t2m_ft.err
#SBATCH --output=cineca/r_t2m_ft.out         

module load profile/deeplrn cuda
source .venv/bin/activate
export WANDB_MODE=offline
# ! --is_continue is used for fine-tuning
# first, copy HumanML3D's data
python train_t2m_transformer.py --name mtrans_t2m_ft --gpu_id 0 --dataset_name t2m_clean --batch_size 64 --vq_name rvq_t2m --is_continue --max_epoch 1000
python train_res_transformer.py --name rtrans_t2m_ft  --gpu_id 0 --dataset_name t2m_clean --batch_size 64 --vq_name rvq_t2m --cond_drop_prob 0.2 --share_weight --is_continue --max_epoch 1000

# * Run
# srun -A IscrC_MU4M -p boost_usr_prod -N 1 -G 2 --pty /bin/bash