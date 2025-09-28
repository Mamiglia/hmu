#! /usr/bin/env bash

#SBATCH --account=IscrC_MU4M
#SBATCH --partition boost_usr_prod
#SBATCH --time=23:59:59
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --job-name=rtrans_t2m_clean
#SBATCH --error=cineca/rtrans_t2m_clean.err
#SBATCH --output=cineca/rtrans_t2m_clean.out         

module load profile/deeplrn cuda
source .venv/bin/activate
export WANDB_MODE=offline
python train_vq.py --name rvq_t2m_clean --gpu_id 0 --dataset_name t2m_clean --batch_size 256 --num_quantizers 6  --max_epoch 50 --quantize_dropout_prob 0.2 --gamma 0.05
python train_t2m_transformer.py --name mtrans_t2m_clean --gpu_id 0 --dataset_name t2m_clean --batch_size 64 --vq_name rvq_t2m_clean
python train_res_transformer.py --name rtrans_t2m_clean  --gpu_id 0 --dataset_name t2m_clean --batch_size 64 --vq_name rvq_t2m_clean --cond_drop_prob 0.2 --share_weight
# * Run
# srun -A IscrC_MU4M -p boost_usr_prod -N 1 -G 1 --pty /bin/bash