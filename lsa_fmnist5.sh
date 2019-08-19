#!/bin/bash
#SBATCH --account=def-yaoliang
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=2G
#SBATCH --time=00-10:00
python test.py  --autoencoder LSA --dataset fmnist --batch_size 256 --code_length 64  --num_blocks 1 --hidden_size 2048 --epochs 2000 --select 5 --log_step 200