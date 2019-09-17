#!/bin/bash
#SBATCH --account=def-yaoliang
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=2
#SBATCH --mem= 10G
#SBATCH --time=00-10:00
python test.py  --autoencoder LSA --estimator EN  --dataset mnist --batch_size 256 --code_length 64  --num_blocks 1 --hidden_size 2048 --MulObj --epochs 2000 --select 2
