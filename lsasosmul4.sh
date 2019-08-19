#!/bin/bash
#SBATCH --account=def-yaoliang
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=10G
#SBATCH --time=00-10:00
python test.py  --autoencoder LSA --estimator SOS --dataset fmnist --batch_size 256 --code_length 64  --num_blocks 1 --hidden_size 2048 --epochs 3000 --select 4 --MulObj
