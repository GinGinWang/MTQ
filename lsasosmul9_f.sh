#!/bin/bash
#SBATCH --account=def-yaoliang
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=5G
#SBATCH --time=00-07:00
python test.py --dataset fmnist --Test  --autoencoder LSA --estimator SOS --MulObj --Train --epoch 2000 --select 7 --lr 0.00001
