#!/bin/bash
#SBATCH --account=def-yaoliang
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=5G
#SBATCH --time=00-07:00
python test.py --dataset mnist --Test  --autoencoder LSA --estimator SOS --lam 0.1 --Train --epoch 2000 --select 5