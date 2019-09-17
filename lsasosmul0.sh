#!/bin/bash
#SBATCH --account=def-yaoliang
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=5G
#SBATCH --time=00-10:00
python test.py --dataset mnist --Test  --autoencoder LSA --estimator SOS --MulObj --Train --epoch 3000