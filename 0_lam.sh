#!/bin/bash
#
#SBATCH --job-name=lam0
#SBATCH --output=lam0.txt
#
#SBATCH --ntasks=4

python test.py --dataset mnist --autoencoder LSA --lr 0.00001  --Train --estimator SOS --lam 0.01 --log_step 1000  --before_log_epoch 200 --batch_size 512 --select 0

python test.py --dataset mnist --autoencoder LSA --lr 0.00001  --Train --estimator SOS --lam 0.1 --log_step 1000  --before_log_epoch 200 --batch_size 512 --select 0

python test.py --dataset mnist --autoencoder LSA --lr 0.00001  --Train --estimator SOS --lam 1 --log_step 1000 --before_log_epoch 200 --batch_size 512 --select 0

python test.py --dataset mnist --autoencoder LSA --lr 0.00001  --Train --estimator SOS --lam 10 --log_step 1000 --before_log_epoch 200 --batch_size 512 --select 0
