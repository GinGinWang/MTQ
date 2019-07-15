#!/bin/bash
#SBATCH --account=def-yaoliang
#SBATCH --nodes=5
#SBATCH --gres=gpu:p100:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem= 10G
#SBATCH --time=0-10:00

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt


python test_cloud.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001  --num_blocks 1 --hidden_size 2048  --n_class 8 --noise3 0.5 --MulObj --NoTest

