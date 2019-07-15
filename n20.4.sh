#!/bin/bash
#SBATCH --account=def-yaoliang
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=4
#SBATCH --mem= 5G
#SBATCH --time=0-10:00


module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index numpy
pip install --no-index torch
pip install --no-index torchvision
pip install --no-index scipy
pip install --no-index pandas
pip install  --no-index scikit-learn
pip install --no-index tqdm

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64    --epochs 1000    --lr 0.0001  --num_blocks 1 --hidden_size 2048  --select 8 --noise2 0.4 --MulObj