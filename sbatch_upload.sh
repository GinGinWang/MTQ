#!/bin/bash
#SBATCH --account=def-yaoliang
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:2
#SBATCH --exclusive
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=1-00:00

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip

# pip install --no-index numpy
# pip install --no-index torch_gpu
# pip install --no-index torchvision
# pip install --no-index scipy
# pip install --no-index pandas
# pip install  --no-index scikit-learn
# pip install --no-index tqdm
# pip install --no-index matplotlib
# pip install --no-index -r requirements.txt

python test.py  --autoencoder LSA --estimator SOS  --dataset mnist --batch_size 256 --code_length 64      --num_blocks 1 --hidden_size 2048  --select 8 --MulObj
